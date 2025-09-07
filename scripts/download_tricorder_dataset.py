#!/usr/bin/env python3
"""
Download and prepare Tricorder competition training dataset
Downloads ISIC datasets and converts them to 10-class format
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import requests
import zipfile
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TricorderDatasetDownloader:
    """Download and prepare datasets for Tricorder competition"""
    
    def __init__(self, output_dir: str = "datasets/tricorder_training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tricorder class mapping
        self.tricorder_classes = {
            "AK": 0,   # Actinic keratosis
            "BCC": 1,  # Basal cell carcinoma
            "SK": 2,   # Seborrheic keratosis
            "SCC": 3,  # Squamous cell carcinoma
            "VASC": 4, # Vascular lesion
            "DF": 5,   # Dermatofibroma
            "NV": 6,   # Benign nevus
            "NON": 7,  # Other non-neoplastic
            "MEL": 8,  # Melanoma
            "ON": 9    # Other neoplastic
        }
        
        # ISIC class mapping to Tricorder classes
        self.isic_to_tricorder = {
            "AK": "AK",
            "BCC": "BCC", 
            "SK": "SK",
            "SCC": "SCC",
            "VASC": "VASC",
            "DF": "DF",
            "NV": "NV",
            "MEL": "MEL",
            "BKL": "SK",  # Benign keratosis-like lesions -> SK
            "UNK": "NON"  # Unknown -> Other non-neoplastic
        }
    
    def download_isic_2019(self) -> bool:
        """Download ISIC 2019 dataset"""
        logger.info("Downloading ISIC 2019 dataset...")
        
        # ISIC 2019 download URLs (these may need to be updated)
        urls = {
            "images": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip",
            "ground_truth": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv",
            "metadata": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv"
        }
        
        dataset_dir = self.output_dir / "isic_2019"
        dataset_dir.mkdir(exist_ok=True)
        
        try:
            # Download ground truth
            logger.info("Downloading ground truth...")
            gt_response = requests.get(urls["ground_truth"])
            gt_path = dataset_dir / "ISIC_2019_Training_GroundTruth.csv"
            with open(gt_path, 'wb') as f:
                f.write(gt_response.content)
            
            # Download metadata
            logger.info("Downloading metadata...")
            meta_response = requests.get(urls["metadata"])
            meta_path = dataset_dir / "ISIC_2019_Training_Metadata.csv"
            with open(meta_path, 'wb') as f:
                f.write(meta_response.content)
            
            # Download images (this is a large file)
            logger.info("Downloading images (this may take a while)...")
            images_response = requests.get(urls["images"], stream=True)
            images_path = dataset_dir / "ISIC_2019_Training_Input.zip"
            
            with open(images_path, 'wb') as f:
                for chunk in images_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract images
            logger.info("Extracting images...")
            with zipfile.ZipFile(images_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            
            logger.info("ISIC 2019 dataset downloaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading ISIC 2019: {e}")
            return False
    
    def download_ham10000(self) -> bool:
        """Download HAM10000 dataset from Kaggle"""
        logger.info("HAM10000 dataset download instructions:")
        logger.info("1. Go to: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        logger.info("2. Download the dataset (requires Kaggle account)")
        logger.info("3. Extract to: datasets/tricorder_training/ham10000/")
        logger.info("4. Run this script again with --prepare flag")
        return False
    
    def process_isic_2019(self) -> bool:
        """Process ISIC 2019 dataset for Tricorder format"""
        logger.info("Processing ISIC 2019 dataset for Tricorder format...")
        
        # Check both possible locations
        dataset_dir = self.output_dir / "isic_2019"
        existing_dataset_dir = Path("datasets/melanoma_dataset")
        
        if existing_dataset_dir.exists():
            # Use existing dataset
            gt_path = existing_dataset_dir / "ISIC_2019_Training_GroundTruth.csv"
            meta_path = existing_dataset_dir / "ISIC_2019_Training_Metadata.csv"
            images_dir = existing_dataset_dir / "images" / "ISIC_2019_Training_Input" / "ISIC_2019_Training_Input"
            logger.info(f"Using existing dataset at: {existing_dataset_dir}")
        else:
            # Use downloaded dataset
            gt_path = dataset_dir / "ISIC_2019_Training_GroundTruth.csv"
            meta_path = dataset_dir / "ISIC_2019_Training_Metadata.csv"
            images_dir = dataset_dir / "ISIC_2019_Training_Input"
        
        if not all([gt_path.exists(), meta_path.exists(), images_dir.exists()]):
            logger.error("ISIC 2019 dataset not found. Please download it first.")
            return False
        
        # Read ground truth
        gt_df = pd.read_csv(gt_path)
        logger.info(f"Ground truth shape: {gt_df.shape}")
        logger.info(f"Columns: {gt_df.columns.tolist()}")
        
        # Read metadata
        meta_df = pd.read_csv(meta_path)
        logger.info(f"Metadata shape: {meta_df.shape}")
        
        # Process ground truth - convert to Tricorder format
        # ISIC 2019 has: MEL, NV, BCC, AK, BKL, DF, VASC
        # We need to map to: AK, BCC, SK, SCC, VASC, DF, NV, NON, MEL, ON
        
        processed_data = []
        
        for idx, row in gt_df.iterrows():
            image_name = row['image']
            
            # Find the class with value 1
            class_found = None
            for col in gt_df.columns:
                if col != 'image' and row[col] == 1:
                    class_found = col
                    break
            
            if class_found is None:
                continue
            
            # Map ISIC class to Tricorder class
            tricorder_class = self.isic_to_tricorder.get(class_found, "NON")
            
            # Get metadata for this image
            meta_row = meta_df[meta_df['image'] == image_name]
            if len(meta_row) > 0:
                age = meta_row.iloc[0].get('age_approx', 50)
                sex = meta_row.iloc[0].get('sex', 'unknown')
                anatom_site = meta_row.iloc[0].get('anatom_site_general', 'torso')
            else:
                age = 50
                sex = 'unknown'
                anatom_site = 'torso'
            
            # Handle NaN values
            if pd.isna(age):
                age = 50
            if pd.isna(sex):
                sex = 'unknown'
            if pd.isna(anatom_site):
                anatom_site = 'torso'
            
            # Map anatomical site to location number
            location_map = {
                'head/neck': 5,
                'upper extremity': 1,
                'lower extremity': 6,
                'torso': 7,
                'palms/soles': 2,
                'oral/genital': 3,
                'anterior torso': 7,
                'posterior torso': 7,
                'lateral torso': 7,
                'upper extremity': 1,
                'lower extremity': 6
            }
            location = location_map.get(str(anatom_site).lower(), 7)
            
            processed_data.append({
                'image_name': f"{image_name}.jpg",
                'class': tricorder_class,
                'class_id': self.tricorder_classes[tricorder_class],
                'age': age,
                'sex': sex,
                'location': location,
                'anatom_site': anatom_site
            })
        
        # Create processed dataset
        processed_df = pd.DataFrame(processed_data)
        
        # Save processed dataset
        processed_path = self.output_dir / "tricorder_processed.csv"
        processed_df.to_csv(processed_path, index=False)
        
        logger.info(f"Processed dataset saved to: {processed_path}")
        logger.info(f"Total images: {len(processed_df)}")
        
        # Print class distribution
        class_counts = processed_df['class'].value_counts()
        logger.info("\nClass distribution:")
        for class_name, count in class_counts.items():
            logger.info(f"  {class_name}: {count} images")
        
        return True
    
    def create_tricorder_dataset(self) -> bool:
        """Create final Tricorder training dataset"""
        logger.info("Creating final Tricorder training dataset...")
        
        processed_path = self.output_dir / "tricorder_processed.csv"
        if not processed_path.exists():
            logger.error("Processed dataset not found. Please run processing first.")
            return False
        
        df = pd.read_csv(processed_path)
        
        # Create train/val/test splits
        from sklearn.model_selection import train_test_split
        
        # Stratified split to maintain class balance
        train_df, temp_df = train_test_split(
            df, 
            test_size=0.3, 
            random_state=42, 
            stratify=df['class_id']
        )
        
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=0.5, 
            random_state=42, 
            stratify=temp_df['class_id']
        )
        
        # Save splits
        train_path = self.output_dir / "train_labels.csv"
        val_path = self.output_dir / "val_labels.csv"
        test_path = self.output_dir / "test_labels.csv"
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Train set: {len(train_df)} images")
        logger.info(f"Validation set: {len(val_df)} images")
        logger.info(f"Test set: {len(test_df)} images")
        
        # Create dataset info
        info = {
            "dataset_name": "Tricorder Training Dataset",
            "total_images": len(df),
            "train_images": len(train_df),
            "val_images": len(val_df),
            "test_images": len(test_df),
            "classes": list(self.tricorder_classes.keys()),
            "class_distribution": df['class'].value_counts().to_dict(),
            "image_directory": "datasets/melanoma_dataset/images/ISIC_2019_Training_Input/ISIC_2019_Training_Input",
            "format": "512x512 RGB images with demographic data"
        }
        
        import json
        with open(self.output_dir / "dataset_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info("Tricorder training dataset created successfully!")
        logger.info(f"Dataset location: {self.output_dir}")
        
        return True
    
    def print_download_instructions(self):
        """Print detailed download instructions"""
        print("\n" + "="*60)
        print("📥 TRICORDER TRAINING DATASET DOWNLOAD INSTRUCTIONS")
        print("="*60)
        
        print("\n🏆 RECOMMENDED: ISIC 2019 Dataset")
        print("-" * 40)
        print("1. Go to: https://challenge.isic-archive.com/")
        print("2. Register for a free account")
        print("3. Navigate to 'ISIC 2019: Lesion Boundary Segmentation'")
        print("4. Download these files:")
        print("   - ISIC_2019_Training_Input.zip (images)")
        print("   - ISIC_2019_Training_GroundTruth.csv (labels)")
        print("   - ISIC_2019_Training_Metadata.csv (demographics)")
        print("5. Extract to: datasets/tricorder_training/isic_2019/")
        
        print("\n🎯 ALTERNATIVE: HAM10000 Dataset")
        print("-" * 40)
        print("1. Go to: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print("2. Download the dataset (requires Kaggle account)")
        print("3. Extract to: datasets/tricorder_training/ham10000/")
        
        print("\n🔬 ADDITIONAL SOURCES")
        print("-" * 40)
        print("• DermNet: https://www.dermnet.com/")
        print("• PH2 Dataset: Available through research institutions")
        print("• ISIC 2020: https://challenge.isic-archive.com/")
        
        print("\n⚡ QUICK START")
        print("-" * 40)
        print("After downloading ISIC 2019:")
        print("python scripts/download_tricorder_dataset.py --download-isic")
        print("python scripts/download_tricorder_dataset.py --process")
        print("python scripts/download_tricorder_dataset.py --create")
        
        print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Download and prepare Tricorder training dataset')
    parser.add_argument('--output-dir', type=str, default='datasets/tricorder_training',
                        help='Output directory for dataset')
    parser.add_argument('--download-isic', action='store_true',
                        help='Download ISIC 2019 dataset')
    parser.add_argument('--download-ham', action='store_true',
                        help='Show HAM10000 download instructions')
    parser.add_argument('--process', action='store_true',
                        help='Process ISIC 2019 dataset for Tricorder format')
    parser.add_argument('--create', action='store_true',
                        help='Create final Tricorder training dataset')
    parser.add_argument('--instructions', action='store_true',
                        help='Show download instructions')
    
    args = parser.parse_args()
    
    downloader = TricorderDatasetDownloader(args.output_dir)
    
    if args.instructions:
        downloader.print_download_instructions()
    elif args.download_isic:
        downloader.download_isic_2019()
    elif args.download_ham:
        downloader.download_ham10000()
    elif args.process:
        downloader.process_isic_2019()
    elif args.create:
        downloader.create_tricorder_dataset()
    else:
        # Show instructions by default
        downloader.print_download_instructions()

if __name__ == "__main__":
    main()
