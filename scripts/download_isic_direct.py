#!/usr/bin/env python3
"""
Direct ISIC Dataset Downloader
Downloads melanoma datasets directly from ISIC Archive
"""

import os
import sys
import argparse
import requests
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import time
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ISICDirectDownloader:
    """Downloads ISIC datasets directly from their source"""
    
    def __init__(self, output_dir: str = "./datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ISIC dataset sources with direct download links
        self.isic_datasets = {
            "isic2019": {
                "name": "ISIC 2019 Challenge Dataset",
                "description": "25,000+ high-quality dermoscopic images",
                "images_url": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip",
                "labels_url": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv",
                "size_mb": 2500,
                "cost": "Free",
                "quality": "Medical-grade, professionally annotated",
                "license": "CC BY-NC-SA 4.0",
                "note": "Direct from ISIC S3"
            },
            "isic2020": {
                "name": "ISIC 2020 Challenge Dataset",
                "description": "33,000+ high-quality dermoscopic images", 
                "images_url": "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_Input.zip",
                "labels_url": "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth.csv",
                "size_mb": 3200,
                "cost": "Free",
                "quality": "Medical-grade, professionally annotated",
                "license": "CC BY-NC-SA 4.0",
                "note": "Direct from ISIC S3"
            },
            "isic2018": {
                "name": "ISIC 2018 Challenge Dataset",
                "description": "2,594 high-quality dermoscopic images",
                "images_url": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip",
                "labels_url": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.csv",
                "size_mb": 250,
                "cost": "Free",
                "quality": "Medical-grade, professionally annotated",
                "license": "CC BY-NC-SA 4.0",
                "note": "Direct from ISIC S3"
            }
        }
    
    def download_file_with_progress(self, url: str, filename: str, expected_size_mb: int = None):
        """Download a file with progress bar and retry logic"""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {filename} (Attempt {attempt + 1}/{max_retries})...")
                logger.info(f"URL: {url}")
                
                if expected_size_mb:
                    logger.info(f"Expected size: ~{expected_size_mb} MB")
                
                # Start download with streaming
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                
                with open(filename, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                        for chunk in response.iter_content(chunk_size=block_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                logger.info(f"Download completed: {filename}")
                return True
                
            except Exception as e:
                logger.error(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"All download attempts failed for {filename}")
                    return False
        
        return False
    
    def download_isic_dataset(self, dataset_key: str = "isic2019", max_images: int = None):
        """Download ISIC dataset directly"""
        if dataset_key not in self.isic_datasets:
            logger.error(f"Unsupported dataset: {dataset_key}")
            return False
        
        dataset_info = self.isic_datasets[dataset_key]
        logger.info(f"Downloading {dataset_info['name']}")
        logger.info(f"Description: {dataset_info['description']}")
        logger.info(f"License: {dataset_info['license']}")
        logger.info(f"Note: {dataset_info['note']}")
        
        # Create dataset directory
        dataset_dir = self.output_dir / f"{dataset_key}_real"
        dataset_dir.mkdir(exist_ok=True)
        
        # Download images
        images_zip = dataset_dir / f"{dataset_key}_images.zip"
        logger.info("Downloading images (this may take a while)...")
        
        if not self.download_file_with_progress(dataset_info['images_url'], images_zip, dataset_info['size_mb']):
            logger.error("Failed to download images")
            return False
        
        # Download labels
        labels_file = dataset_dir / f"{dataset_key}_labels.csv"
        logger.info("Downloading labels...")
        
        if not self.download_file_with_progress(dataset_info['labels_url'], labels_file):
            logger.error("Failed to download labels")
            return False
        
        # Extract images
        logger.info("Extracting images...")
        try:
            with zipfile.ZipFile(images_zip, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir / "images")
            logger.info("Images extracted successfully")
        except Exception as e:
            logger.error(f"Failed to extract images: {e}")
            return False
        
        # Clean up zip file
        images_zip.unlink()
        
        # Process and organize the data
        self.organize_isic_data(dataset_dir, dataset_key, max_images)
        
        logger.info(f"ISIC dataset setup completed: {dataset_dir}")
        return str(dataset_dir)
    
    def organize_isic_data(self, dataset_dir: Path, dataset_key: str, max_images: int = None):
        """Organize downloaded ISIC data into proper structure"""
        images_dir = dataset_dir / "images"
        labels_file = dataset_dir / f"{dataset_key}_labels.csv"
        
        # Read the original labels
        try:
            df = pd.read_csv(labels_file)
            logger.info(f"Original labels shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
        except Exception as e:
            logger.error(f"Failed to read labels: {e}")
            return False
        
        # Get list of image files
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        logger.info(f"Found {len(image_files)} image files")
        
        if len(image_files) == 0:
            logger.error("No image files found")
            return False
        
        # Limit images if requested
        if max_images and len(image_files) > max_images:
            np.random.seed(42)
            image_files = np.random.choice(image_files, max_images, replace=False)
            logger.info(f"Sampled {max_images} images from {len(image_files)} total")
        
        # Create organized CSV with proper structure
        organized_data = []
        
        for img_path in image_files:
            img_name = img_path.name
            
            # Find corresponding label in the CSV
            if 'image' in df.columns:
                # ISIC 2019/2020 format
                label_row = df[df['image'] == img_name]
            elif 'image_name' in df.columns:
                # Alternative format
                label_row = df[df['image_name'] == img_name]
            else:
                # Try to match by filename without extension
                base_name = img_path.stem
                label_row = df[df.iloc[:, 0] == base_name]
            
            if len(label_row) > 0:
                # Extract label information
                row = label_row.iloc[0]
                
                # Determine target (malignant vs benign)
                if 'MEL' in df.columns:
                    # ISIC 2019 format
                    target = 1 if row['MEL'] == 1 else 0
                    diagnosis = 'melanoma' if target == 1 else 'benign'
                elif 'melanoma' in df.columns:
                    # ISIC 2020 format
                    target = 1 if row['melanoma'] == 1 else 0
                    diagnosis = 'melanoma' if target == 1 else 'benign'
                else:
                    # Default to benign if can't determine
                    target = 0
                    diagnosis = 'benign'
                
                organized_data.append({
                    'image_name': img_name,
                    'target': target,
                    'patient_id': f"ISIC_{img_path.stem}",
                    'diagnosis': diagnosis,
                    'source': 'isic',
                    'dataset': dataset_key,
                    'original_path': str(img_path)
                })
            else:
                # If no label found, assume benign
                organized_data.append({
                    'image_name': img_name,
                    'target': 0,
                    'patient_id': f"ISIC_{img_path.stem}",
                    'diagnosis': 'benign',
                    'source': 'isic',
                    'dataset': dataset_key,
                    'original_path': str(img_path)
                })
        
        # Save organized CSV
        organized_df = pd.DataFrame(organized_data)
        organized_csv = dataset_dir / "labels.csv"
        organized_df.to_csv(organized_csv, index=False)
        
        logger.info(f"Organized data saved: {organized_csv}")
        logger.info(f"Total samples: {len(organized_data)}")
        logger.info(f"Benign: {len(organized_df[organized_df['target'] == 0])}")
        logger.info(f"Malignant: {len(organized_df[organized_df['target'] == 1])}")
        
        # Create data splits
        self.create_data_splits(organized_csv)
        
        return organized_csv
    
    def create_data_splits(self, csv_path: Path):
        """Create train/validation/test splits"""
        logger.info("Creating data splits...")
        
        df = pd.read_csv(csv_path)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate splits
        total_samples = len(df)
        train_end = int(total_samples * 0.8)
        val_end = train_end + int(total_samples * 0.1)
        
        # Split the data
        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]
        
        # Save splits
        csv_dir = csv_path.parent
        train_df.to_csv(csv_dir / "train_labels.csv", index=False)
        val_df.to_csv(csv_dir / "val_labels.csv", index=False)
        test_df.to_csv(csv_dir / "test_labels.csv", index=False)
        
        logger.info(f"Data splits created:")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Validation: {len(val_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")
    
    def list_available_datasets(self):
        """List all available ISIC datasets"""
        logger.info("Available ISIC Datasets (Direct Download):")
        logger.info("=" * 60)
        
        for key, info in self.isic_datasets.items():
            logger.info(f"\n🔗 {key.upper()}:")
            logger.info(f"  📋 Name: {info['name']}")
            logger.info(f"  📝 Description: {info['description']}")
            logger.info(f"  📊 Size: ~{info['size_mb']} MB")
            logger.info(f"  💰 Cost: {info['cost']}")
            logger.info(f"  ⭐ Quality: {info['quality']}")
            logger.info(f"  📄 License: {info['license']}")
            logger.info(f"  💡 Note: {info['note']}")

def main():
    parser = argparse.ArgumentParser(description='Download ISIC Datasets Directly')
    parser.add_argument('--dataset', type=str, choices=['isic2019', 'isic2020', 'isic2018'], 
                       default='isic2019', help='ISIC dataset to download')
    parser.add_argument('--output_dir', type=str, default='./datasets', help='Output directory')
    parser.add_argument('--max_images', type=int, help='Maximum number of images to download')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    
    args = parser.parse_args()
    
    downloader = ISICDirectDownloader(args.output_dir)
    
    if args.list:
        downloader.list_available_datasets()
        return
    
    logger.info(f"Starting direct download of {args.dataset}...")
    logger.info("This will download real medical images directly from ISIC!")
    
    dataset_dir = downloader.download_isic_dataset(args.dataset, args.max_images)
    
    if dataset_dir:
        print(f"\n✅ ISIC dataset downloaded successfully!")
        print(f"📁 Dataset directory: {dataset_dir}")
        print(f"🖼️  Images: {dataset_dir}/images")
        print(f"📊 Labels: {dataset_dir}/labels.csv")
        print(f"📈 Train/Val/Test splits: Created")
        print(f"\n🚀 Ready to train on REAL medical data! Run:")
        print(f"python scripts/train_melanoma_model.py --csv_file {dataset_dir}/labels.csv --img_dir {dataset_dir}/images --epochs 50 --convert_onnx")
    else:
        print("❌ Dataset download failed. Please check the logs above.")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Direct ISIC Dataset Downloader")
        print("\nUsage:")
        print("  python scripts/download_isic_direct.py --list")
        print("  python scripts/download_isic_direct.py --dataset isic2019")
        print("  python scripts/download_isic_direct.py --dataset isic2020 --max_images 5000")
        print("\nThis tool downloads REAL medical images directly from ISIC!")
        print("No Kaggle account required - direct from the source.")
    else:
        main()
