#!/usr/bin/env python3
"""
Real Melanoma Dataset Setup Script
Helps you set up real melanoma datasets from various sources
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import shutil
import requests
from PIL import Image, ImageDraw
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealDatasetSetup:
    """Helps set up real melanoma datasets"""
    
    def __init__(self, output_dir: str = "./datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset sources with current working links
        self.dataset_sources = {
            "kaggle_melanoma": {
                "name": "Kaggle Melanoma Classification",
                "url": "https://www.kaggle.com/c/siim-isic-melanoma-classification",
                "description": "Large melanoma dataset with 33K+ images",
                "size": "~3.2 GB",
                "cost": "Free",
                "quality": "High - Medical grade"
            },
            "kaggle_ham10000": {
                "name": "HAM10000 Skin Lesion Dataset",
                "url": "https://www.kaggle.com/datasets/fanconic/skin-cancer-mnist-ham10000",
                "description": "10K curated skin lesion images",
                "size": "~800 MB",
                "cost": "Free",
                "quality": "High - Well curated"
            },
            "isic_archive": {
                "name": "ISIC Archive",
                "url": "https://challenge.isic-archive.com/",
                "description": "Official ISIC challenge datasets",
                "size": "2-5 GB",
                "cost": "Free (registration required)",
                "quality": "Highest - Official medical dataset"
            },
            "dermnet": {
                "name": "DermNet",
                "url": "https://dermnetnz.org/",
                "description": "Professional dermatology image database",
                "size": "Varies",
                "cost": "Free for research",
                "quality": "High - Professional medical"
            }
        }
    
    def create_enhanced_sample_dataset(self, num_samples: int = 1000):
        """Create an enhanced sample dataset with more realistic synthetic images"""
        logger.info(f"Creating enhanced sample dataset with {num_samples} images...")
        
        sample_dir = self.output_dir / "enhanced_sample_melanoma"
        sample_dir.mkdir(exist_ok=True)
        images_dir = sample_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Create more realistic synthetic images
        for i in range(num_samples):
            # Create base image with skin-like background
            img = Image.new('RGB', (224, 224), color=(255, 235, 205))  # Skin tone
            
            # Add some noise and texture
            pixels = img.load()
            for x in range(224):
                for y in range(224):
                    # Add subtle noise
                    noise = random.randint(-20, 20)
                    r, g, b = pixels[x, y]
                    r = max(0, min(255, r + noise))
                    g = max(0, min(255, g + noise))
                    b = max(0, min(255, b + noise))
                    pixels[x, y] = (r, g, b)
            
            draw = ImageDraw.Draw(img)
            
            # Determine lesion type and characteristics
            if i < int(num_samples * 0.8):
                # Benign lesions (80%)
                target = 0
                lesion_types = ['nevus', 'seborrheic_keratosis', 'dermatofibroma']
                diagnosis = random.choice(lesion_types)
                
                if diagnosis == 'nevus':
                    # Brown, round, well-defined
                    color = (139, 69, 19)
                    shape = 'circle'
                elif diagnosis == 'seborrheic_keratosis':
                    # Brown-black, waxy, stuck-on appearance
                    color = (101, 67, 33)
                    shape = 'oval'
                else:  # dermatofibroma
                    # Pink-brown, firm
                    color = (160, 82, 45)
                    shape = 'circle'
            else:
                # Malignant lesions (20%)
                target = 1
                diagnosis = 'melanoma'
                # Asymmetric, irregular borders, multiple colors
                colors = [(128, 0, 0), (139, 69, 19), (0, 0, 0), (255, 0, 0)]
                color = random.choice(colors)
                shape = 'irregular'
            
            # Draw lesion based on type
            if shape == 'circle':
                # Benign: round, well-defined
                x1, y1 = 60, 60
                x2, y2 = 164, 164
                draw.ellipse([x1, y1, x2, y2], fill=color, outline='black', width=2)
            elif shape == 'oval':
                # Seborrheic keratosis: oval, waxy
                x1, y1 = 50, 70
                x2, y2 = 174, 154
                draw.ellipse([x1, y1, x2, y2], fill=color, outline='black', width=2)
            else:
                # Melanoma: irregular, asymmetric
                points = [
                    (70, 60), (90, 50), (120, 55), (150, 70),
                    (160, 100), (155, 130), (140, 150), (110, 155),
                    (80, 140), (60, 120), (65, 90)
                ]
                draw.polygon(points, fill=color, outline='black', width=2)
            
            # Add some texture and variation
            if target == 1:  # Malignant
                # Add irregular borders and color variation
                for _ in range(5):
                    x = random.randint(60, 160)
                    y = random.randint(60, 160)
                    size = random.randint(2, 8)
                    draw.ellipse([x, y, x+size, y+size], fill=color)
            
            # Save image
            img_path = images_dir / f"enhanced_{i:04d}.jpg"
            img.save(img_path, 'JPEG', quality=95)
        
        # Create comprehensive CSV
        csv_data = []
        for i in range(num_samples):
            if i < int(num_samples * 0.8):
                target = 0
                diagnosis = random.choice(['nevus', 'seborrheic_keratosis', 'dermatofibroma'])
                confidence = random.uniform(0.7, 0.95)  # High confidence for benign
            else:
                target = 1
                diagnosis = 'melanoma'
                confidence = random.uniform(0.6, 0.9)  # Lower confidence for malignant
            
            csv_data.append({
                'image_name': f"enhanced_{i:04d}.jpg",
                'target': target,
                'patient_id': f"enhanced_{i:04d}",
                'age': random.randint(20, 80),
                'sex': random.choice(['male', 'female']),
                'anatom_site': random.choice(['head/neck', 'upper extremity', 'lower extremity', 'torso']),
                'diagnosis': diagnosis,
                'confidence': round(confidence, 3),
                'lesion_size_mm': random.randint(3, 25),
                'asymmetry': random.choice(['symmetric', 'asymmetric']) if target == 0 else 'asymmetric',
                'border': random.choice(['regular', 'irregular']) if target == 0 else 'irregular',
                'color': random.choice(['uniform', 'variegated']) if target == 0 else 'variegated',
                'diameter': random.choice(['<6mm', '>=6mm'])
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = sample_dir / "labels.csv"
        df.to_csv(csv_path, index=False)
        
        # Create data splits
        self.create_data_splits(csv_path)
        
        logger.info(f"Enhanced sample dataset created at: {sample_dir}")
        logger.info(f"Images: {images_dir}")
        logger.info(f"CSV: {csv_path}")
        
        return str(sample_dir), str(csv_path)
    
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
    
    def setup_kaggle_instructions(self):
        """Provide instructions for setting up Kaggle datasets"""
        instructions = """
        🚀 SETTING UP KAGGLE DATASETS
        
        Step 1: Install Kaggle CLI
        pip install kaggle
        
        Step 2: Get your Kaggle API credentials
        1. Go to https://www.kaggle.com/account
        2. Click "Create New API Token"
        3. Download kaggle.json
        4. Place it in ~/.kaggle/kaggle.json (Linux/Mac) or C:\\Users\\USERNAME\\.kaggle\\kaggle.json (Windows)
        
        Step 3: Download datasets
        
        # For Melanoma Classification (33K+ images)
        kaggle competitions download -c siim-isic-melanoma-classification
        
        # For HAM10000 (10K images)
        kaggle datasets download -d fanconic/skin-cancer-mnist-ham10000
        
        Step 4: Extract and organize
        unzip siim-isic-melanoma-classification.zip
        unzip skin-cancer-mnist-ham10000.zip
        
        Step 5: Use our scripts to organize the data
        python scripts/setup_real_dataset.py --organize_kaggle --input_dir ./extracted_data
        """
        
        return instructions
    
    def organize_kaggle_data(self, input_dir: str):
        """Organize downloaded Kaggle data into our structure"""
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return False
        
        # Create organized dataset
        organized_dir = self.output_dir / "kaggle_organized"
        organized_dir.mkdir(exist_ok=True)
        images_dir = organized_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Find image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(input_path.rglob(ext))
        
        logger.info(f"Found {len(image_files)} image files")
        
        if len(image_files) == 0:
            logger.error("No image files found. Please check the input directory.")
            return False
        
        # Copy images and create CSV
        csv_data = []
        for i, img_path in enumerate(image_files):
            # Copy image
            new_name = f"kaggle_{i:06d}{img_path.suffix}"
            shutil.copy2(img_path, images_dir / new_name)
            
            # Create label entry (you'll need to update this with real labels)
            if i < int(len(image_files) * 0.8):
                target = 0  # Assume benign for now
                diagnosis = 'nevus'
            else:
                target = 1  # Assume malignant for now
                diagnosis = 'melanoma'
            
            csv_data.append({
                'image_name': new_name,
                'target': target,
                'patient_id': f"kaggle_{i:06d}",
                'age': random.randint(20, 80),
                'sex': random.choice(['male', 'female']),
                'anatom_site': random.choice(['head/neck', 'upper extremity', 'lower extremity', 'torso']),
                'diagnosis': diagnosis,
                'source': 'kaggle',
                'original_path': str(img_path)
            })
        
        # Save CSV
        df = pd.DataFrame(csv_data)
        csv_path = organized_dir / "labels.csv"
        df.to_csv(csv_path, index=False)
        
        # Create splits
        self.create_data_splits(csv_path)
        
        logger.info(f"Kaggle data organized at: {organized_dir}")
        logger.info(f"⚠️  IMPORTANT: Update the CSV file with real labels from the dataset!")
        
        return str(organized_dir), str(csv_path)
    
    def list_dataset_sources(self):
        """List all available dataset sources"""
        logger.info("Available Real Dataset Sources:")
        logger.info("=" * 60)
        
        for key, info in self.dataset_sources.items():
            logger.info(f"\n🔗 {key.upper()}:")
            logger.info(f"  📋 Name: {info['name']}")
            logger.info(f"  📝 Description: {info['description']}")
            logger.info(f"  📊 Size: {info['size']}")
            logger.info(f"  💰 Cost: {info['cost']}")
            logger.info(f"  ⭐ Quality: {info['quality']}")
            logger.info(f"  🌐 URL: {info['url']}")
    
    def create_training_ready_dataset(self, dataset_type: str = "enhanced"):
        """Create a dataset ready for immediate training"""
        if dataset_type == "enhanced":
            dataset_dir, csv_path = self.create_enhanced_sample_dataset(1000)
        else:
            dataset_dir, csv_path = self.create_enhanced_sample_dataset(500)
        
        print(f"\n✅ Training-ready dataset created!")
        print(f"📁 Dataset directory: {dataset_dir}")
        print(f"📊 CSV file: {csv_path}")
        print(f"🖼️  Images directory: {dataset_dir}/images")
        print(f"📈 Train/Val/Test splits: Created")
        
        print(f"\n🚀 Ready to train! Run:")
        print(f"python scripts/train_melanoma_model.py --csv_file {csv_path} --img_dir {dataset_dir}/images --epochs 10 --convert_onnx")
        
        return dataset_dir, csv_path

def main():
    parser = argparse.ArgumentParser(description='Setup Real Melanoma Datasets')
    parser.add_argument('--action', type=str, choices=['enhanced', 'kaggle_instructions', 'organize_kaggle', 'list_sources', 'create_ready'], 
                       default='create_ready', help='Action to perform')
    parser.add_argument('--input_dir', type=str, help='Input directory for organizing Kaggle data')
    parser.add_argument('--dataset_type', type=str, choices=['enhanced', 'standard'], default='enhanced', help='Type of dataset to create')
    
    args = parser.parse_args()
    
    setup = RealDatasetSetup()
    
    if args.action == 'enhanced':
        dataset_dir, csv_path = setup.create_enhanced_sample_dataset(1000)
        print(f"\n✅ Enhanced dataset created successfully!")
        print(f"📁 Dataset directory: {dataset_dir}")
        print(f"📊 CSV file: {csv_path}")
    
    elif args.action == 'kaggle_instructions':
        instructions = setup.setup_kaggle_instructions()
        print(instructions)
    
    elif args.action == 'organize_kaggle':
        if not args.input_dir:
            print("Error: --input_dir is required for organizing Kaggle data")
            return
        result = setup.organize_kaggle_data(args.input_dir)
        if result:
            dataset_dir, csv_path = result
            print(f"\n✅ Kaggle data organized successfully!")
            print(f"📁 Dataset directory: {dataset_dir}")
            print(f"📊 CSV file: {csv_path}")
    
    elif args.action == 'list_sources':
        setup.list_dataset_sources()
    
    elif args.action == 'create_ready':
        dataset_dir, csv_path = setup.create_training_ready_dataset(args.dataset_type)
    
    else:
        print("Invalid action. Use --help to see available options.")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Real Melanoma Dataset Setup Tool")
        print("\nUsage:")
        print("  python scripts/setup_real_dataset.py --action create_ready")
        print("  python scripts/setup_real_dataset.py --action enhanced")
        print("  python scripts/setup_real_dataset.py --action kaggle_instructions")
        print("  python scripts/setup_real_dataset.py --action list_sources")
        print("  python scripts/setup_real_dataset.py --action organize_kaggle --input_dir ./downloaded_data")
        print("\nThis tool helps you:")
        print("1. Create enhanced synthetic datasets for testing")
        print("2. Get instructions for downloading real datasets")
        print("3. Organize downloaded Kaggle datasets")
        print("4. Create training-ready dataset structures")
    else:
        main()
