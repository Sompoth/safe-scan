#!/usr/bin/env python3
"""
Dataset Preparation Script for Melanoma Detection
This script helps prepare melanoma datasets for training the model.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple
import zipfile
import requests
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetPreparator:
    """Class to prepare melanoma datasets for training"""
    
    def __init__(self, output_dir: str = "./datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Common melanoma dataset sources
        self.dataset_sources = {
            "isic2019": {
                "url": "https://storage.googleapis.com/cloud-ai-data/ISIC_2019_Training_Input.zip",
                "description": "ISIC 2019 Challenge Dataset (25K+ images)",
                "license": "CC BY-NC-SA 4.0"
            },
            "isic2020": {
                "url": "https://storage.googleapis.com/cloud-ai-data/ISIC_2020_Training_Input.zip", 
                "description": "ISIC 2020 Challenge Dataset (33K+ images)",
                "license": "CC BY-NC-SA 4.0"
            },
            "ham10000": {
                "url": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T",
                "description": "HAM10000 Dataset (10K+ images)",
                "license": "CC BY-NC-SA 4.0"
            }
        }
    
    def create_sample_dataset(self, num_samples: int = 100):
        """Create a small sample dataset for testing purposes"""
        logger.info(f"Creating sample dataset with {num_samples} images...")
        
        sample_dir = self.output_dir / "sample_melanoma"
        sample_dir.mkdir(exist_ok=True)
        
        # Create sample CSV structure
        sample_data = []
        for i in range(num_samples):
            # Simulate realistic data distribution (80% benign, 20% malignant)
            if i < int(num_samples * 0.8):
                target = 0  # Benign
                image_name = f"benign_{i:03d}.jpg"
            else:
                target = 1  # Malignant
                image_name = f"malignant_{i:03d}.jpg"
            
            sample_data.append({
                'image_name': image_name,
                'target': target,
                'patient_id': f"patient_{i:03d}",
                'age': np.random.randint(20, 80),
                'sex': np.random.choice(['male', 'female']),
                'anatom_site': np.random.choice(['head/neck', 'upper extremity', 'lower extremity', 'torso', 'palms/soles']),
                'diagnosis': 'benign' if target == 0 else 'malignant melanoma'
            })
        
        # Save CSV
        df = pd.DataFrame(sample_data)
        csv_path = sample_dir / "sample_labels.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Sample dataset created at: {sample_dir}")
        logger.info(f"CSV file: {csv_path}")
        logger.info("Note: You need to add actual images to the 'images' subdirectory")
        
        return str(sample_dir), str(csv_path)
    
    def download_isic_sample(self, num_images: int = 1000):
        """Download a sample of ISIC images (this is a simplified version)"""
        logger.info("Note: Full ISIC dataset download requires registration")
        logger.info("Visit: https://challenge.isic-archive.com/")
        
        # Create directory structure
        isic_dir = self.output_dir / "isic_sample"
        isic_dir.mkdir(exist_ok=True)
        
        # Create sample CSV structure
        sample_data = []
        for i in range(num_images):
            # Simulate ISIC data structure
            if i < int(num_images * 0.8):
                target = 0  # Benign
                diagnosis = np.random.choice(['nevus', 'seborrheic keratosis', 'dermatofibroma'])
            else:
                target = 1  # Malignant
                diagnosis = 'melanoma'
            
            sample_data.append({
                'image_name': f"ISIC_{i:07d}.jpg",
                'target': target,
                'diagnosis': diagnosis,
                'patient_id': f"ISIC_{i:07d}",
                'age': np.random.randint(20, 80),
                'sex': np.random.choice(['male', 'female']),
                'anatom_site': np.random.choice(['head/neck', 'upper extremity', 'lower extremity', 'torso']),
                'benign_malignant': 'benign' if target == 0 else 'malignant'
            })
        
        # Save CSV
        df = pd.DataFrame(sample_data)
        csv_path = isic_dir / "isic_labels.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"ISIC sample structure created at: {isic_dir}")
        logger.info(f"CSV file: {csv_path}")
        
        return str(isic_dir), str(csv_path)
    
    def create_data_splits(self, csv_path: str, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """Create train/validation/test splits from a CSV file"""
        logger.info("Creating data splits...")
        
        # Read the CSV
        df = pd.read_csv(csv_path)
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate split indices
        total_samples = len(df)
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)
        
        # Split the data
        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]
        
        # Save splits
        csv_dir = Path(csv_path).parent
        train_df.to_csv(csv_dir / "train_labels.csv", index=False)
        val_df.to_csv(csv_dir / "val_labels.csv", index=False)
        test_df.to_csv(csv_dir / "test_labels.csv", index=False)
        
        logger.info(f"Data splits created:")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Validation: {len(val_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")
        
        return {
            'train': str(csv_dir / "train_labels.csv"),
            'val': str(csv_dir / "val_labels.csv"),
            'test': str(csv_dir / "test_labels.csv")
        }
    
    def validate_dataset(self, csv_path: str, img_dir: str):
        """Validate that all images referenced in CSV exist"""
        logger.info("Validating dataset...")
        
        df = pd.read_csv(csv_path)
        img_dir_path = Path(img_dir)
        
        missing_images = []
        valid_images = []
        
        for idx, row in df.iterrows():
            img_name = row['image_name']
            img_path = img_dir_path / img_name
            
            if img_path.exists():
                valid_images.append(img_name)
            else:
                missing_images.append(img_name)
        
        logger.info(f"Dataset validation complete:")
        logger.info(f"  Total images in CSV: {len(df)}")
        logger.info(f"  Valid images found: {len(valid_images)}")
        logger.info(f"  Missing images: {len(missing_images)}")
        
        if missing_images:
            logger.warning(f"Missing images: {missing_images[:5]}...")
        
        return len(valid_images), len(missing_images)
    
    def create_training_config(self, dataset_path: str, output_path: str = None):
        """Create a training configuration file"""
        if output_path is None:
            output_path = Path(dataset_path) / "training_config.yaml"
        
        config_content = f"""# Training Configuration for Melanoma Detection
# Generated automatically by dataset preparation script

dataset:
  train_csv: {Path(dataset_path) / "train_labels.csv"}
  val_csv: {Path(dataset_path) / "val_labels.csv"}
  test_csv: {Path(dataset_path) / "test_labels.csv"}
  image_dir: {Path(dataset_path) / "images"}
  
model:
  architecture: "resnet50"
  num_classes: 2
  pretrained: true
  
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.0001
  scheduler: "ReduceLROnPlateau"
  
data_augmentation:
  resize: [224, 224]
  horizontal_flip: true
  rotation: 15
  color_jitter: true
  
output:
  model_dir: "./models"
  save_best: true
  save_checkpoints: true
"""
        
        with open(output_path, 'w') as f:
            f.write(config_content)
        
        logger.info(f"Training configuration saved to: {output_path}")
        return str(output_path)

def main():
    parser = argparse.ArgumentParser(description='Prepare Melanoma Dataset')
    parser.add_argument('--action', type=str, choices=['sample', 'isic', 'validate', 'splits', 'config'], 
                       default='sample', help='Action to perform')
    parser.add_argument('--output_dir', type=str, default='./datasets', help='Output directory')
    parser.add_argument('--csv_file', type=str, help='Path to CSV file for validation/splits')
    parser.add_argument('--img_dir', type=str, help='Path to image directory for validation')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples for sample dataset')
    
    args = parser.parse_args()
    
    preparator = DatasetPreparator(args.output_dir)
    
    if args.action == 'sample':
        dataset_dir, csv_path = preparator.create_sample_dataset(args.num_samples)
        print(f"\nSample dataset created!")
        print(f"Dataset directory: {dataset_dir}")
        print(f"CSV file: {csv_path}")
        print("\nNext steps:")
        print("1. Add actual images to the 'images' subdirectory")
        print("2. Update the CSV file with correct image names")
        print("3. Run: python scripts/train_melanoma_model.py --csv_file <csv_path> --img_dir <img_dir> --epochs 10 --convert_onnx")
    
    elif args.action == 'isic':
        dataset_dir, csv_path = preparator.download_isic_sample()
        print(f"\nISIC sample structure created!")
        print(f"Dataset directory: {dataset_dir}")
        print(f"CSV file: {csv_path}")
        print("\nNote: You need to download actual ISIC images from the official website")
    
    elif args.action == 'validate':
        if not args.csv_file or not args.img_dir:
            print("Error: --csv_file and --img_dir are required for validation")
            return
        valid, missing = preparator.validate_dataset(args.csv_file, args.img_dir)
    
    elif args.action == 'splits':
        if not args.csv_file:
            print("Error: --csv_file is required for creating splits")
            return
        splits = preparator.create_data_splits(args.csv_file)
        print(f"\nData splits created:")
        for split_name, split_path in splits.items():
            print(f"  {split_name}: {split_path}")
    
    elif args.action == 'config':
        if not args.csv_file:
            print("Error: --csv_file is required for creating config")
            return
        dataset_path = Path(args.csv_file).parent
        config_path = preparator.create_training_config(str(dataset_path))
        print(f"\nTraining configuration created: {config_path}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Melanoma Dataset Preparation Tool")
        print("\nAvailable actions:")
        print("  sample   - Create a sample dataset structure")
        print("  isic     - Create ISIC dataset structure")
        print("  validate - Validate existing dataset")
        print("  splits   - Create train/val/test splits")
        print("  config   - Create training configuration")
        print("\nExamples:")
        print("  python scripts/prepare_dataset.py --action sample --num_samples 200")
        print("  python scripts/prepare_dataset.py --action validate --csv_file data/labels.csv --img_dir data/images")
        print("  python scripts/prepare_dataset.py --action splits --csv_file data/labels.csv")
    else:
        main()
