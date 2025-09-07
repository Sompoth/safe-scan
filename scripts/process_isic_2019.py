#!/usr/bin/env python3
"""
Process ISIC 2019 dataset for melanoma detection training
This script prepares the real ISIC 2019 dataset for training with DenseNet121
"""

import pandas as pd
import os
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_isic_2019_dataset():
    """Process ISIC 2019 dataset and create proper labels for training"""
    
    # Paths
    base_dir = Path("datasets/melanoma_dataset")
    gt_file = base_dir / "ISIC_2019_Training_GroundTruth.csv"
    meta_file = base_dir / "ISIC_2019_Training_Metadata.csv"
    img_dir = base_dir / "images" / "ISIC_2019_Training_Input" / "ISIC_2019_Training_Input"
    
    # Check if files exist
    if not gt_file.exists():
        logger.error(f"Ground truth file not found: {gt_file}")
        return False
    
    if not meta_file.exists():
        logger.error(f"Metadata file not found: {meta_file}")
        return False
    
    if not img_dir.exists():
        logger.error(f"Image directory not found: {img_dir}")
        return False
    
    logger.info("Processing ISIC 2019 dataset...")
    
    # Read ground truth
    logger.info("Reading ground truth file...")
    gt_df = pd.read_csv(gt_file)
    logger.info(f"Ground truth shape: {gt_df.shape}")
    logger.info(f"Columns: {gt_df.columns.tolist()}")
    
    # Read metadata
    logger.info("Reading metadata file...")
    meta_df = pd.read_csv(meta_file)
    logger.info(f"Metadata shape: {meta_df.shape}")
    
    # Process ground truth - focus on melanoma vs non-melanoma
    # MEL column indicates melanoma (1) vs non-melanoma (0)
    logger.info("Processing labels...")
    
    # Create target column: 1 for melanoma, 0 for non-melanoma
    gt_df['target'] = (gt_df['MEL'] == 1).astype(int)
    
    # Add image extension if missing
    if not gt_df['image'].iloc[0].endswith('.jpg'):
        gt_df['image_name'] = gt_df['image'] + '.jpg'
    else:
        gt_df['image_name'] = gt_df['image']
    
    # Merge with metadata for additional features
    try:
        # Ensure metadata has correct image column
        if 'image' in meta_df.columns:
            if not meta_df['image'].iloc[0].endswith('.jpg'):
                meta_df['image_name'] = meta_df['image'] + '.jpg'
            else:
                meta_df['image_name'] = meta_df['image']
            
            # Select useful metadata columns
            meta_cols = ['image_name', 'age_approx', 'anatom_site_general', 'lesion_id', 'sex']
            meta_cols = [col for col in meta_cols if col in meta_df.columns]
            meta_subset = meta_df[meta_cols]
            
            # Merge ground truth with metadata
            final_df = gt_df.merge(meta_subset, on='image_name', how='left')
            logger.info(f"Merged with metadata, final shape: {final_df.shape}")
        else:
            final_df = gt_df.copy()
            logger.warning("No 'image' column found in metadata, using ground truth only")
    
    except Exception as e:
        logger.warning(f"Could not merge metadata: {e}")
        final_df = gt_df.copy()
    
    # Add required columns for training
    if 'patient_id' not in final_df.columns:
        final_df['patient_id'] = final_df['image_name'].str.replace('.jpg', '', regex=False)
    
    if 'diagnosis' not in final_df.columns:
        final_df['diagnosis'] = final_df['target'].map({0: 'benign', 1: 'melanoma'})
    
    # Reorder columns for consistency
    required_cols = ['image_name', 'target', 'patient_id', 'diagnosis']
    other_cols = [col for col in final_df.columns if col not in required_cols]
    final_df = final_df[required_cols + other_cols]
    
    # Save processed labels
    labels_path = base_dir / "labels.csv"
    final_df.to_csv(labels_path, index=False)
    logger.info(f"Labels saved to: {labels_path}")
    
    # Print dataset statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Total images: {len(final_df)}")
    logger.info(f"Melanoma cases: {final_df['target'].sum()}")
    logger.info(f"Benign cases: {len(final_df) - final_df['target'].sum()}")
    logger.info(f"Melanoma percentage: {100 * final_df['target'].mean():.2f}%")
    
    # Check for missing images
    logger.info("\nChecking for missing images...")
    missing_images = []
    for img_name in final_df['image_name'].head(10):  # Check first 10
        img_path = img_dir / img_name
        if not img_path.exists():
            missing_images.append(img_name)
    
    if missing_images:
        logger.warning(f"Missing images found: {len(missing_images)}")
        logger.warning(f"Examples: {missing_images[:3]}")
    else:
        logger.info("All sample images found!")
    
    # Create train/val/test split
    logger.info("\nCreating data splits...")
    from sklearn.model_selection import train_test_split
    
    # Stratified split to maintain class balance
    train_df, temp_df = train_test_split(
        final_df, 
        test_size=0.3, 
        random_state=42, 
        stratify=final_df['target']
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42, 
        stratify=temp_df['target']
    )
    
    # Save splits
    train_path = base_dir / "train_labels.csv"
    val_path = base_dir / "val_labels.csv"
    test_path = base_dir / "test_labels.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Train set: {len(train_df)} images")
    logger.info(f"Validation set: {len(val_df)} images")
    logger.info(f"Test set: {len(test_df)} images")
    
    logger.info("\nDataset processing complete!")
    logger.info(f"Use this command to train with DenseNet121:")
    logger.info(f"python scripts/train_melanoma_model.py --csv_file {labels_path} --img_dir {img_dir} --epochs 100 --batch_size 16 --convert_onnx")
    
    return True

if __name__ == "__main__":
    process_isic_2019_dataset()
