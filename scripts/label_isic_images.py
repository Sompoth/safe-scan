#!/usr/bin/env python3
"""
ISIC Image Labeling Script
Creates proper labels for ISIC images based on the metadata.csv file
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_isic_labels(metadata_path, images_dir, output_dir, min_samples_per_class=100):
    """
    Create labeled dataset from ISIC images based on metadata
    
    Args:
        metadata_path: Path to metadata.csv file
        images_dir: Directory containing ISIC images
        output_dir: Directory to save labeled dataset
        min_samples_per_class: Minimum samples per class to include
    """
    
    logger.info("Loading metadata...")
    df = pd.read_csv(metadata_path, low_memory=False)
    logger.info(f"Loaded {len(df)} records from metadata")
    
    # Check which images actually exist
    logger.info("Checking which images exist...")
    existing_images = []
    missing_images = []
    
    for idx, row in df.iterrows():
        image_name = f"{row['isic_id']}.jpg"
        image_path = os.path.join(images_dir, image_name)
        
        if os.path.exists(image_path):
            existing_images.append(idx)
        else:
            missing_images.append(idx)
    
    logger.info(f"Found {len(existing_images)} existing images, {len(missing_images)} missing")
    
    # Filter to existing images only
    df_existing = df.iloc[existing_images].copy()
    
    # Create class mapping
    logger.info("Creating class mapping...")
    
    # Define class mapping based on diagnosis_1
    class_mapping = {
        'Benign': 0,
        'Malignant': 1,
        'Indeterminate': 2  # We'll treat this as a separate class or exclude
    }
    
    # Add target column
    df_existing['target'] = df_existing['diagnosis_1'].map(class_mapping)
    
    # Check class distribution
    class_counts = df_existing['target'].value_counts()
    logger.info(f"Class distribution: {class_counts.to_dict()}")
    
    # Filter out classes with too few samples
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    df_filtered = df_existing[df_existing['target'].isin(valid_classes)].copy()
    
    logger.info(f"After filtering: {len(df_filtered)} images")
    logger.info(f"Final class distribution: {df_filtered['target'].value_counts().to_dict()}")
    
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create images subdirectory
    images_output_dir = output_path / "images"
    images_output_dir.mkdir(exist_ok=True)
    
    # Copy images and create labels
    logger.info("Copying images and creating labels...")
    
    # Create the main labels file
    labels_data = []
    
    for idx, row in df_filtered.iterrows():
        image_name = f"{row['isic_id']}.jpg"
        source_path = os.path.join(images_dir, image_name)
        target_path = images_output_dir / image_name
        
        # Copy image
        if os.path.exists(source_path):
            import shutil
            shutil.copy2(source_path, target_path)
            
            # Add to labels
            labels_data.append({
                'image_name': image_name,
                'target': row['target'],
                'patient_id': row['patient_id'],
                'age': row['age_approx'],
                'sex': row['sex'],
                'anatom_site': row['anatom_site_general'],
                'diagnosis': row['diagnosis_1'],
                'diagnosis_2': row['diagnosis_2'],
                'diagnosis_3': row['diagnosis_3'],
                'diagnosis_4': row['diagnosis_4'],
                'diagnosis_5': row['diagnosis_5']
            })
    
    # Create labels DataFrame
    labels_df = pd.DataFrame(labels_data)
    
    # Save labels
    labels_file = output_path / "labels.csv"
    labels_df.to_csv(labels_file, index=False)
    logger.info(f"Saved labels to {labels_file}")
    
    # Create train/val/test splits
    logger.info("Creating train/validation/test splits...")
    
    # Shuffle the data
    labels_df = labels_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split sizes
    total_samples = len(labels_df)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - val_size
    
    # Split the data
    train_df = labels_df[:train_size]
    val_df = labels_df[train_size:train_size + val_size]
    test_df = labels_df[train_size + val_size:]
    
    # Save splits
    train_df.to_csv(output_path / "train_labels.csv", index=False)
    val_df.to_csv(output_path / "val_labels.csv", index=False)
    test_df.to_csv(output_path / "test_labels.csv", index=False)
    
    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Print class distribution for each split
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        class_dist = split_df['target'].value_counts().sort_index()
        logger.info(f"{split_name} class distribution: {class_dist.to_dict()}")
    
    # Create a summary file
    summary = {
        'total_images': len(labels_df),
        'class_distribution': labels_df['target'].value_counts().to_dict(),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'class_mapping': {v: k for k, v in class_mapping.items()}
    }
    
    import json
    with open(output_path / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Dataset created successfully in {output_dir}")
    logger.info(f"Summary saved to {output_path / 'dataset_summary.json'}")
    
    return labels_df

def main():
    parser = argparse.ArgumentParser(description='Label ISIC images based on metadata')
    parser.add_argument('--metadata', default='datasets/ISIC-images/metadata.csv',
                       help='Path to metadata.csv file')
    parser.add_argument('--images_dir', default='datasets/ISIC-images',
                       help='Directory containing ISIC images')
    parser.add_argument('--output_dir', default='datasets/labeled_isic',
                       help='Output directory for labeled dataset')
    parser.add_argument('--min_samples', type=int, default=100,
                       help='Minimum samples per class')
    
    args = parser.parse_args()
    
    # Create labeled dataset
    labels_df = create_isic_labels(
        metadata_path=args.metadata,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        min_samples_per_class=args.min_samples
    )
    
    print(f"\n✅ Successfully created labeled dataset with {len(labels_df)} images")
    print(f"📁 Dataset saved to: {args.output_dir}")
    print(f"📊 Classes: {labels_df['target'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()
