#!/usr/bin/env python3
"""
Class Balancing Strategies for Tricorder Dataset
Handles severe class imbalance in skin lesion classification
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ClassBalancer:
    """Handles class imbalance in tricorder dataset"""
    
    def __init__(self, csv_file: str, img_dir: str):
        self.csv_file = csv_file
        self.img_dir = Path(img_dir)
        self.data = pd.read_csv(csv_file)
        self.class_counts = self.data['class'].value_counts()
        self.class_weights = None
        
    def analyze_imbalance(self):
        """Analyze and visualize class imbalance"""
        print("=== CLASS IMBALANCE ANALYSIS ===")
        print(f"Total samples: {len(self.data)}")
        print(f"Number of classes: {len(self.class_counts)}")
        print(f"Imbalance ratio (max/min): {self.class_counts.max() / self.class_counts.min():.1f}:1")
        print("\nClass distribution:")
        for class_name, count in self.class_counts.items():
            percentage = (count / len(self.data)) * 100
            print(f"  {class_name}: {count:,} ({percentage:.1f}%)")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Bar plot
        plt.subplot(2, 2, 1)
        self.class_counts.plot(kind='bar')
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Log scale
        plt.subplot(2, 2, 2)
        self.class_counts.plot(kind='bar', logy=True)
        plt.title('Class Distribution (Log Scale)')
        plt.xlabel('Class')
        plt.ylabel('Count (Log)')
        plt.xticks(rotation=45)
        
        # Pie chart
        plt.subplot(2, 2, 3)
        self.class_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title('Class Distribution (%)')
        plt.ylabel('')
        
        # Imbalance ratio
        plt.subplot(2, 2, 4)
        ratios = self.class_counts.max() / self.class_counts
        ratios.plot(kind='bar')
        plt.title('Imbalance Ratio (vs Max Class)')
        plt.xlabel('Class')
        plt.ylabel('Ratio')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('class_imbalance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def compute_class_weights(self, method='balanced'):
        """Compute class weights for weighted loss"""
        classes = np.unique(self.data['class'])
        class_weights = compute_class_weight(
            class_weight=method,
            classes=classes,
            y=self.data['class']
        )
        
        self.class_weights = dict(zip(classes, class_weights))
        
        print(f"\n=== CLASS WEIGHTS ({method.upper()}) ===")
        for class_name, weight in self.class_weights.items():
            print(f"  {class_name}: {weight:.3f}")
            
        return self.class_weights
    
    def create_balanced_splits(self, target_samples_per_class=1000, random_state=42):
        """Create balanced train/val/test splits"""
        np.random.seed(random_state)
        
        balanced_data = []
        
        for class_name in self.class_counts.index:
            class_data = self.data[self.data['class'] == class_name]
            
            # Sample up to target_samples_per_class
            if len(class_data) >= target_samples_per_class:
                sampled = class_data.sample(n=target_samples_per_class, random_state=random_state)
            else:
                # If class has fewer samples, use all and oversample
                sampled = class_data
                # Add oversampled data
                oversample_needed = target_samples_per_class - len(class_data)
                oversampled = class_data.sample(n=oversample_needed, replace=True, random_state=random_state)
                sampled = pd.concat([sampled, oversampled])
            
            balanced_data.append(sampled)
        
        balanced_df = pd.concat(balanced_data, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Create splits (70% train, 15% val, 15% test)
        train_size = int(0.7 * len(balanced_df))
        val_size = int(0.15 * len(balanced_df))
        
        train_data = balanced_df[:train_size]
        val_data = balanced_df[train_size:train_size + val_size]
        test_data = balanced_df[train_size + val_size:]
        
        print(f"\n=== BALANCED SPLITS ===")
        print(f"Target samples per class: {target_samples_per_class}")
        print(f"Total samples: {len(balanced_df)}")
        print(f"Train: {len(train_data)} ({len(train_data)/len(balanced_df)*100:.1f}%)")
        print(f"Val: {len(val_data)} ({len(val_data)/len(balanced_df)*100:.1f}%)")
        print(f"Test: {len(test_data)} ({len(test_data)/len(balanced_df)*100:.1f}%)")
        
        return train_data, val_data, test_data
    
    def create_weighted_sampling(self, power=0.75):
        """Create weighted sampling probabilities"""
        # Inverse frequency weighting with power adjustment
        inv_freq = 1.0 / self.class_counts
        sampling_weights = inv_freq ** power
        
        # Normalize to probabilities
        sampling_weights = sampling_weights / sampling_weights.sum()
        
        # Create sampling probabilities for each sample
        sample_weights = []
        for class_name in self.data['class']:
            sample_weights.append(sampling_weights[class_name])
        
        self.data['sampling_weight'] = sample_weights
        
        print(f"\n=== WEIGHTED SAMPLING (power={power}) ===")
        for class_name in self.class_counts.index:
            weight = sampling_weights[class_name]
            print(f"  {class_name}: {weight:.4f}")
            
        return sample_weights
    
    def create_focal_loss_weights(self, alpha=1.0, gamma=2.0):
        """Create weights for focal loss"""
        # Focal loss focuses on hard examples
        class_freq = self.class_counts / len(self.data)
        focal_weights = alpha * (1 - class_freq) ** gamma
        
        self.focal_weights = dict(zip(self.class_counts.index, focal_weights))
        
        print(f"\n=== FOCAL LOSS WEIGHTS (alpha={alpha}, gamma={gamma}) ===")
        for class_name, weight in self.focal_weights.items():
            print(f"  {class_name}: {weight:.3f}")
            
        return self.focal_weights

def main():
    """Example usage"""
    # Initialize balancer
    balancer = ClassBalancer(
        csv_file="datasets/tricorder_training/train_labels.csv",
        img_dir="datasets/melanoma_dataset/images/ISIC_2019_Training_Input/ISIC_2019_Training_Input"
    )
    
    # Analyze imbalance
    balancer.analyze_imbalance()
    
    # Compute different types of weights
    balancer.compute_class_weights('balanced')
    balancer.compute_class_weights('balanced_subsample')
    
    # Create balanced splits
    train_data, val_data, test_data = balancer.create_balanced_splits(target_samples_per_class=1000)
    
    # Save balanced splits
    train_data.to_csv('datasets/tricorder_training/train_labels_balanced.csv', index=False)
    val_data.to_csv('datasets/tricorder_training/val_labels_balanced.csv', index=False)
    test_data.to_csv('datasets/tricorder_training/test_labels_balanced.csv', index=False)
    
    print("\nBalanced datasets saved!")

if __name__ == "__main__":
    main()
