#!/usr/bin/env python3
"""
Evaluate Tricorder Model Performance
Comprehensive evaluation with competition-specific metrics
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import logging
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TricorderEvaluationDataset:
    """Dataset for Tricorder model evaluation"""
    
    def __init__(self, csv_file: str, img_dir: str, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Tricorder class mapping
        self.class_mapping = {
            "AK": 0, "BCC": 1, "SK": 2, "SCC": 3, "VASC": 4,
            "DF": 5, "NV": 6, "NON": 7, "MEL": 8, "ON": 9
        }
        
        self.class_names = list(self.class_mapping.keys())
        
        logger.info(f"Loaded {len(self.data)} samples for evaluation")
        logger.info(f"Class distribution: {self.data['class'].value_counts().to_dict()}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        img_path = self.img_dir / row['image_name']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (512, 512), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        class_name = row['class']
        label = self.class_mapping[class_name]
        
        # Get demographic data
        age = float(row.get('age', 50.0))
        sex = 1.0 if str(row.get('sex', 'male')).lower() == 'male' else 0.0
        
        # Handle anatomical site (convert string to numeric)
        anatom_site_str = str(row.get('anatom_site', 'unknown')).lower()
        anatom_site_mapping = {
            'head/neck': 0.0, 'upper extremity': 1.0, 'lower extremity': 2.0,
            'torso': 3.0, 'palms/soles': 4.0, 'oral/genital': 5.0,
            'unknown': 0.0
        }
        anatom_site = anatom_site_mapping.get(anatom_site_str, 0.0)
        
        # Normalize demographic data
        age_norm = (age - 30.0) / 50.0
        anatom_site_norm = anatom_site / 5.0  # Normalize anatomical site (0-5 range)
        
        demographic = torch.tensor([age_norm, sex, anatom_site_norm], dtype=torch.float32)
        
        return image, demographic, label, class_name, row['image_name']

class TricorderModelEvaluator:
    """Comprehensive evaluator for Tricorder models"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model_path = model_path
        self.class_names = ["AK", "BCC", "SK", "SCC", "VASC", "DF", "NV", "NON", "MEL", "ON"]
        
        # Load model
        self.model = self._load_model()
        
        # Competition-specific class weights
        self.class_weights = {
            "AK": 1.0,    # Benign
            "BCC": 3.0,   # Malignant
            "SK": 2.0,    # Medium Risk
            "SCC": 3.0,   # Malignant
            "VASC": 2.0,  # Medium Risk
            "DF": 1.0,    # Benign
            "NV": 1.0,    # Benign
            "NON": 1.0,   # Benign
            "MEL": 3.0,   # Malignant
            "ON": 1.0     # Benign
        }
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Using device: {self.device}")
    
    def _load_model(self):
        """Load the trained model"""
        # Import the model class
        sys.path.append('scripts')
        from train_tricorder_optimized import TricorderOptimizedModel
        
        model = TricorderOptimizedModel(num_classes=10)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        return model
    
    def get_transforms(self):
        """Get evaluation transforms"""
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict_batch(self, images, demographics):
        """Predict on a batch of data"""
        with torch.no_grad():
            images = images.to(self.device)
            demographics = demographics.to(self.device)
            outputs = self.model(images, demographics)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
        return predictions.cpu().numpy(), probabilities.cpu().numpy()
    
    def evaluate_dataset(self, csv_file: str, img_dir: str, batch_size: int = 16) -> Dict:
        """Evaluate model on a dataset"""
        logger.info(f"Evaluating on dataset: {csv_file}")
        
        # Create dataset and dataloader
        dataset = TricorderEvaluationDataset(csv_file, img_dir, self.get_transforms())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_class_names = []
        all_image_names = []
        inference_times = []
        
        logger.info(f"Evaluating {len(dataset)} samples...")
        
        for batch_idx, (images, demographics, labels, class_names, image_names) in enumerate(dataloader):
            start_time = time.time()
            
            predictions, probabilities = self.predict_batch(images, demographics)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
            all_labels.extend(labels.numpy())
            all_class_names.extend(class_names)
            all_image_names.extend(image_names)
            
            if batch_idx % 50 == 0:
                logger.info(f"Processed {batch_idx * batch_size}/{len(dataset)} samples")
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            all_labels, all_predictions, all_probabilities, 
            all_class_names, inference_times
        )
        
        # Save detailed predictions
        self._save_detailed_predictions(
            all_image_names, all_class_names, all_labels, 
            all_predictions, all_probabilities, csv_file
        )
        
        return metrics
    
    def _calculate_metrics(self, labels, predictions, probabilities, class_names, inference_times):
        """Calculate comprehensive metrics"""
        # Basic accuracy
        accuracy = accuracy_score(labels, predictions)
        
        # Classification report
        report = classification_report(labels, predictions, target_names=self.class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Competition-specific weighted accuracy
        weighted_accuracy = self._calculate_weighted_accuracy(labels, predictions, class_names)
        
        # Malignant vs Benign accuracy
        malignant_accuracy = self._calculate_malignant_accuracy(labels, predictions)
        
        # Inference time metrics
        avg_inference_time = np.mean(inference_times)
        total_inference_time = np.sum(inference_times)
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = np.array(labels) == i
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(
                    np.array(labels)[class_mask], 
                    np.array(predictions)[class_mask]
                )
                per_class_metrics[class_name] = {
                    'accuracy': class_acc,
                    'samples': np.sum(class_mask),
                    'weight': self.class_weights[class_name]
                }
        
        metrics = {
            'overall_accuracy': accuracy,
            'weighted_accuracy': weighted_accuracy,
            'malignant_accuracy': malignant_accuracy,
            'per_class_metrics': per_class_metrics,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'inference_time': {
                'avg_per_batch': avg_inference_time,
                'total_time': total_inference_time,
                'samples_per_second': len(labels) / total_inference_time
            }
        }
        
        return metrics
    
    def _calculate_weighted_accuracy(self, labels, predictions, class_names):
        """Calculate competition-weighted accuracy"""
        correct = 0
        total_weight = 0
        
        for i, (label, pred, class_name) in enumerate(zip(labels, predictions, class_names)):
            weight = self.class_weights[class_name]
            total_weight += weight
            if label == pred:
                correct += weight
        
        return correct / total_weight if total_weight > 0 else 0
    
    def _calculate_malignant_accuracy(self, labels, predictions):
        """Calculate accuracy for malignant classes (BCC, SCC, MEL)"""
        malignant_classes = [1, 3, 8]  # BCC, SCC, MEL
        
        malignant_mask = np.isin(labels, malignant_classes)
        if np.sum(malignant_mask) == 0:
            return 0
        
        malignant_labels = np.array(labels)[malignant_mask]
        malignant_predictions = np.array(predictions)[malignant_mask]
        
        return accuracy_score(malignant_labels, malignant_predictions)
    
    def _save_detailed_predictions(self, image_names, class_names, labels, predictions, probabilities, csv_file):
        """Save detailed predictions to CSV"""
        output_file = f"detailed_predictions_{Path(csv_file).stem}.csv"
        
        results_df = pd.DataFrame({
            'image_name': image_names,
            'true_class': class_names,
            'true_label': labels,
            'predicted_label': predictions,
            'predicted_class': [self.class_names[p] for p in predictions],
            'confidence': [probabilities[i][p] for i, p in enumerate(predictions)],
            'correct': [l == p for l, p in zip(labels, predictions)]
        })
        
        # Add probability columns for each class
        for i, class_name in enumerate(self.class_names):
            results_df[f'prob_{class_name}'] = [prob[i] for prob in probabilities]
        
        results_df.to_csv(output_file, index=False)
        logger.info(f"Detailed predictions saved to {output_file}")
    
    def print_evaluation_report(self, metrics: Dict):
        """Print comprehensive evaluation report"""
        print("\n" + "="*80)
        print("🧬 TRICORDER MODEL EVALUATION REPORT")
        print("="*80)
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
        print(f"  Weighted Accuracy: {metrics['weighted_accuracy']:.4f} ({metrics['weighted_accuracy']*100:.2f}%)")
        print(f"  Malignant Accuracy: {metrics['malignant_accuracy']:.4f} ({metrics['malignant_accuracy']*100:.2f}%)")
        
        print(f"\n⚡ INFERENCE PERFORMANCE:")
        print(f"  Average Inference Time: {metrics['inference_time']['avg_per_batch']:.4f}s per batch")
        print(f"  Total Inference Time: {metrics['inference_time']['total_time']:.2f}s")
        print(f"  Samples per Second: {metrics['inference_time']['samples_per_second']:.1f}")
        
        print(f"\n🎯 PER-CLASS PERFORMANCE:")
        for class_name, class_metrics in metrics['per_class_metrics'].items():
            weight = class_metrics['weight']
            acc = class_metrics['accuracy']
            samples = class_metrics['samples']
            print(f"  {class_name:4s} (Weight: {weight}x): {acc:.4f} ({acc*100:.2f}%) - {samples} samples")
        
        print(f"\n🏆 COMPETITION SCORING:")
        print(f"  The weighted accuracy ({metrics['weighted_accuracy']*100:.2f}%) is the primary metric")
        print(f"  Malignant classes (BCC, SCC, MEL) have 3x weight")
        print(f"  Medium risk classes (SK, VASC) have 2x weight")
        print(f"  Benign classes have 1x weight")
        
        print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Tricorder model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model (.pth file)')
    parser.add_argument('--test_csv', type=str, default='datasets/tricorder_refined/test_labels.csv',
                        help='Test CSV file')
    parser.add_argument('--img_dir', type=str, default='datasets/melanoma_dataset/images/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
                        help='Image directory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = TricorderModelEvaluator(args.model_path, args.device)
    
    # Evaluate model
    metrics = evaluator.evaluate_dataset(args.test_csv, args.img_dir, args.batch_size)
    
    # Print report
    evaluator.print_evaluation_report(metrics)
    
    # Save metrics
    with open('evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
