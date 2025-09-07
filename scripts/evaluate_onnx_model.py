#!/usr/bin/env python3
"""
ONNX Model Evaluation Script for Melanoma Detection
Evaluates the performance of the skin.onnx model on test data
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Tuple, List, Dict
import time

# Deep learning imports
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import onnxruntime as ort

# Image processing
from PIL import Image
import cv2

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MelanomaTestDataset(Dataset):
    """Test dataset for melanoma detection evaluation"""
    
    def __init__(self, csv_file: str, img_dir: str, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations
            img_dir (str): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Filter out missing images
        self.valid_indices = []
        for idx, row in self.data.iterrows():
            img_path = os.path.join(self.img_dir, row['image_name'])
            if os.path.exists(img_path):
                self.valid_indices.append(idx)
        
        logger.info(f"Found {len(self.valid_indices)} valid images out of {len(self.data)} total")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.data.iloc[actual_idx]
        
        img_name = os.path.join(self.img_dir, row['image_name'])
        image = Image.open(img_name).convert('RGB')
        
        # Get label
        label = row['target']
        
        # Prepare tabular features (age, sex, anatomical site)
        tabular_features = []
        
        # Age (normalize to 0-1 range, assume 0-100 years)
        if 'age_approx' in row and pd.notna(row['age_approx']):
            age = min(max(row['age_approx'], 0), 100) / 100.0
        else:
            age = 0.5  # Default middle age
        
        # Sex (0 for male, 1 for female)
        if 'sex' in row and pd.notna(row['sex']):
            sex = 1.0 if row['sex'].lower() in ['female', 'f'] else 0.0
        else:
            sex = 0.5  # Default unknown
        
        # Anatomical site (encode as numeric, normalize)
        if 'anatom_site_general' in row and pd.notna(row['anatom_site_general']):
            # Simple encoding based on common sites
            site_mapping = {
                'head/neck': 0.0,
                'upper extremity': 0.2,
                'lower extremity': 0.4,
                'torso': 0.6,
                'palms/soles': 0.8,
                'oral/genital': 1.0
            }
            site = site_mapping.get(row['anatom_site_general'].lower(), 0.5)
        else:
            site = 0.5  # Default unknown
        
        tabular_features = [age, sex, site]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(tabular_features, dtype=torch.float32), torch.tensor(label, dtype=torch.long), row['image_name']

class ONNXModelEvaluator:
    """Evaluator for ONNX melanoma detection models"""
    
    def __init__(self, onnx_path: str, device: str = 'cpu'):
        self.onnx_path = onnx_path
        self.device = device
        
        # Load ONNX model
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        logger.info(f"Loaded ONNX model: {onnx_path}")
        logger.info(f"Input name: {self.input_name}")
        logger.info(f"Output name: {self.output_name}")
        logger.info(f"Input shape: {self.session.get_inputs()[0].shape}")
        logger.info(f"Output shape: {self.session.get_outputs()[0].shape}")
    
    def get_transforms(self):
        """Get data transforms for evaluation (512x512 for skin.onnx model)"""
        transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Resize to match model input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform
    
    def predict_batch(self, images: torch.Tensor, tabular: torch.Tensor) -> np.ndarray:
        """Predict on a batch of images and tabular features"""
        # Convert to numpy
        if isinstance(images, torch.Tensor):
            images_np = images.numpy()
        else:
            images_np = images
            
        if isinstance(tabular, torch.Tensor):
            tabular_np = tabular.numpy()
        else:
            tabular_np = tabular
        
        # Run inference with both inputs
        outputs = self.session.run([self.output_name], {
            'image': images_np,
            'tabular': tabular_np
        })
        return outputs[0]
    
    def evaluate_dataset(self, csv_file: str, img_dir: str, batch_size: int = 32) -> Dict:
        """Evaluate model on entire dataset"""
        logger.info("Starting model evaluation...")
        
        # Create dataset and dataloader
        transform = self.get_transforms()
        dataset = MelanomaTestDataset(csv_file, img_dir, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Storage for predictions and ground truth
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_image_names = []
        inference_times = []
        
        logger.info(f"Evaluating on {len(dataset)} images...")
        
        for batch_idx, (images, tabular, labels, image_names) in enumerate(dataloader):
            start_time = time.time()
            
            # Get predictions
            outputs = self.predict_batch(images, tabular)
            probabilities = F.softmax(torch.tensor(outputs), dim=1).numpy()
            predictions = np.argmax(outputs, axis=1)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Store results
            all_predictions.extend(predictions)
            # For 10-class model, we need to map to binary (melanoma vs non-melanoma)
            # Assuming class 0 is melanoma, others are non-melanoma
            melanoma_probs = probabilities[:, 0]  # Probability of class 0 (melanoma)
            all_probabilities.extend(melanoma_probs)
            all_labels.extend(labels.numpy())
            all_image_names.extend(image_names)
            
            if batch_idx % 10 == 0:
                logger.info(f"Processed batch {batch_idx}/{len(dataloader)}")
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        # Convert 10-class predictions to binary (0 = melanoma, 1-9 = non-melanoma)
        binary_predictions = (all_predictions == 0).astype(int)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_labels, binary_predictions, all_probabilities)
        
        # Add inference time metrics
        metrics['avg_inference_time'] = np.mean(inference_times)
        metrics['total_inference_time'] = np.sum(inference_times)
        metrics['images_per_second'] = len(dataset) / np.sum(inference_times)
        
        # Store detailed results
        results = {
            'predictions': binary_predictions,  # Store binary predictions
            'probabilities': all_probabilities,
            'labels': all_labels,
            'image_names': all_image_names,
            'metrics': metrics,
            'raw_predictions': all_predictions  # Store original 10-class predictions
        }
        
        return results
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')
        
        # AUC metrics
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['roc_auc'] = 0.0
        
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
            metrics['pr_auc'] = auc(recall_curve, precision_curve)
        except ValueError:
            metrics['pr_auc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Additional metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return metrics
    
    def plot_results(self, results: Dict, output_dir: str):
        """Plot evaluation results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        y_true = results['labels']
        y_pred = results['predictions']
        y_prob = results['probabilities']
        metrics = results['metrics']
        
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Confusion Matrix
        cm = metrics['confusion_matrix']
        im = axes[0, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[0, 0].text(j, i, format(cm[i, j], 'd'),
                              ha="center", va="center",
                              color="white" if cm[i, j] > thresh else "black")
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True)
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        axes[0, 2].plot(recall, precision, color='blue', lw=2,
                       label=f'PR curve (AUC = {metrics["pr_auc"]:.3f})')
        axes[0, 2].set_xlim([0.0, 1.0])
        axes[0, 2].set_ylim([0.0, 1.05])
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curve')
        axes[0, 2].legend(loc="lower left")
        axes[0, 2].grid(True)
        
        # 4. Prediction Distribution
        axes[1, 0].hist(y_prob[y_true == 0], bins=50, alpha=0.7, label='Benign', color='blue')
        axes[1, 0].hist(y_prob[y_true == 1], bins=50, alpha=0.7, label='Melanoma', color='red')
        axes[1, 0].set_xlabel('Predicted Probability (Melanoma)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Prediction Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 5. Metrics Bar Chart
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'Sensitivity']
        metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                        metrics['f1_score'], metrics['specificity'], metrics['sensitivity']]
        
        bars = axes[1, 1].bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 
                                                                 'gold', 'plum', 'lightblue'])
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 6. Class Distribution
        class_counts = np.bincount(y_true)
        axes[1, 2].pie(class_counts, labels=['Benign', 'Melanoma'], autopct='%1.1f%%', 
                      colors=['lightblue', 'lightcoral'])
        axes[1, 2].set_title('Dataset Class Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path / 'evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Evaluation plots saved to {output_path / 'evaluation_results.png'}")
    
    def print_detailed_report(self, results: Dict):
        """Print detailed evaluation report"""
        metrics = results['metrics']
        y_true = results['labels']
        y_pred = results['predictions']
        
        print("\n" + "="*60)
        print("MELANOMA DETECTION MODEL EVALUATION REPORT")
        print("="*60)
        
        print(f"\n📊 DATASET STATISTICS:")
        print(f"   Total images evaluated: {len(y_true)}")
        print(f"   Benign cases: {np.sum(y_true == 0)} ({100*np.sum(y_true == 0)/len(y_true):.1f}%)")
        print(f"   Melanoma cases: {np.sum(y_true == 1)} ({100*np.sum(y_true == 1)/len(y_true):.1f}%)")
        
        print(f"\n🎯 PERFORMANCE METRICS:")
        print(f"   Accuracy:           {metrics['accuracy']:.4f} ({100*metrics['accuracy']:.2f}%)")
        print(f"   Precision:          {metrics['precision']:.4f} ({100*metrics['precision']:.2f}%)")
        print(f"   Recall (Sensitivity): {metrics['recall']:.4f} ({100*metrics['recall']:.2f}%)")
        print(f"   Specificity:        {metrics['specificity']:.4f} ({100*metrics['specificity']:.2f}%)")
        print(f"   F1-Score:           {metrics['f1_score']:.4f}")
        print(f"   ROC AUC:            {metrics['roc_auc']:.4f}")
        print(f"   PR AUC:             {metrics['pr_auc']:.4f}")
        
        print(f"\n⚡ INFERENCE PERFORMANCE:")
        print(f"   Average inference time: {metrics['avg_inference_time']:.4f} seconds")
        print(f"   Total inference time:   {metrics['total_inference_time']:.2f} seconds")
        print(f"   Images per second:      {metrics['images_per_second']:.2f}")
        
        print(f"\n📋 CONFUSION MATRIX:")
        cm = metrics['confusion_matrix']
        print(f"   True Negatives (TN):  {cm[0,0]:4d}")
        print(f"   False Positives (FP): {cm[0,1]:4d}")
        print(f"   False Negatives (FN): {cm[1,0]:4d}")
        print(f"   True Positives (TP):  {cm[1,1]:4d}")
        
        print(f"\n📈 CLASSIFICATION REPORT:")
        print(classification_report(y_true, y_pred, target_names=['Benign', 'Melanoma']))
        
        print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Evaluate ONNX Melanoma Detection Model')
    parser.add_argument('--onnx_path', type=str, default='models/skin.onnx', help='Path to ONNX model')
    parser.add_argument('--csv_file', type=str, default='datasets/melanoma_dataset/test_labels.csv', help='Path to test CSV file')
    parser.add_argument('--img_dir', type=str, default='datasets/melanoma_dataset/images/ISIC_2019_Training_Input/ISIC_2019_Training_Input', help='Directory containing images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.onnx_path):
        logger.error(f"ONNX model not found: {args.onnx_path}")
        return
    
    if not os.path.exists(args.csv_file):
        logger.error(f"CSV file not found: {args.csv_file}")
        return
    
    if not os.path.exists(args.img_dir):
        logger.error(f"Image directory not found: {args.img_dir}")
        return
    
    # Create evaluator
    evaluator = ONNXModelEvaluator(args.onnx_path)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(args.csv_file, args.img_dir, args.batch_size)
    
    # Print detailed report
    evaluator.print_detailed_report(results)
    
    # Plot results
    evaluator.plot_results(results, args.output_dir)
    
    # Save results to file
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([results['metrics']])
    metrics_df.to_csv(output_path / 'evaluation_metrics.csv', index=False)
    
    # Save detailed predictions
    predictions_df = pd.DataFrame({
        'image_name': results['image_names'],
        'true_label': results['labels'],
        'predicted_label_binary': results['predictions'],
        'predicted_label_10class': results['raw_predictions'],
        'melanoma_probability': results['probabilities']
    })
    predictions_df.to_csv(output_path / 'detailed_predictions.csv', index=False)
    
    logger.info(f"Evaluation complete! Results saved to {output_path}")

if __name__ == "__main__":
    main()
