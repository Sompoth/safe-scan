#!/usr/bin/env python3
"""
Tricorder Competition Model Training Script
10-class skin lesion classification with demographic data
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging
import time

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F

# Image processing
from PIL import Image
import cv2

# ONNX conversion
import onnx
import onnxruntime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tricorder class mapping
TRICORDER_CLASSES = [
    "Actinic keratosis (AK)",      # 0 - Benign
    "Basal cell carcinoma (BCC)",  # 1 - Malignant  
    "Seborrheic keratosis (SK)",   # 2 - Medium risk
    "Squamous cell carcinoma (SCC)", # 3 - Malignant
    "Vascular lesion (VASC)",      # 4 - Medium risk
    "Dermatofibroma (DF)",         # 5 - Benign
    "Benign nevus (NV)",           # 6 - Benign
    "Other non-neoplastic (NON)",  # 7 - Benign
    "Melanoma (MEL)",              # 8 - Malignant
    "Other neoplastic (ON)"        # 9 - Benign
]

# Class weights for Tricorder scoring
CLASS_WEIGHTS = {
    'malignant': [1, 3, 8],      # BCC, SCC, MEL (3x weight)
    'medium_risk': [2, 4],       # SK, VASC (2x weight)  
    'benign': [0, 5, 6, 7, 9]    # AK, DF, NV, NON, ON (1x weight)
}

class TricorderDataset(Dataset):
    """Dataset for Tricorder competition with demographic data"""
    
    def __init__(self, csv_file: str, img_dir: str, transform=None, is_training=True):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations
            img_dir (str): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
            is_training (bool): Whether this is for training or validation
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.is_training = is_training
        
        # Filter data if needed
        if is_training:
            # Use 80% of data for training
            self.data = self.data.sample(frac=0.8, random_state=42)
        else:
            # Use 20% of data for validation
            self.data = self.data.sample(frac=0.2, random_state=42)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.data.iloc[idx]
        img_name = os.path.join(self.img_dir, row['image_name'])
        image = Image.open(img_name).convert('RGB')
        
        # Get class label (map to 0-9 range)
        if 'target' in row:
            # Binary target, need to map to 10 classes
            # For now, assume 0=benign (class 6: NV), 1=melanoma (class 8: MEL)
            label = 8 if row['target'] == 1 else 6
        elif 'class' in row:
            # Direct class mapping
            class_name = row['class'].upper()
            if 'MEL' in class_name:
                label = 8
            elif 'BCC' in class_name:
                label = 1
            elif 'SCC' in class_name:
                label = 3
            elif 'AK' in class_name:
                label = 0
            elif 'SK' in class_name:
                label = 2
            elif 'VASC' in class_name:
                label = 4
            elif 'DF' in class_name:
                label = 5
            elif 'NV' in class_name:
                label = 6
            elif 'NON' in class_name:
                label = 7
            elif 'ON' in class_name:
                label = 9
            else:
                label = 6  # Default to benign nevus
        else:
            label = 6  # Default to benign nevus
        
        # Prepare demographic features for Tricorder format
        # Format: [age, gender_encoded, location]
        age = 50.0  # Default age
        if 'age_approx' in row and pd.notna(row['age_approx']):
            age = float(row['age_approx'])
        
        # Gender: 1.0 for male, 0.0 for female
        gender = 0.5  # Default unknown
        if 'sex' in row and pd.notna(row['sex']):
            gender = 1.0 if row['sex'].lower() in ['male', 'm'] else 0.0
        
        # Location: 1-7 mapping
        location = 7.0  # Default to torso
        if 'anatom_site_general' in row and pd.notna(row['anatom_site_general']):
            site = row['anatom_site_general'].lower()
            if 'head' in site or 'neck' in site:
                location = 5.0
            elif 'arm' in site or 'upper' in site:
                location = 1.0
            elif 'leg' in site or 'lower' in site:
                location = 6.0
            elif 'torso' in site or 'trunk' in site:
                location = 7.0
            elif 'hand' in site:
                location = 4.0
            elif 'foot' in site or 'feet' in site:
                location = 2.0
            elif 'genital' in site:
                location = 3.0
        
        demographics = torch.tensor([age, gender, location], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, demographics, torch.tensor(label, dtype=torch.long), row['image_name']

class TricorderCNN(nn.Module):
    """Optimized CNN for Tricorder competition"""
    
    def __init__(self, num_classes=10, num_demographics=3, dropout_rate=0.3):
        super(TricorderCNN, self).__init__()
        
        # Use EfficientNet-B0 for better efficiency
        try:
            import torchvision.models as models
            self.backbone = models.efficientnet_b0(pretrained=True)
            # Replace classifier
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        except:
            # Fallback to DenseNet121
            self.backbone = models.densenet121(pretrained=True)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        
        # Freeze early layers for transfer learning
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        # Demographic data processor
        self.demographics_processor = nn.Sequential(
            nn.Linear(num_demographics, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features + 16, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, image, demographics):
        # Extract image features
        image_features = self.backbone(image)
        
        # Process demographic data
        demo_features = self.demographics_processor(demographics)
        
        # Combine features
        combined_features = torch.cat((image_features, demo_features), dim=1)
        
        # Classify
        logits = self.classifier(combined_features)
        return logits

class TricorderDetector:
    """Main class for Tricorder model training"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = TricorderCNN(num_classes=config['num_classes'])
        self.model.to(self.device)
        
        # Loss and optimizer - optimized for efficiency
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def get_transforms(self):
        """Get data transforms for Tricorder (512x512, [0,512] range)"""
        train_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            # Scale to [0,512] range as required by Tricorder
            transforms.Lambda(lambda x: x * 512.0)
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            # Scale to [0,512] range as required by Tricorder
            transforms.Lambda(lambda x: x * 512.0)
        ])
        
        return train_transform, val_transform
    
    def create_data_loaders(self):
        """Create data loaders for training and validation"""
        train_transform, val_transform = self.get_transforms()
        
        # Create datasets
        train_dataset = TricorderDataset(
            csv_file=self.config['csv_file'],
            img_dir=self.config['img_dir'],
            transform=train_transform,
            is_training=True
        )
        
        val_dataset = TricorderDataset(
            csv_file=self.config['csv_file'],
            img_dir=self.config['img_dir'],
            transform=val_transform,
            is_training=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            num_workers=2
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, demographics, targets, _) in enumerate(train_loader):
            images = images.to(self.device)
            demographics = demographics.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images, demographics)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, demographics, targets, _ in val_loader:
                images = images.to(self.device)
                demographics = demographics.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images, demographics)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def train(self):
        """Main training loop"""
        train_loader, val_loader = self.create_data_loaders()
        
        logger.info("Starting Tricorder model training...")
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            logger.info(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_tricorder_model.pth')
                logger.info("New best Tricorder model saved!")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(f'tricorder_checkpoint_epoch_{epoch+1}.pth')
        
        # Save final model
        self.save_model('final_tricorder_model.pth')
        self.plot_training_history()
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        save_path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, save_path)
        logger.info(f"Tricorder model saved to {save_path}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Tricorder Model Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Tricorder Model Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tricorder_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def convert_to_onnx(self, model_path: str, onnx_path: str):
        """Convert PyTorch model to ONNX format for Tricorder"""
        logger.info("Converting Tricorder model to ONNX format...")
        
        # Load the trained model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Create dummy inputs
        dummy_image = torch.randn(1, 3, 512, 512).to(self.device)
        dummy_demo = torch.tensor([[50.0, 1.0, 7.0]], dtype=torch.float32).to(self.device)  # age, gender, location
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            (dummy_image, dummy_demo),
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['image', 'demographics'],
            output_names=['output'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'demographics': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info(f"Tricorder ONNX model saved to {onnx_path}")
        
        # Test ONNX model
        ort_session = onnxruntime.InferenceSession(onnx_path)
        ort_inputs = {
            'image': dummy_image.cpu().numpy(),
            'demographics': dummy_demo.cpu().numpy()
        }
        ort_outputs = ort_session.run(None, ort_inputs)
        
        logger.info("Tricorder ONNX model verification successful!")
        
        # Print model size
        model_size = os.path.getsize(onnx_path) / (1024 * 1024)
        logger.info(f"Model size: {model_size:.2f} MB")
        
        return onnx_path

def main():
    parser = argparse.ArgumentParser(description='Train Tricorder Competition Model')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file with image labels')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='./tricorder_models', help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes (10 for Tricorder)')
    parser.add_argument('--convert_onnx', action='store_true', help='Convert to ONNX after training')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'csv_file': args.csv_file,
        'img_dir': args.img_dir,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_classes': args.num_classes
    }
    
    # Create Tricorder detector
    detector = TricorderDetector(config)
    
    # Train the model
    detector.train()
    
    # Convert to ONNX if requested
    if args.convert_onnx:
        best_model_path = os.path.join(args.output_dir, 'best_tricorder_model.pth')
        onnx_path = os.path.join(args.output_dir, 'tricorder_model.onnx')
        detector.convert_to_onnx(best_model_path, onnx_path)
        logger.info(f"Tricorder training complete! ONNX model saved to: {onnx_path}")
    else:
        logger.info("Tricorder training complete! Model saved in PyTorch format.")

if __name__ == "__main__":
    main()
