#!/usr/bin/env python3
"""
Melanoma Detection Model Training Script
for Bittensor Subnet 76 Competition

This script creates, trains, and saves a melanoma detection model in ONNX format.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
import logging

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

class MelanomaDataset(Dataset):
    """Custom dataset for melanoma detection"""
    
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
        
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        # Get label (assuming 'target' column exists)
        if 'target' in self.data.columns:
            label = self.data.iloc[idx]['target']
        else:
            # If no target column, assume benign (0) for demo
            label = 0
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

class MelanomaCNN(nn.Module):
    """Custom CNN architecture for melanoma detection"""
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(MelanomaCNN, self).__init__()
        
                # Use DenseNet121 instead of ResNet50 - much better for medical images
        self.backbone = models.densenet121(pretrained=True)
        
        # Freeze early layers for transfer learning
        for param in list(self.backbone.parameters())[:-30]:  # Keep more layers trainable for DenseNet
            param.requires_grad = False
            
        # Get the number of input features from the classifier
        num_features = self.backbone.classifier.in_features
        
        # Replace the classifier with our custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class MelanomaDetector:
    """Main class for melanoma detection model training"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = MelanomaCNN(num_classes=config['num_classes'])
        self.model.to(self.device)
        
        # Loss and optimizer - optimized for DenseNet121
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)  # DenseNet benefits from these beta values
        )
        
        # Learning rate scheduler - more aggressive for DenseNet
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.3, patience=3, min_lr=1e-7
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def get_transforms(self):
        """Get data transforms for training and validation"""
        # DenseNet121 expects 224x224 images by default, but we can optimize for medical images
        # Medical images often benefit from higher resolution, but 224x224 is a good balance
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Slightly larger for better detail
            transforms.RandomCrop((224, 224)),  # Random crop for augmentation
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),  # Medical images can be flipped vertically
            transforms.RandomRotation(degrees=20),  # Slightly more rotation for medical images
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            # Medical-specific augmentations
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            # DenseNet121 normalization (ImageNet stats)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),  # Center crop for validation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def create_data_loaders(self):
        """Create data loaders for training and validation"""
        train_transform, val_transform = self.get_transforms()
        
        # Create datasets
        train_dataset = MelanomaDataset(
            csv_file=self.config['csv_file'],
            img_dir=self.config['img_dir'],
            transform=train_transform,
            is_training=True
        )
        
        val_dataset = MelanomaDataset(
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
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            num_workers=4
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
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
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def train(self):
        """Main training loop"""
        train_loader, val_loader = self.create_data_loaders()
        
        logger.info("Starting training...")
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
                self.save_model('best_model.pth')
                logger.info("New best model saved!")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
        
        # Save final model
        self.save_model('final_model.pth')
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
        logger.info(f"Model saved to {save_path}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def convert_to_onnx(self, model_path: str, onnx_path: str):
        """Convert PyTorch model to ONNX format"""
        logger.info("Converting model to ONNX format...")
        
        # Load the trained model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info(f"ONNX model saved to {onnx_path}")
        
        # Test ONNX model
        ort_session = onnxruntime.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        logger.info("ONNX model verification successful!")
        return onnx_path

def create_sample_dataset():
    """Create a sample dataset structure for demonstration"""
    logger.info("Creating sample dataset structure...")
    
    # Create directories
    os.makedirs("sample_data/images", exist_ok=True)
    os.makedirs("sample_data/labels", exist_ok=True)
    
    # Create sample CSV file
    sample_data = {
        'image_name': ['sample_1.jpg', 'sample_2.jpg', 'sample_3.jpg'],
        'target': [0, 1, 0]  # 0: benign, 1: malignant
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv("sample_data/labels/sample_labels.csv", index=False)
    
    logger.info("Sample dataset structure created in 'sample_data/' directory")
    logger.info("Note: You need to add actual images and update the CSV file")

def main():
    parser = argparse.ArgumentParser(description='Train Melanoma Detection Model')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file with image labels')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='./models', help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (increased for real dataset)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training (reduced for DenseNet121)')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate (optimized for DenseNet121)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (reduced for DenseNet121)')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
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
    
    # Create melanoma detector
    detector = MelanomaDetector(config)
    
    # Train the model
    detector.train()
    
    # Convert to ONNX if requested
    if args.convert_onnx:
        best_model_path = os.path.join(args.output_dir, 'best_model.pth')
        onnx_path = os.path.join(args.output_dir, 'melanoma_model.onnx')
        detector.convert_to_onnx(best_model_path, onnx_path)
        logger.info(f"Training complete! ONNX model saved to: {onnx_path}")
    else:
        logger.info("Training complete! Model saved in PyTorch format.")

if __name__ == "__main__":
    # If no arguments provided, create sample dataset
    if len(sys.argv) == 1:
        create_sample_dataset()
        print("\nTo train a model, use:")
        print("python scripts/train_melanoma_model.py --csv_file sample_data/labels/sample_labels.csv --img_dir sample_data/images --epochs 10 --convert_onnx")
    else:
        main()
