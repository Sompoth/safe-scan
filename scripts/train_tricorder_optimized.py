#!/usr/bin/env python3
"""
Optimized Tricorder Model Training Script
High-performance 10-class skin lesion classification with demographic data
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import logging
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = at * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TricorderOptimizedDataset(Dataset):
    """Optimized dataset for Tricorder training with demographic data"""
    
    def __init__(self, csv_file: str, img_dir: str, transform=None, is_training: bool = True):
        self.data = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_training = is_training
        
        # Tricorder class mapping
        self.class_mapping = {
            "AK": 0, "BCC": 1, "SK": 2, "SCC": 3, "VASC": 4,
            "DF": 5, "NV": 6, "NON": 7, "MEL": 8, "ON": 9
        }
        
        # Class weights for Tricorder competition
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
        
        logger.info(f"Loaded {len(self.data)} samples")
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
            # Return a black image as fallback
            image = Image.new('RGB', (512, 512), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        class_name = row['class']
        label = self.class_mapping[class_name]
        
        # Get demographic data
        age = float(row.get('age', 50.0))  # Default age
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
        age_norm = (age - 30.0) / 50.0  # Normalize around 30, scale by 50
        anatom_site_norm = anatom_site / 5.0  # Normalize anatomical site (0-5 range)
        
        demographic = torch.tensor([age_norm, sex, anatom_site_norm], dtype=torch.float32)
        
        return image, demographic, label, class_name

def compute_class_weights(data: pd.DataFrame, class_column: str = 'class'):
    """Compute class weights for balanced training"""
    classes = data[class_column].unique()
    class_counts = data[class_column].value_counts()
    
    # Compute balanced weights
    weights = compute_class_weight('balanced', classes=classes, y=data[class_column])
    class_weights = dict(zip(classes, weights))
    
    logger.info("Class weights computed:")
    for class_name, weight in class_weights.items():
        logger.info(f"  {class_name}: {weight:.3f}")
    
    return class_weights

def create_weighted_sampler(data: pd.DataFrame, class_column: str = 'class'):
    """Create weighted random sampler for balanced sampling"""
    class_counts = data[class_column].value_counts()
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    
    # Assign weights to each sample
    sample_weights = [class_weights[cls] for cls in data[class_column]]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(data),
        replacement=True
    )

class TricorderOptimizedModel(nn.Module):
    """Optimized Tricorder model with EfficientNet backbone and demographic fusion"""
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        super(TricorderOptimizedModel, self).__init__()
        
        # Use EfficientNet-B3 for better performance
        try:
            import torchvision.models as models
            self.backbone = models.efficientnet_b3(pretrained=True)
            # Remove the classifier
            self.backbone.classifier = nn.Identity()
            feature_dim = 1536  # EfficientNet-B3 feature dimension
        except:
            # Fallback to DenseNet121
            self.backbone = models.densenet121(pretrained=True)
            self.backbone.classifier = nn.Identity()
            feature_dim = 1024
        
        # Demographic processing
        self.demographic_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + 128, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, image, demographic):
        # Extract image features
        img_features = self.backbone(image)
        
        # Process demographic data
        demo_features = self.demographic_net(demographic)
        
        # Fuse features
        combined = torch.cat([img_features, demo_features], dim=1)
        output = self.fusion(combined)
        
        return output

class TricorderOptimizedTrainer:
    """Optimized trainer for Tricorder model"""
    
    def __init__(self, model, train_loader, val_loader, device, class_weights=None, criterion=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimized loss function with class weights
        if criterion is not None:
            self.criterion = criterion
        elif class_weights is not None:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimized optimizer with better parameters
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Advanced learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=1e-3,
            epochs=50,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Early stopping
        self.best_val_acc = 0.0
        self.patience = 10
        self.patience_counter = 0
        
        logger.info(f"Model initialized on {device}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, demographics, labels, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            demographics = demographics.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images, demographics)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                          f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, demographics, labels, _ in self.val_loader:
                images = images.to(self.device)
                demographics = demographics.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images, demographics)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Early stopping check
        if accuracy > self.best_val_acc:
            self.best_val_acc = accuracy
            self.patience_counter = 0
            # Save best model
            torch.save(self.model.state_dict(), 'best_tricorder_model.pth')
        else:
            self.patience_counter += 1
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self, epochs: int = 50) -> Dict[str, List[float]]:
        """Train the model"""
        logger.info(f"Starting training for {epochs} epochs...")
        
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            train_losses.append(train_metrics['loss'])
            train_accs.append(train_metrics['accuracy'])
            
            # Validate
            val_metrics = self.validate(epoch)
            val_losses.append(val_metrics['loss'])
            val_accs.append(val_metrics['accuracy'])
            
            epoch_time = time.time() - start_time
            
            logger.info(f'Epoch {epoch+1}/{epochs} - '
                       f'Train Loss: {train_metrics["loss"]:.4f}, Train Acc: {train_metrics["accuracy"]:.2f}% - '
                       f'Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.2f}% - '
                       f'Time: {epoch_time:.1f}s')
            
            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_tricorder_model.pth'))
        logger.info(f"Training completed. Best validation accuracy: {self.best_val_acc:.2f}%")
        
        return {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs
        }

def get_optimized_transforms(is_training: bool = True):
    """Get optimized transforms for Tricorder training"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def convert_to_onnx(model, device, output_path: str = "tricorder_optimized.onnx"):
    """Convert model to ONNX format"""
    logger.info("Converting model to ONNX format...")
    
    model.eval()
    
    # Create dummy inputs
    dummy_image = torch.randn(1, 3, 512, 512).to(device)
    dummy_demographic = torch.randn(1, 3).to(device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_image, dummy_demographic),
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['image', 'tabular'],
        output_names=['output'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'tabular': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    logger.info(f"Model exported to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Train optimized Tricorder model')
    parser.add_argument('--train_csv', type=str, default='datasets/tricorder_refined/train_labels.csv',
                        help='Training CSV file')
    parser.add_argument('--val_csv', type=str, default='datasets/tricorder_refined/val_labels.csv',
                        help='Validation CSV file')
    parser.add_argument('--img_dir', type=str, default='datasets/melanoma_dataset/images/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
                        help='Image directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--convert_onnx', action='store_true',
                        help='Convert to ONNX after training')
    parser.add_argument('--output_name', type=str, default='tricorder_optimized',
                        help='Output model name')
    
    # Class balancing options
    parser.add_argument('--use_class_weights', action='store_true', 
                        help='Use class weights for balanced training')
    parser.add_argument('--use_weighted_sampling', action='store_true',
                        help='Use weighted random sampling')
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Use focal loss instead of cross entropy')
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                        help='Alpha parameter for focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for focal loss')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = TricorderOptimizedDataset(
        args.train_csv, args.img_dir, 
        transform=get_optimized_transforms(is_training=True), 
        is_training=True
    )
    
    val_dataset = TricorderOptimizedDataset(
        args.val_csv, args.img_dir,
        transform=get_optimized_transforms(is_training=False),
        is_training=False
    )
    
    # Compute class weights if requested
    class_weights = None
    if args.use_class_weights:
        train_data = pd.read_csv(args.train_csv)
        class_weights_dict = compute_class_weights(train_data)
        # Convert to tensor in correct order
        class_order = ['AK', 'BCC', 'SK', 'SCC', 'VASC', 'DF', 'NV', 'NON', 'MEL', 'ON']
        class_weights = [class_weights_dict[cls] for cls in class_order]
        logger.info(f"Using computed class weights: {class_weights}")
    
    # Create weighted sampler if requested
    train_sampler = None
    if args.use_weighted_sampling:
        train_data = pd.read_csv(args.train_csv)
        train_sampler = create_weighted_sampler(train_data)
        logger.info("Using weighted random sampling")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=not args.use_weighted_sampling,  # Don't shuffle if using sampler
        sampler=train_sampler,
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Create model
    model = TricorderOptimizedModel(num_classes=10)
    
    # Create trainer with appropriate loss function
    if args.use_focal_loss:
        # Create focal loss with alpha weights
        if class_weights is not None:
            alpha_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        else:
            alpha_tensor = None
        
        focal_loss = FocalLoss(alpha=alpha_tensor, gamma=args.focal_gamma)
        trainer = TricorderOptimizedTrainer(model, train_loader, val_loader, device, 
                                          class_weights=None, criterion=focal_loss)
        logger.info(f"Using Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    else:
        trainer = TricorderOptimizedTrainer(model, train_loader, val_loader, device, class_weights)
    
    # Train model
    history = trainer.train(args.epochs)
    
    # Save final model
    torch.save(model.state_dict(), f'{args.output_name}.pth')
    logger.info(f"Model saved as {args.output_name}.pth")
    
    # Convert to ONNX if requested
    if args.convert_onnx:
        convert_to_onnx(model, device, f'{args.output_name}.onnx')
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
