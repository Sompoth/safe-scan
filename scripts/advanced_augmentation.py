#!/usr/bin/env python3
"""
Advanced Data Augmentation for Tricorder Dataset
Handles class imbalance through sophisticated augmentation techniques
"""

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
import cv2
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class MedicalImageAugmentation:
    """Advanced augmentation specifically designed for medical skin images"""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (512, 512),
                 intensity_range: Tuple[float, float] = (0.8, 1.2),
                 rotation_range: Tuple[int, int] = (-30, 30),
                 scale_range: Tuple[float, float] = (0.8, 1.2)):
        self.target_size = target_size
        self.intensity_range = intensity_range
        self.rotation_range = rotation_range
        self.scale_range = scale_range
    
    def __call__(self, image: Image.Image, mask: Optional[Image.Image] = None) -> Image.Image:
        """Apply random augmentation to image"""
        # Randomly choose augmentation strategy
        strategy = random.choice([
            'geometric', 'color', 'noise', 'blur', 'elastic', 'perspective'
        ])
        
        if strategy == 'geometric':
            return self._geometric_augmentation(image)
        elif strategy == 'color':
            return self._color_augmentation(image)
        elif strategy == 'noise':
            return self._noise_augmentation(image)
        elif strategy == 'blur':
            return self._blur_augmentation(image)
        elif strategy == 'elastic':
            return self._elastic_augmentation(image)
        elif strategy == 'perspective':
            return self._perspective_augmentation(image)
        
        return image
    
    def _geometric_augmentation(self, image: Image.Image) -> Image.Image:
        """Geometric transformations"""
        # Random rotation
        angle = random.uniform(*self.rotation_range)
        image = image.rotate(angle, fillcolor=(0, 0, 0))
        
        # Random scaling
        scale = random.uniform(*self.scale_range)
        w, h = image.size
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        
        # Random crop or pad to target size
        image = self._crop_or_pad(image, self.target_size)
        
        # Random flip
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.7:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        return image
    
    def _color_augmentation(self, image: Image.Image) -> Image.Image:
        """Color and intensity variations"""
        # Brightness
        brightness_factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        
        # Contrast
        contrast_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        
        # Saturation
        saturation_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation_factor)
        
        # Hue shift
        if random.random() > 0.5:
            image = image.convert('HSV')
            h, s, v = image.split()
            h_array = np.array(h)
            h_shift = random.randint(-20, 20)
            h_array = (h_array + h_shift) % 256
            h = Image.fromarray(h_array.astype('uint8'))
            image = Image.merge('HSV', (h, s, v)).convert('RGB')
        
        return image
    
    def _noise_augmentation(self, image: Image.Image) -> Image.Image:
        """Add realistic noise"""
        img_array = np.array(image)
        
        # Gaussian noise
        noise_std = random.uniform(0.01, 0.05)
        noise = np.random.normal(0, noise_std, img_array.shape)
        img_array = img_array + noise * 255
        
        # Salt and pepper noise
        if random.random() > 0.5:
            salt_pepper_ratio = random.uniform(0.001, 0.01)
            noise_mask = np.random.random(img_array.shape[:2])
            img_array[noise_mask < salt_pepper_ratio] = 0
            img_array[noise_mask > 1 - salt_pepper_ratio] = 255
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def _blur_augmentation(self, image: Image.Image) -> Image.Image:
        """Add realistic blur"""
        blur_type = random.choice(['gaussian', 'motion', 'defocus'])
        
        if blur_type == 'gaussian':
            radius = random.uniform(0.5, 2.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        elif blur_type == 'motion':
            # Simulate motion blur
            img_array = np.array(image)
            kernel_size = random.randint(5, 15)
            angle = random.uniform(0, 180)
            
            # Create motion blur kernel
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = cv2.getRotationMatrix2D((kernel_size//2, kernel_size//2), angle, 1)
            kernel = cv2.warpAffine(kernel, kernel, (kernel_size, kernel_size))
            kernel = kernel / kernel.sum()
            
            # Apply blur
            img_array = cv2.filter2D(img_array, -1, kernel)
            image = Image.fromarray(img_array)
        elif blur_type == 'defocus':
            radius = random.uniform(1.0, 3.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        return image
    
    def _elastic_augmentation(self, image: Image.Image) -> Image.Image:
        """Elastic deformation (simulates skin stretching)"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Create displacement fields
        alpha = random.uniform(50, 150)
        sigma = random.uniform(10, 20)
        
        # Generate random displacement
        dx = cv2.GaussianBlur(np.random.randn(h, w) * alpha, (0, 0), sigma)
        dy = cv2.GaussianBlur(np.random.randn(h, w) * alpha, (0, 0), sigma)
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x_new = x + dx
        y_new = y + dy
        
        # Apply elastic transformation
        img_array = cv2.remap(img_array, x_new.astype(np.float32), y_new.astype(np.float32), 
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return Image.fromarray(img_array)
    
    def _perspective_augmentation(self, image: Image.Image) -> Image.Image:
        """Perspective transformation (simulates camera angle changes)"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Define source points (corners of image)
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Define destination points with perspective distortion
        max_offset = min(w, h) * 0.1
        dst_points = np.float32([
            [random.uniform(-max_offset, max_offset), random.uniform(-max_offset, max_offset)],
            [w + random.uniform(-max_offset, max_offset), random.uniform(-max_offset, max_offset)],
            [w + random.uniform(-max_offset, max_offset), h + random.uniform(-max_offset, max_offset)],
            [random.uniform(-max_offset, max_offset), h + random.uniform(-max_offset, max_offset)]
        ])
        
        # Apply perspective transformation
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        img_array = cv2.warpPerspective(img_array, matrix, (w, h), 
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return Image.fromarray(img_array)
    
    def _crop_or_pad(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Crop or pad image to target size"""
        w, h = image.size
        target_w, target_h = target_size
        
        if w == target_w and h == target_h:
            return image
        
        # Calculate scaling to fit target size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        image = image.resize((new_w, new_h), Image.LANCZOS)
        
        # Create new image with target size
        new_image = Image.new('RGB', target_size, (0, 0, 0))
        
        # Calculate position to center the image
        x = (target_w - new_w) // 2
        y = (target_h - new_h) // 2
        
        # Paste image
        new_image.paste(image, (x, y))
        
        return new_image

class ClassSpecificAugmentation:
    """Class-specific augmentation strategies for different skin lesion types"""
    
    def __init__(self):
        self.class_strategies = {
            'MEL': self._melanoma_augmentation,
            'BCC': self._bcc_augmentation,
            'SCC': self._scc_augmentation,
            'AK': self._ak_augmentation,
            'SK': self._sk_augmentation,
            'VASC': self._vasc_augmentation,
            'DF': self._df_augmentation,
            'NV': self._nv_augmentation,
            'NON': self._non_augmentation,
            'ON': self._on_augmentation
        }
    
    def augment_class(self, image: Image.Image, class_name: str, intensity: float = 1.0) -> Image.Image:
        """Apply class-specific augmentation"""
        if class_name in self.class_strategies:
            return self.class_strategies[class_name](image, intensity)
        else:
            # Default augmentation for unknown classes
            return self._default_augmentation(image, intensity)
    
    def _melanoma_augmentation(self, image: Image.Image, intensity: float) -> Image.Image:
        """Augmentation specific to melanoma (asymmetrical, irregular borders)"""
        # High intensity augmentation for rare but critical class
        aug = MedicalImageAugmentation(
            intensity_range=(0.7, 1.4),
            rotation_range=(-45, 45),
            scale_range=(0.7, 1.3)
        )
        
        # Apply multiple augmentations
        for _ in range(int(2 * intensity)):
            image = aug(image)
        
        return image
    
    def _bcc_augmentation(self, image: Image.Image, intensity: float) -> Image.Image:
        """Augmentation for basal cell carcinoma (pearly, raised lesions)"""
        aug = MedicalImageAugmentation(
            intensity_range=(0.8, 1.3),
            rotation_range=(-30, 30),
            scale_range=(0.8, 1.2)
        )
        
        for _ in range(int(1.5 * intensity)):
            image = aug(image)
        
        return image
    
    def _scc_augmentation(self, image: Image.Image, intensity: float) -> Image.Image:
        """Augmentation for squamous cell carcinoma (scaly, crusted lesions)"""
        aug = MedicalImageAugmentation(
            intensity_range=(0.8, 1.3),
            rotation_range=(-30, 30),
            scale_range=(0.8, 1.2)
        )
        
        for _ in range(int(1.5 * intensity)):
            image = aug(image)
        
        return image
    
    def _ak_augmentation(self, image: Image.Image, intensity: float) -> Image.Image:
        """Augmentation for actinic keratosis (very rare, high intensity)"""
        aug = MedicalImageAugmentation(
            intensity_range=(0.6, 1.5),
            rotation_range=(-45, 45),
            scale_range=(0.6, 1.4)
        )
        
        # Very high intensity for extremely rare class
        for _ in range(int(3 * intensity)):
            image = aug(image)
        
        return image
    
    def _sk_augmentation(self, image: Image.Image, intensity: float) -> Image.Image:
        """Augmentation for seborrheic keratosis (waxy, stuck-on appearance)"""
        aug = MedicalImageAugmentation(
            intensity_range=(0.8, 1.2),
            rotation_range=(-20, 20),
            scale_range=(0.9, 1.1)
        )
        
        for _ in range(int(1.2 * intensity)):
            image = aug(image)
        
        return image
    
    def _vasc_augmentation(self, image: Image.Image, intensity: float) -> Image.Image:
        """Augmentation for vascular lesions (very rare, high intensity)"""
        aug = MedicalImageAugmentation(
            intensity_range=(0.6, 1.5),
            rotation_range=(-45, 45),
            scale_range=(0.6, 1.4)
        )
        
        # Very high intensity for extremely rare class
        for _ in range(int(3 * intensity)):
            image = aug(image)
        
        return image
    
    def _df_augmentation(self, image: Image.Image, intensity: float) -> Image.Image:
        """Augmentation for dermatofibroma (very rare, high intensity)"""
        aug = MedicalImageAugmentation(
            intensity_range=(0.6, 1.5),
            rotation_range=(-45, 45),
            scale_range=(0.6, 1.4)
        )
        
        # Very high intensity for extremely rare class
        for _ in range(int(3 * intensity)):
            image = aug(image)
        
        return image
    
    def _nv_augmentation(self, image: Image.Image, intensity: float) -> Image.Image:
        """Augmentation for benign nevus (common, low intensity)"""
        aug = MedicalImageAugmentation(
            intensity_range=(0.9, 1.1),
            rotation_range=(-15, 15),
            scale_range=(0.95, 1.05)
        )
        
        # Low intensity for common class
        for _ in range(int(0.5 * intensity)):
            image = aug(image)
        
        return image
    
    def _non_augmentation(self, image: Image.Image, intensity: float) -> Image.Image:
        """Augmentation for other non-neoplastic (common, low intensity)"""
        aug = MedicalImageAugmentation(
            intensity_range=(0.9, 1.1),
            rotation_range=(-15, 15),
            scale_range=(0.95, 1.05)
        )
        
        # Low intensity for common class
        for _ in range(int(0.5 * intensity)):
            image = aug(image)
        
        return image
    
    def _on_augmentation(self, image: Image.Image, intensity: float) -> Image.Image:
        """Augmentation for other neoplastic (common, low intensity)"""
        aug = MedicalImageAugmentation(
            intensity_range=(0.9, 1.1),
            rotation_range=(-15, 15),
            scale_range=(0.95, 1.05)
        )
        
        # Low intensity for common class
        for _ in range(int(0.5 * intensity)):
            image = aug(image)
        
        return image
    
    def _default_augmentation(self, image: Image.Image, intensity: float) -> Image.Image:
        """Default augmentation for unknown classes"""
        aug = MedicalImageAugmentation(
            intensity_range=(0.8, 1.2),
            rotation_range=(-20, 20),
            scale_range=(0.9, 1.1)
        )
        
        for _ in range(int(intensity)):
            image = aug(image)
        
        return image

def get_advanced_transforms(is_training: bool = True, 
                          class_name: Optional[str] = None,
                          augmentation_intensity: float = 1.0) -> transforms.Compose:
    """Get advanced transforms with class-specific augmentation"""
    
    if not is_training:
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Base transforms
    base_transforms = [
        transforms.Resize((512, 512)),
    ]
    
    # Add class-specific augmentation if class is specified
    if class_name:
        class_aug = ClassSpecificAugmentation()
        base_transforms.append(
            transforms.Lambda(lambda x: class_aug.augment_class(x, class_name, augmentation_intensity))
        )
    else:
        # General augmentation
        medical_aug = MedicalImageAugmentation()
        base_transforms.append(transforms.Lambda(lambda x: medical_aug(x)))
    
    # Standard transforms
    base_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(base_transforms)

def create_balanced_dataset(csv_file: str, 
                          img_dir: str, 
                          target_samples_per_class: int = 1000,
                          output_dir: str = "datasets/tricorder_balanced") -> None:
    """Create a balanced dataset using advanced augmentation"""
    import pandas as pd
    from pathlib import Path
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    # Load data
    data = pd.read_csv(csv_file)
    class_counts = data['class'].value_counts()
    
    balanced_data = []
    
    for class_name, count in class_counts.items():
        class_data = data[data['class'] == class_name]
        
        if count >= target_samples_per_class:
            # Sample down
            sampled = class_data.sample(n=target_samples_per_class, random_state=42)
        else:
            # Augment up
            sampled = class_data.copy()
            augmentation_needed = target_samples_per_class - count
            
            # Create augmented samples
            class_aug = ClassSpecificAugmentation()
            for i in range(augmentation_needed):
                # Select random sample to augment
                source_sample = class_data.sample(n=1, random_state=42+i).iloc[0]
                
                # Load and augment image
                img_path = Path(img_dir) / source_sample['image_name']
                if img_path.exists():
                    image = Image.open(img_path).convert('RGB')
                    augmented_image = class_aug.augment_class(image, class_name, intensity=2.0)
                    
                    # Save augmented image
                    new_img_name = f"{source_sample['image_name'].split('.')[0]}_aug_{i}.jpg"
                    augmented_image.save(f"{output_dir}/images/{new_img_name}")
                    
                    # Create new row
                    new_row = source_sample.copy()
                    new_row['image_name'] = new_img_name
                    sampled = pd.concat([sampled, new_row.to_frame().T], ignore_index=True)
        
        balanced_data.append(sampled)
    
    # Combine all classes
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save balanced dataset
    balanced_df.to_csv(f"{output_dir}/balanced_labels.csv", index=False)
    
    print(f"Balanced dataset created with {len(balanced_df)} samples")
    print(f"Class distribution:")
    print(balanced_df['class'].value_counts())

if __name__ == "__main__":
    # Example usage
    create_balanced_dataset(
        csv_file="datasets/tricorder_training/train_labels.csv",
        img_dir="datasets/melanoma_dataset/images/ISIC_2019_Training_Input/ISIC_2019_Training_Input",
        target_samples_per_class=1000
    )
