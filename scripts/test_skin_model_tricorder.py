#!/usr/bin/env python3
"""
Test existing skin.onnx model with Tricorder competition format
This script adapts your current model to work with Tricorder requirements
"""

import os
import sys
import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image
import time
import logging

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

class SkinModelTricorderAdapter:
    """Adapter to use existing skin.onnx model with Tricorder format"""
    
    def __init__(self, model_path: str):
        """Initialize ONNX model session"""
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        
        # Get input/output information
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        logger.info(f"Loaded skin.onnx model: {model_path}")
        logger.info(f"Input names: {self.input_names}")
        logger.info(f"Output names: {self.output_names}")
        
        # Print input shapes
        for inp in self.session.get_inputs():
            logger.info(f"Input '{inp.name}' shape: {inp.shape}")
    
    def preprocess_image_tricorder(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for Tricorder competition format
        - Resize to 512x512
        - Convert to [0,512] range as specified in competition
        - Return in BCHW format
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize to 512x512
        img = img.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Scale from [0,255] to [0,512] range as required by Tricorder
        img_array = img_array * (512.0 / 255.0)
        
        # Convert to BCHW format (Batch, Channel, Height, Width)
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def preprocess_demographics_tricorder(self, age: int, gender: str, location: int) -> np.ndarray:
        """
        Preprocess demographic data for Tricorder competition
        Format: [age, gender_encoded, location]
        - age: integer years
        - gender: 'm' -> 1.0, 'f' -> 0.0  
        - location: integer 1-7
        """
        # Validate inputs
        if gender.lower() not in ['m', 'f']:
            raise ValueError(f"Gender must be 'm' or 'f', got: {gender}")
        
        if location not in range(1, 8):
            raise ValueError(f"Location must be 1-7, got: {location}")
        
        # Encode gender: male=1.0, female=0.0
        gender_encoded = 1.0 if gender.lower() == 'm' else 0.0
        
        # Create demographic tensor: [age, gender_encoded, location]
        demo_array = np.array([[float(age), gender_encoded, float(location)]], dtype=np.float32)
        
        return demo_array
    
    def predict_tricorder(self, image_path: str, age: int, gender: str, location: int) -> dict:
        """
        Run inference using existing skin.onnx model with Tricorder format
        """
        start_time = time.time()
        
        # Preprocess inputs in Tricorder format
        image_tensor = self.preprocess_image_tricorder(image_path)
        demo_tensor = self.preprocess_demographics_tricorder(age, gender, location)
        
        # Prepare inputs for ONNX session
        inputs = {
            self.input_names[0]: image_tensor,  # 'image'
            self.input_names[1]: demo_tensor    # 'tabular' or 'demographics'
        }
        
        # Run inference
        outputs = self.session.run(self.output_names, inputs)
        logits = outputs[0].flatten()
        
        # Apply softmax to get probabilities
        probabilities = self.softmax(logits)
        
        inference_time = time.time() - start_time
        
        # Get top predictions
        top3_idx = np.argsort(probabilities)[-3:][::-1]
        top3_predictions = [
            {
                'class_id': int(idx),
                'class_name': TRICORDER_CLASSES[idx],
                'probability': float(probabilities[idx])
            }
            for idx in top3_idx
        ]
        
        # Determine risk level based on Tricorder class weights
        malignant_classes = [1, 3, 8]  # BCC, SCC, MEL (3x weight)
        medium_risk_classes = [2, 4]   # SK, VASC (2x weight)
        benign_classes = [0, 5, 6, 7, 9]  # AK, DF, NV, NON, ON (1x weight)
        
        malignant_prob = sum(probabilities[i] for i in malignant_classes)
        medium_risk_prob = sum(probabilities[i] for i in medium_risk_classes)
        benign_prob = sum(probabilities[i] for i in benign_classes)
        
        risk_level = "Malignant" if malignant_prob > 0.5 else "Medium Risk" if medium_risk_prob > 0.3 else "Benign"
        
        return {
            'all_probabilities': probabilities.tolist(),
            'top3_predictions': top3_predictions,
            'risk_assessment': {
                'level': risk_level,
                'malignant_probability': float(malignant_prob),
                'medium_risk_probability': float(medium_risk_prob),
                'benign_probability': float(benign_prob)
            },
            'inference_time_ms': inference_time * 1000,
            'demographics': {
                'age': age,
                'gender': gender.upper(),
                'location': location
            }
        }
    
    def softmax(self, x):
        """Apply softmax to convert logits to probabilities"""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)

def print_tricorder_results(results: dict):
    """Print formatted Tricorder results"""
    print("\n" + "="*60)
    print("🏆 SKIN.ONNX MODEL - TRICORDER FORMAT TEST")
    print("="*60)
    
    # Demographics
    demo = results['demographics']
    print(f"\n👤 PATIENT DEMOGRAPHICS:")
    print(f"   Age: {demo['age']} years")
    print(f"   Gender: {demo['gender']}")
    print(f"   Body Location: {demo['location']}")
    
    # Risk Assessment
    risk = results['risk_assessment']
    print(f"\n⚠️  RISK ASSESSMENT:")
    print(f"   Overall Risk Level: {risk['level']}")
    print(f"   Malignant Probability: {risk['malignant_probability']:.3f} ({risk['malignant_probability']*100:.1f}%)")
    print(f"   Medium Risk Probability: {risk['medium_risk_probability']:.3f} ({risk['medium_risk_probability']*100:.1f}%)")
    print(f"   Benign Probability: {risk['benign_probability']:.3f} ({risk['benign_probability']*100:.1f}%)")
    
    # Top 3 Predictions
    print(f"\n🎯 TOP 3 PREDICTIONS:")
    print("-" * 50)
    for i, pred in enumerate(results['top3_predictions'], 1):
        print(f"{i}. {pred['class_name']}")
        print(f"   Probability: {pred['probability']:.4f} ({pred['probability']*100:.2f}%)")
    
    # Performance
    print(f"\n⚡ INFERENCE PERFORMANCE:")
    print(f"   Inference Time: {results['inference_time_ms']:.2f} ms")
    
    # All Class Probabilities
    print(f"\n📊 ALL CLASS PROBABILITIES:")
    print("-" * 50)
    for i, (class_name, prob) in enumerate(zip(TRICORDER_CLASSES, results['all_probabilities'])):
        print(f"{i:2d}. {class_name:30s} | {prob:.4f} ({prob*100:5.2f}%)")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Test skin.onnx model with Tricorder format')
    parser.add_argument('--model', type=str, default='models/skin.onnx',
                        help='Path to skin.onnx model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--age', type=int, required=True,
                        help='Patient age in years (e.g., 42)')
    parser.add_argument('--gender', type=str, required=True, choices=['m', 'f'],
                        help='Patient gender: m (male) or f (female)')
    parser.add_argument('--location', type=int, required=True, choices=range(1, 8),
                        help='Body location: 1=Arm, 2=Feet, 3=Genitalia, 4=Hand, 5=Head, 6=Leg, 7=Torso')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed results')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.image):
        logger.error(f"Image file not found: {args.image}")
        return
    
    try:
        # Initialize adapter
        logger.info("Initializing skin.onnx model adapter for Tricorder format...")
        adapter = SkinModelTricorderAdapter(args.model)
        
        # Run prediction
        logger.info("Running prediction with Tricorder format...")
        results = adapter.predict_tricorder(args.image, args.age, args.gender, args.location)
        
        # Print results
        if args.verbose:
            print_tricorder_results(results)
        else:
            # Simple output
            top_pred = results['top3_predictions'][0]
            risk = results['risk_assessment']
            print(f"\nTop Prediction: {top_pred['class_name']} ({top_pred['probability']*100:.2f}%)")
            print(f"Risk Level: {risk['level']}")
            print(f"Inference Time: {results['inference_time_ms']:.2f} ms")
        
        logger.info("Prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        return

if __name__ == "__main__":
    main()
