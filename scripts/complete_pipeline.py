#!/usr/bin/env python3
"""
Complete Melanoma Detection Pipeline
for Bittensor Subnet 76 Competition

This script orchestrates the entire process:
1. Dataset preparation
2. Model training
3. ONNX conversion
4. Model submission to competition
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MelanomaPipeline:
    """Complete pipeline for melanoma detection model"""
    
    def __init__(self, config: dict):
        self.config = config
        self.base_dir = Path.cwd()
        self.scripts_dir = self.base_dir / "scripts"
        self.models_dir = self.base_dir / "models"
        self.datasets_dir = self.base_dir / "datasets"
        
        # Create necessary directories
        self.models_dir.mkdir(exist_ok=True)
        self.datasets_dir.mkdir(exist_ok=True)
        
        logger.info(f"Pipeline initialized with base directory: {self.base_dir}")
    
    def step1_prepare_dataset(self):
        """Step 1: Prepare the dataset"""
        logger.info("=" * 50)
        logger.info("STEP 1: Preparing Dataset")
        logger.info("=" * 50)
        
        # Create sample dataset structure
        cmd = [
            sys.executable, str(self.scripts_dir / "prepare_dataset.py"),
            "--action", "sample",
            "--output_dir", str(self.datasets_dir),
            "--num_samples", str(self.config.get('num_samples', 200))
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Dataset preparation completed successfully!")
            # Extract dataset path from output
            for line in result.stdout.split('\n'):
                if 'Dataset directory:' in line:
                    self.dataset_dir = line.split(': ')[1].strip()
                elif 'CSV file:' in line:
                    self.csv_file = line.split(': ')[1].strip()
            logger.info(f"Dataset directory: {self.dataset_dir}")
            logger.info(f"CSV file: {self.csv_file}")
        else:
            logger.error(f"Dataset preparation failed: {result.stderr}")
            return False
        
        return True
    
    def step2_train_model(self):
        """Step 2: Train the melanoma detection model"""
        logger.info("=" * 50)
        logger.info("STEP 2: Training Model")
        logger.info("=" * 50)
        
        # Get image directory from dataset directory
        img_dir = str(Path(self.dataset_dir) / "images")
        
        cmd = [
            sys.executable, str(self.scripts_dir / "train_melanoma_model.py"),
            "--csv_file", self.csv_file,
            "--img_dir", img_dir,
            "--output_dir", str(self.models_dir),
            "--epochs", str(self.config.get('epochs', 10)),
            "--batch_size", str(self.config.get('batch_size', 32)),
            "--learning_rate", str(self.config.get('learning_rate', 1e-4)),
            "--convert_onnx"  # Always convert to ONNX
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        logger.info("Note: This step requires actual images in the dataset directory")
        logger.info("If you don't have images yet, please add them and run this step manually")
        
        # Check if images exist
        if not Path(img_dir).exists() or not any(Path(img_dir).iterdir()):
            logger.warning(f"No images found in {img_dir}")
            logger.warning("Please add images and run training manually:")
            logger.warning(f"python scripts/train_melanoma_model.py --csv_file {self.csv_file} --img_dir {img_dir} --epochs 10 --convert_onnx")
            return False
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Model training completed successfully!")
            # Find the ONNX model
            onnx_files = list(self.models_dir.glob("*.onnx"))
            if onnx_files:
                self.onnx_model = str(onnx_files[0])
                logger.info(f"ONNX model: {self.onnx_model}")
            else:
                logger.error("No ONNX model found after training")
                return False
        else:
            logger.error(f"Model training failed: {result.stderr}")
            return False
        
        return True
    
    def step3_evaluate_model(self):
        """Step 3: Evaluate the model locally"""
        logger.info("=" * 50)
        logger.info("STEP 3: Local Model Evaluation")
        logger.info("=" * 50)
        
        cmd = [
            sys.executable, "neurons/miner.py",
            "--action", "evaluate",
            "--competition_id", "melanoma-1",
            "--model_path", self.onnx_model
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        logger.info("This will download the competition dataset and evaluate your model")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Local evaluation completed successfully!")
            logger.info("Check the output for model performance metrics")
        else:
            logger.error(f"Local evaluation failed: {result.stderr}")
            return False
        
        return True
    
    def step4_upload_to_huggingface(self):
        """Step 4: Upload model to Hugging Face"""
        logger.info("=" * 50)
        logger.info("STEP 4: Upload to Hugging Face")
        logger.info("=" * 50)
        
        if not self.config.get('hf_repo_id') or not self.config.get('hf_token'):
            logger.warning("Hugging Face credentials not provided")
            logger.warning("Please run this step manually with your credentials:")
            logger.warning(f"python neurons/miner.py --action upload --competition_id melanoma-1 --model_path {self.onnx_model} --code_directory ./ --hf_model_name melanoma_model.onnx --hf_repo_id YOUR_USERNAME/YOUR_REPO --hf_token YOUR_TOKEN")
            return False
        
        cmd = [
            sys.executable, "neurons/miner.py",
            "--action", "upload",
            "--competition_id", "melanoma-1",
            "--model_path", self.onnx_model,
            "--code_directory", "./",
            "--hf_model_name", "melanoma_model.onnx",
            "--hf_repo_id", self.config['hf_repo_id'],
            "--hf_token", self.config['hf_token']
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Upload to Hugging Face completed successfully!")
        else:
            logger.error(f"Upload to Hugging Face failed: {result.stderr}")
            return False
        
        return True
    
    def step5_submit_to_competition(self):
        """Step 5: Submit model to competition"""
        logger.info("=" * 50)
        logger.info("STEP 5: Submit to Competition")
        logger.info("=" * 50)
        
        if not self.config.get('wallet_name') or not self.config.get('wallet_hotkey'):
            logger.warning("Wallet credentials not provided")
            logger.warning("Please run this step manually with your wallet:")
            logger.warning(f"python neurons/miner.py --action submit --competition_id melanoma-1 --hf_code_filename code.zip --hf_model_name melanoma_model.onnx --hf_repo_id {self.config.get('hf_repo_id', 'YOUR_USERNAME/YOUR_REPO')} --wallet.name YOUR_WALLET --wallet.hotkey YOUR_HOTKEY --netuid 76 --subtensor.network finney")
            return False
        
        cmd = [
            sys.executable, "neurons/miner.py",
            "--action", "submit",
            "--competition_id", "melanoma-1",
            "--hf_code_filename", "code.zip",
            "--hf_model_name", "melanoma_model.onnx",
            "--hf_repo_id", self.config['hf_repo_id'],
            "--wallet.name", self.config['wallet_name'],
            "--wallet.hotkey", self.config['wallet_hotkey'],
            "--netuid", "76",
            "--subtensor.network", "finney"
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Model submission completed successfully!")
            logger.info("Your model is now in the competition!")
        else:
            logger.error(f"Model submission failed: {result.stderr}")
            return False
        
        return True
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        logger.info("Starting Complete Melanoma Detection Pipeline")
        logger.info("This pipeline will guide you through the entire process")
        
        # Step 1: Prepare dataset
        if not self.step1_prepare_dataset():
            logger.error("Pipeline failed at dataset preparation")
            return False
        
        # Step 2: Train model (if images are available)
        if not self.step2_train_model():
            logger.warning("Model training step requires manual intervention")
            logger.warning("Please add images to the dataset and run training manually")
            return False
        
        # Step 3: Evaluate model
        if not self.step3_evaluate_model():
            logger.error("Pipeline failed at model evaluation")
            return False
        
        # Step 4: Upload to Hugging Face
        if not self.step4_upload_to_huggingface():
            logger.warning("Upload step requires Hugging Face credentials")
            logger.warning("Please run this step manually")
            return False
        
        # Step 5: Submit to competition
        if not self.step5_submit_to_competition():
            logger.warning("Submission step requires wallet credentials")
            logger.warning("Please run this step manually")
            return False
        
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        logger.info("Your melanoma detection model has been:")
        logger.info("✓ Trained and converted to ONNX")
        logger.info("✓ Evaluated locally")
        logger.info("✓ Uploaded to Hugging Face")
        logger.info("✓ Submitted to the competition")
        logger.info("")
        logger.info("Good luck in the competition!")
        
        return True

def create_config_template():
    """Create a configuration template file"""
    config = {
        "num_samples": 200,
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "hf_repo_id": "your_username/your_repo",
        "hf_token": "your_huggingface_token",
        "wallet_name": "your_wallet_name",
        "wallet_hotkey": "your_hotkey_name"
    }
    
    config_path = "pipeline_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration template created: {config_path}")
    logger.info("Please edit this file with your actual credentials before running the pipeline")

def main():
    parser = argparse.ArgumentParser(description='Complete Melanoma Detection Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--create_config', action='store_true', help='Create configuration template')
    parser.add_argument('--num_samples', type=int, default=200, help='Number of samples for dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_config_template()
        return
    
    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override with command line arguments
    if args.num_samples:
        config['num_samples'] = args.num_samples
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    
    # Set defaults for missing values
    config.setdefault('num_samples', 200)
    config.setdefault('epochs', 10)
    config.setdefault('batch_size', 32)
    config.setdefault('learning_rate', 1e-4)
    
    # Create and run pipeline
    pipeline = MelanomaPipeline(config)
    success = pipeline.run_pipeline()
    
    if not success:
        logger.error("Pipeline completed with warnings/errors")
        logger.error("Please check the logs above and complete any manual steps")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Complete Melanoma Detection Pipeline")
        print("\nUsage:")
        print("  python scripts/complete_pipeline.py --create_config")
        print("  python scripts/complete_pipeline.py --config pipeline_config.json")
        print("  python scripts/complete_pipeline.py --num_samples 500 --epochs 20")
        print("\nThis pipeline will:")
        print("1. Prepare a sample dataset structure")
        print("2. Train a melanoma detection model")
        print("3. Convert to ONNX format")
        print("4. Evaluate locally")
        print("5. Upload to Hugging Face")
        print("6. Submit to competition")
        print("\nFirst, create a config file:")
        print("  python scripts/complete_pipeline.py --create_config")
    else:
        main()
