# 🏥 Melanoma Detection Model Training Scripts

This directory contains complete scripts to build, train, and submit a melanoma detection model for the **Bittensor Subnet 76 Cancer Detection Competition**.

## 📋 What You'll Get

✅ **Complete training pipeline** from scratch to ONNX  
✅ **Dataset preparation tools** for melanoma images  
✅ **State-of-the-art CNN architecture** (ResNet50-based)  
✅ **Automatic ONNX conversion** for competition submission  
✅ **Local evaluation** against competition datasets  
✅ **Step-by-step guidance** through the entire process  

## 🚀 Quick Start

### 1. **Create Configuration File**
```bash
python scripts/complete_pipeline.py --create_config
```
Edit the generated `pipeline_config.json` with your credentials.

### 2. **Run Complete Pipeline**
```bash
python scripts/complete_pipeline.py --config pipeline_config.json
```

### 3. **Manual Step-by-Step** (if you prefer)
```bash
# Step 1: Prepare dataset
python scripts/prepare_dataset.py --action sample --num_samples 200

# Step 2: Train model (after adding images)
python scripts/train_melanoma_model.py --csv_file datasets/sample_melanoma/sample_labels.csv --img_dir datasets/sample_melanoma/images --epochs 10 --convert_onnx

# Step 3: Evaluate locally
python neurons/miner.py --action evaluate --competition_id melanoma-1 --model_path models/melanoma_model.onnx

# Step 4: Upload to Hugging Face
python neurons/miner.py --action upload --competition_id melanoma-1 --model_path models/melanoma_model.onnx --code_directory ./ --hf_model_name melanoma_model.onnx --hf_repo_id YOUR_USERNAME/YOUR_REPO --hf_token YOUR_TOKEN

# Step 5: Submit to competition
python neurons/miner.py --action submit --competition_id melanoma-1 --hf_code_filename code.zip --hf_model_name melanoma_model.onnx --hf_repo_id YOUR_USERNAME/YOUR_REPO --wallet.name YOUR_WALLET --wallet.hotkey YOUR_HOTKEY --netuid 76 --subtensor.network finney
```

## 📁 Scripts Overview

### **`complete_pipeline.py`** - 🎯 **Main Pipeline**
- Orchestrates the entire process from start to finish
- Handles dataset preparation, training, evaluation, and submission
- Provides step-by-step guidance and error handling

### **`train_melanoma_model.py`** - 🧠 **Model Training**
- Creates a custom CNN architecture based on ResNet50
- Implements transfer learning for melanoma detection
- Includes data augmentation and validation
- Automatically converts trained model to ONNX format

### **`prepare_dataset.py`** - 📊 **Dataset Management**
- Creates sample dataset structures
- Generates realistic CSV files with proper formatting
- Creates train/validation/test splits
- Validates dataset integrity

## 🏗️ Model Architecture

The melanoma detection model uses:

- **Backbone**: Pre-trained ResNet50 (ImageNet weights)
- **Transfer Learning**: Frozen early layers, trainable final layers
- **Custom Head**: 3-layer fully connected network with dropout
- **Input Size**: 224x224 RGB images
- **Output**: Binary classification (benign vs. malignant)
- **Optimizer**: AdamW with weight decay
- **Scheduler**: ReduceLROnPlateau for adaptive learning

## 📊 Dataset Requirements

### **CSV Format**
```csv
image_name,target,patient_id,age,sex,anatom_site,diagnosis
image1.jpg,0,patient_001,45,male,torso,benign
image2.jpg,1,patient_002,52,female,upper extremity,malignant melanoma
```

### **Image Requirements**
- **Format**: JPG, PNG, or other common formats
- **Size**: Any size (will be resized to 224x224)
- **Channels**: RGB (3 channels)
- **Content**: Dermatological images (skin lesions)

### **Label Encoding**
- **0**: Benign (nevus, seborrheic keratosis, dermatofibroma)
- **1**: Malignant (melanoma)

## ⚙️ Training Configuration

### **Default Parameters**
```yaml
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.0001
  
data_augmentation:
  resize: [224, 224]
  horizontal_flip: true
  rotation: 15
  color_jitter: true
```

### **Customization**
You can override any parameter:
```bash
python scripts/train_melanoma_model.py \
  --epochs 100 \
  --batch_size 64 \
  --learning_rate 0.0005 \
  --convert_onnx
```

## 🔧 Installation & Dependencies

### **🚀 GPU Training (Recommended)**
```bash
# Install GPU-optimized requirements
pip install -r scripts/requirements_training.txt

# Verify GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### **💻 CPU Training (Fallback)**
```bash
# Install CPU-only requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r scripts/requirements_training.txt
```

### **📦 Required Packages**
- **PyTorch**: Deep learning framework (GPU-optimized with CUDA support)
- **TorchVision**: Pre-trained models and transforms
- **ONNX**: Model format conversion (GPU runtime included)
- **PIL/OpenCV**: Image processing
- **Pandas/NumPy**: Data handling
- **Matplotlib**: Training visualization
- **TensorBoard**: Training monitoring
- **GPU Monitoring**: nvidia-ml-py3, gpustat
- **Acceleration**: accelerate, transformers

## 📈 Training Process

### **Phase 1: Data Preparation**
1. Create dataset structure
2. Add melanoma images
3. Create train/val/test splits
4. Validate dataset integrity

### **Phase 2: Model Training**
1. Load pre-trained ResNet50
2. Apply transfer learning
3. Train with data augmentation
4. Monitor validation metrics
5. Save best model

### **Phase 3: ONNX Conversion**
1. Load trained PyTorch model
2. Convert to ONNX format
3. Verify ONNX model
4. Test inference

### **Phase 4: Competition Submission**
1. Local evaluation
2. Hugging Face upload
3. Blockchain submission
4. Competition entry

## 🚀 GPU Training Guide

### **Hardware Requirements**
- **Minimum**: 8GB RAM, CPU training
- **Recommended**: 16GB+ RAM, RTX 3060+ GPU
- **Optimal**: 32GB+ RAM, RTX 3080+ GPU
- **Professional**: 64GB+ RAM, RTX 4090/A100 GPU

### **GPU Setup & Verification**
```bash
# Check NVIDIA GPU
nvidia-smi

# Verify CUDA installation
nvcc --version

# Test PyTorch GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### **GPU Training Commands**

#### **Tricorder Model (10-class classification)**
```bash
# GPU training with optimized settings
python scripts/train_tricorder_optimized.py \
    --train_csv datasets/tricorder_training/train_labels.csv \
    --val_csv datasets/tricorder_training/val_labels.csv \
    --test_csv datasets/tricorder_training/test_labels.csv \
    --img_dir datasets/melanoma_dataset/images/ISIC_2019_Training_Input/ISIC_2019_Training_Input \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --convert_onnx \
    --device cuda
```

#### **Melanoma Model (Binary classification)**
```bash
# GPU training for melanoma detection
python scripts/train_melanoma_model.py \
    --csv_file datasets/melanoma_dataset/labels.csv \
    --img_dir datasets/melanoma_dataset/images \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --convert_onnx \
    --device cuda
```

### **GPU Performance Optimization**

#### **Batch Size Guidelines**
- **RTX 3060 (8GB)**: batch_size 16-24
- **RTX 3070 (8GB)**: batch_size 24-32
- **RTX 3080 (10GB)**: batch_size 32-48
- **RTX 4090 (24GB)**: batch_size 64-128

#### **Memory Optimization**
```bash
# Use gradient accumulation for larger effective batch sizes
--gradient_accumulation_steps 4

# Enable mixed precision training
--mixed_precision

# Clear GPU cache if needed
python -c "import torch; torch.cuda.empty_cache()"
```

#### **Multi-GPU Training**
```bash
# Train on multiple GPUs
python scripts/train_tricorder_optimized.py \
    --train_csv datasets/tricorder_training/train_labels.csv \
    --val_csv datasets/tricorder_training/val_labels.csv \
    --img_dir datasets/melanoma_dataset/images/ISIC_2019_Training_Input/ISIC_2019_Training_Input \
    --epochs 30 \
    --batch_size 64 \
    --learning_rate 0.0001 \
    --convert_onnx \
    --device cuda \
    --multi_gpu
```

### **Training Monitoring**
```bash
# Real-time GPU monitoring
gpustat -i 1

# TensorBoard for training visualization
tensorboard --logdir runs/

# System resource monitoring
htop
```

### **Expected Training Times**
- **RTX 4090**: 1-2 hours (30 epochs)
- **RTX 3080**: 2-3 hours (30 epochs)
- **RTX 3060**: 4-6 hours (30 epochs)
- **CPU Only**: 12-24 hours (30 epochs)

### **Training Tips**
- Start with fewer epochs (5-10) for testing
- Use smaller batch sizes if memory limited
- Monitor validation loss to prevent overfitting
- Use learning rate scheduling for better convergence
- Enable early stopping to prevent overtraining

### **Model Improvements**
- Experiment with different architectures (EfficientNet, Vision Transformer)
- Try ensemble methods
- Use advanced data augmentation techniques
- Implement cross-validation
- Use transfer learning from medical imaging models

## 🚨 Common Issues & Solutions

### **GPU Issues**

#### **CUDA Out of Memory**
```bash
# Reduce batch size
--batch_size 16  # or 8, or 4

# Use gradient accumulation
--gradient_accumulation_steps 4

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### **CUDA Not Available**
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA support
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Update NVIDIA drivers
sudo apt update && sudo apt install nvidia-driver-525
```

#### **CUDA Version Mismatch**
```bash
# Check CUDA version
nvcc --version

# Install correct PyTorch version
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### **Training Issues**

#### **Out of Memory (CPU/GPU)**
```bash
# Reduce batch size
--batch_size 16

# Use gradient accumulation
--gradient_accumulation_steps 4

# Enable mixed precision
--mixed_precision
```

#### **Training Not Converging**
```bash
# Increase learning rate
--learning_rate 0.001

# Reduce weight decay
--weight_decay 0.00001

# Use learning rate scheduling
--lr_scheduler
```

#### **Slow Training**
```bash
# Use GPU if available
--device cuda

# Increase batch size (if memory allows)
--batch_size 64

# Use multiple workers for data loading
--num_workers 4
```

### **ONNX Conversion Errors**
```bash
# Ensure model is in evaluation mode
model.eval()

# Check input/output tensor shapes
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")

# Verify ONNX opset version compatibility
--onnx_opset 11
```

### **Dataset Issues**

#### **Images Not Found**
```bash
# Check image directory path
ls datasets/melanoma_dataset/images/ISIC_2019_Training_Input/ISIC_2019_Training_Input/

# Verify CSV file paths
head -5 datasets/tricorder_training/train_labels.csv
```

#### **CSV Format Errors**
```bash
# Check CSV structure
python -c "import pandas as pd; df = pd.read_csv('datasets/tricorder_training/train_labels.csv'); print(df.head())"
```

## 📊 Evaluation Metrics

The model will be evaluated on:
- **Accuracy**: Overall classification accuracy
- **Sensitivity**: True positive rate for melanoma detection
- **Specificity**: True negative rate for benign detection
- **AUC-ROC**: Area under the receiver operating characteristic curve

## 🏆 Competition Strategy

### **Model Quality**
- Focus on **sensitivity** (catching all melanomas)
- Balance with **specificity** (avoiding false positives)
- Use ensemble methods for robustness

### **Submission Timing**
- Submit **30+ minutes** before competition start
- Test locally first to ensure model works
- Verify ONNX model compatibility

### **Code Quality**
- Include **complete training code**
- Document data preprocessing steps
- Explain model architecture decisions
- Provide training configuration

## 🔗 Useful Resources

- **ISIC Archive**: [https://challenge.isic-archive.com/](https://challenge.isic-archive.com/)
- **HAM10000 Dataset**: [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- **PyTorch Documentation**: [https://pytorch.org/docs/](https://pytorch.org/docs/)
- **ONNX Documentation**: [https://onnx.ai/](https://onnx.ai/)

## 🆘 Getting Help

1. **Check the logs** for detailed error messages
2. **Verify dataset format** matches requirements
3. **Test individual components** before running full pipeline
4. **Join Discord** for community support
5. **Review competition documentation** for specific requirements

## 🎉 Success Checklist

Before submitting to competition, ensure:
- [ ] GPU setup verified (`nvidia-smi`, `torch.cuda.is_available()`)
- [ ] Model trains successfully
- [ ] ONNX conversion works
- [ ] Local evaluation passes
- [ ] Hugging Face upload succeeds
- [ ] Model submission completes
- [ ] Extrinsic record documented

## 🚀 Quick Reference

### **GPU Training Commands**
```bash
# Tricorder (10-class)
python scripts/train_tricorder_optimized.py \
    --train_csv datasets/tricorder_training/train_labels.csv \
    --val_csv datasets/tricorder_training/val_labels.csv \
    --img_dir datasets/melanoma_dataset/images/ISIC_2019_Training_Input/ISIC_2019_Training_Input \
    --epochs 30 --batch_size 32 --device cuda --convert_onnx

# Melanoma (Binary)
python scripts/train_melanoma_model.py \
    --csv_file datasets/melanoma_dataset/labels.csv \
    --img_dir datasets/melanoma_dataset/images \
    --epochs 50 --batch_size 32 --device cuda --convert_onnx
```

### **GPU Monitoring**
```bash
# Real-time GPU usage
gpustat -i 1

# Training visualization
tensorboard --logdir runs/
```

### **Troubleshooting**
```bash
# CUDA out of memory
--batch_size 16 --gradient_accumulation_steps 4

# CUDA not available
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

**Good luck in the competition! 🏆**

Your model could help save lives by improving early melanoma detection. Every submission contributes to the advancement of cancer detection technology.
