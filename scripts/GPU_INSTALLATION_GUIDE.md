# 🚀 GPU Training Setup Guide

## 📋 **Prerequisites**

### **1. Check NVIDIA GPU**
```bash
nvidia-smi
```

### **2. Check CUDA Version**
```bash
nvcc --version
# or
cat /usr/local/cuda/version.txt
```

## 🔧 **Installation Options**

### **Option 1: Automatic Installation (Recommended)**
```bash
# Install from requirements file
pip install -r scripts/requirements_training.txt
```

### **Option 2: Manual PyTorch Installation**

#### **For CUDA 11.8 (Most Common)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### **For CUDA 12.1**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### **For CPU Only (Fallback)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## ✅ **Verify GPU Installation**

### **Test PyTorch GPU Support**
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### **Test ONNX GPU Support**
```python
import onnxruntime as ort
print(f"ONNX Runtime version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")
```

## 🎯 **GPU Training Commands**

### **Basic GPU Training**
```bash
python3 scripts/train_tricorder_optimized.py \
    --train_csv datasets/tricorder_training/train_labels.csv \
    --val_csv datasets/tricorder_training/val_labels.csv \
    --img_dir datasets/melanoma_dataset/images/ISIC_2019_Training_Input/ISIC_2019_Training_Input \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --convert_onnx \
    --device cuda
```

### **Multi-GPU Training (if available)**
```bash
python3 scripts/train_tricorder_optimized.py \
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

## 📊 **GPU Monitoring**

### **Real-time GPU Usage**
```bash
# Install gpustat
pip install gpustat

# Monitor GPU usage
gpustat -i 1

# Or use nvidia-smi
watch -n 1 nvidia-smi
```

### **Memory Optimization**
```bash
# Clear GPU cache if needed
python3 -c "import torch; torch.cuda.empty_cache()"
```

## 🚨 **Troubleshooting**

### **CUDA Out of Memory**
```bash
# Reduce batch size
--batch_size 16  # or 8, or 4

# Use gradient accumulation
--gradient_accumulation_steps 4
```

### **CUDA Version Mismatch**
```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### **Driver Issues**
```bash
# Update NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-525  # or latest version
sudo reboot
```

## 🎯 **Performance Tips**

1. **Batch Size**: Start with 32, increase if you have more GPU memory
2. **Mixed Precision**: Use automatic mixed precision for faster training
3. **Data Loading**: Use multiple workers for data loading
4. **Memory**: Monitor GPU memory usage and adjust batch size accordingly

## 📈 **Expected Performance**

- **RTX 3080**: ~2-3 hours for 30 epochs
- **RTX 4090**: ~1-2 hours for 30 epochs  
- **RTX 2080**: ~4-5 hours for 30 epochs
- **CPU Only**: ~12-24 hours for 30 epochs
