#!/usr/bin/env python3
"""
Monitor Tricorder Training Progress
"""

import os
import time
import glob
from pathlib import Path

def check_training_progress():
    """Check if training is progressing"""
    print("🔍 Checking Tricorder Training Progress...")
    print("="*50)
    
    # Check for model files
    model_files = glob.glob("*.pth")
    onnx_files = glob.glob("*.onnx")
    
    print(f"📁 Model files found: {len(model_files)}")
    for f in model_files:
        size = os.path.getsize(f) / (1024*1024)  # MB
        print(f"  - {f} ({size:.1f} MB)")
    
    print(f"📁 ONNX files found: {len(onnx_files)}")
    for f in onnx_files:
        size = os.path.getsize(f) / (1024*1024)  # MB
        print(f"  - {f} ({size:.1f} MB)")
    
    # Check for log files
    log_files = glob.glob("*.log")
    if log_files:
        print(f"📄 Log files: {log_files}")
    
    # Check if training is still running
    import subprocess
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True)
        if 'python.exe' in result.stdout:
            print("✅ Python training process is running")
        else:
            print("❌ No Python training process found")
    except:
        print("⚠️  Could not check process status")
    
    print("="*50)

if __name__ == "__main__":
    check_training_progress()
