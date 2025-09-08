# 🎯 Class Imbalance Handling Guide for Tricorder Dataset

## 📊 **Problem Analysis**

The tricorder dataset has severe class imbalance:
- **NV (Benign nevus)**: 12,875 samples (50.8%)
- **MEL (Melanoma)**: 4,522 samples (17.8%)
- **BCC (Basal cell carcinoma)**: 3,323 samples (13.1%)
- **SK (Seborrheic keratosis)**: 2,624 samples (10.4%)
- **AK (Actinic keratosis)**: 867 samples (3.4%)
- **SCC (Squamous cell carcinoma)**: 628 samples (2.5%)
- **VASC (Vascular lesion)**: 253 samples (1.0%)
- **DF (Dermatofibroma)**: 239 samples (0.9%)

**Imbalance ratio**: 54:1 (most vs least frequent class)

## 🛠️ **Solutions Implemented**

### **1. Class Weights (Recommended)**
```bash
python scripts/train_tricorder_optimized.py \
    --train_csv datasets/tricorder_training/train_labels.csv \
    --val_csv datasets/tricorder_training/val_labels.csv \
    --img_dir datasets/melanoma_dataset/images/ISIC_2019_Training_Input/ISIC_2019_Training_Input \
    --epochs 30 \
    --batch_size 32 \
    --use_class_weights \
    --convert_onnx
```

### **2. Weighted Random Sampling**
```bash
python scripts/train_tricorder_optimized.py \
    --train_csv datasets/tricorder_training/train_labels.csv \
    --val_csv datasets/tricorder_training/val_labels.csv \
    --img_dir datasets/melanoma_dataset/images/ISIC_2019_Training_Input/ISIC_2019_Training_Input \
    --epochs 30 \
    --batch_size 32 \
    --use_weighted_sampling \
    --convert_onnx
```

### **3. Focal Loss (Advanced)**
```bash
python scripts/train_tricorder_optimized.py \
    --train_csv datasets/tricorder_training/train_labels.csv \
    --val_csv datasets/tricorder_training/val_labels.csv \
    --img_dir datasets/melanoma_dataset/images/ISIC_2019_Training_Input/ISIC_2019_Training_Input \
    --epochs 30 \
    --batch_size 32 \
    --use_focal_loss \
    --focal_gamma 2.0 \
    --convert_onnx
```

### **4. Combined Approach (Best Results)**
```bash
python scripts/train_tricorder_optimized.py \
    --train_csv datasets/tricorder_training/train_labels.csv \
    --val_csv datasets/tricorder_training/val_labels.csv \
    --img_dir datasets/melanoma_dataset/images/ISIC_2019_Training_Input/ISIC_2019_Training_Input \
    --epochs 30 \
    --batch_size 32 \
    --use_class_weights \
    --use_weighted_sampling \
    --convert_onnx
```

## 📈 **Expected Improvements**

### **Without Balancing**
- **Accuracy**: High (biased toward majority class)
- **Recall (Minority classes)**: Very low
- **F1-Score**: Poor for rare classes
- **Clinical relevance**: Low (misses important cases)

### **With Balancing**
- **Accuracy**: Slightly lower but more meaningful
- **Recall (Minority classes)**: Significantly improved
- **F1-Score**: Better overall balance
- **Clinical relevance**: High (catches important cases)

## 🔧 **Advanced Techniques**

### **1. Data Augmentation for Minority Classes**
```python
# Increase augmentation for rare classes
rare_classes = ['VASC', 'DF', 'SCC', 'AK']
for class_name in rare_classes:
    # Apply more aggressive augmentation
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3)
```

### **2. Synthetic Data Generation**
```bash
# Generate synthetic samples for rare classes
python scripts/class_balancing_strategies.py
```

### **3. Ensemble Methods**
```bash
# Train multiple models with different balancing strategies
python scripts/train_tricorder_optimized.py --use_class_weights --output_name model_weights
python scripts/train_tricorder_optimized.py --use_focal_loss --output_name model_focal
python scripts/train_tricorder_optimized.py --use_weighted_sampling --output_name model_sampling
```

## 📊 **Monitoring and Evaluation**

### **Key Metrics to Watch**
1. **Per-class F1-Score**: Most important for medical diagnosis
2. **Confusion Matrix**: Visualize class performance
3. **Precision-Recall Curves**: For each class
4. **ROC-AUC**: Overall model performance

### **Class-Specific Targets**
- **MEL (Melanoma)**: High recall (don't miss cancer)
- **BCC, SCC**: High precision (avoid false alarms)
- **Rare classes**: Balanced precision/recall

## 🎯 **Recommended Training Strategy**

### **Phase 1: Baseline**
```bash
# Train without balancing to establish baseline
python scripts/train_tricorder_optimized.py \
    --epochs 10 \
    --output_name baseline
```

### **Phase 2: Class Weights**
```bash
# Add class weights
python scripts/train_tricorder_optimized.py \
    --epochs 20 \
    --use_class_weights \
    --output_name with_weights
```

### **Phase 3: Advanced Balancing**
```bash
# Add weighted sampling
python scripts/train_tricorder_optimized.py \
    --epochs 30 \
    --use_class_weights \
    --use_weighted_sampling \
    --output_name balanced
```

### **Phase 4: Focal Loss**
```bash
# Try focal loss for hard examples
python scripts/train_tricorder_optimized.py \
    --epochs 30 \
    --use_focal_loss \
    --focal_gamma 2.0 \
    --output_name focal
```

## 🚨 **Common Pitfalls**

### **1. Overfitting to Minority Classes**
- **Solution**: Use validation set to monitor per-class performance
- **Sign**: Validation loss increases while training loss decreases

### **2. Losing Majority Class Performance**
- **Solution**: Balance class weights carefully
- **Sign**: Overall accuracy drops significantly

### **3. Computational Overhead**
- **Solution**: Use efficient sampling strategies
- **Sign**: Training becomes very slow

## 📋 **Evaluation Checklist**

Before submitting your model:

- [ ] **Per-class F1-scores** are balanced
- [ ] **MEL (melanoma) recall** is high (>0.8)
- [ ] **Rare classes** have reasonable performance
- [ ] **Confusion matrix** shows no major biases
- [ ] **Validation curves** are stable
- [ ] **Model size** is within limits

## 🏆 **Competition Strategy**

### **Priority Order**
1. **MEL (Melanoma)**: Highest priority - don't miss cancer
2. **BCC, SCC**: High priority - other cancers
3. **Rare classes**: Medium priority - improve overall score
4. **Common classes**: Lower priority - already well-represented

### **Scoring Focus**
- **Sensitivity** for malignant classes
- **Specificity** for benign classes
- **Balanced F1** for overall performance

## 🔬 **Research Directions**

### **Advanced Techniques**
1. **Cost-sensitive learning**: Different costs for different misclassifications
2. **SMOTE**: Synthetic minority oversampling
3. **Adversarial training**: Generate hard examples
4. **Meta-learning**: Learn to balance classes

### **Architecture Improvements**
1. **Attention mechanisms**: Focus on important features
2. **Multi-task learning**: Auxiliary tasks for rare classes
3. **Hierarchical classification**: Group similar classes

---

**Remember**: In medical AI, missing a cancer case is much worse than a false alarm. Prioritize recall for malignant classes!
