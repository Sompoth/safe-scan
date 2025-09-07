# Tricorder Competition Submission

## Model Information
- **Model Size**: 42.45 MB
- **Input Format**: 512x512 RGB image + demographics [age, gender, location]
- **Output Format**: 10 class probabilities (sum to 1.0)
- **Average Inference Time**: 186.39 ms

## Usage
```bash
python inference.py --image <path> --age <age> --gender <m|f> --location <1-7>
```

## Example
```bash
python inference.py --image sample.jpg --age 50 --gender f --location 7
```

## Classes (in order)
0. Actinic keratosis (AK) - Benign
1. Basal cell carcinoma (BCC) - Malignant
2. Seborrheic keratosis (SK) - Medium risk
3. Squamous cell carcinoma (SCC) - Malignant
4. Vascular lesion (VASC) - Medium risk
5. Dermatofibroma (DF) - Benign
6. Benign nevus (NV) - Benign
7. Other non-neoplastic (NON) - Benign
8. Melanoma (MEL) - Malignant
9. Other neoplastic (ON) - Benign

## Performance Metrics
- Mean inference time: 186.39 ms
- Standard deviation: 66.29 ms
- Model size: 42.45 MB

## Competition Ready ✅
This submission meets all Tricorder competition requirements.
