#!/usr/bin/env python3
"""
Tricorder Competition Inference Script
Usage: python inference.py --image <path> --age <age> --gender <m|f> --location <1-7>
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tricorder_submission import TricorderSubmission
import argparse

def main():
    parser = argparse.ArgumentParser(description='Tricorder Competition Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--age', type=int, required=True, help='Patient age in years')
    parser.add_argument('--gender', type=str, required=True, choices=['m', 'f'], help='Patient gender')
    parser.add_argument('--location', type=int, required=True, choices=range(1, 8), help='Body location 1-7')
    
    args = parser.parse_args()
    
    # Initialize model
    submission = TricorderSubmission('tricorder_model.onnx')
    
    # Run prediction
    probabilities = submission.predict(args.image, args.age, args.gender, args.location)
    
    # Print results (competition format)
    print(probabilities)

if __name__ == "__main__":
    main()
