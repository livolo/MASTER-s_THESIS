# Mammography AI Detection System

## Overview
This project focuses on developing an Artificial Intelligence (AI) system for detecting and analyzing abnormalities in mammography images. The goal is to assist in early breast cancer detection by combining deep learning models for lesion detection and classification.

## Project Objective
To design a hybrid deep learning pipeline capable of:
- Detecting suspicious regions in mammography images
- Classifying breast cancer patterns
- Assisting medical image analysis using AI

## Methodology
The system integrates multiple deep learning techniques:

1. Image preprocessing and dataset preparation
2. Object detection using Faster R-CNN
3. Classification using ResNet-50
4. Hybrid CNN pipeline for improved prediction
5. Model evaluation using confusion matrix, ROC curve, and precision-recall analysis

## Technologies Used
- Python
- PyTorch
- TensorFlow
- Computer Vision
- Deep Learning
- Jupyter Notebook

## Model Architecture
The pipeline includes:

Detection Model:
- Faster R-CNN

Classification Model:
- ResNet-50

Hybrid Pipeline:
- Feature extraction
- Model fusion
- Prediction visualization

## Results
The model performance was evaluated using multiple metrics:

- Confusion Matrix
- Precision-Recall Curve
- ROC Curve
- Feature Importance Analysis

## Dataset
The mammography dataset used in this project is approximately **64 GB** and therefore cannot be hosted directly on GitHub.

Dataset access information will be provided separately.

## Project Structure
Mammography-AI
│
├── apphybrid.py
├── train_resnet50.py
├── train_faster_rcnn.py
├── requirements.txt
├── notebooks/
│   ├── evaluation.ipynb
│
└── results/
    ├── confusion_matrix.png
    ├── roc_curve.png
    
## Applications
- Breast cancer early detection
- Medical image analysis
- Clinical decision support systems
- AI-assisted diagnostics

## Future Work
- Improve model accuracy with larger datasets
- Integrate explainable AI methods
- Develop a web-based diagnostic interface

## Author
Kirti Vishwakarma  
Bioinformatics / AI Research