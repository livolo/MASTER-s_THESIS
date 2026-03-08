# Mammography AI Detection System

## Overview
This project focuses on developing an Artificial Intelligence system for detecting and analyzing abnormalities in mammography images. The goal is to assist in early breast cancer detection by combining deep learning models for lesion detection and classification.

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
Downloaded three CSV files from IBIA: details.csv, Image_details.csv, and Indian Biological Images Archive_labelling.csv.
Combined the data from multiple databases:
Step 1: Merged the details.csv with labelling information on Sample ID.
Step 2: Merged result with Image_details.csv on Experiment ID.
Essential columns retained: image_path & label (binary: "Cancer" vs. "Not cancer").
All DICOM images in a_data/ were converted to PNG format in the converted_png/ directory.
Made a stratified data-split for the images into cancer and non-cancer categories to avoid any leakages. 
Views: 118
Images: 3577
Downloads: 203
Data size: 63GB
Generated final_dataset/ with train/validation/test splits-e.g., 70/15/15 ratio-from dataset/ folder.
Made sure class balance by stratified sampling.


## Project Structure
```
Mammography-AI
│
├── apphybrid.py
├── train_resnet50.py
├── train_faster_rcnn.py
├── requirements.txt
│
├── notebooks/
│   └── evaluation.ipynb
│
└── results/
    ├── confusion_matrix.png
    └── roc_curve.png
```

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