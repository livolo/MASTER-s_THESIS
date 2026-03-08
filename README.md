# Development of a Three-Stage Deep Learning–Based Computer-Aided Diagnosis Software for Breast Cancer Classification and Tumor Localization in Mammography

## Overview
This project presents a deep learning–based pipeline for analyzing mammography images from the **Indian Biological Images Archive (IBIA)**. The objective is to develop an interpretable AI system capable of detecting suspicious breast cancer regions and assisting clinical diagnosis.

The workflow integrates:

- Clinical tabular data analysis  
- CNN-based image classification  
- Grad-CAM interpretability  
- Pseudo bounding box generation  
- Object detection using Faster R-CNN  

The pipeline is designed to address key challenges such as:

- Data leakage prevention  
- Overfitting control  
- Dependence on clinical metadata  
- Interpretable tumor localization  

---

## Project Objective
The goal of this project is to design a hybrid deep learning pipeline capable of:

- Detecting suspicious regions in mammography images  
- Classifying breast cancer patterns  
- Providing interpretable predictions using Grad-CAM  
- Assisting medical image analysis using AI  

---

## Methodology
The system integrates several deep learning components:

1. Image preprocessing and dataset preparation  
2. CNN-based classification using **ResNet-50**  
3. Grad-CAM visualization for interpretability  
4. Pseudo bounding box generation from activation maps  
5. Object detection refinement using **Faster R-CNN**  
6. Model evaluation using:
   - Confusion Matrix
   - ROC Curve
   - Precision–Recall analysis

---

## Technologies Used

- Python  
- PyTorch  
- TensorFlow  
- Computer Vision  
- Deep Learning  
- Jupyter Notebook  

---

## Model Architecture

### Detection Model
- Faster R-CNN

### Classification Model
- ResNet-50

### Hybrid Pipeline
- Feature extraction  
- Model fusion  
- Prediction visualization  

---

## Results
Model performance was evaluated using multiple metrics:

- Confusion Matrix  
- Precision–Recall Curve  
- ROC Curve  
- Feature Importance Analysis  

These evaluations demonstrate the effectiveness of deep learning models for mammography-based breast cancer detection.

---

## Dataset

The dataset was obtained from the **Indian Biological Images Archive (IBIA)**.

### Data Preparation

1. Downloaded three CSV files:
   - `details.csv`
   - `Image_details.csv`
   - `Indian Biological Images Archive_labelling.csv`

2. Data merging process:
   - `details.csv` merged with labeling information using **Sample ID**
   - Result merged with `Image_details.csv` using **Experiment ID**

3. Essential fields retained:
   - `image_path`
   - `label` (Cancer / Not Cancer)

4. Image preprocessing:
   - DICOM images converted to PNG format

```
a_data/ → converted_png/
```

5. Dataset splitting:

```
Train : 70%
Validation : 15%
Test : 15%
```

### Dataset Statistics

| Property | Value |
|--------|------|
Views | 118 |
Images | 3577 |
Downloads | 203 |
Dataset Size | ~63 GB |

Due to the large dataset size, the full dataset is **not included in this repository**.

---

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

---

## Applications

- Breast cancer early detection  
- Medical image analysis  
- Clinical decision support systems  
- AI-assisted diagnostic tools  

---

## Future Work

- Improve model performance with larger datasets  
- Integrate explainable AI techniques  
- Develop a graphical interface for diagnosis  
- Build a web-based clinical decision support tool  

---

## Author

**Kirti Vishwakarma**  
Bioinformatics & AI Research