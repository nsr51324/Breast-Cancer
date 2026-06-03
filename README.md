# Breast Cancer Classification using Machine Learning

## Overview
This project predicts whether a tumor is malignant or benign using machine learning techniques based on medical features extracted from cell nuclei.

## Dataset
- Breast Cancer Dataset
- Features include:
  - texture_mean
  - perimeter_mean
  - area_mean
  - and other statistical measurements

## Workflow

### 1. Data Preprocessing
- Removed unnecessary column (`id`)
- Checked duplicates
- Handled missing values
- Encoded target variable using LabelEncoder

### 2. Handling Imbalance
- Balanced dataset using oversampling technique

### 3. Feature Scaling
- Applied MinMaxScaler to normalize feature values

## Model Used
- Logistic Regression

## Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision / Recall / F1-score)

## Results
- Model achieves good classification performance on balanced dataset
- Visualized using confusion matrix heatmap

## Model Saving
Model saved using Joblib:

- cancer_model.pkl

## How to Run
### Install dependencies
```bash
pip install -r requirements.txt
