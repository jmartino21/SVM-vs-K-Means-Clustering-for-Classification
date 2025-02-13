# SVM vs KNN vs K-Means Classification

## Overview
This project compares different classification models for predicting seed types based on their physical properties. The models include **Support Vector Machines (SVM)**, **K-Nearest Neighbors (KNN)**, and **K-Means Clustering**.

## Features
- **Area** – Surface area of the seed
- **Perimeter** – Boundary length of the seed
- **Compactness** – Shape compactness ratio
- **Length of Kernel** – Kernel length measurement
- **Width of Kernel** – Kernel width measurement
- **Asymmetry Coefficient** – Measure of seed asymmetry
- **Length of Kernel Groove** – Measurement of groove length
- **Class** – Label for seed type (Filtered for classes 1 and 2)

## Models Implemented
- **Support Vector Machines (SVM)**:
  - Linear SVM
  - Gaussian SVM
  - Polynomial SVM (Degree = 3)
- **K-Nearest Neighbors (KNN)**
- **K-Means Clustering** (Unsupervised Learning)

## Dataset
This project requires the **seeds_dataset_1.csv** dataset. Ensure it is placed in the same directory as the script. If missing, you can download it from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/seeds).

## Usage
### Running the Script
Execute the script using:
```bash
python svm_knn_kmeans_classification_6.677.py
```

### Steps Performed
1. Loads and preprocesses the dataset.
2. Splits the dataset into training and testing sets.
3. Standardizes feature values.
4. Trains and evaluates SVM, KNN, and K-Means models.
5. Displays accuracy scores and confusion matrices.

## Output
- Accuracy scores for each classification model.
- Confusion matrices for SVM and KNN models.
- K-Means clustering accuracy.

## Findings
- **K-Means clustering achieved the highest accuracy (85%)**, outperforming supervised models.
- **KNN performed better than all SVM models**, reaching **54% accuracy**.
- **Gaussian SVM had the highest TPR (94%) but lowest classification accuracy (20%)**, indicating overfitting.

## License
This project is open-source and available for modification and use.

****
