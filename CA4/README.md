# Boston Housing Price Prediction with Machine Learning

## Overview

This project implements a comprehensive machine learning pipeline for predicting housing prices in Boston using the Boston Housing Dataset. The analysis covers data exploration, preprocessing, and multiple modeling techniques including linear regression, polynomial regression, gradient descent, decision trees, K-Nearest Neighbors (KNN), and Support Vector Machines (SVM). The project demonstrates the application of various AI/ML algorithms to understand housing price determinants and evaluate model performance through appropriate metrics.

## Key Features

- **Data Exploration**: Comprehensive EDA including correlation analysis, feature distributions, and missing value assessment
- **Data Preprocessing**: Multiple imputation techniques, feature scaling, and data splitting strategies
- **Model Development**: Implementation of regression and classification algorithms from scratch and using scikit-learn
- **Model Evaluation**: Performance comparison using RMSE, R² score, accuracy, precision, recall, F1-score, and ROC-AUC
- **Visualization**: Extensive plotting for data understanding and model interpretation

## Technologies Used

- **Programming Language**: Python 3.8+
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Development Environment**: Jupyter Notebook

## Installation/Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/tahamajs/AI-CA3_-Clustering_PCA_Unsupervised_Learning_Solutions.git
   cd AI-CA3_-Clustering_PCA_Unsupervised_Learning_Solutions/CA4/Project
   ```

2. Install required packages:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. Ensure the dataset file `DataSet.xlsx` is in the project directory.

## Data Summary

- **Dataset**: Boston Housing Dataset
- **Source**: Excel file (`DataSet.xlsx`)
- **Size**: 506 samples, 14 features
- **Target Variable**: MEDV (Median value of owner-occupied homes in $1000's)
- **Features**: Crime rate, zoning, industry, air quality, rooms, age, employment access, taxes, pupil-teacher ratio, demographics

## How to Run

1. Open the Jupyter notebook:

   ```bash
   jupyter notebook CA4.ipynb
   ```

2. Execute cells sequentially to reproduce the analysis

3. The notebook is self-contained and includes all necessary code for data loading, preprocessing, modeling, and evaluation

## Results Summary

### Regression Models

- **Linear Regression**: Manual implementation and scikit-learn comparison
- **Polynomial Regression**: Degree optimization and performance analysis
- **Gradient Descent**: Custom implementation for polynomial regression

### Classification Models

- **Decision Trees**: Pruning analysis and visualization
- **K-Nearest Neighbors**: Distance metrics comparison and hyperparameter tuning
- **Support Vector Machines**: Kernel comparison and ROC curve analysis

### Key Findings

- RM (average rooms) and LSTAT (% lower status) are strongest predictors
- Polynomial regression shows overfitting with high degrees
- SVM with RBF kernel achieves best classification performance
- KNN performance sensitive to distance metrics and k-value

## Licensing

This project is licensed under the MIT License - see the LICENSE file in the root directory for details.

## Author

**Mohammad Taha Majlesi** (810101504)
Artificial Intelligence Course 2024
University of Tehran
