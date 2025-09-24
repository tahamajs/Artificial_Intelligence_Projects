# AI-S03-A4 & HW-4: Machine Learning Solutions

## Project Overview

This repository includes solutions for **AI-S03-A4** (Artificial Intelligence - Assignment 4) and **HW-4** (Homework 4) focused on **Machine Learning algorithms** and  **data analysis** . The project applies various techniques such as  **K-Nearest Neighbors (KNN)** ,  **Support Vector Machines (SVM)** ,  **Linear Regression** , and **Logistic Regression** on real-world datasets.

This project will help you understand and apply foundational machine learning concepts through the following main tasks:

* **Data Preprocessing** and **Feature Engineering**
* **Modeling** using various algorithms
* **Performance Evaluation** using metrics like  **Accuracy, Precision, Recall** , and **F1-Score**

---

## Key Concepts and Techniques

### 1. **K-Nearest Neighbors (KNN)**

* **KNN** is a non-parametric algorithm used for classification and regression tasks.
* The classification of a data point is determined based on the majority class of its **k** closest neighbors.
* **Distance metrics** like **Euclidean Distance** and **Manhattan Distance** are used to calculate the proximity between data points.

### 2. **Support Vector Machine (SVM)**

* **SVM** is a powerful classifier that works by finding the optimal hyperplane that separates data points of different classes.
* It uses  **support vectors** , which are the points closest to the hyperplane and are crucial for defining the decision boundary.
* **Kernels** such as  **linear** ,  **polynomial** , and **RBF** help map non-linear data to higher dimensions for linear separation.

### 3. **Linear Regression**

* Linear Regression models the relationship between a dependent variable and one or more independent variables by fitting a linear equation.
* It aims to minimize the  **mean squared error (MSE)** , which measures the difference between predicted and actual values.
* Methods like **Gradient Descent** can be used for optimization in large datasets.

### 4. **Logistic Regression**

* **Logistic Regression** is used for binary classification tasks, predicting the probability that a given input belongs to a certain class (spam or not spam).
* It uses a **logistic function** to model the decision boundary, and the output is a probability between 0 and 1.

### 5. **Model Evaluation Metrics**

* **Confusion Matrix** : Provides insights into the true positives, true negatives, false positives, and false negatives.
* **Accuracy** : The fraction of correctly classified data points.
* **Precision** : The proportion of true positive results among all the positive predictions.
* **Recall** : The proportion of true positive results among all the actual positives.
* **F1-Score** : The harmonic mean of precision and recall, used for imbalanced classes.
* **AUC-ROC Curve** : Helps evaluate classification performance, especially in imbalanced datasets.

---

Key Questions Addressed in the Assignments

1. **How to handle missing data in datasets?**
   * Solutions include removing or imputing missing values using techniques like  **mean/mode imputation** , or using more advanced models like  **Random Forest** .
2. **How to deal with imbalanced datasets?**
   * Techniques like **resampling** (oversampling and undersampling), using algorithms like **Balanced Random Forest** or  **SMOTE** , and adjusting class weights in models are commonly applied.
3. **What metrics are best for evaluating classification models?**
   * **Precision, Recall, F1-Score** , and **ROC-AUC** are all essential for assessing classification models, especially in the case of imbalanced datasets.
4. **What are Support Vectors in SVM?**
   * Support vectors are data points that are closest to the decision boundary. These points are critical in defining the hyperplane in  **SVM** .
5. **How do kernels work in SVM?**
   * **Kernels** map data into higher-dimensional space where linear separation becomes easier. The most commonly used kernels include  **linear** ,  **polynomial** , and  **RBF** .

---

## Results & Performance Evaluation

The solutions in this repository include model training and evaluation using multiple metrics.

* **Accuracy, Precision, Recall, and F1-Score** for classification models
* **Mean Squared Error (MSE)** for regression models
* **Confusion Matrix and ROC-AUC** for evaluating classifier performance
* Results can be found in the `results/` directory, which includes performance graphs and comparison of different models.

---
