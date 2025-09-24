
###  **AI-S03-A3 Clustering & Unsupervised Learning Solutions**

## **Project Overview**

This repository contains solutions for  **AI-S03-A3** , focusing on  **Clustering Algorithms, Feature Engineering, and Dimensionality Reduction** . The project covers different  **unsupervised learning techniques** , including:

* **K-Means Clustering**
* **DBSCAN Clustering**
* **Agglomerative (Hierarchical) Clustering**
* **Dimensionality Reduction using PCA**
* **Feature Extraction for Image Clustering (VGG16)**

This repository is useful for understanding how different clustering techniques work, how to evaluate their effectiveness, and how they can be applied to **real-world datasets** such as image classification and customer segmentation.

---

---

## **Key Topics & Techniques Covered**

### **1. K-Means Clustering**

* Assigns **K clusters** based on centroid initialization.
* Uses **Euclidean Distance** to classify points.
* Iteratively updates cluster centers until  **convergence** .
* **Key steps implemented:**
  * **Initialize cluster centers**
  * **Assign data points to nearest cluster**
  * **Recalculate cluster centroids**
  * **Repeat until convergence**

### **2. DBSCAN Clustering (Density-Based)**

* Finds **clusters based on density** instead of centroids.
* Can detect **outliers** and  **non-spherical clusters** .
* Requires tuning of two parameters:
  * **Epsilon (ε)** - Defines neighborhood radius.
  * **MinPoints** - Minimum points required for dense region.
* **Implemented cases:**
  * ε = 2, MinPoints = 2
  * ε = 10, MinPoints = 2

### **3. Hierarchical Clustering (Agglomerative)**

* Builds a **hierarchical structure** of clusters.
* Uses **distance matrices** to merge closest points.
* Dendrogram used to  **visualize hierarchical grouping** .
* **Complete, Single, and Average Linkage** methods tested.

### **4. Feature Extraction using VGG16**

* Uses **pre-trained Convolutional Neural Network (CNN)** for feature extraction.
* Extracts **high-level features** from image datasets.
* Fully-connected layers removed to get feature vectors.
* Helps improve clustering performance on image datasets.

### **5. Dimensionality Reduction using PCA**

* Principal Component Analysis ( **PCA** ) used to reduce feature dimensions.
* Reduces **computational complexity** and improves visualization.
* Allows plotting  **clusters in 2D or 3D** .
* Used to  **compare K-Means and DBSCAN clustering results** .

---

---

## **Key Questions Addressed in the Assignment**

### **K-Means Clustering**

* How do you determine the number of  **K clusters** ?
* How does **centroid initialization** affect convergence?
* How many iterations are required for convergence?

### **DBSCAN Clustering**

* How does **epsilon (ε) and MinPoints** impact clustering?
* How does DBSCAN handle **outliers** compared to K-Means?
* What are the advantages of  **density-based clustering** ?

### **Hierarchical Clustering**

* What is the difference between  **Agglomerative & Divisive Clustering** ?
* How can **Dendrograms** be used for cluster visualization?
* How does **linkage method** affect hierarchical clustering results?

### **Feature Extraction & PCA**

* Why is feature extraction important for clustering?
* How does  **VGG16 improve feature representation** ?
* How does  **PCA help in reducing dimensionality** ?

---

## **Results & Performance Analysis**

* **Cluster Assignments for K-Means**
* **Cluster Stability with Different Initializations**
* **Effect of PCA on Clustering Accuracy**
* **Comparison of K-Means vs DBSCAN vs Hierarchical Clustering**
* **Silhouette Score & Homogeneity Score for Cluster Evaluation**

Results are stored in the `results/` directory, containing  **cluster visualizations, PCA plots, and performance metrics** .
