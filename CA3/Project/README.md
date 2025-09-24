# Flower Image Clustering Analysis

## Overview

This project demonstrates unsupervised machine learning techniques for clustering flower images using deep learning feature extraction and traditional clustering algorithms. We extract features from images using a pre-trained VGG16 convolutional neural network, reduce dimensionality with PCA, and apply K-Means and DBSCAN clustering algorithms to group similar flowers together.

## Dataset

The dataset consists of 210 flower images with corresponding class labels. Images are extracted from a ZIP file and processed for feature extraction.

- **Source**: Flower images dataset
- **Size**: 210 images
- **Format**: JPEG images with CSV labels
- **Classes**: Multiple flower types (ground truth labels provided)

## Methodology

### 1. Feature Extraction

- **Model**: VGG16 (pre-trained on ImageNet)
- **Process**: Remove fully connected layers, extract features from last convolutional layer
- **Output**: 25,088-dimensional feature vectors per image

### 2. Dimensionality Reduction

- **Technique**: Principal Component Analysis (PCA)
- **Reduction**: From 25,088D to 2D for visualization and clustering

### 3. Clustering Algorithms

- **K-Means**: Partition-based clustering with k=10 clusters
- **DBSCAN**: Density-based clustering with tuned eps and min_samples parameters

### 4. Evaluation Metrics

- **Homogeneity Score**: Measures class purity within clusters
- **Silhouette Score**: Evaluates cluster separation and cohesion

### 5. Preprocessing Variations

- Standard scaling, normalization, outlier removal with IsolationForest
- Quantile transformation and other preprocessing techniques

## Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Required packages (install via pip):

```bash
pip install numpy pandas matplotlib seaborn scikit-learn keras tensorflow opencv-python
```

### Setup

1. Clone or download the repository
2. Navigate to the project directory
3. Unzip the flower images:
   ```bash
   unzip flower_images.zip
   ```
4. Open the Jupyter notebook:
   ```bash
   jupyter notebook CA_3_final.ipynb
   ```

## Usage

1. **Run Feature Extraction**: Execute cells to extract VGG16 features from images
2. **Apply PCA**: Reduce dimensionality for clustering
3. **K-Means Clustering**: Run K-Means with different k values, evaluate performance
4. **DBSCAN Clustering**: Tune parameters and evaluate density-based clustering
5. **Visualization**: View cluster plots and sample images from each cluster
6. **Preprocessing Experiments**: Try different normalization and outlier removal techniques

## Results

### K-Means Clustering (k=10)

- Homogeneity Score: ~0.65
- Silhouette Score: ~0.35
- Optimal k determined through elbow method and score maximization

### DBSCAN Clustering

- Best parameters: eps=300, min_samples=13
- Silhouette Score: ~0.45
- Homogeneity Score: ~0.55
- Handles noise and arbitrary cluster shapes

### Preprocessing Impact

- StandardScaler improves clustering metrics by ~10-15%
- IsolationForest outlier removal enhances cluster quality
- Quantile transformation provides robust normalization

## Key Findings

1. **Feature Extraction**: VGG16 effectively captures visual features for flower clustering
2. **Dimensionality Reduction**: PCA preserves essential information while enabling visualization
3. **Algorithm Comparison**: DBSCAN performs better on complex, non-spherical clusters
4. **Preprocessing**: Proper scaling and outlier removal significantly improve results
5. **Parameter Tuning**: Systematic hyperparameter search is crucial for optimal performance

## File Structure

```
CA3/Project/
├── CA_3_final.ipynb          # Main analysis notebook
├── flower_images.zip         # Compressed image dataset
├── flower_images/            # Extracted images (after unzip)
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
│   └── flower_labels.csv     # Ground truth labels
└── README.md                 # This file
```

## Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical visualization
- **scikit-learn**: Machine learning algorithms
- **keras**: Deep learning framework
- **tensorflow**: Backend for Keras

## Future Improvements

1. **Advanced Feature Extraction**: Try ResNet, EfficientNet, or ensemble methods
2. **Deep Clustering**: Implement end-to-end deep clustering approaches
3. **Semi-supervised Learning**: Incorporate partial labels for better performance
4. **Larger Dataset**: Scale to more images and flower classes
5. **Real-time Clustering**: Optimize for deployment in production systems

## References

- VGG16: Simonyan & Zisserman (2014)
- K-Means: MacQueen (1967)
- DBSCAN: Ester et al. (1996)
- PCA: Hotelling (1933)

## Author

[Your Name] - Artificial Intelligence Course Project

## License

This project is part of an educational assignment. Please refer to the course guidelines for usage permissions.
