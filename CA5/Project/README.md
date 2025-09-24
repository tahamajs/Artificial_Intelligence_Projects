# Artificial Intelligence Course Assignment 5: Convolutional Neural Networks for Suicidal Tweet Classification

## Project Overview

This project implements a Convolutional Neural Network (CNN) for detecting suicidal ideation in Twitter posts. The system preprocesses text data, converts it to word embeddings using Word2Vec, and classifies tweets as either indicating suicidal thoughts (label 1) or not (label 0).

## Authors

- Mohammad Taha Majlesi (810101504)
- Course: Artificial Intelligence
- Instructor: Dr. Fadaei and Dr. Yaghoobzaadeh

## Table of Contents

1. [Introduction](#introduction)
2. [Data Acquisition](#data-acquisition)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Feature Engineering](#feature-engineering)
5. [Model Architecture and Definition](#model-architecture-and-definition)
6. [Training Configuration](#training-configuration)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Conclusion and Future Work](#conclusion-and-future-work)
9. [Installation and Setup](#installation-and-setup)
10. [Usage](#usage)
11. [Results](#results)

## Introduction

### Project Objective
Develop a deep learning model for detecting suicidal thoughts in social media posts, specifically Twitter tweets, classifying them as suicidal (1) or non-suicidal (0).

### Problem Domain
Suicidal ideation detection is critical in mental health. Social media provides valuable data for early intervention, but noisy text requires sophisticated NLP techniques.

### Dataset Description
- **Source**: Twitter social media platform
- **Size**: Approximately 2,000+ tweets
- **Features**: Raw tweet text
- **Labels**: Binary classification (0: non-suicidal, 1: suicidal)
- **Characteristics**: Informal language, abbreviations, emojis, varying lengths

### AI/ML Technique Used
- **Primary Technique**: Convolutional Neural Networks (CNNs) for text classification
- **Word Embeddings**: Word2Vec for text-to-vector conversion
- **Preprocessing**: Text cleaning, tokenization, lemmatization, stop word removal
- **Optimization**: Stochastic Gradient Descent (SGD) with momentum
- **Evaluation**: Precision, Recall, F1-Score, Confusion Matrix

## Data Acquisition

### Installing Required Libraries
The project requires several Python libraries for deep learning, NLP, and data processing. Install them using pip.

### Loading Libraries and Setting Up Environment
All necessary libraries are imported at the beginning for clean code organization. SSL context is configured for secure downloads.

## Exploratory Data Analysis (EDA)

The EDA section explores the processed dataset, including sequence lengths, token validity, and data structure validation.

## Feature Engineering

### Text Preprocessing
- Convert to lowercase
- Remove punctuation, numbers, URLs, user mentions
- Handle emojis by converting to text descriptions
- Normalize whitespaces
- Tokenize and lemmatize
- Remove stop words

### Word Embeddings
- Use pre-trained Word2Vec model (Google News 300D)
- Convert tokens to 300-dimensional vectors
- Handle out-of-vocabulary words with zero vectors
- Pad/truncate sequences to fixed length (64 tokens)

## Model Architecture and Definition

### CNN Architecture
- **Input**: 64x300 matrix (sequence length x embedding dimension)
- **Conv1D Layer 1**: 64 filters, kernel size 3, padding 1
- **MaxPool1D**: Kernel size 2, stride 2
- **Conv1D Layer 2**: 128 filters, kernel size 3, padding 1
- **MaxPool1D**: Kernel size 2, stride 2
- **Conv1D Layer 3**: 256 filters, kernel size 3, padding 1
- **MaxPool1D**: Kernel size 2, stride 2
- **Fully Connected 1**: 64 units
- **Fully Connected 2**: 1 unit (sigmoid for binary classification)

## Training Configuration

### Training Setup
- **Loss Function**: Binary Cross-Entropy Loss
- **Optimizer**: Stochastic Gradient Descent with momentum
- **Batch Size**: Configurable (default 32)
- **Epochs**: Configurable
- **Device**: Auto-detect GPU/CPU

### Data Splitting
- 80% training, 20% validation
- Stratified split to maintain class balance

## Evaluation Metrics

### Metrics Used
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Evaluation Process
- Model evaluation on validation set during training
- Final evaluation with detailed classification report
- Confusion matrix visualization

## Conclusion and Future Work

### Key Achievements
- Successfully implemented CNN for text classification
- Achieved robust preprocessing pipeline
- Demonstrated effective word embedding usage
- Provided comprehensive evaluation framework

### Future Improvements
- Experiment with different embedding techniques (BERT, GloVe)
- Implement attention mechanisms
- Add regularization techniques (Dropout, Batch Normalization)
- Explore larger datasets and transfer learning
- Deploy as web service for real-time detection

## Installation and Setup

### Prerequisites
- Python 3.7+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Installation
```bash
pip install torch torchvision torchaudio
pip install gensim nltk emoji tqdm seaborn scikit-learn matplotlib pandas numpy
```

### Setup
1. Clone the repository
2. Download the dataset (twitter-suicidal-data.csv)
3. Run the Jupyter notebook

## Usage

### Running the Notebook
1. Open `CA5 (1).ipynb` in Jupyter or Google Colab
2. Execute cells in order
3. Monitor training progress and evaluation metrics

### Key Functions
- `preprocess_data()`: Text preprocessing
- `Twitter()`: Dataset class for data loading
- `CNN()`: Neural network model
- `train_model()`: Training loop
- `generate_confusion_matrix()`: Model evaluation

## Results

### Performance Metrics
- **Accuracy**: [Insert final accuracy]
- **F1-Score**: [Insert F1-score]
- **Precision**: [Insert precision]
- **Recall**: [Insert recall]

### Training History
- Loss curves and accuracy plots available in notebook
- Model checkpoints saved for best validation performance

## File Structure
```
CA5/
├── Project/
│   ├── CA5 (1).ipynb          # Main notebook
│   ├── twitter-suicidal-data.csv  # Dataset
│   └── README.md               # This file
├── Description/
│   └── [PDF files with project description]
└── ...
```

## License

This project is part of the Artificial Intelligence course assignment and is intended for educational purposes.

## Acknowledgments

- Dr. Fadaei and Dr. Yaghoobzaadeh for course instruction
- University of Tehran for providing the course framework
- Open-source community for PyTorch, Gensim, and other libraries