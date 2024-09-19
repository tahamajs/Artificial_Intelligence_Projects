# University of Tehran - Artificial Intelligence Course

This repository contains the coursework and assessments for the **Artificial Intelligence** course at the **University of Tehran**. Each folder represents a different coursework assignment (referred to as "CA"), and additional resources are provided for reference.

## Repository Structure

- [**CA1/**](#coursework-assignment-1-search-and-genetic-algorithms) - **Coursework Assignment 1:** Search and Genetic Algorithms.
- [**CA2/**](#coursework-assignment-2-bayesian-networks-and-hidden-markov-models) - **Coursework Assignment 2:** Bayesian Networks and Hidden Markov Models.
- [**CA3/**](#coursework-assignment-3-clustering-and-dimensionality-reduction) - **Coursework Assignment 3:** Clustering and Dimensionality Reduction.
- [**CA4/**](#coursework-assignment-4-data-analysis-and-machine-learning-algorithms) - **Coursework Assignment 4:** Data Analysis and Machine Learning Algorithms.
- [**CA5/**](#coursework-assignment-5-convolutional-neural-networks-cnns-and-word-embeddings) - **Coursework Assignment 5:** Convolutional Neural Networks (CNNs) and Word Embeddings.
- [**CA6/**](#coursework-assignment-6-reinforcement-learning-and-deep-q-networks-dqn) - **Coursework Assignment 6:** Reinforcement Learning and Deep Q-Networks (DQN).
- **Extra/** - **Additional Material:** Extra material related to the course, including code snippets, research papers, or supplementary assignments.
- **REF/** - **Reference Material:** Contains reference documents, papers, or other resources used throughout the course.
- **Slides/** - **Course Slides:** Lecture slides provided by the instructors.

---

## Coursework Assignments

### Coursework Assignment 1: Search and Genetic Algorithms

This assignment focuses on search algorithms and the application of genetic algorithms to solve the Knapsack problem.

#### Project Overview

##### Part 1: Search Algorithms

1. **BFS (Breadth-First Search) and DFS (Depth-First Search):**
   - Implemented to find the optimal path on a given graph.
   - Compared both algorithms based on different types of graphs.
   - Included tracking of explored and frontier sets.

2. **Uniform Cost Search:**
   - Implemented to find the minimum cost path from a start node (S) to a goal node (G).
   - Calculated the total cost considering both the path cost and heuristic values.

3. **A\* Search:**
   - Implemented a heuristic search algorithm to find the shortest path on a graph.
   - Evaluated the admissibility and consistency of the heuristic function used.

##### Part 2: Genetic Algorithm for the Knapsack Problem

1. **Knapsack Problem:**
   - Implemented a genetic algorithm to maximize the value of items packed in a knapsack without exceeding its weight limit.

2. **Genetic Algorithm Implementation:**
   - **Chromosome Representation:** Each chromosome represents a possible selection of items.
   - **Population Initialization:** Randomly generated initial population.
   - **Fitness Function:** Evaluates chromosomes based on total value and weight of selected items.
   - **Crossover and Mutation:** Used to produce new generations of solutions.
   - **Selection:** Best solutions are selected based on the fitness function to form the next generation.

---

### Coursework Assignment 2: Bayesian Networks and Hidden Markov Models

This assignment covers Bayesian Networks and Hidden Markov Models (HMM), focusing on probability calculations, forward/backward algorithms, and model evaluation.

#### Project Overview

##### Part 1: Bayesian Networks

1. **Bayesian Network Design:**
   - Evaluated the probability of car accidents based on factors like car age, weather conditions, and maintenance.
   - Implemented probability tables for each variable.

2. **Probability Calculations:**
   - Computed joint and conditional probabilities using the Bayesian Network.
   - Implemented the Variable Elimination algorithm for specific probabilities.

3. **Independence Analysis:**
   - Analyzed independence and conditional independence of variables.
   - Justified dependencies based on network structure.

##### Part 2: Hidden Markov Models (HMM)

1. **HMM Design for Music Genre Prediction:**
   - Hidden states represent user emotions; observations represent music genres.
   - Configured transition, emission, and initial probability matrices.

2. **Algorithm Implementation:**
   - **Forward Algorithm:** Calculated the probability of an observation sequence.
   - **Smoothing Algorithm:** Computed the probability of a hidden state at a given time.
   - **Viterbi Algorithm:** Determined the most likely sequence of hidden states.

3. **Practical Implementation:**
   - Implemented HMM using both `hmmlearn` library and a custom implementation.
   - Compared results and performance.

##### Data Processing and Feature Extraction

1. **Feature Extraction:**
   - Extracted features like MFCC, Zero Crossing Rate, and chroma features from audio data.
   - Investigated their impact on HMM performance.

2. **Evaluation Metrics:**
   - Evaluated model performance using Accuracy, Precision, Recall, and F1 Score.
   - Implemented metrics from scratch and compared with library results.

---

### Coursework Assignment 3: Clustering and Dimensionality Reduction

This assignment involves implementing clustering algorithms, feature extraction, dimensionality reduction, and cluster evaluation.

#### Project Overview

##### Part 1: Clustering Algorithms

1. **K-Means Clustering:**
   - Clustered points into three clusters using K-Means.
   - Calculated new cluster centers and visualized clusters.
   - Determined iterations needed for convergence.

2. **DBSCAN Clustering:**
   - Applied DBSCAN with different `ε` (epsilon) and `minPoints`.
   - Visualized clusters and identified core, border, and noise points.

3. **Agglomerative Clustering:**
   - Performed clustering using a distance matrix.
   - Visualized hierarchical clustering with a dendrogram.

##### Part 2: Practical Implementation - Clustering Images

1. **Dataset:**
   - Used an image dataset associated with different colors.

2. **Feature Extraction:**
   - Employed VGG16 for high-level feature extraction.
   - Researched techniques and preprocessing required.

3. **Clustering Implementation:**
   - Applied K-Means and DBSCAN to feature vectors.
   - Experimented with different parameters.
   - Compared clustering results.

##### Part 3: Dimensionality Reduction and Visualization

1. **PCA:**
   - Reduced dimensionality using Principal Component Analysis.
   - Visualized clusters in 2D/3D space.

2. **Evaluation:**
   - Used silhouette and homogeneity scores to evaluate clusters.
   - Compared results before and after dimensionality reduction.

##### Part 4: Analysis and Discussion

- **Comparison of Methods:**
  - Discussed pros and cons of K-Means vs. DBSCAN.
  - Provided insights on algorithm preferences.

- **Cluster Evaluation:**
  - Detailed calculation methods for evaluation metrics.
  - Suggested improvements for cluster performance.

---

### Coursework Assignment 4: Data Analysis and Machine Learning Algorithms

This assignment covers data preprocessing, regression, classification, model evaluation, and ensemble methods.

#### Project Overview

##### Part 1: Data Analysis and Preprocessing

1. **Exploratory Data Analysis (EDA):**
   - Analyzed data distribution and characteristics.
   - Visualized relationships between features.

2. **Data Preprocessing:**
   - Handled missing values.
   - Performed feature scaling.
   - Managed categorical and numerical features.
   - Split data into training, validation, and testing sets.

##### Part 2: Regression Analysis

1. **Linear Regression:**
   - Implemented from scratch without libraries.
   - Used gradient descent and least squares.
   - Evaluated using MSE, RMSE, and R² score.

2. **Polynomial Regression (Optional):**
   - Captured non-linear relationships.
   - Compared with linear model.

##### Part 3: Classification Algorithms

1. **K-Nearest Neighbors (KNN):**
   - Implemented KNN for classification.
   - Tested different distance metrics.
   - Optimized value of K.

2. **Support Vector Machine (SVM):**
   - Implemented with linear and RBF kernels.
   - Evaluated using confusion matrix and other metrics.

##### Part 4: Decision Trees and Ensemble Methods

1. **Decision Trees:**
   - Implemented for regression and classification.
   - Explored pruning techniques.

2. **Random Forests:**
   - Used as an ensemble method combining multiple trees.
   - Analyzed hyperparameter impacts.

3. **Ensemble Methods:**
   - Compared bagging and boosting.
   - Implemented Random Forests and XGBoost.

##### Part 5: Model Evaluation and Optimization

1. **Evaluation Metrics:**
   - Used confusion matrix, precision, recall, F1-score, and ROC-AUC.
   - Performed hyperparameter tuning with Grid and Random Search.

2. **ROC Curve Analysis:**
   - Generated and analyzed ROC curves.
   - Explained AUC significance.

##### Part 6: Final Report

- **Report:** Detailed implementation, results, and analysis in `report.pdf`.
- **Contents:** Visualizations and discussions on model outcomes.

---

### Coursework Assignment 5: Convolutional Neural Networks (CNNs) and Word Embeddings

This assignment focuses on deep learning techniques, specifically CNNs and Word Embeddings.

#### Project Overview

##### Part 1: Convolutional Neural Networks (CNNs)

- **Introduction to CNNs:**
  - Structure including convolutional, pooling, and fully connected layers.
  - Processing and feature extraction from images.

- **Implementation:**
  - Designed architecture and selected activation functions.
  - Understood roles of different layers.
  - Performed hyperparameter tuning.

##### Part 2: Word Embeddings

- **Word2Vec and GloVe:**
  - Transformed words into dense vector representations.
  - Captured semantic relationships.

- **Implementation:**
  - Used pre-trained models to extract word vectors.
  - Evaluated effectiveness in capturing semantic similarities.

##### Part 3: Text Classification with CNNs

- **Building a CNN:**
  - Applied to text classification using word embeddings.
  - Designed suitable architecture for text data.
  - Tuned context window size and hyperparameters.

- **Model Training and Evaluation:**
  - Trained on the given dataset.
  - Evaluated using accuracy, precision, recall, and F1-score.
  - Visualized training with loss and accuracy curves.

##### Part 4: Regularization Techniques

- **Addressing Overfitting:**
  - Explored Dropout and Batch Normalization.
  - Implemented within CNN models.

- **Performance Analysis:**
  - Compared models with and without regularization.
  - Provided detailed findings.

---

### Coursework Assignment 6: Reinforcement Learning and Deep Q-Networks (DQN)

This assignment delves into reinforcement learning, focusing on MDPs, TD-Learning, and DQNs.

#### Project Overview

##### Part 1: Markov Decision Processes (MDP)

1. **Value Iteration:**
   - Calculated optimal policy for a grid environment.
   - Derived Value function and optimal policy.

2. **Monte Carlo and Q-Learning:**
   - Explored as model-free approaches.
   - Compared convergence and efficiency.

##### Part 2: Temporal Difference Learning (TD-Learning)

1. **Implementation:**
   - Estimated value function for MDP.
   - Examined learning rates and exploration-exploitation tradeoff.
   - Compared with Monte Carlo methods.

##### Part 3: Deep Q-Networks (DQN)

1. **Overview and Applications:**
   - Researched DQN applications.
   - Implemented a basic DQN model.

2. **Practical Implementation - Snake Game:**
   - Created an AI agent for Snake using Q-Learning and DQN.
   - Trained to maximize score through reinforcement learning.

##### Part 4: Model Training and Evaluation

1. **Training:**
   - Used various hyperparameters.
   - Implemented Epsilon Decay for exploration-exploitation balance.

2. **Evaluation:**
   - Assessed cumulative reward and episode length.
   - Visualized Q-values convergence and total reward.

3. **Hyperparameter Tuning:**
   - Experimented with configurations.
   - Saved and compared models.

---

## Additional Directories

- **Extra/** - **Additional Material:**
  - Code snippets, research papers, or supplementary assignments.

- **REF/** - **Reference Material:**
  - Reference documents and resources used in the course.

- **Slides/** - **Course Slides:**
  - Lecture slides from the instructors.

---

## Acknowledgements

- **Instructors:** Dr. Fadayee and Dr. Yaghoobzadeh
- **University:** University of Tehran
- **Course:** Artificial Intelligence

This repository was created as part of the coursework for the Artificial Intelligence course at the University of Tehran. **All rights to the content are reserved.**

---
