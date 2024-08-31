# Coursework Assignment 1: Search and Genetic Algorithms

This directory contains the implementation of algorithms required for the first coursework assignment of the Artificial Intelligence course at the University of Tehran. The assignment focuses on search algorithms and the application of genetic algorithms to solve the Knapsack problem.

## Project Overview

### Part 1: Search Algorithms

1. **BFS (Breadth-First Search) and DFS (Depth-First Search):**

   - Implemented to find the optimal path on a given graph.
   - Comparison of both algorithms based on different types of graphs.
   - Implementation includes the tracking of explored and frontier sets.
2. **Uniform Cost Search:**

   - An implementation to find the minimum cost path from a start node (S) to a goal node (G).
   - The algorithm calculates the total cost considering both the path cost and heuristic values.
3. **A* Search:**

   - A heuristic search algorithm implemented to find the shortest path on a graph.
   - Evaluates the admissibility and consistency of the heuristic function used.

### Part 2: Genetic Algorithm for the Knapsack Problem

1. **Knapsack Problem:**

   - Implementation of a genetic algorithm to solve the Knapsack problem, where the goal is to maximize the value of items packed in a knapsack without exceeding its weight limit.
2. **Genetic Algorithm Implementation:**

   - **Chromosome Representation:** Each chromosome represents a possible solution (selection of items).
   - **Population Initialization:** Initial population is randomly generated.
   - **Fitness Function:** Evaluates the suitability of each chromosome based on the total value and weight of selected items.
   - **Crossover and Mutation:** Genetic operators used to produce new generations of solutions.
   - **Selection:** The algorithm selects the best solutions based on the fitness function to form the next generation.

## How to Run

- Ensure you have Python and required libraries installed.
- Execute the search algorithms by running `search_algorithms.py`.
- Run the genetic algorithm implementation using `genetic_algorithm.py`.
- Test cases for both parts are provided in `test_cases/`.

---

# Coursework Assignment 2: Bayesian Networks and Hidden Markov Models

This directory contains the implementation and analysis of algorithms required for the second coursework assignment of the Artificial Intelligence course at the University of Tehran. The assignment covers Bayesian Networks and Hidden Markov Models (HMM), with a focus on probability calculations, forward/backward algorithms, and model evaluation.

## Project Overview

### Part 1: Bayesian Networks

1. **Bayesian Network Design:**

   - Created a Bayesian Network to evaluate the probability of car accidents based on various factors such as the age of the car, weather conditions, and car maintenance.
   - Implemented probability tables for each variable in the network.
2. **Probability Calculations:**

   - Computed the joint and conditional probabilities using the designed Bayesian Network.
   - Implemented the Variable Elimination algorithm to compute specific probabilities related to the network.
3. **Independence and Conditional Independence:**

   - Analyzed the independence and conditional independence of various variables in the Bayesian Network.
   - Provided justifications for the independence or dependence of certain events based on the network structure.

### Part 2: Hidden Markov Models (HMM)

1. **HMM Design for Music Genre Prediction:**

   - Defined an HMM where the hidden states represent the emotional state of users and the observations represent the music genres they listen to.
   - Configured the transition, emission, and initial probability matrices.
2. **Algorithm Implementation:**

   - Implemented the Forward algorithm to calculate the probability of an observation sequence.
   - Used the Smoothing algorithm to compute the probability of a hidden state at a given time.
   - Applied the Viterbi algorithm to determine the most likely sequence of hidden states given an observation sequence.
3. **Practical Implementation:**

   - Implemented the HMM using both a pre-built library (`hmmlearn`) and a custom implementation from scratch.
   - Compared the results and performance of the two approaches.

### Data Processing and Feature Extraction

1. **Feature Extraction:**

   - Processed audio data to extract features such as MFCC, Zero Crossing Rate, and chroma features.
   - Investigated the impact of various features on the performance of the HMM.
2. **Evaluation Metrics:**

   - Evaluated model performance using metrics such as Accuracy, Precision, Recall, and F1 Score.
   - Implemented these metrics from scratch and compared them against the results obtained from the library.

---

# Coursework Assignment 3: Clustering and Dimensionality Reduction

This directory contains the implementation and analysis of clustering algorithms and related tasks for the third coursework assignment of the Artificial Intelligence course at the University of Tehran. The assignment covers K-Means, DBSCAN, Agglomerative Clustering, feature extraction, dimensionality reduction, and cluster evaluation.

## Project Overview

### Part 1: Clustering Algorithms

1. **K-Means Clustering:**

   - Implemented the K-Means algorithm to cluster a set of given points into three clusters.
   - The initial cluster centers were provided, and the algorithm was run for one epoch.
   - Calculated the new cluster centers and visualized the clusters.
   - Determined the number of iterations needed for the algorithm to converge.
2. **DBSCAN Clustering:**

   - Applied the DBSCAN algorithm to the same set of points with different `ε` (epsilon) and `minPoints` values.
   - Visualized the clusters formed by DBSCAN and identified the core, border, and noise points.
3. **Agglomerative Clustering:**

   - Performed agglomerative clustering on a set of points using a distance matrix.
   - Visualized the hierarchical clustering process through a dendrogram.

### Part 2: Practical Implementation - Clustering Images

1. **Dataset:**

   - Used a dataset of images, each associated with different colors.
   - Extracted relevant features from the images using a pre-trained VGG16 model.
2. **Feature Extraction:**

   - Employed VGG16, a pre-trained Convolutional Neural Network (CNN), to extract high-level features from images.
   - Explained the rationale behind using pre-trained models for feature extraction.
   - Conducted research on the specific techniques used for feature extraction and the preprocessing required for the images.
3. **Clustering Implementation:**

   - Applied K-Means and DBSCAN clustering algorithms to the extracted feature vectors.
   - Experimented with different values of `K` in K-Means and different parameters for DBSCAN to identify optimal clustering.
   - Compared the clusters produced by K-Means and DBSCAN.

### Part 3: Dimensionality Reduction and Cluster Visualization

1. **Dimensionality Reduction using PCA:**

   - Reduced the dimensionality of the feature vectors using Principal Component Analysis (PCA).
   - Visualized the clusters in 2D or 3D space after dimensionality reduction.
2. **Evaluation of Clustering Results:**

   - Evaluated the clustering results using metrics like silhouette score and homogeneity score.
   - Compared the clustering results before and after dimensionality reduction.

### Part 4: Analysis and Discussion

- **Comparison of Clustering Methods:**

  - Discussed the advantages and disadvantages of K-Means and DBSCAN.
  - Provided insights on when one algorithm might be preferable over the other.
- **Cluster Evaluation:**

  - Detailed the calculation methods for silhouette and homogeneity scores.
  - Analyzed the results of these evaluations and suggested ways to improve cluster performance.

---

# Coursework Assignment 4: Data Analysis and Machine Learning Algorithms

This directory contains the implementation and analysis of various data analysis techniques and machine learning algorithms for the fourth coursework assignment of the Artificial Intelligence course at the University of Tehran. The assignment covers data preprocessing, regression, classification, model evaluation, and ensemble methods.

## Project Overview

### Part 1: Data Analysis and Preprocessing

1. **Exploratory Data Analysis (EDA):**

   - Performed initial data analysis to understand the distribution and characteristics of the dataset.
   - Visualized data using scatter plots, hexbin plots, and other relevant methods to identify relationships between features.
2. **Data Preprocessing:**

   - Addressed missing values using various imputation techniques.
   - Conducted feature scaling (normalization and standardization) where appropriate.
   - Handled categorical and numerical features differently based on their nature.
   - Split the dataset into training, validation, and testing sets.

### Part 2: Regression Analysis

1. **Linear Regression:**

   - Implemented linear regression from scratch without using any libraries.
   - Calculated model coefficients using gradient descent and least squares methods.
   - Evaluated the model using metrics such as MSE, RMSE, and R² score.
2. **Polynomial Regression (Optional):**

   - Extended the linear regression model to polynomial regression to capture non-linear relationships.
   - Evaluated and compared the polynomial regression model with the linear model.

### Part 3: Classification Algorithms

1. **K-Nearest Neighbors (KNN):**

   - Implemented KNN for classification tasks.
   - Compared the performance of the model using different distance metrics (Euclidean and Manhattan).
   - Experimented with various values of K to determine the optimal configuration.
2. **Support Vector Machine (SVM):**

   - Implemented SVM with linear and RBF kernels.
   - Evaluated the model using confusion matrix and other classification metrics such as accuracy, precision, recall, and F1-score.

### Part 4: Decision Trees and Ensemble Methods

1. **Decision Trees:**

   - Implemented decision trees for both regression and classification tasks.
   - Explored the concept of pruning to avoid overfitting and improve model generalization.
2. **Random Forests:**

   - Implemented random forests as an ensemble method that combines multiple decision trees.
   - Discussed the impact of hyperparameters such as the number of trees and max depth on model performance.
3. **Ensemble Methods:**

   - Compared bagging and boosting methods to understand their strengths and weaknesses.
   - Implemented and analyzed the performance of Random Forests and XGBoost.

### Part 5: Model Evaluation and Optimization

1. **Evaluation Metrics:**

   - Evaluated model performance using various metrics such as confusion matrix, precision, recall, F1-score, and ROC-AUC.
   - Conducted hyperparameter tuning using Grid Search and Random Search to optimize model performance.
2. **ROC Curve Analysis:**

   - Generated and analyzed ROC curves to assess the performance of classification models.
   - Explained the significance of AUC (Area Under the Curve) and its impact on model evaluation.

### Part 6: Final Report

- A comprehensive report detailing the implementation, results, and analysis is included in the `report.pdf` file.
- The report covers all the tasks mentioned above, with visualizations and discussions on the outcomes of various models.

---

# Coursework Assignment 6: Reinforcement Learning and Deep Q-Networks (DQN)

This directory contains the implementation and analysis of reinforcement learning algorithms, particularly focusing on Markov Decision Processes (MDP), Temporal Difference Learning, and Deep Q-Networks (DQN) for the sixth coursework assignment of the Artificial Intelligence course at the University of Tehran.

## Project Overview

### Part 1: Markov Decision Processes (MDP)

1. **MDP Value Iteration:**

   - Implemented the Value Iteration algorithm to calculate the optimal policy for a given grid-based environment.
   - The environment is defined with specific reward and transition dynamics, and the agent needs to determine the best actions to maximize rewards.
   - Calculated the Value function for each state using a given discount factor and then derived the optimal policy.
2. **Monte Carlo and Q-Learning:**

   - Explored Monte Carlo methods and Q-Learning as model-free approaches to learning optimal policies.
   - Compared the performance of these methods in terms of convergence and efficiency in different scenarios.

### Part 2: Temporal Difference Learning (TD-Learning)

1. **TD-Learning Implementation:**
   - Implemented Temporal Difference Learning to estimate the value function for the given MDP.
   - Examined the effects of different learning rates and the exploration-exploitation tradeoff.
   - Compared the performance of TD-Learning with Monte Carlo methods.

### Part 3: Deep Q-Networks (DQN)

1. **DQN Overview and Applications:**

   - Conducted research on the applications of Deep Q-Networks (DQN) in various domains.
   - Implemented a basic DQN model for a given game environment to demonstrate its capability in learning complex strategies.
2. **Practical Implementation - Snake Game:**

   - Designed and implemented an AI agent to play the classic Snake game using Q-Learning and DQN.
   - The environment includes different states representing the position of the snake, its body, and the food item.
   - The agent was trained to maximize the score by learning the optimal strategy through reinforcement learning techniques.

### Part 4: Model Training and Evaluation

1. **Model Training:**

   - Trained the DQN model using various hyperparameters, including different learning rates, discount factors, and exploration strategies.
   - Implemented Epsilon Decay to balance exploration and exploitation during training.
2. **Performance Evaluation:**

   - Evaluated the performance of the trained model using metrics such as cumulative reward and episode length.
   - Visualized the training process with plots showing the convergence of the Q-values and the total reward over episodes.
3. **Hyperparameter Tuning:**

   - Experimented with different hyperparameters to optimize the model's performance.
   - Saved and compared models trained with different configurations to identify the most effective approach.
