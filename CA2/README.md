

# AI-S03-A2: Bayesian Networks and Hidden Markov Models Solutions

## **Project Overview**

This repository contains solutions for  **AI-S03-A2** , which focuses on **Bayesian Networks (Bayes Nets)** and  **Hidden Markov Models (HMM)** . The project includes mathematical formulations and practical implementations using these powerful tools for **probabilistic reasoning** and  **sequence prediction** . Key topics covered in the assignment include:

1. **Bayesian Networks** : Conditional probability tables, marginalization, and inference.
2. **Hidden Markov Models (HMM)** : Modeling temporal data, state transitions, emissions, and the Viterbi algorithm.

---

## **Key Concepts & Techniques**

### **1. Bayesian Networks**

* **Bayesian Networks** are graphical models that represent probabilistic relationships among a set of variables.
* The assignment explores various **inference tasks** in a Bayesian Network:
  * **Calculating conditional probabilities** given evidence (e.g., finding the probability of a vehicle having a mechanical issue given specific conditions).
  * **Marginalization** and **Variable Elimination** for handling multiple random variables.
  * Conditional probability table (CPT) construction for each node in the network.

#### **Tasks:**

* Calculating specific conditional probabilities such as  **P(+O, -W, +F, -R, +A)** .
* Performing **inference** by applying Bayesian rules to calculate probabilities under different conditions.

### **2. Hidden Markov Models (HMM)**

* **HMMs** are used for modeling sequential data with hidden states and observed outputs (emissions).
* The assignment focuses on applying the **Viterbi algorithm** for finding the most likely sequence of hidden states given a sequence of observations.
* The **Forward-Backward algorithm** is also used for smoothing and calculating state probabilities.

#### **Tasks:**

* **State Transitions and Emissions** : Modeling hidden states and observable events like music genres in the example.
* **Forward Algorithm** : Calculating the probability of observing a sequence of events.
* **Smoothing and Viterbi Algorithm** : Finding the most likely sequence of hidden states for given observations.

---

---

## **Key Questions Addressed in the Assignment**

### **Bayesian Networks:**

* **How do you calculate the conditional probability of one variable given the others?**
  * Example: **P(W|O)** (probability of having a mechanical issue given the observation of wet conditions).
* **How do you perform variable elimination for inference?**
  * Use **marginalization** and **summation over unobserved variables** to simplify the calculations.
* **How do you interpret conditional probability tables (CPT)?**
  * Understanding the relationships between variables and how the conditional probabilities are defined.

### **Hidden Markov Models (HMM):**

* **What are the steps involved in the Viterbi algorithm?**
  * **Initialization** ,  **Recursion** , and **Termination** to find the most probable sequence of hidden states.
* **How does the Forward-Backward algorithm help in state estimation?**
  * **Forward algorithm** calculates the probability of observed events given the hidden states, while the **Backward algorithm** helps refine the estimate.
* **How do you calculate the likelihood of an observation sequence using HMM?**
  * Applying the **Forward algorithm** and **Smoothing** techniques to estimate the probability of the sequence of observations.

---

## **Results & Performance Evaluation**

* **Bayesian Network Inference Results** : Results from calculating conditional probabilities and marginalization using the Bayes Nets.
* **HMM Results** :
* Viterbi algorithm outputs the most likely hidden states given a sequence of observations.
* Forward-backward probabilities and **smoothing** for state estimates.
