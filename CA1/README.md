# AI-S03-A1: Agents and Search Algorithms Solutions

## **Project Overview**

This repository contains solutions for  **AI-S03-A1** , which focuses on  **agents** ,  **search algorithms** , and **problem-solving** in  **Artificial Intelligence (AI)** . The project involves concepts like  **state-space search** ,  **search trees** , and algorithms such as  **DFS** ,  **BFS** ,  **A*** , and **Hill Climbing** for finding optimal solutions. The assignment also covers **agents' behaviors** in different environments (e.g., **Sudoku** and  **CS:GO** ).

The key tasks include:

* Classifying problem environments.
* Implementing  **graph-based search algorithms** .
* Using **heuristic functions** for **A*** search.
* Implementing **local search algorithms** like  **Hill Climbing** .

---

---

## **Key Concepts & Techniques Covered**

### **1. Agents and Environments**

* **Fully/Partially Observable** : An agent's perception of the environment.
* **Deterministic/Stochastic** : Whether the environment is predictable or has randomness.
* **Episodic/Sequential** : Whether actions depend on past actions.
* **Static/Dynamic** : Whether the environment changes over time.
* **Discrete/Continuous** : Whether the environment's state space is discrete or continuous.
* **Single-Agent/Multi-Agent** : Whether the environment involves one or multiple agents.

#### **Example Environments** :

* **Sudoku** (Deterministic, Fully Observable, Episodic, Discrete)
* **CS:GO** (Stochastic, Partially Observable, Sequential, Dynamic)

### **2. Search Algorithms**

* **Depth-First Search (DFS)** : Explores as far as possible along each branch before backtracking.
* **Breadth-First Search (BFS)** : Explores all nodes at the present depth before moving on to nodes at the next level.
* **Uniform Cost Search** : Expands the least costly node.
* **A* Search** : Uses a heuristic function to find the optimal path.
* **Hill Climbing** : Local search method, continuously moving towards the goal by choosing the best neighboring state.

### **3. Heuristic Search (A*)**

* **Admissible Heuristic** : Never overestimates the cost to reach the goal.
* **Consistent Heuristic** : Ensures the heuristic satisfies the triangle inequality.
* **A* Algorithm** : Combines the cost to reach a node and the estimated cost to reach the goal (using the heuristic).

### **4. Local Search Algorithms (Hill Climbing)**

* **Hill Climbing** : Starts with an arbitrary solution and iteratively moves to the neighboring state with the best evaluation.
* **Local Maximum/Minimum** : The algorithm might get stuck at a local maximum or minimum.
* **Simulated Annealing** : A more robust local search algorithm that avoids getting stuck at local maxima.

---

## **Installation & Setup**

### **1. Clone the Repository**

```sh
git clone https://github.com/your-repo-name.git
cd your-repo-name
```

### **2. Install Dependencies**

```sh
pip install -r requirements.txt
```

### **3. Run the Code**

```sh
python main.py
```

Alternatively, you can run the Jupyter notebooks for interactive analysis:

```sh
jupyter notebook notebooks/AI-S03-A1-Analysis.ipynb
```

---

## **Key Questions Addressed in the Assignment**

### **Agents and Environment Classification:**

* How do you classify an environment based on observability, determinism, and other features?
* How do you define agent behaviors in different environments like **CS:GO** and  **Sudoku** ?

### **Search Algorithms:**

* When should you use **DFS** vs  **BFS** ? How do the algorithms differ in performance?
* How does **A*** search differ from  **Uniform Cost Search** ?
* What are the advantages of using **A*** with an admissible heuristic?
* What are the possible issues with **Hill Climbing** and how can you mitigate them with techniques like  **Simulated Annealing** ?

---

## **Results & Performance Evaluation**

* **Search Trees and Path Costs** : Evaluating the effectiveness of search algorithms like **BFS** and  **DFS** .
* **A* Search Results** : Pathfinding with **heuristics** in environments such as Sudoku and CS:GO.
* **Local Search (Hill Climbing)** : Evaluation of how **local maxima** can affect algorithm performance.


## **Contributors**

* **Dr. Yaqubzadeh**
* **Haji Heidari**
* **Mohammad Sadek Aboofazeli**
* **Pasha Barahimi**
