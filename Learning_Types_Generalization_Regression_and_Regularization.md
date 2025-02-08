# Learning from data

At a high level, machine learning algorithms can be grouped based on the type of data they work with: labeled or unlabeled data. Let's go deeper into each of the learning types and their application to classification and regression tasks.

### 1. **Supervised Learning (for labeled data)**

Supervised learning refers to the scenario where we have labeled data, meaning that each data point has a corresponding label or target value. The goal is to learn a mapping from input features to these labels so that we can make predictions on unseen data.

#### a) **Classification**

In classification, the goal is to predict a category or class label from the data. For example, determining whether an email is spam or not, based on features like words used in the email.

- **Classification** is used when the target variable (Y) is **categorical** (e.g., spam or not spam).
- The output Y can take one of a finite set of values (classes).
  
##### Formula:

If we have a dataset $\( D \)$ with features $\( X = \{ x_1, x_2, ..., x_n \} \)$ and a target label $\( Y \)$ that belongs to a set of classes $\( C = \{c_1, c_2, ..., c_k \} \)$, then the task is to find a function $\( f(X) \)$ that maps $\( X \)$ to one of the classes $\( c \)$.

Mathematically, the problem is:
```math
f: X \rightarrow C
```
where $\( f \)$ is the classification function, and $\( C \)$ is the set of possible class labels.

Common algorithms for classification:
- Logistic Regression
- Decision Trees
- Support Vector Machines (SVM)
- k-Nearest Neighbors (k-NN)

#### b) **Regression**

In regression, the objective is to predict a continuous value (real number). For example, predicting the price of a house based on its features (square footage, number of bedrooms, etc.).

- **Regression** is used when the target variable $\( Y \)$ is **continuous**.
- The output Y is a real number, and we aim to approximate the relationship between the input features and the continuous target.

##### Formula:

Given a dataset $\( D \)$ with features $\( X = \{ x_1, x_2, ..., x_n \} \)$ and continuous target values $\( Y \)$, the objective is to learn a function $\( f(X) \)$ that predicts a continuous value for Y.

Mathematically:
```math
f: X \rightarrow Y
``` 
where $\( Y \)$ is a continuous variable.

In linear regression, for example, the model would look like this:
```math
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon
``` 
where:
- $\( y \)$ is the predicted continuous value (e.g., house price),
- $\( x_1, x_2, ..., x_n \)$ are the features (e.g., square footage, number of bedrooms),
- $\( \beta_0, \beta_1, ..., \beta_n \)$ are the coefficients (parameters) that we learn from the data,
- $\( \epsilon \)$ is the error term (difference between predicted and actual value).

Common algorithms for regression:
- Linear Regression
- Decision Trees
- Random Forests
- Support Vector Machines for regression (SVR)

### 2. **Unsupervised Learning (for unlabeled data)**

Unsupervised learning is used when you have data without explicit labels or target values. The goal here is often to find patterns or structures in the data.

#### a) **Clustering**

In clustering, the goal is to group similar data points together, so that points within the same group (cluster) are more similar to each other than to points in other groups. This is typically used for customer segmentation, document clustering, or image segmentation.

- **Clustering** algorithms do not rely on labels.
- The output is a grouping of data points based on similarity.

One of the most common clustering algorithms is **k-means**. The idea is to partition data points into $\( k \)$ clusters, where each cluster is represented by the mean (centroid) of the points in that cluster.

##### Formula for k-means:
1. Choose $\( k \)$ initial centroids (randomly or using some heuristic).
2. Assign each data point to the nearest centroid:
```math
c_i = \arg \min_{j} \| x_i - \mu_j \|^2
``` 
where $\( \mu_j \)$ is the centroid of cluster $\( j \)$, and $\( x_i \)$ is a data point.
3. Recalculate the centroids by averaging the points in each cluster:
```math
\mu_j = \frac{1}{|C_j|} \sum_{i \in C_j} x_i
``` 
where $\( C_j \)$ is the set of points assigned to cluster $\( j \)$, and $\( |C_j| \)$ is the number of points in cluster $\( j \)$.

### 3. **Reinforcement Learning (for reward-based data)**

Reinforcement learning involves an agent that interacts with an environment and learns to make decisions based on rewards or penalties. The agent receives feedback in the form of rewards (positive) or penalties (negative) and adjusts its behavior to maximize cumulative rewards over time.

- **Reinforcement learning** typically involves sequential decision-making.
- It uses **reward signals** to learn.

Common algorithms:
- Q-Learning
- Deep Q Networks (DQN)

### 4. **Self-Supervised Learning (for both labeled and unlabeled data)**

Self-supervised learning is a hybrid approach that combines aspects of supervised and unsupervised learning. The algorithm uses parts of the data as "pseudo-labels" and tries to predict them. For instance, predicting the next word in a sentence given the previous ones (in natural language processing) or predicting missing parts of an image.

- **Self-supervised learning** can work with both labeled and unlabeled data.
- It can be seen as a way to create labels from unlabeled data.

### **Summary**

- **Supervised learning** is when you have labeled data and can either classify (categorical target) or regress (continuous target).
- **Unsupervised learning** deals with unlabeled data and is used for tasks like clustering.
- **Reinforcement learning** is based on reward data, where the goal is to maximize cumulative rewards.
- **Self-supervised learning** is a technique that blends labeled and unlabeled data to create pseudo-labels and learn from them.
