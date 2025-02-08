# Learning from data

At a high level, machine learning algorithms can be grouped based on the type of data they work with: labeled or unlabeled data. Let's go deeper into each of the learning types and their application to classification and regression tasks.

### 1. **Supervised Learning (for labeled data)**

Supervised learning refers to the scenario where we have labeled data, meaning that each data point has a corresponding label or target value. The goal is to learn a mapping from input features to these labels so that we can make predictions on unseen data.

#### a) **Classification**

In classification, the goal is to predict a category or class label from the data. For example, determining whether an email is spam or not, based on features like words used in the email.

- **Classification** is used when the target variable $Y$ is **categorical** (e.g., spam or not spam).
- The output $Y$ can take one of a finite set of values (classes).
  
##### Formula:

If we have a dataset $D$ with features $X = \{ x_1, x_2, ..., x_n \}$ and a target label $Y$ that belongs to a set of classes $C = \{c_1, c_2, ..., c_k \}$, then the task is to find a function $f(X)$ that maps $X$ to one of the classes $c$.

Mathematically, the problem is:
```math
f: X \rightarrow C
```
where $f$ is the classification function, and $C$ is the set of possible class labels.

Common algorithms for classification:
- Logistic Regression
- Decision Trees
- Support Vector Machines (SVM)
- k-Nearest Neighbors (k-NN)

#### b) **Regression**

In regression, the objective is to predict a continuous value (real number). For example, predicting the price of a house based on its features (square footage, number of bedrooms, etc.).

- **Regression** is used when the target variable $Y$ is **continuous**.
- The output Y is a real number, and we aim to approximate the relationship between the input features and the continuous target.

##### Formula:

Given a dataset $D$ with features $X = \{ x_1, x_2, ..., x_n \}$ and continuous target values $Y$, the objective is to learn a function $f(X)$ that predicts a continuous value for Y.

Mathematically:
```math
f: X \rightarrow Y
``` 
where $Y$ is a continuous variable.

In linear regression, for example, the model would look like this:
```math
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon
``` 
where:
- $y$ is the predicted continuous value (e.g., house price),
- $x_1, x_2, ..., x_n$ are the features (e.g., square footage, number of bedrooms),
- $\beta_0, \beta_1, ..., \beta_n$ are the coefficients (parameters) that we learn from the data,
- $\epsilon$ is the error term (difference between predicted and actual value).

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

One of the most common clustering algorithms is **k-means**. The idea is to partition data points into $k$ clusters, where each cluster is represented by the mean (centroid) of the points in that cluster.

##### Formula for k-means:
1. Choose $k$ initial centroids (randomly or using some heuristic).
2. Assign each data point to the nearest centroid:
```math
c_i = \arg \min_{j} \| x_i - \mu_j \|^2
``` 
where $\mu_j$ is the centroid of cluster $j$, and $x_i$ is a data point.
3. Recalculate the centroids by averaging the points in each cluster:
```math
\mu_j = \frac{1}{|C_j|} \sum_{i \in C_j} x_i
``` 
where $C_j$ is the set of points assigned to cluster $j$, and $|C_j|$ is the number of points in cluster $j$.

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

# Generative models vs Discriminative models

Generative and discriminative models are two fundamental approaches in machine learning, each with distinct objectives and methods for handling data. Let's break them down further, focusing on their key differences, how they work, and their applications.

### **Generative Models**

Generative models focus on understanding the *underlying process* that generates the data. In other words, these models aim to model the joint probability distribution of the input features $X$ and the output labels $Y$.

#### **Key Concepts:**
- **Joint Probability Distribution:** Generative models estimate the joint probability $P(X, Y)$, which represents how likely the input $X$ and the output $Y$ are to occur together. From this, they can also calculate the conditional probability $P(Y|X)$ using **Bayes' Theorem**:
  
  ![image](https://github.com/user-attachments/assets/832a2221-d643-4aea-81b0-e62872b46edb)

  
  Here:
  - $P(Y|X)$ is the posterior probability (what we're interested in: the probability of $Y$ given $X$),
  - $P(X|Y)$ is the likelihood (how likely the data $X$ is given the label $Y$),
  - $P(Y)$ is the prior probability (how likely label $Y$ is before observing $X$),
  - $P(X)$ is the marginal likelihood (the total probability of the data).
  
- **Modeling the Data:** Generative models try to learn how the data is generated, which allows them to simulate new instances of data. By modeling both the input and output together, these models can handle missing data or generate new samples (for example, in image generation tasks).

#### **Example Algorithms:**
1. **Naïve Bayes:** Assumes that the features $X$ are conditionally independent given the class $Y$. This simplifies the computation of $P(X|Y)$, allowing for efficient classification. Despite its simplicity, Naïve Bayes can perform well for certain tasks like text classification.
2. **Hidden Markov Models (HMM):** Used for sequential data (like time series or speech), where the system is assumed to be a Markov process with hidden states. The model learns the transitions between these hidden states and generates observations accordingly.
3. **Dimensionality Reduction Techniques (e.g., PCA):** These techniques try to model how the data is distributed in high-dimensional space and reduce it to a lower-dimensional representation. While they focus on the data generation process, they can be used for feature extraction in classification tasks.

#### **Advantages:**
- Can generate new data points or handle missing data.
- Provide insight into how data is distributed across different classes.
  
#### **Disadvantages:**
- Sensitive to outliers, since these models learn the entire distribution, and outliers can skew the estimates of $P(X|Y)$ and $P(Y)$.
- Can be computationally intensive, especially when modeling complex distributions.

### **Discriminative Models**

Discriminative models, on the other hand, focus on directly learning the decision boundary between classes. These models do not attempt to model the data generation process; instead, they focus on modeling the conditional probability $P(Y|X)$, which is the probability of the label $Y$ given the input $X$.

#### **Key Concepts:**
- **Conditional Probability:** Discriminative models directly estimate $P(Y|X)$, the probability that a given input $X$ belongs to a particular class $Y$. By modeling this directly, discriminative models can make more accurate predictions because they focus only on distinguishing between classes rather than modeling how the data is generated.
  
  Unlike generative models, discriminative models **do not attempt to model $P(X)$**, the distribution of the data, and therefore they focus only on the class boundaries.

- **Learning Decision Boundaries:** Discriminative models essentially "draw" a decision boundary in the feature space that separates different classes. This boundary minimizes classification error by maximizing the margin between classes (in the case of SVMs) or directly optimizing for the likelihood of correct classification (in the case of logistic regression).

#### **Example Algorithms:**
1. **Logistic Regression:** A linear model for binary classification, where the output is modeled as a logistic (sigmoid) function of the input features. It directly estimates $P(Y=1|X)$ for binary outcomes.
2. **Support Vector Machines (SVM):** A classifier that tries to find the optimal hyperplane that maximizes the margin between the classes in the feature space. SVMs are effective in high-dimensional spaces and are robust to overfitting, especially with a small number of training examples.
3. **Decision Trees:** A hierarchical model that recursively splits the feature space based on feature values to separate the classes. It is a powerful and interpretable model for classification tasks.
4. **Random Forest:** An ensemble of decision trees that aggregates the predictions of multiple trees to improve accuracy and reduce overfitting.

#### **Advantages:**
- Typically more accurate for classification tasks because they focus only on the decision boundary.
- More robust to outliers, since they are not trying to model the entire data distribution but rather just distinguishing between classes.

#### **Disadvantages:**
- Cannot generate new data points.
- Prone to misclassification in cases where the classes overlap or are not linearly separable (especially for simple models like logistic regression or decision trees).

### **Generative vs. Discriminative Models: Summary**

| **Aspect**                        | **Generative Models**                                  | **Discriminative Models**                               |
|-----------------------------------|--------------------------------------------------------|--------------------------------------------------------|
| **Focus**                         | Models the joint distribution $P(X, Y)$               | Models the conditional distribution $P(Y\|X)$          |
| **Key Objective**                 | Explains how data was generated                        | Classifies data by finding a decision boundary          |
| **Example Algorithms**            | Naïve Bayes, HMM, PCA, Dimensionality Reduction         | Logistic Regression, SVM, Decision Trees, Random Forest |
| **Robustness to Outliers**        | Sensitive to outliers                                  | More robust to outliers                                |
| **Ability to Handle Missing Data**| Can generate new data points or handle missing data    | Cannot generate new data                               |
| **Computational Complexity**      | Often more computationally expensive                   | Often more computationally efficient                   |
| **Usage**                         | Data generation, density estimation                    | Classification, predicting labels                      |

### **Choosing Between Generative and Discriminative Models**

- **Generative models** are typically more useful when you want to simulate new data or when you have a small amount of data and need to understand the distribution of features and labels. They're also useful for tasks where data is missing or incomplete.
  
- **Discriminative models** are generally preferred when the primary task is classification, and you're aiming for higher accuracy in distinguishing between classes. They are usually more robust and simpler to train than generative models, especially for large datasets where the generation process isn't as important.

# Linear Regression

### What is Linear Regression?

Linear regression is a method used to model the relationship between a set of input variables and a continuous output variable. It's called "linear" because the relationship between the inputs and the output is represented as a straight line (or hyperplane in higher dimensions).

In simple terms, linear regression tries to find the best straight line that fits the data points. Imagine you have a scatter plot, and you want to draw a line that goes through the middle of the points, as close as possible to all of them.

### The Linear Regression Formula

The general form of a linear regression model is:

```math
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_d x_d
``` 

Here:
- $\hat{y}$ is the predicted output (the value you want to estimate).
- $\beta_0$ is the intercept (the value of $y$ when all inputs are zero).
- $\beta_1, \beta_2, \dots, \beta_d$ are the coefficients (slopes) for each input variable.
- $x_1, x_2, \dots, x_d$ are the input features (the data you have for each observation).

## Simple Linear Regression (1D Case)

Let’s start with a simple example where there is just one input variable (like $x_1$) and one output variable $y$ (i.e., $d = 1$ and $m = 1$).

The model would look like this:

```math
Y = \beta_0 + \beta_1 x_1
``` 

This equation means that $Y$ (the output) is a straight line that depends on the input $x_1$. The term $\beta_0\$ is the starting point (the intercept) on the $y$-axis when $x_1 = 0$, and $\beta_1$ is the slope of the line, which tells us how much $Y$ changes when $x_1$ changes.

### Training the Model

To use this equation to make predictions, we need to find the best values for $\beta_0$ and $\beta_1$. This is where the **training data** comes in. You have a set of real data points with known values of $x_1$ (inputs) and $y$ (outputs). Your goal is to find $\beta_0$ and $\beta_1$ such that the line fits the data as well as possible.

To do this, we use a method called **Least Squares Estimation**, which minimizes the difference between the predicted values and the actual values of $y$ (the real data points).

### Mean Squared Error (MSE)

The difference between the predicted value $\hat{y}$ and the actual value $y$ is called the **error**. To measure how well our model fits the data, we use **Mean Squared Error (MSE)**, which is calculated as:

```math
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
``` 

Where:
- $n$ is the total number of data points,
- $y_i$ is the actual value of the output for the $i$-th data point,
- $\hat{y}_i$ is the predicted value for the $i$-th data point.

The MSE calculates the average of the squared differences between the actual values and the predicted values. The reason we square the differences is to ensure that both positive and negative errors are treated equally, and also to penalize larger errors more.

### Goal: Minimize the MSE

The goal of linear regression is to adjust $\beta_0$ and $\beta_1$ so that the MSE is as small as possible. In other words, we want the predicted values $\hat{y}_i$ to be as close as possible to the actual values $y_i$.

### After Training: Predictions

Once we have the best values for $\beta_0$ and $\beta_1$ (the ones that minimize the MSE), we can use the equation $Y$ = $\beta_0$ + $\beta_1$ $x_1$ to make predictions on new data. For any new value of $x_1$, you can plug it into the equation and get the predicted $y$ value.

### Recap of Steps:
1. **Define the model:** $Y$ = $\beta_0$ + $\beta_1$ $x_1$.
2. **Train the model:** Find $\beta_0$ and $\beta_1$ that minimize the MSE, i.e., minimize the difference between the predicted and actual $y$ values.
3. **Make predictions:** Once the model is trained, use it to predict the output for new input values.

### Why is it called "Curve Fitting"?

Linear regression is also known as **curve fitting** because it fits a curve (or in this case, a straight line) to the data. It tries to find the best-fitting line that minimizes the error between the predicted and actual values.

### Simple Example:

Suppose you have the following data points for $x_1$ and $y$:

| $x_1$ | $y$ |
|--------|------|
| 1      | 2    |
| 2      | 3    |
| 3      | 5    |
| 4      | 6    |

You would use linear regression to find the best line that fits these points. After applying the least squares method, you might get something like this:

```math
\hat{y} = 1 + 1.2 x_1
``` 

This means that for every increase of 1 in $x_1$, $y$ increases by 1.2 units, starting from a value of 1 when $x_1 = 0$.

### Conclusion

Linear regression is a simple yet powerful method for predicting continuous values. It works by finding the line (or curve in higher dimensions) that best fits the data, minimizing the errors between predicted and actual values. The process of training the model involves adjusting parameters to minimize the error and make the best possible predictions.

## Multiple Linear Regression (Higher-Dimensional Case)
 
This case is an extension of linear regression to handle multiple input features (i.e., $d > 1$), making it a **multiple linear regression** problem.

### Setup and Notation

- You have $n$ data points, each with $d$ features. For each data point $i$, the input $x_i$ is a vector in $\mathbb{R}^d$, and the output $y_i$ is a scalar in $\mathbb{R}$.
- We represent the input data as a matrix $X$, where each row corresponds to a data point, and each column corresponds to a specific feature. The vector $y$ contains the corresponding output values.

#### The Model:
The general multiple linear regression model is represented as:

```math
Y = X B
```

Where:
- $Y$ is the vector of outputs (size $n$ by 1),
- $X$ is the design matrix (size $n$ by ($d+1$), including a column of ones to account for the intercept $\beta_0$ and $d$ columns for the features $x_1, x_2, \dots, x_d$,
- $B$ is the vector of coefficients (size $d+1$ by 1), where each element corresponds to a specific parameter: $\beta_0, \beta_1, \dots, \beta_d$.

So, the matrix form of the equation is:

```math
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n
\end{bmatrix}
=
\begin{bmatrix}
1 & x_{11} & x_{12} & \dots & x_{1d} \\
1 & x_{21} & x_{22} & \dots & x_{2d} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \dots & x_{nd}
\end{bmatrix}
\begin{bmatrix}
\beta_0 \\
\beta_1 \\
\vdots \\
\beta_d
\end{bmatrix}
```

Here:
- $Y$ is a column vector containing the actual output values for all data points.
- $X$ is the design matrix, where the first column is all 1's (for the intercept term $\beta_0$), and the remaining columns are the feature values.
- $B$ is the vector of coefficients (the parameters we want to estimate).

### Objective: Minimize the Cost Function

The goal of linear regression is to find the values of $B$ (the coefficients) that minimize the difference between the predicted values $\hat{y}_i$ and the actual values $y_i$ for all data points.

The cost function (also known as the **loss function**) used to measure the error is the **Mean Squared Error (MSE)**, given by:

```math
J(B) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
```

Where:
- $y_i$ is the actual value for the $i$-th data point,
- $\hat{y}_i = X_i B$ is the predicted value for the $i$-th data point (using the current estimate of $B$).

Since the model $Y = X B$ uses matrix notation, we can rewrite the cost function in matrix form as:

```math
J(B) = \frac{1}{2n} (Y - X B)^T (Y - X B)
```

This is the sum of squared residuals (the differences between the actual and predicted values), averaged over all data points.

### Minimizing the Cost Function

To minimize the cost function $J(B)$, we need to find the value of $B$ that minimizes this expression. This can be done by taking the **derivative** of $J(B)$ with respect to $B$ and setting it equal to zero (this is a standard method in optimization). The solution is:

```math
B = (X^T X)^{-1} X^T Y
```

#### Explanation:
- $X^T$ is the transpose of the matrix $X$.
- $X^T X$ is the matrix product of $X^T$ and $X$, which is a square matrix of size ($d+1$) $\times$ ($d+1$).
- $(X^T X)^{-1}$ is the inverse of the matrix $X^T X$, and it exists as long as $X^T X$ is **non-singular** (i.e., it has full rank).
- $X^T Y$ is the matrix product of $X^T$ and the vector $Y$, which gives a vector of size $(d+1)$ $\times$ $1$.

So, by multiplying $(X^T X)^{-1} X^T Y$, we get the vector $B$ that minimizes the cost function and provides the **best-fit** coefficients for the regression model.

### Conclusion

In summary:
1. **Model**: $Y = X B$, where $Y$ is the output, $X$ is the design matrix, and $B$ is the vector of coefficients.
2. **Cost Function**: The objective is to minimize $J(B) = \frac{1}{2n} (Y - X B)^T (Y - X B)$, the Mean Squared Error between the actual and predicted values.
3. **Solution**: The optimal coefficients $B$ are found using the formula $B = (X^T X)^{-1} X^T Y$.

This formula is the core of multiple linear regression, and it allows you to compute the coefficients that give the best fit to your data.

## Multiple-Output Regression

This case describes an extension of the previous **multiple linear regression** to a situation where you have **multiple outputs** (i.e., \($Y$ $\in$ $\mathbb{R}^{n \times 2}$)) instead of just a single output. This type of model is also known as **multivariate linear regression**.

### Notation and Setup

- You have $n$ data points, each with $d$ features.
- Each data point has **2 output values** (so $Y$ is now a matrix of size \($n \times 2$).
- Each output value corresponds to a vector in $\mathbb{R}^2$, so the target output for each data point is a pair of values $(y_{i1}, y_{i2})$.

### Model Representation

The linear model is:

```math
Y = X B
```

Where:
- $Y$ is the matrix of outputs (size $n \times 2$),
- $X$ is the **design matrix** (size $n$ $\times$ ($d+1$), which includes a column of ones (for the intercept term \($\beta_0\$)) and $\(d\)$ columns for the input features \($x_1$, $x_2$, $\dots$, $x_d$\),
- $B$ is the coefficient matrix (size \($d+1$) $\times$ $2$\) where each column corresponds to the coefficients of one of the two output variables.

The equation can be written as:

```math
\begin{bmatrix}
y_{11} & y_{12} \\
y_{21} & y_{22} \\
\vdots & \vdots \\
y_{n1} & y_{n2}
\end{bmatrix}
=
\begin{bmatrix}
1 & x_{11} & x_{12} & \dots & x_{1d} \\
1 & x_{21} & x_{22} & \dots & x_{2d} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \dots & x_{nd}
\end{bmatrix}
\begin{bmatrix}
\beta_{10} & \beta_{20} \\
\beta_{11} & \beta_{21} \\
\vdots & \vdots \\
\beta_{1d} & \beta_{2d}
\end{bmatrix}
```

Where:
- Each $y_{i1}$ and $y_{i2}$ represent the two outputs for the $i$-th data point,
- The matrix $B$ contains the coefficients that will be estimated.

### Objective: Minimize the Cost Function

The cost function (also known as the loss function) measures how well the model's predictions match the actual outputs. The cost function for this model is the **Mean Squared Error (MSE)**, given by:

```math
J(B) = \frac{1}{2n} \sum_{i=1}^{n} \left( (y_{i1} - \hat{y}_{i1})^2 + (y_{i2} - \hat{y}_{i2})^2 \right)
```

Where:
- $y_{i1}$ and $y_{i2}$ are the actual values for the $i$-th data point,
- $`\hat{y}_{i1}`$ and $\hat{y}_{i2}$  are the predicted values for the $i$-th data point.

In matrix form, this can be written as:

```math
J(B) = \frac{1}{2n} \left( Y - X B \right)^T \left( Y - X B \right)
```

Where:
- $Y$ is the matrix of actual outputs \($n \times 2$\),
- $X$ is the design matrix \($n$ $\times$ ($d+1$)\),
- $B$ is the matrix of coefficients \($(d+1) \times 2$\).

### Minimizing the Cost Function

The goal of linear regression is to find the value of $B$ that minimizes the cost function $J(B)$. To do this, we take the derivative of $J(B)$ with respect to $B$ and set it equal to zero. This yields the solution:

```math
B = (X^T X)^{-1} X^T Y
```

#### Explanation:
- $X^T$ is the transpose of the matrix $X$,
- $X^T X$ is the matrix product of $X^T$ and $X$, which results in a square matrix of size $\(d+1) \times (d+1)$,
- $\(X^T X)^{-1}$ is the inverse of $X^T X$ (it must be invertible for this solution to exist),
- $X^T Y$ is the matrix product of $X^T$ and $Y$, which is a matrix of size $\(d+1) \times 2$.

The result is that the coefficient matrix $B$ (of size \($d+1$) $\times$ 2\) can be computed using the formula:

```math
B = (X^T X)^{-1} X^T Y
```

### How the Formula Works

The formula $\(X^T X)^{-1} X^T Y$ gives you the optimal values for the coefficient matrix $B$ by minimizing the squared differences between the predicted and actual values. Since there are two output variables, $B$ has two columns, each corresponding to the coefficients for one of the output variables.

### Conclusion

To summarize:
1. **Model**: The linear model is $Y = X B$, where $Y$ is the matrix of outputs, $X$ is the design matrix, and $B$ is the coefficient matrix.
2. **Cost Function**: The goal is to minimize the Mean Squared Error, $J(B)$, between the predicted and actual outputs.
3. **Solution**: The optimal coefficients are computed using the formula $B = (X^T X)^{-1} X^T Y$.

This approach allows you to estimate the coefficients for multiple output variables at once, and it works similarly to the single-output case, except that the coefficient matrix $B$ has multiple columns, one for each output variable.


# Non-Linear Regression Using Linear Basis Function   

Non-linear regression is a fundamental technique in data mining, used to model complex relationships between input and output variables. While linear regression assumes a direct linear relationship, non-linear regression expands this by transforming the input data using *basis functions*.  

A **linear basis function model** transforms the input variable $x$ into a new representation using a set of basis functions $\phi_j(x)$. These functions help map non-linear relationships into a higher-dimensional space where linear regression can be effectively applied.

---

## **1. Basis Functions in Non-Linear Regression**
Basis functions help transform data into a more suitable form for modeling complex patterns. Below are some commonly used basis functions:

### **A. Polynomial Basis Functions**  
These are simple power functions of $x$:  
```math
\phi_j (x) = x^j
``` 
For example, a polynomial of degree 2 (quadratic) would use basis functions:  
- $\phi_0(x) = 1$  (bias term)
- $\phi_1(x) = x$  (linear term)
- $\phi_2(x) = x^2$  (quadratic term)  

#### **Use in Data Mining:**
- Good for capturing smooth curves in data.
- Works well for problems where the trend follows a known polynomial pattern.
- However, higher-degree polynomials can lead to overfitting.

---

### **B. Gaussian Basis Functions**  
These are bell-shaped functions centered at $\mu_j$:  
```math
\phi_j (x) = \exp \left( -\frac{(x - \mu_j)^2}{2s^2} \right)
``` 
where:
- $\mu_j$ is the center of the Gaussian,
- $s$ is the standard deviation (spread).  

#### **Use in Data Mining:**
- Useful for capturing local variations in data.
- Common in clustering algorithms and kernel-based regression models.
- Suitable for datasets where data points are grouped into localized patterns.

---

### **C. Sigmoidal Basis Functions**  
These use the sigmoid function:  
```math
\phi_j (x) = \sigma \left( \frac{x - \mu_j}{s} \right)
``` 
where:
```math
\sigma(a) = \frac{1}{1 + \exp(-a)}
```
This function smoothly transitions between 0 and 1.

#### **Use in Data Mining:**
- Common in neural networks (activation functions).
- Helps model step-like changes in data.
- Useful in classification tasks where a decision boundary needs to be learned.

---

## **2. How Linear Basis Function Models Work**
A linear combination of basis functions is used to model a target variable $y$:  
```math
y(x) = w_0 + w_1 \phi_1(x) + w_2 \phi_2(x) + ... + w_n \phi_n(x)
``` 
where:
- $w_j$ are the model parameters (weights),
- $\phi_j(x)$ are the basis functions.

This approach allows us to use linear regression techniques while capturing non-linear relationships.

---

## **3. Applications in Data Mining**
- **Predictive Analytics**: Basis function models help in making accurate predictions by capturing non-linear dependencies.
- **Pattern Recognition**: Gaussian basis functions are used in anomaly detection and clustering.
- **Neural Networks**: Sigmoidal basis functions are the foundation of deep learning models.
- **Financial Forecasting**: Polynomial basis functions help in modeling trends in stock prices.

---

## **Conclusion**
Linear basis function models offer a powerful way to apply regression in data mining, making non-linear relationships easier to analyze. By choosing the right basis functions, data scientists can model complex data patterns effectively while maintaining interpretability.
 

# **Regression with Basis Functions**

The concept of **regression with basis functions** involves using mathematical transformations (basis functions) to enhance the predictive power of a regression model, especially when the relationship between inputs and outputs is complex or non-linear. 

Let’s break down the components and the mathematical formulation step-by-step.

---

### **Basic Regression with Basis Functions**
In simple linear regression, we try to predict the output $Y$ based on the input features $X$ using a linear relationship. For example, in simple linear regression:

```math
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_d X_d
``` 

where $X_1$, $X_2$, $\dots$, $X_d$ are input features, and $\beta_0$, $\beta_1$, $\dots$, $\beta_d$ are the coefficients (weights) that we want to learn.

**But what if the relationship is non-linear?**  
We can apply **basis functions** (like squares, exponentials, or other functions) to transform the input features so that a linear relationship becomes applicable in a new feature space. This allows us to capture more complex patterns in the data.

---

### **Understanding the Notation**

In the equation:

```math
Y = \Phi(X) B
``` 

- **$X$ $\in$ $\mathbb{R}^{n \times (d+1)}$**: This is the **input matrix**, where $n$ is the number of data points (samples), and $d$ is the number of features. The "+1" accounts for the bias term $\beta_0$.
  
- **$Y$ $\in$ $\mathbb{R}^{n \times 2}$**: This is the **output matrix**, where $n$ is the number of samples, and there are two output values for each sample (multi-output regression). It could be used to predict multiple outcomes at once, such as both height and weight based on the same input features.
  
- **$B$ $\in$ $\mathbb{R}^{(d+1) \times 2}$**: This is the matrix of **coefficients (weights)**, where each element corresponds to the weight for a particular feature after applying basis functions, and the "+1" again accounts for the intercept term.

- **$\Phi(X)$**: This represents the **transformed feature matrix** where each feature in $X$ is transformed by the chosen basis functions (e.g., polynomials, exponentials, or other non-linear transformations).

---

### **Goal of the Regression:**

The goal of this regression model is to find the best matrix **$B$** (which contains the coefficients $\beta_0$, $\beta_1$, $\dots$, $\beta_d$) that minimizes the error between the predicted and actual values of $Y$.

The error is measured by the **Mean Squared Error (MSE)**:

```math
J(B) = \frac{1}{2n} \sum_{i=1}^{n} (Y_i - \Phi(X_i) B)^2
``` 

Where:
- $Y_i$ is the actual output for the \(i\)-th sample,
- $\Phi(X_i)$ $B$ is the predicted output for the \(i\)-th sample,
- The error is squared to penalize larger errors more, and then averaged over all the samples.

---

### **Matrix Form of the Model**

Now, let’s represent the model in matrix form to handle multiple inputs and outputs efficiently.

#### Input Matrix: $\Phi(X)$

The input matrix **$\Phi(X)$** contains the transformed features for all data points. Each row corresponds to one data point, and each column corresponds to a transformed version of the features.

For example, if you have three data points and two features $X_1$ and $X_2$, the matrix looks like this:

```math
\Phi(X) =
\begin{pmatrix}
1 & \phi(x_{11}) & \phi(x_{12}) \\
1 & \phi(x_{21}) & \phi(x_{22}) \\
1 & \phi(x_{31}) & \phi(x_{32}) \\
\end{pmatrix}
``` 

Where:
- The first column of ones corresponds to the bias term ($\beta_0$).
- The other columns represent the transformed features ($\phi(x)$), which could be squares, exponentials, etc.

#### Output Matrix: $Y$

The output matrix $Y$ contains the actual values we want to predict for each data point. For example, if we want to predict two values for each data point (like height and weight), the output matrix would look like:

```math
Y =
\begin{pmatrix}
y_{11} & y_{12} \\
y_{21} & y_{22} \\
y_{31} & y_{32} \\
\end{pmatrix}
``` 

Where:
- $y_{i1}$ and $y_{i2}$ represent the two output values for the \(i\)-th data point.

#### Coefficients Matrix: $B$

The coefficients matrix $B$ holds the weights that we are trying to learn. The matrix will look like this:

```math
B =
\begin{pmatrix}
\beta_0 & \beta_1 \\
\beta_0 & \beta_2 \\
\beta_1 & \beta_1 \\
\end{pmatrix}
``` 

Where:
- $\beta_0$ represents the intercept (bias term),
- The other $\beta$ values correspond to the weights of the transformed features.

---

### **Estimation of $B$**

To find the best values for the coefficients matrix $B$, we use the following formula:

```math
\hat{B} = (\Phi(X)^T \Phi(X))^{-1} \Phi(X)^T Y
``` 

Where:
- $\Phi(X)^T$ is the **transpose** of the matrix $\Phi(X)$.
- $(\Phi(X)^T \Phi(X))^{-1}$ is the **inverse** of the matrix $\Phi(X)^T$ $\Phi(X)$.
- This formula allows us to solve for $B$ in a way that minimizes the **mean squared error**.

---

### **Summary**

- **Basis functions** allow us to transform the input features into a higher-dimensional space, making it easier for the model to capture complex relationships.
- The model seeks the **best coefficients (weights)** that minimize the difference between the predicted values and the actual values.
- This is done by solving for **$B$** using matrix operations, specifically using the formula $\hat{B} = (\Phi(X)^T \Phi(X))^{-1} \Phi(X)^T Y$.

By using basis functions, the regression model becomes more flexible and can handle more complex, non-linear relationships in the data.

# **Probabilistic Linear Predictions:**

In this section, we are diving into **probabilistic models** in the context of linear regression, specifically how we model the uncertainty of our predictions using **Gaussian distributions**.

### **1. Gaussian Distribution (Univariate)**

First, let’s define the **univariate Gaussian distribution**. It gives us the probability of observing a particular value \( x \) given the mean \( \mu \) and the variance \( \sigma^2 \):

\[
f(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
\]

- **\( \mu \)** is the mean (the center of the distribution).
- **\( \sigma^2 \)** is the variance (a measure of spread).
- The equation describes a bell curve, where the highest point is at \( \mu \), and the spread is controlled by \( \sigma^2 \).

### **2. Multivariate Gaussian Distribution**

Now, when we deal with **multivariate data**, where the input \( X \) is a vector in \( \mathbb{R}^d \) (meaning multiple features), we use the **multivariate Gaussian distribution**.

The probability density function for a **multivariate Gaussian distribution** is given by:

\[
f(X; \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (X - \mu)^T \Sigma^{-1} (X - \mu) \right)
\]

Where:

- **\( X \)** is a vector of observations (a point in the \( d \)-dimensional space).
- **\( \mu \)** is the mean vector of size \( d \times 1 \).
- **\( \Sigma \)** is the covariance matrix (a \( d \times d \) matrix) that describes the correlation between the features.
- **\( |\Sigma| \)** is the determinant of the covariance matrix.
- **\( \Sigma^{-1} \)** is the inverse of the covariance matrix.

The multivariate Gaussian distribution describes how likely the vector \( X \) is, given the mean vector \( \mu \) and the covariance matrix \( \Sigma \).

### **3. Likelihood Estimation in Linear Regression**

Now, let’s connect the probabilistic model to linear regression:

In linear regression, we model the output \( y_i \) for each data point as a linear function of the input features \( X_i \), plus some noise (error):

\[
y_i = X_i B + \epsilon_i
\]

Where:

- \( y_i \) is the observed output (target variable),
- \( X_i \) is the input vector (features) for the \( i \)-th data point,
- \( B \) is the vector of model parameters (coefficients),
- \( \epsilon_i \) is the error term.

We assume that the errors \( \epsilon_i \) are distributed as a **Gaussian distribution** with **mean 0** and **variance \( \sigma^2 \)**:

\[
\epsilon_i \sim \mathcal{N}(0, \sigma^2)
\]

Thus, the output \( y_i \) is also normally distributed with mean \( X_i B \) (the linear prediction) and variance \( \sigma^2 \):

\[
y_i \sim \mathcal{N}(X_i B, \sigma^2)
\]

This implies that the likelihood of observing \( y_i \) given \( X_i \) and the model parameters \( B \) is:

\[
P(y_i | X_i, B) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(y_i - X_i B)^2}{2\sigma^2} \right)
\]

### **4. Likelihood Function for the Entire Dataset**

Given the assumption that each data point is independent, the likelihood function for the entire dataset of size \( n \) (with observations \( (X_1, y_1), (X_2, y_2), \dots, (X_n, y_n) \)) is the product of the individual likelihoods:

\[
P(Y | X, B, \sigma^2) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(y_i - X_i B)^2}{2\sigma^2} \right)
\]

Taking the natural logarithm of the likelihood function gives us the **log-likelihood**:

\[
\log P(Y | X, B, \sigma^2) = -\frac{n}{2} \log(2\pi \sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (y_i - X_i B)^2
\]

This log-likelihood function is what we want to **maximize** in order to estimate the parameters \( B \) and \( \sigma^2 \).

### **5. Maximum Likelihood Estimation**

The maximum likelihood estimation (MLE) of the parameters \( B \) and \( \sigma^2 \) can be found by:

- **Maximizing** the log-likelihood function with respect to \( B \), which will give us the best-fitting coefficients.
- The term inside the summation \( (y_i - X_i B)^2 \) resembles the residual sum of squares (RSS) used in least squares regression.

By differentiating the log-likelihood function with respect to \( B \) and setting it to zero, we get the same **normal equation** that we saw earlier for least squares:

\[
B = (X^T X)^{-1} X^T Y
\]

Thus, the probabilistic approach leads to the same solution as the least squares method.

### **Summary:**

- In **probabilistic linear regression**, we assume that the target variable \( y \) is normally distributed with mean \( X_i B \) and variance \( \sigma^2 \).
- We model the likelihood of the observations and use **maximum likelihood estimation (MLE)** to find the best parameters \( B \) and \( \sigma^2 \).
- The resulting coefficients \( B \) are the same as those found through the traditional least squares method.

This approach provides a probabilistic interpretation of linear regression, helping to quantify the uncertainty around the predictions.
 
 
