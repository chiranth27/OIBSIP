import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv("C:\\Users\\chira\\Desktop\\Oasis_Infobyte\\Advertising.csv")

print(df.head())
print(df.describe())
print(df.info())
print(df.shape)

# Plotting
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Scatter Plot of TV vs Sales
sb.scatterplot(data=df, x='TV', y='Sales', ax=axs[0, 0])
axs[0, 0].set_title('Scatter Plot of TV vs Sales')
axs[0, 0].set_xlabel('TV')
axs[0, 0].set_ylabel('Sales')

# Scatter Plot of Radio vs Sales
sb.scatterplot(data=df, x='Radio', y='Sales', ax=axs[0, 2])
axs[0, 2].set_title('Scatter Plot of Radio Vs Sales')
axs[0, 2].set_xlabel('Radio')
axs[0, 2].set_ylabel('sales')

# Hide the middle subplot in the first row
axs[0, 1].axis('off')

# Scatter Plot of Newspaper vs Sales
sb.scatterplot(data=df, x='Newspaper', y='Sales', ax=axs[1, 1])
axs[1, 1].set_title('Scatter Plot of Newspaper Vs Sales')
axs[1, 1].set_xlabel('Newspaper')
axs[1, 1].set_ylabel('Sales')

# Hide the empty subplots
axs[1, 0].axis('off')
axs[1, 2].axis('off')
plt.tight_layout()

# Rename columns
df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
df.drop(df.columns[[0]], axis=1, inplace=True)

# Correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix)
plt.figure(figsize=(15, 8))
sb.heatmap(correlation_matrix, annot=True, cmap='YlGnBu')
plt.show()

# Prepare data for training
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Add intercept
X = np.c_[np.ones(X.shape[0]), X]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9927)

# Initialize weights
def initialize_parameters(n):
    W = np.zeros(n)
    return W

W = initialize_parameters(X_train.shape[1])

# Hypothesis function
def hypothesis(X, W):
    return np.dot(X, W)

# Cost function
def compute_cost(X, y, W):
    J = (1 / (2 * len(y))) * np.sum((hypothesis(X, W) - y) ** 2)
    return J

# Gradient descent algorithm
def gradient_descent(X, y, W, learning_rate, num_iterations):
    m = len(y)
    cost_history = []

    for i in range(num_iterations):
        gradients = (1 / m) * np.dot(X.T, (hypothesis(X, W) - y))
        W = W - learning_rate * gradients
        cost = compute_cost(X, y, W)
        cost_history.append(cost)

    return W, cost_history

# Training the model
learning_rate = 0.00001
num_iterations = 2000
W, cost_history = gradient_descent(X_train, y_train, W, learning_rate, num_iterations)

# Plot the cost function
plt.plot(range(num_iterations), cost_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.title('Cost function over iterations')
plt.show()

# Make predictions on the training and test sets
y_pred_train = hypothesis(X_train, W)
y_pred_test = hypothesis(X_test, W)

# Compute the mean squared error on the training set
mse_train = np.mean((y_pred_train - y_train) ** 2)
print(f"Mean Squared Error on Training Set: {mse_train}")

# Compute the mean squared error on the test set
mse_test = np.mean((y_pred_test - y_test) ** 2)
print(f"Mean Squared Error on Test Set: {mse_test}")

# R^2 function
def r2(X, y, W):
    y_pred = hypothesis(X, W)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2

# Calculate R^2 for the training set
r2_train = r2(X_train, y_train, W)
print("R^2 on training set:", r2_train)

# Calculate R^2 for the test set
r2_test = r2(X_test, y_test, W)
print("R^2 on test set:", r2_test)

# Check accuracy of the codes
print(r2_score(y_test, y_pred_test))
print(r2_score(y_train, y_pred_train))

# K-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=9926)
mse_scores = []
r2_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    W = initialize_parameters(X_train.shape[1])
    W, cost_folds = gradient_descent(X_train, y_train, W, learning_rate, num_iterations)

    y_pred_test = hypothesis(X_test, W)
    mse = np.mean((y_pred_test - y_test) ** 2)
    r2 = r2_score(y_test, y_pred_test)

    mse_scores.append(mse)
    r2_scores.append(r2)

print(f"Mean Squared Error for each fold: {mse_scores}")
print(f"Average Mean Squared Error: {np.mean(mse_scores)}")
print(f"R^2 Scores for each fold: {r2_scores}")
print(f"Average R^2 Score: {np.mean(r2_scores)}")
