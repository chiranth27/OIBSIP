import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


iris = load_iris(as_frame=True)
df = iris.frame

# Split data into features and target
X = df.drop(columns=['target'])
y = df['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9927)

# Building decision trees using bootstrap sampling and Gini impurity
def evaluate_trees(num_trees, num_features):
    trees = []
    out_of_bag_errors = []

    for _ in range(num_trees):
        # Bootstrap sampling
        sample_indices = np.random.choice(len(X_train), len(X_train), replace=True)
        out_of_bag_indices = [i for i in range(len(X_train)) if i not in sample_indices]
        
        X_sample = X_train.iloc[sample_indices, :num_features]
        y_sample = y_train.iloc[sample_indices]

        # Train decision tree
        tree = DecisionTreeClassifier(criterion='gini', random_state=9926)
        tree.fit(X_sample, y_sample)
        trees.append(tree)

        # Compute out-of-bag error
        if out_of_bag_indices:
            X_out_of_bag = X_train.iloc[out_of_bag_indices, :num_features]
            y_out_of_bag = y_train.iloc[out_of_bag_indices]
            out_of_bag_predictions = tree.predict(X_out_of_bag)
            out_of_bag_error = 1 - accuracy_score(y_out_of_bag, out_of_bag_predictions)
            out_of_bag_errors.append(out_of_bag_error)

    
    return trees, out_of_bag_errors

# Evaluating trees using 3 and 4 features
trees_3_features, out_of_bag_errors_3_features = evaluate_trees(10, 3)
trees_4_features, out_of_bag_errors_4_features = evaluate_trees(10, 4)

# Selecting the best trees based on out-of-bag error
best_tree_3_features = trees_3_features[np.argmin(out_of_bag_errors_3_features)]
best_tree_4_features = trees_4_features[np.argmin(out_of_bag_errors_4_features)]

print("Best tree with 3 features Out-of-Bag error:", min(out_of_bag_errors_3_features))
print("Best tree with 4 features Out-of-Bag error:", min(out_of_bag_errors_4_features))

# Comparing out-of-bag errors
if min(out_of_bag_errors_3_features) < min(out_of_bag_errors_4_features):
    best_tree = best_tree_3_features
    print("Selected tree: Best tree with 3 features")
else:
    best_tree = best_tree_4_features
    print("Selected tree: Best tree with 4 features")

# Evaluating the best tree on the test set
test_predictions = best_tree.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Test set accuracy of the selected tree:", test_accuracy)

# Important feature weightage
importances = best_tree.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(X_train.shape[1]), [X_train.columns[i] for i in indices])
plt.xlim([-1, X_train.shape[1]])
plt.show()

# Plot the best trees
plt.figure(figsize=(20, 10))
plot_tree(best_tree_3_features, feature_names=X_train.columns[:3], class_names=iris.target_names, filled=True)
plt.title("Best Decision Tree with 3 Features")
plt.show()

plt.figure(figsize=(20, 10))
plot_tree(best_tree_4_features, feature_names=X_train.columns, class_names=iris.target_names, filled=True)
plt.title("Best Decision Tree with 4 Features")
plt.show()
