import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.special import expit

# Load the dataset
file_path = "C:\\Users\\chira\\Desktop\\Oasis_Infobyte\\spam.csv"
df = pd.read_csv(file_path, encoding='latin-1')

# Inspect the data
print(df.head())
print(df.info())
print(df.describe())

# Drop unnecessary columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

# Rename columns
df.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)


# Visualize the distribution of spam and ham
plt.figure(figsize=(6, 4))
sb.countplot(data=df, x='label')
plt.title('Distribution of Spam and Ham Emails')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()



# Convert labels to binary values
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=9926)
 
# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# Initialize weights and bias
def initialize_params(n_features):
    W = np.zeros(n_features)
    b = 0
    return W, b

# Sigmoid function
def sigmoid(z):
    return expit(z)

# Logistic regression model
def logistic_regression(X, y, W, b, learning_rate, num_iterations):
    m = X.shape[0]
    cost_history = []

    for i in range(num_iterations):
        # Compute linear combination of inputs and weights
        z = np.dot(X, W) + b
        # Apply sigmoid activation function
        y_pred = sigmoid(z)

        # Compute cost
        cost = (-1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        cost_history.append(cost)

        # Compute gradients
        dW = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)

        # Update weights and bias
        W -= learning_rate * dW
        b -= learning_rate * db

        
            

    return W, b, cost_history

# Initialize parameters
n_features = X_train_tfidf.shape[1]
W, b = initialize_params(n_features)

# Train the model
learning_rate = 0.01
num_iterations = 1000
W, b, cost_history = logistic_regression(X_train_tfidf, y_train, W, b, learning_rate, num_iterations)

# Plot the cost function
plt.plot(range(num_iterations), cost_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.title('Cost function over iterations')
plt.show()

# Make predictions
def predict(X, W, b):
    z = np.dot(X, W) + b
    y_pred = sigmoid(z)
    return [1 if i > 0.5 else 0 for i in y_pred]

y_pred_train = predict(X_train_tfidf, W, b)
y_pred_test = predict(X_test_tfidf, W, b)

# Evaluate the model
print("Accuracy on training set:", accuracy_score(y_train, y_pred_train))
print("Accuracy on test set:", accuracy_score(y_test, y_pred_test))
print("Classification Report:\n", classification_report(y_test, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))

# Visualize the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(6, 4))
sb.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
