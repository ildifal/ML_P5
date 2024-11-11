import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline

# Load the dataset
dataset = pd.read_csv('dataset.csv')

# Data Preprocessing
numerical_features = dataset.select_dtypes(include=['int64', 'float64']).columns.drop('Turnout')
categorical_features = dataset.select_dtypes(include=['object']).columns

# Handle missing values
dataset.fillna(dataset.median(numeric_only=True), inplace=True)

# One-hot encode categorical variables
dataset_encoded = pd.get_dummies(dataset, columns=categorical_features, drop_first=True)

# Convert target variable to binary classification (using median as a threshold for Turnout)
threshold = dataset_encoded['Turnout'].median()
dataset_encoded['Turnout_Binary'] = (dataset_encoded['Turnout'] > threshold).astype(int)

# Define Features (X) and Target (y)
X = dataset_encoded.drop(columns=['Turnout', 'Turnout_Binary'])
y = dataset_encoded['Turnout_Binary']

# Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression Model with L1 Regularization
logistic_model = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='liblinear', C=1.0))
logistic_model.fit(X_train, y_train)

# Make Predictions
y_pred = logistic_model.predict(X_test)
y_pred_proba = logistic_model.predict_proba(X_test)[:, 1]  # Probability scores for ROC-AUC

# Calculate Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print Metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC Score:", roc_auc)

# Confusion Matrix Visualization
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve Visualization
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

# Detailed Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
