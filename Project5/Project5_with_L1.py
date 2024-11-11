import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
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

# Define Features (X) and Target (y)
X = dataset_encoded.drop(columns=['Turnout'])
y = dataset_encoded['Turnout']

# Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Lasso Regression Model with L1 Regularization
lasso_model = make_pipeline(StandardScaler(), Lasso(alpha=0.1, random_state=42))
lasso_model.fit(X_train, y_train)

# Make Predictions and Evaluate the Model
y_train_pred = lasso_model.predict(X_train)
y_test_pred = lasso_model.predict(X_test)

# Calculate Mean Squared Error and R² Score
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print evaluation metrics
print("Train MSE:", train_mse)
print("Test MSE:", test_mse)
print("Train R² Score:", train_r2)
print("Test R² Score:", test_r2)

# Visualize the Actual vs Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.6, label='Predictions', color='orange')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='blue', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual Turnout')
plt.ylabel('Predicted Turnout')
plt.title('Lasso Regression Predictions')
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance Visualization (Lasso Coefficients)
lasso_coefficients = lasso_model.named_steps['lasso'].coef_
feature_names = X.columns

# Filter non-zero coefficients for better visualization
non_zero_coefficients = lasso_coefficients != 0
feature_names = feature_names[non_zero_coefficients]
lasso_coefficients = lasso_coefficients[non_zero_coefficients]

plt.figure(figsize=(12, 8))
plt.barh(feature_names, lasso_coefficients)
plt.xlabel("Lasso Coefficient")
plt.title("Feature Importance (Lasso Regression)")
plt.grid(True)
plt.show()
