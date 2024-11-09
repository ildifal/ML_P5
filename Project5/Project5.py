import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the data
turnout_init = pd.read_csv('dataset.csv')

# Split into training and testing sets
train_set, test_set = train_test_split(turnout_init, test_size=0.2, random_state=42)

# Process the training set
# Handle outliers and scale numerical data
std_scaler = StandardScaler()
train_set_num = train_set.select_dtypes(include=[np.number]).drop(columns=["Year"])
train_set_num_scaled = std_scaler.fit_transform(train_set_num)

# Convert scaled array back to DataFrame
train_set_num_scaled = pd.DataFrame(train_set_num_scaled, columns=train_set_num.columns, index=train_set.index)

# Handle missing numerical data
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(train_set_num_scaled)
train_set_tr = pd.DataFrame(X_train, columns=train_set_num.columns, index=train_set.index)

# Encode categorical data in training set
cat_encoder = OneHotEncoder(sparse_output=False)
train_cat = train_set[["House Majority", "Senate Majority", "Current Presidential Party"]]
train_cat_encoded = pd.DataFrame(cat_encoder.fit_transform(train_cat),
                                 columns=cat_encoder.get_feature_names_out(train_cat.columns),
                                 index=train_set.index)

# Combine all processed columns for the training set
train_set_final = pd.concat([train_set_tr, train_cat_encoded], axis=1)

# Process the test set using the same transformations
# Scale numerical data in test set
test_set_num = test_set.select_dtypes(include=[np.number]).drop(columns=["Year"])
test_set_num_scaled = std_scaler.transform(test_set_num)  # Use transform (not fit_transform) on the test set
test_set_num_scaled = pd.DataFrame(test_set_num_scaled, columns=test_set_num.columns, index=test_set.index)

# Impute missing numerical data in test set
X_test = imputer.transform(test_set_num_scaled)
test_set_tr = pd.DataFrame(X_test, columns=test_set_num.columns, index=test_set.index)

# Encode categorical data in test set using the same encoder
test_cat = test_set[["House Majority", "Senate Majority", "Current Presidential Party"]]
test_cat_encoded = pd.DataFrame(cat_encoder.transform(test_cat),
                                columns=cat_encoder.get_feature_names_out(train_cat.columns),
                                index=test_set.index)

# Combine all processed columns for the test set
test_set_final = pd.concat([test_set_tr, test_cat_encoded], axis=1)

# Now train_set_final and test_set_final are ready for model training and evaluation
# print(train_set_final.head())
# print(test_set_final.head())

# VISUALIZING THE DATA
"""Here we are looking at all data in a large scatter matrix"""
scatter_matrix(train_set_final, alpha=0.2, figsize=(20, 20), diagonal='hist')
plt.show()

"""Here we are only looking at a few of the features and Turnout (label)"""
train_set_final.plot(kind="scatter", x="Registered Voters", y="Turnout")
plt.show()
train_set_final.plot(kind="scatter", x="Voting Age Population (VAP)", y="Turnout")
plt.show()
train_set_final.plot(kind="scatter", x="Average Income", y="Turnout")
plt.show()
train_set_final.plot(kind="scatter", x="Age 18-24", y="Turnout")
plt.show()

# Checking correlations
corr_matrix = train_set_final.corr()
print(corr_matrix["Turnout"].sort_values(ascending=False))

# Generate a heat map
plt.figure(figsize=(15, 15))
sns.heatmap(train_set_final.corr(), annot=True, fmt='.2f')
plt.show()
"""Based on the heatmap, the categorical columns have 0 correlation, and ages have 1"""
