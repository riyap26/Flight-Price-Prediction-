import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso

# Load the dataset from a CSV file in your Downloads folder
file_path = "newestest_itineraries4.csv"  # Replace with the correct file path
data = pd.read_csv(file_path)

# Extract the features (X) and the target variable (y)
X = data.iloc[:, 1:]  # Exclude the first column (target) for features
y = data.iloc[:, 0]  # Use the first column as the target variable

# Create a Lasso regression model
lasso = Lasso(alpha=0.1)  # Adjust alpha as needed

# Fit the model to the data
lasso.fit(X, y)

# Get the feature coefficients (importances)
feature_importances = lasso.coef_

# Identify the least important features (zero coefficients)
least_important_features = [i for i, importance in enumerate(feature_importances) if importance == 0]

# Remove the least important features from the dataset
X_filtered = X.drop(X.columns[least_important_features], axis=1)

# Print the original and filtered data shapes
print(least_important_features)
print("Original data shape:", X.shape)
print("Filtered data shape:", X_filtered.shape)
