# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate a simple synthetic dataset (for demonstration purposes)
# Let's create a dataset with a single feature (X) and a target (y)
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # Random feature, shape (100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relation with some noise

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the evaluation metrics
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Visualize the regression line and the actual vs predicted values
plt.figure(figsize=(12, 6))

# Plot the regression line (for training data)
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.plot(X_train, model.predict(X_train), color='red', label='Regression Line')
plt.title('Training Data and Regression Line')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.legend()

# Plot actual vs predicted values (for test data)
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, color='green', label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='black', linestyle='--', label='Ideal Prediction Line')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()
