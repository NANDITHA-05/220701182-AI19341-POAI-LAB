# Import necessary libraries
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the Boston Housing dataset
data = load_boston()
X = data['data']  # Features: Various factors affecting house price
y = data['target']  # House price (continuous target variable)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the ANN model for regression
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),  # Input layer with 13 features (for Boston dataset)
    Dense(32, activation='relu'),                             # Hidden layer
    Dense(1)                                                  # Output layer (single continuous value)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='mean_squared_error',  # Loss function for regression
              metrics=['mean_absolute_error'])  # Metric to track

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Mean Absolute Error: {mae:.2f}")

# Make predictions
predictions = model.predict(X_test)

# Display predictions vs true values
for i in range(10):  # Display the first 10 predictions
    print(f"Predicted: {predictions[i][0]:.2f}, Actual: {y_test[i]:.2f}")
