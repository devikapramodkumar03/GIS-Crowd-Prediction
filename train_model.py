import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
dataset_path = "data/dataset.csv"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please check the path or file.")

# Load the CSV file
df = pd.read_csv(dataset_path)

# Check columns
required_columns = ["latitude", "longitude", "ndvi", "temperature", "crowd_density"]
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Dataset must include columns: {required_columns}")

# Prepare input features (X) and target variable (y)
X = df[["latitude", "longitude", "ndvi", "temperature"]]
y = df["crowd_density"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and clip predictions between 0 and 10 people/m² (physical limits)
y_pred = np.clip(model.predict(X_test), 0, 10)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
print(f"Model Mean Absolute Error: {mae:.2f} people/m²")

# Save trained model
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "trained_model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Trained model saved at {model_path}")
