from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os
from sentinel_api import get_satellite_data

app = Flask(__name__, template_folder="templates")

# Load trained model
model_path = "models/trained_model.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found at {model_path}. Train the model first.")

with open(model_path, "rb") as f:
    model = pickle.load(f)
    
@app.route("/")
def root():
    return render_template("landing.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/form")
def index():
    """Serve the HTML page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict crowd density based on input latitude, longitude, temperature, and NDVI.
    """
    try:
        data = request.json
        lat = float(data["latitude"])
        lon = float(data["longitude"])
        temp = float(data["temperature"])

        # Fetch NDVI data using the Sentinel API
        ndvi = get_satellite_data(lat, lon)

        # Validate NDVI
        if not isinstance(ndvi, (int, float)):
            raise ValueError(f"Invalid NDVI value: {ndvi}")

        # Prepare input for the model
        input_features = np.array([[lat, lon, ndvi, temp]])
        predicted_crowd = model.predict(input_features)[0]

        # Clip the value to realistic range (optional)
        predicted_crowd = max(0, min(predicted_crowd, 10))

        return jsonify({
            "predicted_crowd": round(predicted_crowd, 2),
            "unit": "people/m²"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
