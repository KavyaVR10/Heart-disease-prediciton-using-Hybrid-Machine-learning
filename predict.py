import numpy as np
import joblib

# Load the hybrid model
hybrid_model = joblib.load("model/hybrid_model.pkl")

scaler = hybrid_model["scaler"]
rf = hybrid_model["random_forest"]
gb = hybrid_model["gradient_boosting"]

# Example input data (replace with real values)
input_data = np.array([[67, 1, 2, 152, 212, 0, 0, 150, 0, 0.8, 1, 0, 3]])  # Replace with actual data

# Preprocess input
input_scaled = scaler.transform(input_data)

# Get RF probabilities
rf_probs = rf.predict_proba(input_scaled)[:, 1].reshape(-1, 1)

# Combine with original features
input_combined = np.hstack((input_scaled, rf_probs))

# Make final prediction
prediction = gb.predict(input_combined)
print("\n🔍 Final Prediction:", "Heart Disease" if prediction[0] == 1 else "No Heart Disease")
