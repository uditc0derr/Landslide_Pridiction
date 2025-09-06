from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import joblib
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

# --- Load Model, Scaler, and Feature Means ---
try:
    model = tf.keras.models.load_model("landslide_cnn_model.h5")
    scaler = joblib.load("scaler.gz")
    # Load the average values for all 225 features
    feature_means = np.load("feature_means.npy") 
    PREDICTION_THRESHOLD = 0.3
    print("Model, scaler, and feature means loaded successfully.")
except Exception as e:
    print(f" Error loading files: {e}")
    model = scaler = feature_means = None

@app.route("/")
def home():
    return render_template("landslide_app.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None or feature_means is None:
        return jsonify({"error": "Model files not loaded properly."}), 500

    data = request.get_json()
    print(f"➡️ Received data: {data}")

    try:
        # --- FINAL FIX: Use feature means as the base ---
        # Create a copy of the means array to use as our input template
        input_features = feature_means.copy()

        # Replicate the alphabetical sorting to find correct indices
        feature_names = sorted([f"{i:02d}_{name}" for i in range(1, 26) for name in ['aspect', 'elevation', 'geology', 'lsfactor', 'placurv', 'procurv', 'sdoif', 'slope', 'twi']])
        feature_index_map = {name: i for i, name in enumerate(feature_names)}

        # Overwrite the mean values for the central cell (13) with the user's input
        input_features[feature_index_map['13_elevation']] = float(data['elevation'])
        input_features[feature_index_map['13_slope']] = float(data['slope'])
        input_features[feature_index_map['13_aspect']] = float(data['aspect'])
        input_features[feature_index_map['13_placurv']] = float(data['placurv'])
        input_features[feature_index_map['13_procurv']] = float(data['procurv'])
        input_features[feature_index_map['13_lsfactor']] = float(data['lsfactor'])
        input_features[feature_index_map['13_twi']] = float(data['twi'])
        input_features[feature_index_map['13_geology']] = float(data['geology'])
        input_features[feature_index_map['13_sdoif']] = float(data['sdoif'])
        
        # --- Preprocessing & Prediction ---
        scaled_features = scaler.transform(input_features.reshape(1, -1))
        reshaped_features = scaled_features.reshape(1, 5, 5, 9)
        prediction_proba = model.predict(reshaped_features)[0][0]
        is_landslide = prediction_proba > PREDICTION_THRESHOLD
        
        result = {
            "prediction": "High Risk: Landslide Likely" if is_landslide else "Low Risk: Landslide Unlikely",
            "confidence_score": f"{prediction_proba:.4f}",
            "is_landslide": bool(is_landslide)
        }
        print(f"⬅️ Sending prediction: {result}")
        return jsonify(result)

    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input data: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {e}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
