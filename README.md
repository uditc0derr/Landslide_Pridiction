# Landslide Risk Predictor

This project is a web-based application that predicts the risk of a landslide in a specific area based on various geological and topographical factors. It utilizes a trained Convolutional Neural Network (CNN) model to analyze the input data and provide a real-time risk assessment.

## How it Works

The application takes nine input parameters for a central cell of a 5x5 grid representing an area. These parameters are:

* **Elevation**: The altitude of the area in meters.
* **Slope**: The steepness of the terrain in degrees.
* **Aspect**: The direction the slope faces in degrees (0-360).
* **Plan Curvature**: The curvature of the contour lines.
* **Profile Curvature**: The curvature of the slope in the direction of the steepest descent.
* **LS Factor**: A combination of slope length and steepness.
* **Topographic Wetness Index (TWI)**: A measure of where water is likely to accumulate.
* **Geology**: A categorical code representing the type of rock and soil.
* **SDOIF**: A measure of the distance to the nearest fault line.

The Flask backend receives this data, preprocesses it using a pre-trained scaler, and feeds it into the CNN model. The model then calculates the probability of a landslide. If the probability is above a set threshold (0.3), the application will classify the area as "High Risk".

## Features

* **User-friendly web interface**: A simple form to input the required data points.
* **Real-time predictions**: Get instant risk assessments from the trained CNN model.
* **Model Confidence Score**: See the model's confidence in its prediction.
* **Clear Results**: The prediction is displayed as either "High Risk: Landslide Likely" or "Low Risk: Landslide Unlikely".

## How to Run the Project

1.  **Clone the repository.**
2.  **Install the required dependencies:**
    ```bash
    pip install Flask flask-cors tensorflow numpy joblib
    ```
3.  **Run the Flask application:**
    ```bash
    python app.py
    ```
4.  **Open the `landslide_app.html` file in your web browser.** You can now enter the required data and get a prediction.

## Files

* `app.py`: The main Flask application file that handles the backend logic.
* `landslide_cnn_model.h5`: The pre-trained Keras model file.
* `scaler.gz`: The saved scikit-learn scaler object.
* `feature_means.npy`: A NumPy file containing the mean values for all 225 features used in the model.
* `templates/landslide_app.html`: The HTML file for the user interface.
* `.gitattributes`: A Git attributes file to handle large files with Git LFS.
