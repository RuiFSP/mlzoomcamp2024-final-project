import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow GPU warnings

import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import logging
from datetime import datetime
from typing import Dict, Any

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define base directory for models
base_dir = os.path.dirname(__file__)

# Load the model and preprocessing objects
model_path = os.path.join(base_dir, '..', 'models', 'best_model.keras')
feature_info_path = os.path.join(base_dir, '..', 'models', 'feature_info.npy')
ct_path = os.path.join(base_dir, '..', 'models', 'column_transformer.pkl')
label_encoder_path = os.path.join(base_dir, '..', 'models', 'label_encoder.pkl')

try:
    best_model = tf.keras.models.load_model(model_path)
    feature_info = np.load(feature_info_path, allow_pickle=True).item()
    ct = joblib.load(ct_path)
    label_encoder = joblib.load(label_encoder_path)
    logger.info("Model and preprocessing objects loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or preprocessing objects: {str(e)}")
    raise

app = Flask("predict")

def validate_team_name(team_name: str, data: pd.DataFrame) -> bool:
    """Validate if the team name exists in the dataset."""
    return team_name in data['home_team'].unique() or team_name in data['away_team'].unique()

def predict_match_outcome(data: pd.DataFrame, home_team: str, away_team: str, date_str: str, 
                          ct: Any, best_model: tf.keras.Model, label_encoder: Any, 
                          feature_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict the outcome of a football match based on historical data and features.

    Parameters:
        data (pd.DataFrame): The historical match data.
        home_team (str): Home team's name.
        away_team (str): Away team's name.
        date_str (str): Date of the match in 'YYYY-MM-DD' format.
        ct (ColumnTransformer): Preprocessing pipeline from training.
        best_model_dense (keras.Model): Trained neural network model.
        label_encoder (LabelEncoder): Encoder for target classes.
        feature_info (dict): Dictionary containing `categorical_features` and `numerical_features`.

    Returns:
        dict: Predicted probabilities and the predicted result.
    """
    # Extract feature info
    categorical_features = feature_info['categorical_features']
    numerical_features = feature_info['numerical_features']

    # Step 1: Define home and away team columns
    team_columns_home = [col for col in data.columns if col.startswith('home_')]
    team_columns_away = [col for col in data.columns if col.startswith('away_')]

    # Step 2: Get the latest features for both teams
    team_data_home = data[data['home_team'] == home_team].tail(1)[team_columns_home]
    team_data_away = data[data['away_team'] == away_team].tail(1)[team_columns_away]

    # Combine features into a single row
    combined_features = pd.concat([team_data_home.reset_index(drop=True), 
                                    team_data_away.reset_index(drop=True)], axis=1)

    # Step 3: Add date-related features
    date = pd.to_datetime(date_str, format='%Y-%m-%d')
    day_of_week = date.dayofweek
    month = date.month
    day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7.0)
    day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7.0)
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)

    date_features = pd.DataFrame([{
        'day_of_week': day_of_week,
        'month': month,
        'day_of_week_sin': day_of_week_sin,
        'day_of_week_cos': day_of_week_cos,
        'month_sin': month_sin,
        'month_cos': month_cos
    }])

    # Combine all features
    final_features = pd.concat([combined_features, date_features], axis=1)

    # Step 4: Add missing features
    # Add missing categorical features with a default "missing" value
    for feature in categorical_features:
        if feature not in final_features.columns:
            final_features[feature] = "missing"

    # Add missing numerical features with a default value of 0
    for feature in numerical_features:
        if feature not in final_features.columns:
            final_features[feature] = 0

    # Reorder the columns to match the training feature order
    final_features = final_features[categorical_features + numerical_features]

    # Step 5: Preprocess the features
    X_new_processed = ct.transform(final_features)

    # Step 6: Predict probabilities
    y_pred = best_model.predict(X_new_processed)

    # Map probabilities to outcomes
    prob_home_win = y_pred[0][label_encoder.transform(['H'])[0]]
    prob_draw = y_pred[0][label_encoder.transform(['D'])[0]]
    prob_away_win = y_pred[0][label_encoder.transform(['A'])[0]]

    # Predicted class
    y_pred_classes = np.argmax(y_pred, axis=1)
    predicted_label = label_encoder.inverse_transform(y_pred_classes)[0]

    # Map predicted label to match result
    result_mapping = {'H': 'Home_Win', 'D': 'Draw', 'A': 'Away_Win'}
    match_result = result_mapping[predicted_label]

    return {
        'Match_Result': match_result,
        'Prob_Home_Win': prob_home_win,
        'Prob_Draw': prob_draw,
        'Prob_Away_Win': prob_away_win
    }

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()
        logger.info(f"Received input data: {input_data}")

        required_keys = {'home_team', 'away_team', 'date'}
        if not all(key in input_data for key in required_keys):
            logger.error(f"Invalid input format. Missing keys: {set(required_keys) - set(input_data.keys())}")
            return jsonify({"error": "Invalid input format. Expected keys: 'home_team', 'away_team', 'date'."}), 400

        home_team = input_data['home_team']
        away_team = input_data['away_team']
        date_str = input_data['date']

        # Validate date format
        try:
            pd.to_datetime(date_str, format='%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid date format: {date_str}")
            return jsonify({"error": "Invalid date format. Expected 'YYYY-MM-DD'."}), 400

        data = pd.read_csv(os.path.join(base_dir, '..', 'data', 'processed', 'data_for_model.csv'))

        # Validate team names
        if not validate_team_name(home_team, data):
            logger.error(f"Invalid home team name: {home_team}")
            return jsonify({"error": f"Invalid home team name: {home_team}"}), 400
        if not validate_team_name(away_team, data):
            logger.error(f"Invalid away team name: {away_team}")
            return jsonify({"error": f"Invalid away team name: {away_team}"}), 400

        prediction = predict_match_outcome(data, home_team, away_team, date_str, ct, best_model, label_encoder, feature_info)

        # Convert float32 values to float
        prediction = {k: float(v) if isinstance(v, np.float32) else v for k, v in prediction.items()}

        logger.info(f"Prediction: {prediction}")
        return jsonify(prediction)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)