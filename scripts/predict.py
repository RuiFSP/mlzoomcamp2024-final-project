import joblib
import os
import pandas as pd
from io import StringIO
from scripts.teams_data import get_teams_data
import logging
from flask import Flask, request, jsonify

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to the saved model and additional validation data
path_model = os.path.join(os.path.dirname(__file__),"models", "final_model_pipeline.pkl")
path_column_names = os.path.join(os.path.dirname(__file__),"models", "column_names.pkl")
path_dtypes = os.path.join(os.path.dirname(__file__),"models", "dtypes.pkl")

# Load the model with error handling
try:
    with open(path_model, 'rb') as f:
        loaded_model = joblib.load(f)
    logger.info(f"Model loaded successfully from {path_model}")
except FileNotFoundError:
    logger.error(f"Model file not found at {path_model}")
    raise Exception(f"Model file not found at {path_model}")

# Load column names and dtypes for validation
try:
    with open(path_column_names, 'rb') as f:
        column_names = joblib.load(f)
    with open(path_dtypes, 'rb') as f:
        dtypes = joblib.load(f)
except FileNotFoundError:
    raise Exception(f"Column names or dtypes file not found.")

app = Flask("predict")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()
        logger.info(f"Received input data: {input_data}")

        # Validate input data
        required_keys = {'home_team', 'away_team', 'date'}
        if not all(key in input_data for key in required_keys):
            logger.error(f"Invalid input format. Missing keys: {set(required_keys) - set(input_data.keys())}")
            return jsonify({"error": "Invalid input format. Expected keys: 'home_team', 'away_team', 'date'."}), 400

        # Get the teams data
        data_json = get_teams_data(input_data['home_team'], input_data['away_team'], input_data['date']).to_json(orient='records')
        logger.info(f"Fetched team data for {input_data['home_team']} vs {input_data['away_team']} on {input_data['date']}")

        # Convert JSON string to dictionary
        data = pd.read_json(StringIO(data_json), orient='records')

        # Validate that the data is in the expected format
        if not isinstance(data, pd.DataFrame):
            logger.error("Invalid data format received. Expected a JSON object.")
            return jsonify({"error": "Invalid input format. Expected a JSON object."}), 400

        logger.info(f"Input data successfully parsed into DataFrame with {data.shape[0]} rows and {data.shape[1]} columns.")

        # Model prediction
        result_dict = {
            0: 'Away_Win',
            1: 'Draw',
            2: 'Home_Win'
        }

        # Making predictions
        raw_prediction = loaded_model.predict(data)[0]
        probabilities = loaded_model.predict_proba(data)[0]

        logger.info(f"Prediction made: {result_dict[raw_prediction]}")
        logger.info(f"Prediction probabilities: {probabilities}")

        # Format the prediction and probabilities for the response
        prediction_prob_df = {
            'Prob_Away_Win': round(float(probabilities[0]), 3),
            'Prob_Draw': round(float(probabilities[1]), 3),
            'Prob_Home_Win': round(float(probabilities[2]), 3),
            'Match_Result': result_dict[raw_prediction]
        }

        logger.info(f"Prediction result: {prediction_prob_df}")

        return jsonify(prediction_prob_df)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)