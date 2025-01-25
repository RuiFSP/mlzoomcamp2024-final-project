import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from io import StringIO
from scripts.predict import predict_match_outcome, validate_team_name
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "models", "best_model.keras")
feature_info_path = os.path.join(base_dir, "models", "feature_info.npy")
ct_path = os.path.join(base_dir, "models", "column_transformer.pkl")
label_encoder_path = os.path.join(base_dir, "models", "label_encoder.pkl")
data_path = os.path.join(base_dir, "data", "processed", "data_for_model.csv")

# Load the model and additional data
try:
    best_model = tf.keras.models.load_model(model_path)
    feature_info = np.load(feature_info_path, allow_pickle=True).item()
    ct = joblib.load(ct_path)
    label_encoder = joblib.load(label_encoder_path)
    data = pd.read_csv(data_path)
    logger.info("Model and additional data loaded successfully")
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    st.error(f"File not found: {e}")
except Exception as e:
    logger.error(f"Error loading model or data: {str(e)}")
    st.error(f"Error loading model or data: {str(e)}")

# List of all team names
list_of_teams = data['home_team'].unique().tolist()

# Streamlit UI
st.set_page_config(page_title="Premier League Match Predictor", layout="wide")
st.title("Premier League Match Outcome Predictor - Deep Learning")

st.sidebar.header("Match Details")
home_team = st.sidebar.selectbox("Home Team", list_of_teams)
away_team = st.sidebar.selectbox("Away Team", list_of_teams)
match_date = st.sidebar.date_input("Match Date")

st.sidebar.markdown("### Prediction")
if st.sidebar.button("Predict Outcome"):
    if home_team and away_team and match_date:
        try:
            # Validate team names
            if not validate_team_name(home_team, data):
                st.error(f"Invalid home team name: {home_team}")
            elif not validate_team_name(away_team, data):
                st.error(f"Invalid away team name: {away_team}")
            else:
                # Making predictions
                prediction = predict_match_outcome(data, home_team, away_team, str(match_date), ct, best_model, label_encoder, feature_info)

                # Log the prediction result for debugging
                logger.info(f"Prediction result: {prediction}")

                # Check the keys in the prediction dictionary
                logger.debug(f"Prediction keys: {prediction.keys()}")
                logger.debug(f"Match_Result value: {prediction.get('Match_Result')}")

                # Format prediction result for display
                match_result = prediction.get('Match_Result', 'Unknown')
                logger.debug(f"Mapped Match_Result: {match_result}")

                prob_home_win = round(float(prediction.get('Prob_Home_Win', 0)), 3)
                prob_draw = round(float(prediction.get('Prob_Draw', 0)), 3)
                prob_away_win = round(float(prediction.get('Prob_Away_Win', 0)), 3)

                prediction_result = {
                    'Match Result': match_result,
                    'Probability Home Win': prob_home_win,
                    'Probability Draw': prob_draw,
                    'Probability Away Win': prob_away_win
                }

                st.subheader("Prediction Result")
                st.write(prediction_result)

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            st.error(f"Error during prediction: {str(e)}")
    else:
        st.warning("Please enter all the required fields.")
