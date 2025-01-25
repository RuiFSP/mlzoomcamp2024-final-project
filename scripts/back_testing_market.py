# imports
import os
import warnings
import logging
from typing import Tuple, List

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress TensorFlow and CUDA logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

# Suppress specific TensorFlow warnings
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Disable GPU usage (if not using GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Suppress all TensorFlow logs and CUDA warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

#notebook settings
pd.set_option('display.max_columns', None)

# Define base directory for models
base_dir = os.path.dirname(__file__)

def load_data(base_dir: str) -> pd.DataFrame:
    """Load the data from the specified directory."""
    data_path = os.path.join(base_dir, '..', 'data', 'processed', 'prepared_football_data.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    logging.info(f"Loading data from {data_path}")
    return pd.read_csv(data_path)

def load_model_and_transformers(base_dir: str) -> Tuple[tf.keras.Model, dict, ColumnTransformer, LabelEncoder]:
    """Load the model and transformers from the specified directory."""
    logging.info("Loading model and transformers")
    model = tf.keras.models.load_model(os.path.join(base_dir, '..', 'models', 'best_model.keras'))
    feature_info = np.load(os.path.join(base_dir, '..', 'models', 'feature_info.npy'), allow_pickle=True).item()
    ct = joblib.load(os.path.join(base_dir, '..', 'models', 'column_transformer.pkl'))
    label_encoder = joblib.load(os.path.join(base_dir, '..', 'models', 'label_encoder.pkl'))
    return model, feature_info, ct, label_encoder

def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Preprocess the data and split it into training and testing sets."""
    logging.info("Preprocessing data")
    X = data.drop(columns=['full_time_result'])
    y = data['full_time_result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    return X_train, X_test, y_train, y_test

def predict_results(model: tf.keras.Model, ct: ColumnTransformer, X_test: pd.DataFrame, label_encoder: LabelEncoder) -> Tuple[np.ndarray, np.ndarray]:
    """Predict the results using the model and transformers."""
    logging.info("Predicting results")
    y_pred_prob = model.predict(ct.transform(X_test))
    y_pred = label_encoder.inverse_transform(np.argmax(y_pred_prob, axis=1))
    return y_pred_prob, y_pred

def create_team_names_df(X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray, y_pred_prob: np.ndarray, data_for_back_testing: pd.DataFrame) -> pd.DataFrame:
    """Create a DataFrame to store team names and predictions."""
    logging.info("Creating team names DataFrame")
    team_names = X_test.loc[:, ['home_team', 'away_team']].copy()
    team_names['true_results'] = y_test
    team_names['predicted_results'] = y_pred
    team_names[['away_prob', 'draw_prob', 'home_prob']] = pd.DataFrame(y_pred_prob, index=team_names.index)
    team_names[['implied_home_win_prob', 'implied_draw_prob', 'implied_away_win_prob']] = data_for_back_testing.loc[team_names.index, ['implied_home_win_prob', 'implied_draw_prob', 'implied_away_win_prob']]
    team_names = team_names[['home_team', 'away_team', 'true_results', 'predicted_results', 'home_prob', 'draw_prob', 'away_prob', 'implied_home_win_prob', 'implied_draw_prob', 'implied_away_win_prob']]
    return team_names

def calculate_brier_score(predictions: np.ndarray, true_outcome: List[int]) -> float:
    """Calculate the Brier score for the given predictions and true outcomes."""
    return brier_score_loss(true_outcome, predictions)

def encode_result(result: str) -> List[int]:
    """Encode the result as a one-hot encoded list."""
    if result == 'H':
        return [1, 0, 0]
    elif result == 'D':
        return [0, 1, 0]
    elif result == 'A':
        return [0, 0, 1]

def calculate_brier_scores(team_names: pd.DataFrame) -> Tuple[float, float]:
    """Calculate the Brier scores for the market and the model."""
    logging.info("Calculating Brier scores")
    team_names['true_predictions_brier'] = team_names['true_results'].apply(encode_result)
    team_names['brier_score_market'] = team_names.apply(lambda row: calculate_brier_score(np.array([row['home_prob'], row['draw_prob'], row['away_prob']]), row['true_predictions_brier']), axis=1)
    team_names['brier_score_model'] = team_names.apply(lambda row: calculate_brier_score(np.array([row['implied_home_win_prob'], row['implied_draw_prob'], row['implied_away_win_prob']]), row['true_predictions_brier']), axis=1)
    average_brier_score_market = team_names['brier_score_market'].mean()
    average_brier_score_model = team_names['brier_score_model'].mean()
    return average_brier_score_market, average_brier_score_model

def main():
    """Main function to run the back-testing script."""
    logging.info("Starting back-testing script")
    base_dir = os.path.dirname(__file__)
    data = load_data(base_dir)
    model, feature_info, ct, label_encoder = load_model_and_transformers(base_dir)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    y_pred_prob, y_pred = predict_results(model, ct, X_test, label_encoder)
    data_for_back_testing = data[['implied_home_win_prob', 'implied_draw_prob', 'implied_away_win_prob']]
    team_names = create_team_names_df(X_test, y_test, y_pred, y_pred_prob, data_for_back_testing)
    average_brier_score_market, average_brier_score_model = calculate_brier_scores(team_names)
    logging.info(f"Average Brier Score Market: {average_brier_score_market}")
    logging.info(f"Average Brier Score Model: {average_brier_score_model}")

if __name__ == "__main__":
    main()
