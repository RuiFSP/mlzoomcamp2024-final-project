import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import keras_tuner as kt
import logging
import argparse
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Define base directory for logs and models
base_dir = os.path.dirname(__file__)  # The root of your project folder

# Function to create log directories
def create_log_dirs() -> str:
    """
    Create directories for logging and return the log directory path.
    """
    log_dir = os.path.join(base_dir, 'logs', 'model_training', datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

# Function to create tuner directories
def create_tuner_dirs() -> str:
    """
    Create directories for the tuner and return the tuner directory path.
    """
    tuner_dir = os.path.join(base_dir, 'tuner', 'dense_model_tuning')
    os.makedirs(tuner_dir, exist_ok=True)
    return tuner_dir

# Function to delete directories
def delete_dirs(dirs: list) -> None:
    """
    Delete specified directories.
    
    Args:
        dirs (list): List of directory paths to delete.
    """
    for dir_path in dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            logger.info(f"Deleted directory: {dir_path}")

# Function to load data
def load_data() -> pd.DataFrame:
    """
    Load data from a CSV file and return it as a DataFrame.
    """
    logger.info("Loading data from CSV...")
    return pd.read_csv(os.path.join(base_dir, '..', 'data', 'processed', 'data_for_model.csv'))

# Function to preprocess data
def preprocess_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Preprocess the data by encoding categorical features and scaling numerical features.
    
    Args:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target data.
    
    Returns:
        tuple: Processed features, encoded target, column transformer, label encoder, and feature info.
    """
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.drop(columns=categorical_features).columns.tolist()
    
    feature_info = {
        'categorical_features': categorical_features,
        'numerical_features': numerical_features
    }
    
    ct = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                           ('num', StandardScaler(), numerical_features)])
    X_processed = ct.fit_transform(X)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = tf.keras.utils.to_categorical(y_encoded)
    
    return X_processed, y_categorical, ct, label_encoder, feature_info

# Function to split data into training and testing
def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Split the data into training and test sets.
    
    Args:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target data.
    
    Returns:
        tuple: Training and test sets for features and target.
    """
    logger.info("Splitting data into training and test sets...")
    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Function to save preprocessing objects
def save_preprocessing_objects(feature_info: dict, ct: ColumnTransformer, label_encoder: LabelEncoder) -> None:
    """
    Save preprocessing objects to disk.
    
    Args:
        feature_info (dict): Information about features.
        ct (ColumnTransformer): Column transformer.
        label_encoder (LabelEncoder): Label encoder.
    """
    logger.info("Saving preprocessing objects...")
    np.save(os.path.join(base_dir, '..', 'models', 'feature_info.npy'), feature_info)
    joblib.dump(ct, os.path.join(base_dir, '..', 'models', 'column_transformer.pkl'))
    joblib.dump(label_encoder, os.path.join(base_dir, '..', 'models', 'label_encoder.pkl'))

# Function to compute class weights
def compute_class_weights(y_train: np.ndarray) -> dict:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        y_train (np.ndarray): Training target data.
    
    Returns:
        dict: Class weights.
    """
    logger.info("Computing class weights...")
    return dict(enumerate(class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)))

# Function to build the model
def build_model(hp: kt.HyperParameters, input_dim: int) -> tf.keras.Model:
    """
    Build a Keras model with hyperparameter tuning.
    
    Args:
        hp (kt.HyperParameters): Hyperparameters for tuning.
        input_dim (int): Input dimension.
    
    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(hp.Int('units_1', min_value=128, max_value=256, step=64), 
              activation=hp.Choice('activation_1', values=['relu', 'swish']),
              kernel_regularizer=l2(hp.Float('l2_1', min_value=0.001, max_value=0.01, step=0.001))),
        Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.6, step=0.1)),
        Dense(hp.Int('units_2', min_value=128, max_value=256, step=64), 
              activation=hp.Choice('activation_2', values=['relu', 'swish']),
              kernel_regularizer=l2(hp.Float('l2_2', min_value=0.001, max_value=0.01, step=0.001))),
        Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.6, step=0.1)),
        Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Define a learning rate schedule function
def lr_schedule(epoch, lr):
    if epoch < 10:
        return float(lr)  # Ensuring that the returned value is a plain float
    else:
        # Use exponential decay to reduce the learning rate
        new_lr = float(lr * tf.math.exp(-0.1))  # Convert Tensor to Python float
        return new_lr

# Function to perform hyperparameter tuning
def hyperparameter_tuning(tuner: kt.Tuner, X_train_processed: np.ndarray, y_train_processed: np.ndarray, class_weights: dict, tensorboard_callback: TensorBoard) -> None:
    """
    Perform hyperparameter tuning using the tuner.
    
    Args:
        tuner (kt.Tuner): Keras tuner object.
        X_train_processed (np.ndarray): Processed training features.
        y_train_processed (np.ndarray): Processed training target.
        class_weights (dict): Class weights.
        tensorboard_callback (TensorBoard): TensorBoard callback.
    """
    logger.info("Starting hyperparameter tuning...")
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    checkpoint_callback = ModelCheckpoint(os.path.join(base_dir, '..', 'models','model_checkpoint.keras'), save_best_only=True)
    tuner.search(X_train_processed, y_train_processed, epochs=20, batch_size=32, validation_split=0.2,
                 class_weight=class_weights, callbacks=[early_stopping, tensorboard_callback, checkpoint_callback, LearningRateScheduler(lr_schedule)])

# Function to evaluate the best model
def evaluate_best_model(best_model: tf.keras.Model, X_test_processed: np.ndarray, y_test_processed: np.ndarray) -> tuple:
    """
    Evaluate the best model on the test set.
    
    Args:
        best_model (tf.keras.Model): Best Keras model.
        X_test_processed (np.ndarray): Processed test features.
        y_test_processed (np.ndarray): Processed test target.
    
    Returns:
        tuple: Test loss and test accuracy.
    """
    logger.info("Evaluating the best model on the test set...")
    test_loss, test_accuracy = best_model.evaluate(X_test_processed, y_test_processed)
    logger.info(f"Best Model Test Accuracy: {test_accuracy:.2f}")
    return test_loss, test_accuracy

# Function to save the best model
def save_best_model(best_model: tf.keras.Model) -> None:
    """
    Save the best model to disk.
    
    Args:
        best_model (tf.keras.Model): Best Keras model.
    """
    model_save_path = os.path.join(base_dir, '..', 'models', 'best_model.keras')
    best_model.save(model_save_path)
    logger.info(f"Model saved at {model_save_path}")

# Main function to tie everything together
def main(delete_existing: bool = False) -> None:
    """
    Main function to execute the model training pipeline.
    
    Args:
        delete_existing (bool): Whether to delete existing logs and tuner directories.
    """
    if delete_existing:
        delete_dirs([os.path.join(base_dir, 'logs'), os.path.join(base_dir, 'tuner')])
    
    # Create directories
    log_dir = create_log_dirs()
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    tuner_dir = create_tuner_dirs()
    
    # Load and preprocess data
    data = load_data()
    X = data.drop(columns=['full_time_result'])
    y = data['full_time_result']
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    X_train_processed, y_train_processed, ct, label_encoder, feature_info = preprocess_data(X_train, y_train)
    X_test_processed = ct.transform(X_test)
    y_test_encoded = label_encoder.transform(y_test)
    y_test_processed = tf.keras.utils.to_categorical(y_test_encoded)
    
    # Save preprocessing objects
    save_preprocessing_objects(feature_info, ct, label_encoder)
    
    # Compute class weights
    class_weights = compute_class_weights(y_train)
    
    # Define the tuner
    tuner = kt.Hyperband(lambda hp: build_model(hp, X_train_processed.shape[1]),
                         objective='val_accuracy', max_epochs=20, factor=3, executions_per_trial=2,
                         directory=tuner_dir, project_name='dense_model_tuning')
    
    # Hyperparameter tuning
    hyperparameter_tuning(tuner, X_train_processed, y_train_processed, class_weights, tensorboard_callback)
    
    # Get best model
    best_model = tuner.get_best_models(num_models=1)[0]
    
    # Evaluate and save the best model
    evaluate_best_model(best_model, X_test_processed, y_test_processed)
    save_best_model(best_model)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train the model with optional deletion of existing logs and tuner directories.")
    parser.add_argument('--delete-existing', action='store_true', help="Delete existing logs and tuner directories before training.")
    args = parser.parse_args()

    main(delete_existing=args.delete_existing)
