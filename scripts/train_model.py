import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Print statement to confirm the script is running
print("Script is running...")

# Load the dataset
data = pd.read_csv(
    os.path.join('..', 'data', 'processed', 'all_concat_football_data.csv'), 
    parse_dates=['Date']
)

# Print statement to confirm the dataset is loaded
print("Dataset loaded successfully")

# Convert to datetime explicitly with format
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d', errors='coerce')

# drop missing values
data = data.dropna()


# Define the time window
time_window = 5  # Number of previous matches to consider

# Function to create sequences
def create_sequences(data, team_col, feature_cols, time_window):
    sequences = []
    for team in data[team_col].unique():
        team_data = data[data[team_col] == team].sort_values(by='Date')
        team_features = team_data[feature_cols].values
        for i in range(len(team_features) - time_window):
            sequences.append(team_features[i:i + time_window])
    return np.array(sequences)

# Features to include in sequences
feature_cols = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']

# Generate sequences
home_sequences = create_sequences(data, 'HomeTeam', feature_cols, time_window)
away_sequences = create_sequences(data, 'AwayTeam', feature_cols, time_window)

# Concatenate home and away sequences
X = np.concatenate([home_sequences, away_sequences], axis=2)

# Target variable
y = data['FTR']

# Normalize features
scaler = StandardScaler()
X_reshaped = X.reshape(-1, X.shape[-1])  # Flatten the time dimension
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(X.shape)

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)  # Converts to integers: H=0, D=1, A=2

# Print unique values of y to check encoding
print("Unique values in y after encoding:", np.unique(y))

# Ensure X and y have the same number of samples
num_samples = X.shape[0]
y = y[:num_samples]

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

# Build the LSTM model
model = Sequential([
    LSTM(64, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax'),  # 3 output classes: H, D, A
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")
