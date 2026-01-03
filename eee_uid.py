import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pyswarm import pso  # Particle Swarm Optimization
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Load and Preprocess Data
file_path = r"C:\Users\aparn\Downloads\EEEdataset_processed.csv" # Change this if needed
df = pd.read_csv(file_path)

# Ensure data is valid
if df.isnull().values.any():
    print("Warning: Missing values detected!")
    df = df.fillna(0)

# Select Features & Target
X = df.iloc[:, 5:-1].values  # Use all years except last
y = df.iloc[:, -1].values    # Predict last year's CO₂ emissions

# Normalize Data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# LSTM Model
X_train_LSTM = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_LSTM = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model_lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train_LSTM.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train_LSTM, y_train, epochs=30, batch_size=16, validation_data=(X_test_LSTM, y_test))

# Predict with LSTM
y_pred_lstm = model_lstm.predict(X_test_LSTM)
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm)

# MLPNN Model
model_mlpn = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

model_mlpn.compile(optimizer='adam', loss='mse')
model_mlpn.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Predict with MLPNN
y_pred_mlpn = model_mlpn.predict(X_test)
y_pred_mlpn = scaler_y.inverse_transform(y_pred_mlpn)

# MCOA (PSO Optimization)
def objective_function(weights):
    weights = np.array(weights).reshape(X_train.shape[1], 1)
    X_train_weighted = X_train * weights.T
    X_test_weighted = X_test * weights.T
    model_mlpn.fit(X_train_weighted, y_train, epochs=5, batch_size=16, verbose=0)
    y_pred = model_mlpn.predict(X_test_weighted)
    mse = np.mean((y_test - y_pred) ** 2)
    return mse

lb = [-1] * X_train.shape[1]
ub = [1] * X_train.shape[1]
best_weights, _ = pso(objective_function, lb, ub, swarmsize=10, maxiter=5)

X_train_best = X_train * best_weights
X_test_best = X_test * best_weights

model_mlpn.fit(X_train_best, y_train, epochs=30, batch_size=16, validation_data=(X_test_best, y_test))
y_pred_mcoa = model_mlpn.predict(X_test_best)
y_pred_mcoa = scaler_y.inverse_transform(y_pred_mcoa)

# Evaluation
mse_lstm = mean_squared_error(y_test, y_pred_lstm)
r2_lstm = r2_score(y_test, y_pred_lstm)

mse_mlpn = mean_squared_error(y_test, y_pred_mlpn)
r2_mlpn = r2_score(y_test, y_pred_mlpn)

mse_mcoa = mean_squared_error(y_test, y_pred_mcoa)
r2_mcoa = r2_score(y_test, y_pred_mcoa)

print(f"LSTM: MSE = {mse_lstm:.4f}, R² = {r2_lstm:.4f}")
print(f"MLPNN: MSE = {mse_mlpn:.4f}, R² = {r2_mlpn:.4f}")
print(f"MLPNN-MCOA: MSE = {mse_mcoa:.4f}, R² = {r2_mcoa:.4f}")

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual CO₂ Emissions", color="black")
plt.plot(y_pred_lstm, label="LSTM Predictions", color="blue")
plt.plot(y_pred_mlpn, label="MLPNN Predictions", color="green")
plt.plot(y_pred_mcoa, label="MLPNN-MCOA Predictions", color="red")
plt.xlabel("Test Samples")
plt.ylabel("CO₂ Emissions")
plt.legend()
plt.title("Comparison of Predictions")
plt.show()