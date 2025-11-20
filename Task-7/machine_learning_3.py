"""
Task 6 — Ensemble Forecasting

This module implements an extensible ensemble system that combines:
- ARIMA / SARIMA
- Deep Learning models (LSTM, GRU, RNN, Bidirectional, CNN-LSTM)
- Random Forest Regressors
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from data_processing_2 import load_stock_data
from machine_learning_1 import build_model, train_model
from graph_plot import plot_predictions
import matplotlib.pyplot as plt

# Utility: Create DL Sequences
def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, 3])  # Close price
    return np.array(X), np.array(y)

# ARIMA Model Wrapper
def train_arima(train_close, order=(5, 1, 2)):
    print("ARIMA Training...")
    model = ARIMA(train_close, order=order)
    model_fit = model.fit()
    return model_fit


def forecast_arima(model_fit, steps):
    return model_fit.forecast(steps=steps)


# Random Forest Wrapper
def train_random_forest(X_train_flat, y_train):
    print("[RandomForest] Training...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )
    rf.fit(X_train_flat, y_train)
    return rf

# Ensemble Method (Weighted Averaging)
def ensemble_predictions(pred_arima, pred_dl, pred_rf=None):
    if pred_rf is None:
        return (pred_arima * 0.4) + (pred_dl * 0.6)   # Two-model ensemble

    return (pred_arima * 0.3) + (pred_dl * 0.5) + (pred_rf * 0.2)  # Three-model ensemble

def plot_ensemble_results(dates, actual, lstm_pred, arima_pred, ensemble_pred):
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual, label="Actual Close")
    plt.plot(dates, lstm_pred, label="LSTM Prediction")
    plt.plot(dates, arima_pred, label="ARIMA Prediction")
    plt.plot(dates, ensemble_pred, label="Ensemble Prediction", linewidth=3)

    plt.title("Ensemble Model Predictions")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Main Ensemble 
def run_ensemble(ticker="NVDA", seq_len=60, dl_config_id=2, arima_order=(5,1,2)):
    """
    Ensemble of:
       1. ARIMA
       2. Deep-Learning (LSTM/GRU/... from machine_learning_1)
       3. Random Forest (optional)
    """

    # Load data
    df, train_df, test_df, scaler = load_stock_data(
        ticker=ticker,
        start_date="2020-01-01",
        end_date="2025-07-31",
        split_by_date=True,
        test_size=0.2,
        scale=True
    )

    # Prepare DL sequences
    X_train, y_train = create_sequences(train_df.values, seq_len)
    X_test, y_test = create_sequences(test_df.values, seq_len)

    # (1) Train ARIMA
    close_train = train_df["Close"].values
    arima_model = train_arima(close_train, order=arima_order)

    pred_arima = forecast_arima(arima_model, len(y_test))
    pred_arima = np.array(pred_arima)

    # (2) Train Deep Learning Model
    input_shape = (X_train.shape[1], X_train.shape[2])
    from machine_learning_1 import TEST_CONFIGS

    dl_model = build_model(TEST_CONFIGS[dl_config_id], input_shape=input_shape)
    history = train_model(dl_model, X_train, y_train, X_test, y_test, epochs=75, batch_size=32)

    pred_dl = dl_model.predict(X_test).flatten()

    # (3) Train Random Forest
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    X_test_flat = X_test.reshape((X_test.shape[0], -1))

    rf_model = train_random_forest(X_train_flat, y_train)
    pred_rf = rf_model.predict(X_test_flat)

    # Perform Ensemble
    final_pred = ensemble_predictions(pred_arima, pred_dl, pred_rf)


    # Inverse-scale for interpretability
    dummy = np.zeros((len(final_pred), train_df.shape[1]))
    dummy[:, 3] = final_pred
    final_inv = scaler.inverse_transform(dummy)[:, 3]

    return final_inv, y_test, history, pred_arima, pred_dl, pred_rf


if __name__ == "__main__":
    final_pred, y_test, history, pred_arima, pred_dl, pred_rf = run_ensemble(
        ticker="NVDA",
        seq_len=60,
        dl_config_id=1,
        arima_order=(5,1,2)
    )
    

    print("Ensemble Prediction Completed.")
    print("Sample Predictions:", final_pred[:5])