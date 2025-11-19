"""
Task 6 - Machine Learning 3
Multivariate + Multistep LSTM forecasting using data from data_processing_2.py
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from data_processing_2 import load_stock_data
from graph_plot import plot_predictions


# -----------------------------------------------------------
# Build LSTM Model
# -----------------------------------------------------------

def build_lstm_model(input_shape, output_steps=1):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(output_steps)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# -----------------------------------------------------------
# Main Task 6 Pipeline
# -----------------------------------------------------------

def run_task6(
    ticker="AAPL",
    start_date="2018-02-01",
    end_date="2024-08-31",
    seq_len=20,
    pred_steps=5
):
    print("=== Task 6: Multivariate Time-Series Forecasting ===")

    # Load using your existing function
    df, train_df, test_df, scaler = load_stock_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        split_by_date=True,
        test_size=0.2,
        scale=True
    )

    # Your feature columns
    feature_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    target_col = "Close"

    scaled_values = train_df[feature_cols].values

    # -----------------------------
    # Create Sequences
    # -----------------------------
    X = []
    y = []

    for i in range(seq_len, len(scaled_values) - pred_steps):
        X.append(scaled_values[i - seq_len:i])
        y.append(scaled_values[i:i + pred_steps, feature_cols.index(target_col)])

    X = np.array(X)
    y = np.array(y)

    # -----------------------------
    # Train/Validation Split
    # -----------------------------
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # -----------------------------
    # Train Model
    # -----------------------------
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        output_steps=pred_steps
    )

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=[es],
        verbose=1
    )

    # -----------------------------
    # Make Predictions
    # -----------------------------
    predictions = model.predict(X_val)

    # Only the last predicted close price per sequence (optional)
    last_step_pred = predictions[:, -1]

    # Inverse scale prediction for CLOSE using dummy array
    dummy = np.zeros((len(last_step_pred), len(feature_cols)))
    dummy[:, feature_cols.index(target_col)] = last_step_pred
    inv_preds = scaler.inverse_transform(dummy)[:, feature_cols.index(target_col)]

    # Inverse scale actual for CLOSE
    X_val_last = X_val[:, -1, :]
    true_close = scaler.inverse_transform(X_val_last)[:, feature_cols.index(target_col)]

    # -----------------------------
    # Save Output
    # -----------------------------
    os.makedirs("output", exist_ok=True)

    result_df = pd.DataFrame({
        "True_Close": true_close,
        "Pred_Close": inv_preds
    })

    result_path = "output/task6_predictions.csv"
    result_df.to_csv(result_path, index=False)

    print(f"Predictions saved to {result_path}")

    # -----------------------------
    # Plot the results
    # -----------------------------
    plot_predictions(result_df["True_Close"], result_df["Pred_Close"],
                     title="Task 6 Forecast vs Actual")


# -----------------------------------------------------------
# Run if executed directly
# -----------------------------------------------------------

if __name__ == "__main__":
    run_task6()
