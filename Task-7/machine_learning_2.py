"""
Task 5 — Multivariate and Multistep Forecasting
Uses data preprocessing from Task-4 
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data_processing_2 import load_stock_data 
from machine_learning_1 import build_model, train_model, plot_metric


# Multi Sequence utils
def create_multistep_sequences(df: pd.DataFrame, feature_cols: list, seq_len: int, pred_steps: int, target_col: str = "Close"):
    arr = df[feature_cols].values
    tgt_idx = feature_cols.index(target_col)
    X, y = [], []
    for i in range(seq_len, len(arr) - pred_steps + 1):
        X.append(arr[i - seq_len:i, :])
        y.append(arr[i:i + pred_steps, tgt_idx])
    return np.array(X), np.array(y)

def inverse_transform_multistep(preds_scaled: np.ndarray, scaler, feature_idx: int, n_features: int):
    inv = []
    for seq in preds_scaled:
        seq_inv = []
        for val in seq:
            dummy = np.zeros((1, n_features))
            dummy[0, feature_idx] = val
            seq_inv.append(scaler.inverse_transform(dummy)[0, feature_idx])
        inv.append(seq_inv)
    return np.array(inv)

# single day prediction utility
def predict_single_day_multivariate(model, ticker: str,predict_date: str, feature_cols: list, seq_len: int = 60, target_col: str = "Close"):
        # Convert to datetime
    predict_dt = datetime.strptime(predict_date, "%Y-%m-%d")
    start_dt = predict_dt - timedelta(days=730)

    print(f"Preparing multivariate data from {start_dt.date()} -> {predict_dt.date()}")

    # Load abd scale using task4 preprocessing
    df, _, _, scaler = load_stock_data(
        ticker=ticker,
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=predict_dt.strftime("%Y-%m-%d"),
        split_by_date=False,
        scale=True
    )

    # Ensure all required features exist
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required feature '{col}' in data.")

    # Convert to scaled numpy array
    arr = df[feature_cols].values

    if len(arr) < seq_len:
        raise ValueError(f"Not enough data rows {len(arr)} for sequence length {seq_len}")

    # Create model input (last seq_len rows)
    X_input = arr[-seq_len:]
    X_input = np.expand_dims(X_input, axis=0)     # shape: (1, seq_len, n_features)

    # Predict scaled
    pred_scaled = model.predict(X_input)[0][0]    # model outputs 1 number

    # Inverse-scale only the target feature
    feature_idx = feature_cols.index(target_col)
    dummy = np.zeros((1, len(feature_cols)))
    dummy[0, feature_idx] = pred_scaled

    pred_inv = scaler.inverse_transform(dummy)[0, feature_idx]

    print(f"Predicted Close on {predict_date}: {pred_inv:.2f}")
    return pred_inv


# main (format inconsistancy because i worte this later)
def main(train_df, test_df, scaler, feature_cols, seq_len=60, pred_steps=30, target_col="Close"):
    X_train, y_train = create_multistep_sequences(train_df, feature_cols, seq_len, pred_steps, target_col)
    X_test, y_test = create_multistep_sequences(test_df, feature_cols, seq_len, pred_steps, target_col)

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(
        layer_configs=[
            {'type': 'LSTM', 'units': 128, 'return_sequences': True, 'dropout': 0.2},
            {'type': 'LSTM', 'units': 64, 'return_sequences': False, 'dropout': 0.2},
        ],
        input_shape=input_shape,
        output_units=pred_steps
    )
    hist = train_model(model, X_train, y_train, X_test, y_test, epochs=75, batch_size=32)

    preds_scaled = model.predict(X_test)
    preds_inv = inverse_transform_multistep(
        preds_scaled,
        scaler,
        feature_idx=feature_cols.index(target_col),
        n_features=len(feature_cols)
    )

    plot_metric(hist.history['loss'], hist.history['val_loss'], 'Train vs Validation Loss')
    return model, preds_inv, y_test


if __name__ == "__main__":
    print("Multivariate + Multistep Forecasting")

    # Load data
    df, train_df, test_df, scaler = load_stock_data(
        ticker="NVDA", #select company
        start_date="2020-01-01", #start date
        end_date="2025-7-31", #end date
        split_by_date=True,
        test_size=0.2,
        scale=True,
        pred_steps=10
    )

    feature_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    model, preds_inv, y_test = main(train_df, test_df, scaler, feature_cols)

    print(f"Produced {preds_inv.shape[0]} forecast sequences")