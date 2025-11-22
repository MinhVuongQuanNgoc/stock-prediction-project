"""
Task 6 — Ensemble Forecasting (FINAL CLEAN VERSION)
 - ARIMA uses SAME train/test split as LSTM/GRU/RF
 - No dummy files, no fake loads
 - Works with any real ticker
 - Supports ARIMA, LSTM, GRU, BiLSTM, RF
 - Optional weight search
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from data_processing_2 import load_stock_data
from machine_learning_1 import build_model, train_model
from typing import List, Dict, Any
import itertools
import json

# ------------------------------------------------------------
# Utility: Create DL Sequences
# ------------------------------------------------------------

def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, 3])  # Close column
    return np.array(X), np.array(y)


# ------------------------------------------------------------
# ARIMA
# ------------------------------------------------------------

def train_arima(train_close, order=(5,1,2)):
    model = ARIMA(train_close, order=order)
    model_fit = model.fit()
    return model_fit


def forecast_arima(model_fit, steps):
    return np.asarray(model_fit.forecast(steps=steps)).reshape(-1)


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

def train_random_forest(X_train_flat, y_train):
    rf = RandomForestRegressor(
        n_estimators=200, 
        max_depth=12, 
        random_state=42
    )
    rf.fit(X_train_flat, y_train)
    return rf


# ------------------------------------------------------------
# Deep Learning Model Configurations
# ------------------------------------------------------------

MODEL_CONFIGS = {
    "LSTM": [
        {'type': 'LSTM', 'units': 128, 'return_sequences': True, 'dropout': 0.2},
        {'type': 'LSTM', 'units': 64, 'return_sequences': False, 'dropout': 0.2},
    ],
    "GRU": [
        {'type': 'GRU', 'units': 128, 'return_sequences': True, 'dropout': 0.2},
        {'type': 'GRU', 'units': 64, 'return_sequences': False, 'dropout': 0.2},
    ],
    "BiLSTM": [
        {'type': 'Bidirectional(LSTM)', 'units': 128, 'return_sequences': True, 'dropout': 0.2},
        {'type': 'Bidirectional(LSTM)', 'units': 64, 'return_sequences': False, 'dropout': 0.2},
    ]
}


def build_and_train_dl(config_name, input_shape, X_train, y_train, X_val, y_val, epochs=75, batch_size=32):
    cfg = MODEL_CONFIGS.get(config_name)
    if cfg is None:
        raise ValueError(f"Unknown DL model config: {config_name}")

    model = build_model(cfg, input_shape=input_shape)
    history = train_model(model, X_train, y_train, X_val, y_val,
                          epochs=epochs, batch_size=batch_size)
    return model, history


# ------------------------------------------------------------
# Inverse-scaling helper (DL & RF)
# ------------------------------------------------------------

def inverse_scale_predictions(preds_scaled, scaler, close_idx, n_feats):
    inv = []
    for v in preds_scaled:
        dummy = np.zeros((1, n_feats))
        dummy[0, close_idx] = float(np.squeeze(v))
        inv_val = scaler.inverse_transform(dummy)[0, close_idx]
        inv.append(inv_val)
    return np.array(inv)


# ------------------------------------------------------------
# Weighted Ensemble
# ------------------------------------------------------------

def ensemble_weighted(preds_list, weights):
    stacked = np.vstack(preds_list)
    return np.tensordot(weights, stacked, axes=(0, 0))


# ------------------------------------------------------------
# Core Ensemble Function
# ------------------------------------------------------------

def run_ensemble(
    ticker="NVDA",
    seq_len=60,
    model_set=["ARIMA","LSTM"],
    dl_choice="LSTM",
    arima_order=(5,1,2),
    weight_search=False,
    output_dir="output_task6"
):

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # 1) LOAD DATA
    # --------------------------------------------------------

    df_raw, train_raw, test_raw, _ = load_stock_data(
        ticker=ticker,
        start_date="2020-01-01",
        end_date="2025-07-31",
        split_by_date=True,
        test_size=0.2,
        scale=False
    )

    df_scaled, train_scaled, test_scaled, scaler = load_stock_data(
        ticker=ticker,
        start_date="2020-01-01",
        end_date="2025-07-31",
        split_by_date=True,
        test_size=0.2,
        scale=True
    )

    # --------------------------------------------------------
    # 2) PREPARE SEQUENCES
    # --------------------------------------------------------

    X_train, y_train = create_sequences(train_scaled.values, seq_len)
    X_test,  y_test  = create_sequences(test_scaled.values, seq_len)

    # Ground truth (raw close)
    y_true = test_raw["Close"].iloc[seq_len:].values

    feature_idx_close = list(train_scaled.columns).index("Close")
    n_feats = train_scaled.shape[1]

    # --------------------------------------------------------
    # 3) ARIMA
    # --------------------------------------------------------

    preds_list = []
    names_list = []

    if "ARIMA" in model_set:
        raw_close_train = train_raw["Close"].values
        model_arima = train_arima(raw_close_train, order=arima_order)
        preds_arima = forecast_arima(model_arima, len(y_true))
        preds_list.append(preds_arima)
        names_list.append("ARIMA")

    # --------------------------------------------------------
    # 4) DEEP LEARNING (LSTM/GRU/BILSTM)
    # --------------------------------------------------------

    if any(m in model_set for m in ["LSTM","GRU","BiLSTM"]):

        input_shape = (X_train.shape[1], X_train.shape[2])
        dl_model, dl_hist = build_and_train_dl(
            dl_choice, input_shape,
            X_train, y_train,
            X_test,  y_test,
            epochs=50, batch_size=32
        )

        preds_dl_scaled = dl_model.predict(X_test).flatten()
        preds_dl = inverse_scale_predictions(
            preds_dl_scaled, scaler,
            feature_idx_close, n_feats
        )

        preds_list.append(preds_dl)
        names_list.append(dl_choice)

    # --------------------------------------------------------
    # 5) RANDOM FOREST
    # --------------------------------------------------------

    if "RF" in model_set:
        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        X_test_flat  = X_test.reshape((X_test.shape[0], -1))

        model_rf = train_random_forest(X_train_flat, y_train)
        preds_rf_scaled = model_rf.predict(X_test_flat)
        preds_rf = inverse_scale_predictions(
            preds_rf_scaled, scaler,
            feature_idx_close, n_feats
        )

        preds_list.append(preds_rf)
        names_list.append("RF")

    # --------------------------------------------------------
    # 6) WEIGHT SEARCH OR EQUAL WEIGHTS
    # --------------------------------------------------------

    summary_rows = []

    def evaluate(weights):
        pred = ensemble_weighted(preds_list, weights)
        mse = mean_squared_error(y_true, pred)
        return mse, pred

    if weight_search and (2 <= len(preds_list) <= 3):

        grid = np.linspace(0, 1, 11)
        best_mse = np.inf
        best_w = None

        if len(preds_list) == 2:
            for w in grid:
                weights = [w, 1-w]
                mse, _ = evaluate(weights)
                if mse < best_mse:
                    best_mse = mse
                    best_w = weights

        else:  # 3-model ensemble
            for w1 in grid:
                for w2 in grid:
                    if w1 + w2 > 1:
                        continue
                    w3 = 1 - w1 - w2
                    weights = [w1, w2, w3]
                    mse, _ = evaluate(weights)
                    if mse < best_mse:
                        best_mse = mse
                        best_w = weights

        weights_final = best_w
        mse_final, ensemble_pred = evaluate(best_w)

        summary_rows.append({
            "ensemble": "+".join(names_list),
            "weights": json.dumps(weights_final),
            "mse": mse_final,
            "type": "weight_search"
        })

    else:
        # Equal weighting fallback
        n = len(preds_list)
        weights_final = [1/n] * n
        mse_final, ensemble_pred = evaluate(weights_final)

        summary_rows.append({
            "ensemble": "+".join(names_list),
            "weights": json.dumps(weights_final),
            "mse": mse_final,
            "type": "equal_weight"
        })

    # --------------------------------------------------------
    # 7) SAVE SUMMARY
    # --------------------------------------------------------

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "ensemble_summary.csv"), index=False)

    # --------------------------------------------------------
    # 8) PLOT
    # --------------------------------------------------------

    plt.figure(figsize=(15,6))
    plt.plot(y_true, label="Actual", linewidth=2)

    for name, preds in zip(names_list, preds_list):
        plt.plot(preds, label=name)

    plt.plot(ensemble_pred, label="Ensemble", linewidth=3, linestyle='--', color='black')
    plt.legend()
    plt.title(f"Ensemble: {'+'.join(names_list)} | MSE={mse_final:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ensemble_prediction_plot.png"))
    plt.show()

    # --------------------------------------------------------
    # 9) RETURN OBJECT
    # --------------------------------------------------------

    return {
        "names": names_list,
        "preds_list": preds_list,
        "ensemble_pred": ensemble_pred,
        "weights": weights_final,
        "mse": mse_final,
        "y_true": y_true,
        "summary_df": summary_df
    }



# ------------------------------------------------------------
# EXAMPLE RUNS
# ------------------------------------------------------------

if __name__ == "__main__":

    # ARIMA + LSTM
    out1 = run_ensemble(
        ticker="NVDA",
        model_set=["ARIMA","LSTM"],
        dl_choice="LSTM",
        weight_search=False
    )

    # ARIMA + LSTM (weight search ON)
    out2 = run_ensemble(
        ticker="NVDA",
        model_set=["ARIMA","LSTM"],
        dl_choice="LSTM",
        weight_search=True
    )

    # ARIMA + RF + LSTM
    out3 = run_ensemble(
        ticker="NVDA",
        model_set=["ARIMA","RF","LSTM"],
        dl_choice="LSTM",
        weight_search=True
    )
