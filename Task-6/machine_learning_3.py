# machine_learning_3.py
"""
Task 6: Ensemble Forecasting
Featuring:
Auto-SARIMA (pmdarima) + DL (LSTM/GRU/BiLSTM) + RandomForest
Multivariate input (Open,High,Low,Close,Adj Close,Volume)
Multistep output (k days)
Ensemble weight grid-search
Saves summary CSV, per-horizon MSE, sample preds, and numpy arrays
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# pmdarima for auto-sarima
try:
    import pmdarima as pm
    PM_AVAILABLE = True
except Exception:
    PM_AVAILABLE = False

from data_processing_2 import load_stock_data
from machine_learning_1 import build_model, train_model, plot_metric


# Utilities for multistep sequences
def create_multistep_sequences(df: pd.DataFrame, feature_cols: List[str], seq_len: int, pred_steps: int, target_col: str="Close") -> Tuple[np.ndarray, np.ndarray]:
    arr = df[feature_cols].values
    tgt_idx = feature_cols.index(target_col)
    X, y = [], []
    for i in range(seq_len, len(arr) - pred_steps + 1):
        X.append(arr[i - seq_len:i, :])
        y.append(arr[i:i + pred_steps, tgt_idx])
    return np.array(X, dtype=float), np.array(y, dtype=float)

def inverse_transform_multistep(preds_scaled: np.ndarray, scaler: MinMaxScaler, feature_idx: int, n_features: int) -> np.ndarray:
    """
    preds_scaled: (samples, pred_steps) in scaled space
    return: (samples, pred_steps) in raw price space
    """
    samples, pred_steps = preds_scaled.shape
    flat = np.zeros((samples * pred_steps, n_features), dtype=float)
    for i in range(samples):
        for j in range(pred_steps):
            flat[i*pred_steps + j, feature_idx] = float(preds_scaled[i, j])
    inv = scaler.inverse_transform(flat)
    inv = inv[:, feature_idx].reshape(samples, pred_steps)
    return inv

# SARIMA wrapper
def train_sarima_auto(series: np.ndarray, max_p=3, max_q=3, max_P=2, max_Q=2, m=7, seasonal=True, maxiter=50, suppress_warnings=True):
    if not PM_AVAILABLE:
        print("SARIMA pmdarima not available.")
        return None
    try:
        model = pm.auto_arima(
            series,
            start_p=0, start_q=0,
            max_p=max_p, max_q=max_q,
            start_P=0, start_Q=0,
            max_P=max_P, max_Q=max_Q,
            seasonal=seasonal, m=m,
            stepwise=True, suppress_warnings=suppress_warnings,
            error_action='ignore', trace=False,
            maxiter=maxiter
        )
        return model
    except Exception as e:
        print(f"SARIMA auto_arima failed: {e}")
        return None

def forecast_sarima(model, steps: int):
    if model is None:
        return None
    try:
        f = model.predict(n_periods=steps)
        return np.asarray(f, dtype=float).reshape(-1)
    except Exception as e:
        print(f"SARIMA forecast failed: {e}")
        return None


# Random Forest multi-step helpers
def train_rf_multistep(X_train: np.ndarray, y_train: np.ndarray, n_estimators=150) -> List[RandomForestRegressor]:
    pred_steps = y_train.shape[1]
    flat_X = X_train.reshape((X_train.shape[0], -1))
    rf_models = []
    for step in range(pred_steps):
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        rf.fit(flat_X, y_train[:, step])
        rf_models.append(rf)
    return rf_models

def rf_predict_multistep(rf_models: List[RandomForestRegressor], X: np.ndarray) -> np.ndarray:
    flat = X.reshape((X.shape[0], -1))
    preds = [m.predict(flat) for m in rf_models]
    return np.vstack(preds).T  # (samples, pred_steps)


# DL builder/trainer wrapper (uses machine_learning_1.py)
MODEL_CONFIGS = {
    "LSTM": [
        {'type':'LSTM','units':128,'return_sequences':True,'dropout':0.2},
        {'type':'LSTM','units':64,'return_sequences':False,'dropout':0.2},
    ],
    "GRU": [
        {'type':'GRU','units':128,'return_sequences':True,'dropout':0.2},
        {'type':'GRU','units':64,'return_sequences':False,'dropout':0.2},
    ],
    "BiLSTM": [
        {'type':'Bidirectional(LSTM)','units':128,'return_sequences':True,'dropout':0.2},
        {'type':'Bidirectional(LSTM)','units':64,'return_sequences':False,'dropout':0.2},
    ]
}

def build_and_train_dl(config_key: str, input_shape: Tuple[int,int], output_steps: int, X_train, y_train, X_val, y_val, epochs=75, batch_size=32):
    if config_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown DL config: {config_key}")
    cfg = MODEL_CONFIGS[config_key]
    model = build_model(cfg, input_shape=input_shape, output_units=output_steps)
    hist = train_model(model, X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
    return model, hist


# Ensemble utilities-
def ensemble_weighted(preds_list: List[np.ndarray], weights: List[float]) -> np.ndarray:
    preds_list = [np.asarray(p, dtype=float) for p in preds_list]
    shapes = [p.shape for p in preds_list]
    if not all(s==shapes[0] for s in shapes):
        # align by trimming to minimum
        min_s0 = min(s[0] for s in shapes)
        min_s1 = min(s[1] for s in shapes)
        preds_list = [p[:min_s0, :min_s1] for p in preds_list]
    stacked = np.stack(preds_list, axis=0)
    weighted = np.tensordot(weights, stacked, axes=(0,0))
    return weighted


def run_ensemble(
    ticker: str = "AMZN",
    seq_len: int = 60,
    pred_steps: int = 10,
    model_set: List[str] = ["SARIMA","RF","LSTM"],
    dl_choice: str = "LSTM",
    weight_search: bool = False,
    test_size: float = 0.2,
    start_date: str = "2020-01-01",
    end_date: str = "2025-08-31",
    output_dir: str = "output_ensemble",
    dl_epochs: int = 75,
    dl_batch_size: int = 32,
    sarima_m: int = 7
) -> Dict[str, Any]:
    """
    Runs ensemble and returns structured output.
    model_set: any subset of ["SARIMA", "LSTM", "GRU", "BiLSTM", "RF"]
    """

    os.makedirs(output_dir, exist_ok=True)

    # Load data: raw (scale=False) and scaled (if scale=True)
    df_raw, train_raw, test_raw, _ = load_stock_data(
        ticker=ticker, start_date=start_date, end_date=end_date, split_by_date=True, test_size=test_size, scale=False
    )
    df_scaled, train_scaled, test_scaled, scaler = load_stock_data(
        ticker=ticker, start_date=start_date, end_date=end_date, split_by_date=True, test_size=test_size, scale=True
    )

    feature_cols = list(train_scaled.columns)
    n_features = len(feature_cols)
    close_idx = feature_cols.index("Close")

    # Create sequences
    X_train, y_train = create_multistep_sequences(train_scaled, feature_cols, seq_len, pred_steps)
    X_test,  y_test  = create_multistep_sequences(test_scaled, feature_cols, seq_len, pred_steps)

    if X_test.size == 0:
        raise ValueError("Not enough data to create test sequences with given seq_len/pred_steps.")

    # SARIMA: fit on raw_close_all with auto_arima (pmdarima). If fails, skip SARIMA
    sarima_preds = None
    if "SARIMA" in model_set:
        raw_close_all = pd.concat([train_raw["Close"], test_raw["Close"]], axis=0).values
        print("[SARIMA] fitting auto_arima (may take time)...")
        sarima_model = train_sarima_auto(raw_close_all, m=sarima_m)
        if sarima_model is None:
            print("[SARIMA] auto_arima failed or pmdarima missing - SARIMA disabled for this run.")
            sarima_preds = None
        else:
            all_preds = []
            # rolling predictions for each valid sequence start:
            total_seq = len(raw_close_all) - seq_len - pred_steps + 1
            for i in range(seq_len, seq_len + (len(raw_close_all) - seq_len - pred_steps + 1)):
                # Used the fitted model to forecast a sliding set starting at the end of train portion.
                break
            sarima_all = []
            raw = raw_close_all
            for i in range(seq_len, len(raw) - pred_steps + 1):
                hist = raw[:i]
                try:
                    # fit a small auto_arima on hist
                    m = pm.auto_arima(hist, start_p=0, start_q=0, max_p=3, max_q=3, seasonal=True, m=sarima_m,
                                      stepwise=True, suppress_warnings=True, error_action='ignore', maxiter=20)
                    f = m.predict(n_periods=pred_steps)
                    sarima_all.append(np.asarray(f, dtype=float))
                except Exception:
                    sarima_all.append(np.asarray([hist[-1]] * pred_steps, dtype=float))
            sarima_all = np.array(sarima_all, dtype=float)
            n_train_seq = X_train.shape[0]
            if sarima_all.shape[0] <= n_train_seq:
                print("[SARIMA] Not enough rolling preds produced; disabling SARIMA.")
                sarima_preds = None
            else:
                sarima_preds = sarima_all[n_train_seq:]
                print(f"[SARIMA] produced {sarima_preds.shape[0]} test sequence forecasts.")

    # DL model
    dl_preds_scaled = None
    dl_history = None
    if any(m in model_set for m in ["LSTM","GRU","BiLSTM"]):
        if dl_choice not in MODEL_CONFIGS:
            raise ValueError("dl_choice must be one of: " + ", ".join(MODEL_CONFIGS.keys()))
        input_shape = (X_train.shape[1], X_train.shape[2])
        dl_model, dl_history = build_and_train_dl(dl_choice, input_shape, pred_steps, X_train, y_train, X_test, y_test, epochs=dl_epochs, batch_size=dl_batch_size)
        dl_preds_scaled = dl_model.predict(X_test)  # (n_test_seq, pred_steps)

    # Random Forest
    rf_preds_scaled = None
    if "RF" in model_set:
        rf_models = train_rf_multistep(X_train, y_train, n_estimators=150)
        rf_preds_scaled = rf_predict_multistep(rf_models, X_test)

    # Prepare components in raw price
    preds_components = []
    names = []
    if sarima_preds is not None:
        preds_components.append(np.asarray(sarima_preds, dtype=float))
        names.append("SARIMA")
    if dl_preds_scaled is not None:
        preds_dl_inv = inverse_transform_multistep(np.squeeze(dl_preds_scaled), scaler, close_idx, n_features)
        preds_components.append(np.asarray(preds_dl_inv, dtype=float))
        names.append(dl_choice)
    if rf_preds_scaled is not None:
        preds_rf_inv = inverse_transform_multistep(rf_preds_scaled, scaler, close_idx, n_features)
        preds_components.append(np.asarray(preds_rf_inv, dtype=float))
        names.append("RF")

    if len(preds_components) == 0:
        raise ValueError("No component predictions available. Choose different models or check data.")

    # Ground truth raw for test sequences
    raw_arr = test_raw[["Open","High","Low","Close","Adj Close","Volume"]].values
    y_true_raw = []
    for i in range(seq_len, len(raw_arr) - pred_steps + 1):
        y_true_raw.append(raw_arr[i:i+pred_steps, 3])
    y_true_raw = np.array(y_true_raw, dtype=float)

    # Align shapes to minimum
    all_shapes = [p.shape for p in preds_components] + [y_true_raw.shape]
    min_samples = min(s[0] for s in all_shapes)
    min_steps = min(s[1] for s in all_shapes)
    preds_components = [p[:min_samples, :min_steps] for p in preds_components]
    y_true_raw = y_true_raw[:min_samples, :min_steps]

    # Weight search or equal weight
    n_models = len(preds_components)
    def compute_metrics(ensemble_pred, y_true):
        per = [float(mean_squared_error(y_true[:,h], ensemble_pred[:,h])) for h in range(y_true.shape[1])]
        return {"mse_per_horizon": per, "avg_mse": float(np.mean(per))}

    final_ensemble = None
    best_config = None
    results_summary = []
    if weight_search and 2 <= n_models <= 3:
        grid = np.linspace(0.0, 1.0, 11)
        best_mse = np.inf; best_w = None; best_pred = None
        if n_models == 2:
            for w in grid:
                weights = [w, 1.0-w]
                ens = ensemble_weighted(preds_components, weights)
                m = compute_metrics(ens, y_true_raw)
                if m["avg_mse"] < best_mse:
                    best_mse = m["avg_mse"]; best_w = weights; best_pred = ens
        else:
            for w1 in grid:
                for w2 in grid:
                    if w1 + w2 > 1.0: continue
                    w3 = 1.0 - w1 - w2
                    weights = [w1, w2, w3]
                    ens = ensemble_weighted(preds_components, weights)
                    m = compute_metrics(ens, y_true_raw)
                    if m["avg_mse"] < best_mse:
                        best_mse = m["avg_mse"]; best_w = weights; best_pred = ens
        if best_pred is None:
            # fallback to equal weights
            weights = [1.0/n_models]*n_models
            final_ensemble = ensemble_weighted(preds_components, weights)
            best_config = {"weights": weights, "avg_mse": compute_metrics(final_ensemble,y_true_raw)["avg_mse"]}
            results_summary.append({"ensemble":"+".join(names),"weights":json.dumps(weights),"avg_mse":best_config["avg_mse"],"type":"fallback_equal"})
        else:
            final_ensemble = best_pred
            best_config = {"weights": best_w, "avg_mse": best_mse}
            results_summary.append({"ensemble":"+".join(names),"weights":json.dumps(best_w),"avg_mse":best_mse,"type":"weight_search"})
    else:
        weights = [1.0/n_models]*n_models
        final_ensemble = ensemble_weighted(preds_components, weights)
        best_config = {"weights": weights, "avg_mse": compute_metrics(final_ensemble,y_true_raw)["avg_mse"]}
        results_summary.append({"ensemble":"+".join(names),"weights":json.dumps(weights),"avg_mse":best_config["avg_mse"],"type":"equal_weight"})

    # Save outputs
    pd.DataFrame(results_summary).to_csv(os.path.join(output_dir, "ensemble_experiments_summary.csv"), index=False)
    mse_per_horizon = [float(mean_squared_error(y_true_raw[:,h], final_ensemble[:,h])) for h in range(y_true_raw.shape[1])]
    pd.DataFrame({"horizon": list(range(1, y_true_raw.shape[1]+1)), "mse": mse_per_horizon}).to_csv(os.path.join(output_dir, "ensemble_per_horizon_mse.csv"), index=False)

    # Save arrays
    np.save(os.path.join(output_dir, "y_true_raw.npy"), y_true_raw)
    np.save(os.path.join(output_dir, "final_ensemble.npy"), final_ensemble)
    for nm, comp in zip(names, preds_components):
        np.save(os.path.join(output_dir, f"preds_{nm}.npy"), comp)

    # Save sample CSV
    out_df = pd.DataFrame({
        "actual_t+1": y_true_raw[:,0],
        **{f"{name}_t+1": preds[:,0] for name, preds in zip(names, preds_components)},
        "ensemble_t+1": final_ensemble[:,0]
    })
    out_df.to_csv(os.path.join(output_dir, "ensemble_predictions_sample.csv"), index=False)

    # Plot loss if DL trained
    if dl_history is not None:
        try:
            plot_metric(dl_history.history['loss'], dl_history.history['val_loss'], f"{ticker} - {dl_choice} loss")
        except Exception:
            pass

    # Horizon-1 line plot for quick visual
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    plt.plot(y_true_raw[:,0], label="Actual (t+1)", linewidth=2)
    for nm, comp in zip(names, preds_components):
        plt.plot(comp[:,0], label=f"{nm} (t+1)")
    plt.plot(final_ensemble[:,0], label="Ensemble (t+1)", linestyle="--", color="k", linewidth=2)
    plt.legend()
    plt.title(f"{ticker} Ensemble (t+1) avg_mse={best_config['avg_mse']:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ensemble_horizon1_plot.png"))
    plt.show()

    return {
        "names": names,
        "preds_components": preds_components,
        "final_ensemble": final_ensemble,
        "y_true_raw": y_true_raw,
        "weights": best_config["weights"],
        "avg_mse": best_config["avg_mse"],
        "mse_per_horizon": mse_per_horizon,
        "dl_history": dl_history,
        "output_dir": output_dir
    }

# CLI
if __name__ == "__main__":
    out = run_ensemble(
        ticker="AMZN",
        seq_len=60,
        pred_steps=10,
        model_set=["SARIMA","RF","LSTM"],
        dl_choice="LSTM",
        weight_search=True,
        output_dir="output_ensemble", #specify output directory
        dl_epochs=75,
        dl_batch_size=32
    )
    print("Results saved to:", out["output_dir"])