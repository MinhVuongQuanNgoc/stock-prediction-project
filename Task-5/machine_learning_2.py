"""
Task 5: Multivariate and Multistep Forecasting
Produces:
    multistep forecasts (k steps)
    per-horizon MSE table
    prediction map for k days
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from data_processing_2 import load_stock_data
from machine_learning_1 import build_model, train_model, plot_metric
from sklearn.metrics import mean_squared_error
import mplfinance as mpf

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
    # scaler expects shape (n_samples, n_features)
    for seq in preds_scaled:
        seq_inv = []
        for val in seq:
            sample = np.zeros((1, n_features))
            sample[0, feature_idx] = float(np.squeeze(val))
            seq_inv.append(scaler.inverse_transform(sample)[0, feature_idx])
        inv.append(seq_inv)
    return np.array(inv)


def plot_multistep_errors(y_true_inv, y_pred_inv, pred_steps, show=True, save_path=None):
    # y_true_inv and y_pred_inv shapes: (n_sequences, pred_steps)
    n_steps = pred_steps
    mse_per_step = [mean_squared_error(y_true_inv[:, i], y_pred_inv[:, i]) for i in range(n_steps)]
    # bar plot
    plt.figure(figsize=(8,4))
    plt.bar(range(1, n_steps+1), mse_per_step)
    plt.xlabel("Prediction step")
    plt.ylabel("MSE")
    plt.title("Multistep MSE per horizon")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace(".png","_mse_bar.png"))
    if show:
        plt.show()
    return mse_per_step

def plot_predictions(test_df_raw, y_pred_inv, seq_len, pred_steps, show=True, save_path=None):

    # get the last historical window for candles
    hist_df = test_df_raw.iloc[-seq_len:].copy()
    hist_df.index = pd.to_datetime(hist_df.index)

    # future prediction dates (pred_steps days)
    last_hist_date = hist_df.index[-1]
    future_dates = []
    day = last_hist_date
    while len(future_dates) < pred_steps:
        day += timedelta(days=1)
        if day.weekday() < 5:  # skip weekends
            future_dates.append(day)

    # create the master index spanning historical and future periods
    master_index = hist_df.index.append(pd.DatetimeIndex(future_dates))

    # create the full prediction series (NaN for history, value for future)
    predicted_close = y_pred_inv[-1] # Get the last prediction sequence

    # Create the prediction line series
    full_pred_series = pd.Series(np.nan, index=master_index)
    full_pred_series.loc[future_dates] = predicted_close

    # create the extended plot DataFrame
    # Fill ohlc with NaN for the future period
    future_df = pd.DataFrame(np.nan, index=pd.DatetimeIndex(future_dates), columns=hist_df.columns)
    
    # Concatenate the historical data and the future placeholder data
    mpf_df_extended = pd.concat([hist_df, future_df])

    # Create the addplot
    addplot = mpf.make_addplot(full_pred_series, type="line", width=2, color='gold', label='Predicted Close')

    # Plot
    fig, axlist = mpf.plot(
        mpf_df_extended,
        type="candle",
        style="yahoo",
        volume=True,
        addplot=addplot,
        figsize=(14,6),
        returnfig=True,
        tight_layout=True
    )

    fig.suptitle(f"Multistep Forecast Overlay ({pred_steps} days)",fontsize=16,y=0.98)

    if save_path:
        plt.savefig(save_path.replace(".png"," prediction_plots.png"))

    if show:
        plt.show()
    
    plt.close(fig)

def main(train_df, test_df, scaler, feature_cols, seq_len=60, pred_steps=30, target_col="Close", output_dir="multi_output"):
    os.makedirs(output_dir, exist_ok=True)

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

    preds_scaled = model.predict(X_test)  # shape (n_sequences, pred_steps)
    # preds_scaled shape may be (n, pred_steps) already; ensure correct shape
    preds_scaled = np.array(preds_scaled)

    # inverse transform scaled sequences to price units
    n_features = scaler.n_features_in_ if hasattr(scaler, "n_features_in_") else len(feature_cols)
    feature_idx = feature_cols.index(target_col)
    preds_inv = inverse_transform_multistep(preds_scaled, scaler, feature_idx, n_features)

    # true values inverse transform 
    y_test_inv = inverse_transform_multistep(y_test, scaler, feature_idx, n_features)

    # compute per-step MSE
    mse_per_step = [mean_squared_error(y_test_inv[:, i], preds_inv[:, i]) for i in range(pred_steps)]

    # save results table
    results_df = pd.DataFrame({
        "step": list(range(1, pred_steps+1)),
        "mse": mse_per_step
    })
    results_df.to_csv(os.path.join(output_dir, "mse_per_step_table.csv"), index=False)

    # plots
    plot_multistep_errors(y_test_inv, preds_inv, pred_steps, show=True, save_path=os.path.join(output_dir, " prediction_plots.png"))
    plot_predictions(test_df_raw=test_df,y_pred_inv=preds_inv,seq_len=seq_len,pred_steps=pred_steps,show=True,save_path=os.path.join(output_dir, " plots.png")
    )

    plot_metric(hist.history['loss'], hist.history['val_loss'], 'Train vs Validation Loss')

    return model, preds_inv, y_test_inv, results_df


if __name__ == "__main__":
    print("Multivariate + Multistep Forecasting")

    df, train_df, test_df, scaler = load_stock_data(
        ticker="NVDA",
        start_date="2020-01-01",
        end_date="2025-07-31",
        split_by_date=True,
        test_size=0.2,
        scale=True
    )

    feature_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    model, preds_inv, y_test_inv, results_df = main(train_df, test_df, scaler, feature_cols, seq_len=120, pred_steps=10)

    print("Produced", preds_inv.shape[0], "forecast sequences")