import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data_processing_2 import load_stock_data
from google_trends import get_google_trends
from machine_learning_1 import build_model, train_model, plot_metric
from machine_learning_2 import inverse_transform_multistep
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def merge_trends(prices, trends):

    trends = trends.copy()
    trends.index = pd.to_datetime(trends.index)

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)

    merged = prices.merge(
        trends[['GoogleTrend']],
        left_index=True,
        right_index=True,
        how='left'
    )

    merged['GoogleTrend'] = merged['GoogleTrend'].interpolate(limit_direction="both")
    merged['GoogleTrend'] = merged['GoogleTrend'].fillna(method='bfill').fillna(method='ffill')
    merged['GoogleTrend'] = merged['GoogleTrend'].astype(float)

    return merged

def create_multivariate_sequences(df, feature_cols, seq_len):
    arr = df[feature_cols].values
    X, y = [], []
    for i in range(seq_len, len(arr)):
        X.append(arr[i-seq_len:i, :])
        y.append(arr[i, feature_cols.index("Close")])
    return np.array(X), np.array(y)


def build_trends_model(input_shape):
    return build_model(
        [
            {'type': 'LSTM', 'units': 128, 'return_sequences': True, 'dropout': 0.2},
            {'type': 'LSTM', 'units': 64, 'return_sequences': False, 'dropout': 0.2},
        ],
        input_shape=input_shape,
        output_units=1
    )


def train(ticker, start_date, end_date, seq_len=60):

    # LOAD RAW PRICE DATA 
    df, train_df, test_df, _ = load_stock_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        split_by_date=True,
        test_size=0.2,
        scale=False       
    )

    # 1. Get Google Trends
    trends = get_google_trends(ticker, start_date, end_date)

    # 2. Merge trends into price df
    df = merge_trends(df, trends)

    # 3. Re-split AFTER merging trends
    train_df = df.iloc[:len(train_df)]
    test_df = df.iloc[len(train_df):]

    # 4. Apply correct scaling
    feature_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "GoogleTrend"]
    price_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(train_df[price_cols])
    scaled_prices_test = scaler.transform(test_df[price_cols])

    # Rebuild scaled dfs
    train_scaled = pd.DataFrame(scaled_prices, columns=price_cols, index=train_df.index)
    test_scaled = pd.DataFrame(scaled_prices_test, columns=price_cols, index=test_df.index)

    # Add GoogleTrend WITHOUT scaling
    train_scaled["GoogleTrend"] = train_df["GoogleTrend"]
    test_scaled["GoogleTrend"] = test_df["GoogleTrend"]

    # 5. Create sequences
    X_train, y_train = create_multivariate_sequences(train_scaled, feature_cols, seq_len)
    X_test, y_test = create_multivariate_sequences(test_scaled, feature_cols, seq_len)

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_trends_model(input_shape)

    # 6. Train
    hist = train_model(model, X_train, y_train, X_test, y_test, epochs=75, batch_size=32)

    # 7. Predictions (inverse transform using ONLY price cols)
    pred_scaled = model.predict(X_test)
    pred_prices = []

    for val in pred_scaled:
        close_scaled = float(val.squeeze())  # <-- FIX
        dummy = np.zeros((1, len(price_cols)))
        dummy[0, price_cols.index("Close")] = close_scaled
        close_price = scaler.inverse_transform(dummy)[0, price_cols.index("Close")]
        pred_prices.append(close_price)

    pred_prices = np.array(pred_prices)

    # Plot Loss
    plot_metric(hist.history['loss'], hist.history['val_loss'], 'Train vs Validation loss')

    return model, pred_prices, y_test

def plot_model_comparison(dates, actual, predictions_dict):
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual, label="Actual", linewidth=3)

    for name, pred in predictions_dict.items():
        plt.plot(dates, pred, label=name)

    plt.title("Google Trend")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model, preds, actual = train(
        ticker="NVDA",
        start_date="2020-01-01",
        end_date="2025-07-31"
    )
    print("Training complete.")
    
    dates = actual.index[-len(preds):] if hasattr(actual, "index") else np.arange(len(preds))
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual, label="Actual", linewidth=3)
    plt.plot(dates, preds, label="LSTM")
    plt.title("Actual vs LSTM Prediction")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.bar(["GT-LSTM"], [np.mean((actual - preds) ** 2)])
    plt.title("MSE")
    plt.tight_layout()
    plt.show()

