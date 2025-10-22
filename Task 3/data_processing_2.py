"""
Target features:
Visualize stock data with matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

def plot_candlestick(df, n_days=1, title="Candlestick Chart"):
    """
    Show stock price data as a candlestick chart.
    Each candle can represent `n_days` combined (resampled).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DateTime index")

    # Resample if n_days > 1
    if n_days > 1:
        df = df.resample(f"{n_days}D").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna()

    mpf.plot(
        df,
        type="candle",
        style="yahoo",
        title=f"{title} (each candle={n_days} days)",
        ylabel="Price",
        volume=True,
        figratio=(12,6)
    )

def plot_boxplot(df, n_days=5, price_col="Close"):
    """
    Display boxplots for moving windows of `n_days`.
    Helps visualize price distribution and volatility.
    """
    if price_col not in df.columns:
        raise KeyError(f"{price_col} not found in DataFrame")

    series = df[price_col]
    windows = [series[i:i+n_days] for i in range(0, len(series), n_days)]
    plt.figure(figsize=(12,6))
    plt.boxplot(windows, patch_artist=True)
    plt.title(f"{price_col} Boxplot ({n_days}-day windows)")
    plt.xlabel("Window index")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    from data_processing_1 import load_stock_data
    df, train, test, _ = load_stock_data("AAPL", "2020-01-01", "2024-01-01")
    plot_candlestick(df, n_days=5)
    plot_boxplot(df, n_days=10)