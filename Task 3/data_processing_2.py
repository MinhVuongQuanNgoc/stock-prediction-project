"""
Target features:
Visualizing stock data using candlestick and boxplot charts.
Libray: matplotlib
Loaded data from data_processing_1.py
"""

import os
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from data_processing_1 import load_stock_data

# ...existing code...

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure columns are flat (no MultiIndex) and 'Price' column (if present)
    is converted to the datetime index. Return a cleaned DataFrame.
    """
    # Flatten MultiIndex columns (e.g. ('Close','AAPL') -> 'Close')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # If 'Price' column holds date strings, set it as index and drop the column
    if 'Price' in df.columns and df['Price'].dtype == object:
        try:
            df.index = pd.to_datetime(df['Price'])
            df = df.drop(columns=['Price'])
        except Exception:
            # If conversion fails, leave as-is (caller can handle)
            pass

    # Ensure index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    return df

def plot_candlestick(df: pd.DataFrame, ticker: str = "AAPL", n_days: int = 1, save=False):
    """
    Plot a candlestick chart for given stock data.
    """
    # Normalize columns and index up-front
    df = _normalize_df(df.copy())
    
    # Resample if >1 trading day per candle
    if n_days > 1:
        df = df.resample(f'{n_days}D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

    # Plot candlestick
    kwargs = {
        'type': 'candle',
        'style': 'charles',
        'title': f"{ticker} Candlestick Chart ({n_days}-Day)",
        'ylabel': 'Price (USD)',
        'volume': True,
        'mav': (5, 10, 20)
    }
    
    if save:
        kwargs['savefig'] = f"{ticker}_candlestick_{n_days}day.png"
    
    mpf.plot(df, **kwargs)

def plot_boxplot(df: pd.DataFrame, feature: str = "Close", n_days: int = 5, ticker: str = "AAPL", save=False):
    """
    Display a boxplot for n consecutive trading days.
    """
    df = _normalize_df(df.copy())

    if feature not in df.columns:
        raise ValueError(f"'{feature}' column not found in dataset.")

    # Ensure we pass a 1-D array/Series slices to boxplot
    series = df[feature]
    if isinstance(series, pd.DataFrame):
        # squeeze to Series if single-column DataFrame
        series = series.iloc[:, 0]

    # Group into chunks of n_days
    grouped = [series.iloc[i:i + n_days].values for i in range(0, len(series), n_days) if len(series.iloc[i:i + n_days]) > 0]

    plt.figure(figsize=(10, 6))
    plt.boxplot(grouped, patch_artist=True)
    plt.title(f"{ticker} {feature} Price Distribution (every {n_days} days)")
    plt.xlabel("Window Number")
    plt.ylabel("Price (USD)")
    if save:
        plt.savefig(f"{ticker}_boxplot_{feature}_{n_days}day.png")
    plt.show()

# ...existing code...
if __name__ == "__main__":
    # Example usage
    df, train, test, _ = load_stock_data(
        ticker="AAPL",
        start_date="2022-02-01",
        end_date="2024-08-31",
        scale=False
    )
    
    # Normalize once in main too
    df = _normalize_df(df)

    print("\nDataFrame Info after processing:")
    print(df.info())
    
    plot_candlestick(df, ticker="AAPL", n_days=3)
    #if 
    plot_boxplot(df, feature="Close", n_days=10)
