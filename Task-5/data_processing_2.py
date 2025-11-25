"""
Target features:
    Visualizing stock data using candlestick and boxplot charts.
    Libray: matplotlib
    Extention of data_processing_1.py
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import mplfinance as mpf
#import matplotlib as mpl


"""
data_1 functions start here
"""

#load selection
def load_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    split_by_date: bool = True,
    test_size: float = 0.2,
    scale: bool = True,
    cache: bool = True,
    data_dir: str = "data"
):

    # Ensure local data directory exists
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, f"{ticker}_{start_date}_{end_date}.csv")

    #Load or download data
    if cache and os.path.exists(local_path):
        print(f"Data already downloaded, load that instead: {local_path}")
        try:
            df = pd.read_csv(local_path, index_col="Date", parse_dates=True)
        except Exception:
            # fallback if file structure is unexpected (because of multi-header csv. Not sure why this isnt a problem on Data 1)
            df = pd.read_csv(local_path, skiprows=[1, 2]) #these two keep causing problems
            df.columns = ["Price", "Adj Close", "Close", "High", "Low", "Open", "Volume"]
            df.set_index(pd.to_datetime(df["Price"]), inplace=True)
            df.drop(columns=["Price"], inplace=True)
    else:
        print(f"Downloading {ticker} data")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        df.to_csv(local_path)

    # Cleaning NaN value
    df.dropna(inplace=True)

    #scaling
    feature_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    scaler = None
    if scale:
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Train/test split
    if split_by_date:
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
    else:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    print(f"Split -> Train {len(train_df)} rows  |  Test {len(test_df)} rows")

    # Save processed versions
    train_df.to_csv(os.path.join(data_dir, f"{ticker}_train.csv"))
    test_df.to_csv(os.path.join(data_dir, f"{ticker}_test.csv"))

    return df, train_df, test_df, scaler
"""
# data_1 functions end here
"""

"""
# data_2 candlestick
"""
def plot_candlestick_chart(df: pd.DataFrame, ticker: str, n_days: int = 1):
    # Make sure the index is in datetime format (required by mplfinance)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Resample data if user wants multiday candlesticks
    if n_days >= 1:
        df = df.resample(f'{n_days}D').agg({
            'Open': 'first', # First opening price in the period
            'High': 'max', # Highest price
            'Low': 'min',  # Lowest price
            'Close': 'last', # Closing price
            'Volume': 'sum' # Total trading volume
        })
        df.dropna(inplace=True)  # Remove incomplete periods

    # Plot
    # cm = 1/2.54  # centimeters to inches conversion
    mpf.plot(
        df,
        type='candle', # candlestick style
        style='yahoo',
        title=f"{ticker} Candlestick Chart ({n_days}-Day Candles)",
        volume=True, # Include volume subplot
        mav=(5, 10, 20), # 5, 10, 20 days moving averages
        figsize=(10, 6), # Figure size in inches (metrics need conversion line above)
        tight_layout=True # Avoid layout overlap
    )