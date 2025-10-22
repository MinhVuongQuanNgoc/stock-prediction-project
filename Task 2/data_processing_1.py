"""
Target features:
    Start, end date and ticker
    Automatically download from yfinance or load CSV if exists
    Handle NaN values
    Split train/test (by date or randomly)
    Optional scaling
    Save dataset CSV locally
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
        df = pd.read_csv(local_path, index_col=0)
    else:
        print(f"Downloading {ticker} data")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        df.to_csv(local_path)

    # Basic cleaning
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)

    #scaling
    feature_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    scaler = None
    if scale:
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        print("Scaled numeric feature columns.")

    # Train/test split
    if split_by_date:
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
    else:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    print(f"Split → Train {len(train_df)} rows  |  Test {len(test_df)} rows")

    # Save processed versions
    train_df.to_csv(os.path.join(data_dir, f"{ticker}_train.csv"))
    test_df.to_csv(os.path.join(data_dir, f"{ticker}_test.csv"))

    return df, train_df, test_df, scaler


if __name__ == "__main__":
    #Usage example here
    df, train, test, scaler = load_stock_data(
        ticker="AAPL", #select company
        start_date="2018-02-01", #start date
        end_date="2024-08-31", #end date
        split_by_date=True, #whether to split the dataset into training/testing by date
        test_size=0.2, #ratio for test data
        scale=True 
    )
    print(df.head())
