"""
Target features:
    Start, end date and ticker
    Automatically download from yfinance or load CSV if exists
    Deal with NaN values
    Split train/test (by date or randomly)
    Optional scaling
    Save dataset CSV locally, read existing if already downloaded
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
        try:
            df = pd.read_csv(local_path, index_col="Date", parse_dates=True)
        except Exception:
            # fallback if file structure is unexpected (because of multi-header csv. old logic seems fine with it for some reason)
            df = pd.read_csv(local_path, skiprows=[1, 2]) #no more NAN
            df.columns = ["Price", "Adj Close", "Close", "High", "Low", "Open", "Volume"]
            df.set_index(pd.to_datetime(df["Price"]), inplace=True)
            df.drop(columns=["Price"], inplace=True)
    else:
        print(f"Downloading {ticker} data")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        df.to_csv(local_path)

    # Cleaning empty rows
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


if __name__ == "__main__":
    #To run, change the parameters as needed
    df, train, test, scaler = load_stock_data(
        ticker="NVDA", #select company
        start_date="2020-01-01", #start date
        end_date="2025-7-31", #end date
        split_by_date=True, #whether to split the dataset into training/testing by date
        test_size=0.2, #ratio for test data
        scale=True #whether to scale the data or not
    )
    print(df.head())
