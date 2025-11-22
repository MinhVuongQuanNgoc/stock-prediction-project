import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
# Import the necessary functions from your existing modules
from data_processing_2 import load_stock_data 
from machine_learning_1 import build_model, train_model, plot_metric # Assuming build_model and train_model are here
from machine_learning_2 import main as ml2_main
# Assuming your google_trends.py is accessible for import
from google_trends import get_google_trends 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import mplfinance as mpf
# Note: You need to ensure machine_learning_1.py and data_processing_2.py are in the same folder, 
# and the build_model/train_model functions are correctly structured for import.

# --- Data Processing and Merging ---

def merge_trends(prices, trends):
  """Merges price data (df_raw) with Google Trends data."""
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

  # Fill NaNs (due to price data being daily and trend data often being less frequent)
  merged['GoogleTrend'] = merged['GoogleTrend'].interpolate(limit_direction="both")
  merged['GoogleTrend'] = merged['GoogleTrend'].fillna(method='bfill').fillna(method='ffill')
  merged['GoogleTrend'] = merged['GoogleTrend'].astype(float)

  return merged

def create_multivariate_sequences(df, feature_cols, seq_len):
  """Creates sequences for multivariate input."""
  arr = df[feature_cols].values
  X, y = [], []
  # y is the 'Close' price of the next day/step
  for i in range(seq_len, len(arr)):
    X.append(arr[i-seq_len:i, :])
    y.append(arr[i, feature_cols.index("Close")])
  return np.array(X), np.array(y)


def build_trends_model(input_shape):
  """Defines the LSTM model structure using the imported build_model."""
  return build_model(
    [
      {'type': 'LSTM', 'units': 128, 'return_sequences': True, 'dropout': 0.2},
      {'type': 'LSTM', 'units': 64, 'return_sequences': False, 'dropout': 0.2},
    ],
    input_shape=input_shape,
    output_units=1
  )

# --- Plotting Function (FIXED mplfinance issue) ---

def plot_candlestick_predicted_with_trends(test_df_with_trends: pd.DataFrame, predicted_prices: np.ndarray, n: int = 4, seq_len: int = 60):
    """
    Overlay predicted closing prices and Google Trend on the main candlestick chart (panel=0).
    FIXED (Task-7):
       - Prevent Google Trend secondary axis from overlapping the title
       - Clamp secondary axis to safe vertical bounds
       - Add top padding to title area
    """
    
    # 1. Align the data for plotting
    df = test_df_with_trends.iloc[seq_len:].copy()
        
    if len(predicted_prices) != len(df):
        print(f"Warning: Prediction length ({len(predicted_prices)}) does not match plot data length ({len(df)}).")
        return
        
    df['Predicted Close'] = predicted_prices
    df.index = pd.to_datetime(df.index)

    # 2. Resample (Aggregate) Data
    resample_agg = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'Predicted Close': 'last',
        'GoogleTrend': 'mean'
    }
    
    if n > 1:
        df_resampled = df.resample(f'{n}D').agg(resample_agg).dropna()
    else:
        df_resampled = df.dropna()

    # 3. Create Additional Plots (AddPlots)
    apds = [
        mpf.make_addplot(df_resampled['Predicted Close'], color='blue', panel=0, width=1.5, 
                         type='line', secondary_y=False),

        mpf.make_addplot(df_resampled['GoogleTrend'], color='purple', panel=0, 
                         type='line', secondary_y=True)
    ]
    
    # 4. Plot the Chart
    fig, axes = mpf.plot(
        df_resampled,
        type='candle',
        style='yahoo',
        title="Prediction with Google Trend Overlay ({}-Day Candlesticks)".format(n),
        volume=True,
        mav=(5, 10, 20),
        addplot=apds,
        panel_ratios=(6, 2),
        figscale=1.5,
        tight_layout=True,
        returnfig=True 
    )

    ax_price = axes[0]                 # Main candlestick axis
    ax_volume = axes[1]                # Volume axis
    ax_price = axes[0]
    ax_volume = axes[1]

    # --- UNIVERSAL SECONDARY AXIS DETECTION ---
    # mpf always places the secondary y-axis as the LAST axis created
    ax_google = axes[-1]

    # In some versions, volume panel is created last → adjust
    if ax_google is ax_volume:
        ax_google = axes[-2]     # Secondary axis created by mpf

    # -------------------------------------------------------------------
    # ⭐⭐⭐ TASK-7 FIX: Prevent Google Trend Secondary Axis From Overlapping Title ⭐⭐⭐
    # -------------------------------------------------------------------

    # 1) Get safe y-limits for price
    p_min, p_max = ax_price.get_ylim()
    price_range = p_max - p_min

    # 2) Clamp Google Trend axis to stay *within* price axis bounds
    g_min = df_resampled["GoogleTrend"].min()
    g_max = df_resampled["GoogleTrend"].max()

    # Scale trend into safe region
    scale_factor = price_range * 0.35     # Trend will occupy bottom 35% of chart
    g_scaled_low = p_min + price_range * 0.02
    g_scaled_high = g_scaled_low + scale_factor

    ax_google.set_ylim(g_scaled_low, g_scaled_high)

    # 3) Add padding to avoid title collision
    ax_price.figure.subplots_adjust(top=0.92)

    # -------------------------------------------------------------------
    # ⭐⭐⭐ END OF FIX ⭐⭐⭐
    # -------------------------------------------------------------------

    # 5. Manual Legend
    proxy_predicted = plt.Line2D([0], [0], color='blue', linewidth=1.5, label='Predicted Close')
    proxy_trend = plt.Line2D([0], [0], color='purple', linewidth=1.5, label='Google Trend')

    current_handles, current_labels = ax_price.get_legend_handles_labels()

    final_handles = current_handles + [proxy_predicted, proxy_trend]
    final_labels = current_labels + ['Predicted Close', 'Google Trend']

    ax_price.legend(final_handles, final_labels, loc='upper left')

    plt.show()


# --- Main Training Function (FIXED Scaling/Inverse Transform) ---

def train(ticker, start_date, end_date, seq_len=60):
  
  # 1. LOAD RAW PRICE DATA (scale=False is critical)
  df_raw, train_df_raw, test_df_raw, _ = load_stock_data(
    ticker=ticker, start_date=start_date, end_date=end_date, 
    split_by_date=True, test_size=0.2, scale=False
  )

  # Get Google Trends & Merge
  trends = get_google_trends(ticker, start_date, end_date)
  df_merged = merge_trends(df_raw, trends)

  # 3. Re-split AFTER merging trends
  train_df = df_merged.iloc[:len(train_df_raw)]
  test_df = df_merged.iloc[len(train_df_raw):] # The unscaled DF needed for plotting

  # 4. Apply correct scaling (Scale ALL 7 features together)
  feature_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "GoogleTrend"]
  
  scaler = MinMaxScaler() 
  
  # Fit on training data (all features), transform train and test
  train_scaled = pd.DataFrame(scaler.fit_transform(train_df[feature_cols]),
                columns=feature_cols, index=train_df.index)
  
  test_scaled = pd.DataFrame(scaler.transform(test_df[feature_cols]),
               columns=feature_cols, index=test_df.index)

  # 5. Create sequences
  X_train, y_train = create_multivariate_sequences(train_scaled, feature_cols, seq_len)
  X_test, y_test = create_multivariate_sequences(test_scaled, feature_cols, seq_len)
  
  # 6. Train model
  input_shape = (X_train.shape[1], X_train.shape[2])
  model = build_trends_model(input_shape)
  hist = train_model(model, X_train, y_train, X_test, y_test, epochs=75, batch_size=32)

  # 7. Predictions (Correct Inverse-Transform)
  pred_scaled = model.predict(X_test)
  pred_prices = []
  
  close_idx_in_full_features = feature_cols.index("Close") 
  
  for val in pred_scaled:
    close_scaled = float(val.squeeze()) 
    # Dummy array must have 7 columns (all features)
    dummy = np.zeros((1, len(feature_cols))) 
    dummy[0, close_idx_in_full_features] = close_scaled
    # Inverse transform the full dummy array and pull out the 'Close' value
    close_price = scaler.inverse_transform(dummy)[0, close_idx_in_full_features]
    pred_prices.append(close_price)

  pred_prices = np.array(pred_prices)

  # Plot Loss
  plot_metric(hist.history['loss'], hist.history['val_loss'], 'Train vs Validation loss')

  # Align the actuals for the metric plot
  actual_unscaled = test_df["Close"].iloc[seq_len:] 
  
  return model, pred_prices, actual_unscaled, test_df


if __name__ == "__main__":
  # Define seq_len for use in the plotting function
  SEQ_LEN = 60
  
  # Unpack the 4 returned values
  model, preds, actual_unscaled, test_df_with_trends = train( 
    ticker="NVDA",
    start_date="2020-01-01",
    end_date="2025-07-31",
    seq_len=SEQ_LEN
  )
  print("Training complete.")
  
  dates = actual_unscaled.index
  
  # --- 1. Line Plot Actual vs Prediction ---
  plt.figure(figsize=(14, 7))
  plt.plot(dates, actual_unscaled.values, label="Actual", linewidth=3)
  plt.plot(dates, preds, label="GT-LSTM Prediction")
  plt.title("Actual vs Google Trend-Augmented LSTM Prediction (Line Plot)")
  plt.xlabel("Date")
  plt.ylabel("Price")
  plt.legend()
  plt.tight_layout()
  plt.show()

  # --- 2. MSE Bar Plot ---
  mse_value = np.mean((actual_unscaled.values - preds) ** 2)
  
  plt.figure(figsize=(8, 5))
  plt.bar(["GT-LSTM"], [mse_value])
  plt.title("Mean Squared Error (MSE)")
  plt.tight_layout()
  plt.show()

  # --- 3. Candlestick/Trend Overlay Plot (The final visualization) ---
  plot_candlestick_predicted_with_trends(
    test_df_with_trends, 
    preds, 
    n=4, 
    seq_len=SEQ_LEN
  )