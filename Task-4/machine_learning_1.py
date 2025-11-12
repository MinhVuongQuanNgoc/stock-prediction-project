"""
machine_learning_1.py
Task-4: loads preprocessed CSVs, builds dynamic RNN model (LSTM/GRU/RNN),
trains, predicts and plots metrics.

Place this script in Task-4/ and ensure Task-4/data contains:
  - AAPL_train.csv
  - AAPL_test.csv
  - (optionally) AAPL_... main csv

This script is defensive: it autodetects feature/target columns and prints helpful diagnostics.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, Bidirectional, Input, Activation
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------
# Config
# ---------------------------
DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "AAPL_train.csv")
TEST_FILE  = os.path.join(DATA_DIR, "AAPL_test.csv")


STEP_SIZE = 30          # look-back window length (timesteps)
BATCH_SIZE = 32
EPOCHS = 30             # keep small for quick runs; increase for real experiments
PATIENCE = 6

# Default expected features; we'll intersect with DataFrame columns present
EXPECTED_FEATURES = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'RSI', 'EMAF', 'EMAM', 'EMAS']
# Preferred target column names (in priority order)
PREFERRED_TARGETS = ['TargetNextClose', 'Target', 'Adj Close', 'Close']

# Example dynamic layer configuration (you can swap with other tests from the notebook)
LAYER_CONFIG = [
    { 'type': 'LSTM', 'units': 200, 'return_sequences': True, 'dropout': 0.2 },
    { 'type': 'LSTM', 'units': 150, 'return_sequences': True, 'dropout': 0.2 },
    { 'type': 'LSTM', 'units': 100, 'return_sequences': False, 'dropout': 0.2 }
]
# ---------------------------


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Make sure CSVs from Task-2 exist in {DATA_DIR}")

    # Try normal load
    df = pd.read_csv(path)

    # 🧩 If no numeric columns — likely wrong header from metadata rows
    if df.select_dtypes(include=[np.number]).shape[1] == 0:
        print(f"⚠️ {os.path.basename(path)} has no numeric columns — retrying with skiprows=[1, 2]")
        df = pd.read_csv(path, skiprows=[1, 2])

    # ✅ Try to fix index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.set_index('Date', inplace=True)
    else:
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    # ✅ Drop completely empty or non-numeric columns
    for col in df.columns:
        if df[col].dtype == object and not pd.to_numeric(df[col], errors='coerce').notna().any():
            df.drop(columns=col, inplace=True)

    print(f"✅ Loaded {os.path.basename(path)} | Shape: {df.shape} | Columns: {df.columns.tolist()}")
    return df


def choose_features_and_target(train_df):
    # Find feature columns intersection
    available = list(train_df.columns)
    features = [c for c in EXPECTED_FEATURES if c in available]
    if len(features) == 0:
        # fallback: use any numeric columns except obvious non-feature columns
        features = train_df.select_dtypes(include=[np.number]).columns.tolist()
    # pick target
    target = None
    for t in PREFERRED_TARGETS:
        if t in available:
            target = t
            break
    if target is None:
        # fallback to last numeric column
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            raise RuntimeError("No numeric columns found to use as target/feature.")
        target = numeric_cols[-1]
    # ensure target not in features (we will allow target to stay in df but exclude from X)
    features = [f for f in features if f != target]
    return features, target


def scale_train_test(train_df, test_df, feature_cols, target_col):
    # Fit scalers on train only
    feat_scaler = MinMaxScaler()
    targ_scaler = MinMaxScaler()

    X_train_raw = train_df[feature_cols].values.astype(float)
    y_train_raw = train_df[[target_col]].values.astype(float)

    X_test_raw = test_df[feature_cols].values.astype(float)
    y_test_raw = test_df[[target_col]].values.astype(float)

    X_train_scaled = feat_scaler.fit_transform(X_train_raw)
    X_test_scaled = feat_scaler.transform(X_test_raw)

    y_train_scaled = targ_scaler.fit_transform(y_train_raw)
    y_test_scaled = targ_scaler.transform(y_test_raw)

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, feat_scaler, targ_scaler


def create_sequences(X_scaled, y_scaled, step_size):
    Xs, ys = [], []
    total = len(X_scaled)
    for i in range(step_size, total):
        Xs.append(X_scaled[i-step_size:i])
        ys.append(y_scaled[i])          # single-step forecast (next timestep target)
    return np.array(Xs), np.array(ys)


def create_dynamic_model(input_shape, layer_configs, output_units=1):
    """
    Build model using explicit Input so model is built immediately.
    layer_configs is list of dicts: {'type': 'LSTM'|'GRU'|'RNN'|'Bidirectional(LSTM)', 'units':int, 'return_sequences':bool, 'dropout':float}
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    for i, cfg in enumerate(layer_configs):
        layer_type = cfg['type']
        units = cfg.get('units', 64)
        ret_seq = cfg.get('return_sequences', False)

        # Support "Bidirectional(...)" syntax from notebook
        if layer_type.startswith('Bidirectional(') and layer_type.endswith(')'):
            inner = layer_type.replace('Bidirectional(', '').replace(')', '')
            if inner == 'LSTM':
                r = LSTM(units, return_sequences=ret_seq)
            elif inner == 'GRU':
                r = GRU(units, return_sequences=ret_seq)
            elif inner in ('RNN', 'SimpleRNN'):
                r = SimpleRNN(units, return_sequences=ret_seq)
            else:
                raise ValueError(f"Unsupported inner type: {inner}")
            from tensorflow.keras.layers import Bidirectional as B
            model.add(B(r))
        else:
            if layer_type.upper() in ('LSTM',):
                model.add(LSTM(units, return_sequences=ret_seq))
            elif layer_type.upper() in ('GRU',):
                model.add(GRU(units, return_sequences=ret_seq))
            elif layer_type.upper() in ('RNN', 'SIMPLERNN'):
                model.add(SimpleRNN(units, return_sequences=ret_seq))
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        if cfg.get('dropout', 0) and cfg.get('dropout', 0) > 0:
            model.add(Dropout(cfg['dropout']))

        if 'activation' in cfg:
            model.add(Activation(cfg['activation']))

    # final dense output
    model.add(Dense(output_units))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def plot_training(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='train loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


def main():
    print("Loading CSVs from", DATA_DIR)
    train_df = load_csv(TRAIN_FILE)
    test_df  = load_csv(TEST_FILE)
    
    # ✅ Drop non-numeric and unwanted columns
    drop_cols = ['Ticker', 'Symbol', 'Date', 'Unnamed: 0']
    for col in drop_cols:
        if col in train_df.columns:
            print(f"Dropping {col} from train_df")
            train_df.drop(columns=col, inplace=True)
        if col in test_df.columns:
            print(f"Dropping {col} from test_df")
            test_df.drop(columns=col, inplace=True)

    print("Train shape:", train_df.shape, "Test shape:", test_df.shape)

    features, target = choose_features_and_target(train_df)
    print(train_df.head(3))
    print("Using features:", features)
    print("Using target:", target)

    # Remove NaNs and sync indices (drop rows with NaN in either file)
    train_df = train_df.dropna(subset=features + [target])
    test_df  = test_df.dropna(subset=features + [target])

    X_train_s, y_train_s, X_test_s, y_test_s, feat_scaler, targ_scaler = scale_train_test(train_df, test_df, features, target)

    # Create sequences (X shaped: samples x timesteps x features)
    X_train, y_train = create_sequences(X_train_s, y_train_s, STEP_SIZE)
    X_test, y_test   = create_sequences(X_test_s, y_test_s, STEP_SIZE)

    print("Sequence shapes -> X_train:", X_train.shape, "y_train:", y_train.shape, "X_test:", X_test.shape, "y_test:", y_test.shape)

    if X_train.size == 0:
        raise RuntimeError("No training samples created. Check STEP_SIZE vs data length.")

    input_shape = (X_train.shape[1], X_train.shape[2])

    # Build model
    print("Building model with input_shape:", input_shape)
    model = create_dynamic_model(input_shape, LAYER_CONFIG, output_units=1)
    model.summary()   # now model is built because Input() set shape

    # Train
    early = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early], verbose=2)

    plot_training(history)

    # Predict (scaled), then inverse transform
    preds_scaled = model.predict(X_test)
    # preds_scaled shape (samples, 1)
    preds = targ_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)
    y_true = targ_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    # Simple printing / diagnostics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae  = mean_absolute_error(y_true, preds)
    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    # Plot last N actual vs predicted
    N = min(200, len(y_true))
    plt.figure(figsize=(10,4))
    plt.plot(y_true[-N:], label='actual')
    plt.plot(preds[-N:], label='predicted', linestyle='--')
    plt.title("Actual vs Predicted (last {} samples)".format(N))
    plt.legend()
    plt.show()

    # Optionally save model
    model.save(os.path.join(DATA_DIR, "rnn_model.h5"))
    print("Model saved to:", os.path.join(DATA_DIR, "rnn_model.h5"))


if __name__ == "__main__":
    main()
