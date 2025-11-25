"""
Configurable DL Model Builder
Featuring: LSTM, GRU, RNN
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, Bidirectional, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mplfinance as mpf

from data_processing_2 import load_stock_data


#build model function
def build_model(layer_configs, input_shape, output_units=1):
    model = Sequential()
    for i, cfg in enumerate(layer_configs):
        layer_type = cfg.get('type', 'LSTM')
        units = cfg.get('units', 50)
        return_sequences = cfg.get('return_sequences', False)
        dropout = cfg.get('dropout', 0.0)
        activation = cfg.get('activation', None)
        kwargs = {}
        if i == 0:
            kwargs['input_shape'] = input_shape

        # Bidirectional
        if 'Bidirectional' in layer_type:
            base_type = layer_type.replace('Bidirectional(', '').replace(')', '').strip()
            if base_type == 'LSTM':
                model.add(Bidirectional(LSTM(units, return_sequences=return_sequences, activation=activation, **kwargs)))
            elif base_type == 'GRU':
                model.add(Bidirectional(GRU(units, return_sequences=return_sequences, activation=activation, **kwargs)))
            elif base_type == 'RNN':
                model.add(Bidirectional(SimpleRNN(units, return_sequences=return_sequences, activation=activation, **kwargs)))
        else:
            if layer_type == 'LSTM':
                model.add(LSTM(units, return_sequences=return_sequences, activation=activation, **kwargs))
            elif layer_type == 'GRU':
                model.add(GRU(units, return_sequences=return_sequences, activation=activation, **kwargs))
            elif layer_type == 'RNN':
                model.add(SimpleRNN(units, return_sequences=return_sequences, activation=activation, **kwargs))
            elif layer_type == 'Conv1D':
                filters = cfg.get('filters', 64)
                kernel_size = cfg.get('kernel_size', 3)
                act = cfg.get('activation', 'relu')
                model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation=act, **kwargs))
            elif layer_type == 'Flatten':
                model.add(Flatten())
            elif layer_type == 'Dense':
                model.add(Dense(units, activation=activation))
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        if dropout > 0:
            model.add(Dropout(dropout))

    model.add(Dense(output_units))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


#train model function
def train_model(model, x_train, y_train, x_val, y_val, epochs=75, batch_size=32):
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )
    return history


#output training visual
def plot_metric(train_metric, val_metric, title):
    """Plot training vs validation loss."""
    plt.figure(figsize=(8, 4))
    plt.plot(train_metric, label='Train Loss', linewidth=2)
    plt.plot(val_metric, label='Validation Loss', linewidth=2)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_candlestick_predicted(test_df, predicted_prices, n=4):
    """Overlay predicted closing prices on a candlestick chart."""

    df = test_df.copy()
    df = df.iloc[-len(predicted_prices):]        # Align sizes
    df['Predicted Close'] = predicted_prices
    df.index = pd.to_datetime(df.index)

    # Aggregate into multi-day candlesticks
    df_resampled = df.resample(f'{n}D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'Predicted Close': 'mean'
    }).dropna()

    try:
        add_plot = mpf.make_addplot(df_resampled['Predicted Close'], color='blue',label='Predicted Close')
    except TypeError:
        add_plot = mpf.make_addplot(df_resampled['Predicted Close'], color='blue', width=1.2)

    # Plot candlestick + prediction line
    mpf.plot(
        df_resampled,
        type='candle',
        style='yahoo',
        title=f"Predicted Close Overlay ({n}-Day Candles)",
        volume=True,
        addplot=add_plot,
        figsize=(10, 6),
        tight_layout=True
    )

#model test
TEST_CONFIGS = {

    1: [# Mix LSTM, GRU, RNN, diffrent activation methods
        {'type': 'LSTM', 'units': 120, 'return_sequences': True, 'dropout': 0.25, 'activation': 'tanh'},
        {'type': 'LSTM', 'units': 100, 'return_sequences': True, 'dropout': 0.25},
        {'type': 'GRU', 'units': 80, 'return_sequences': True, 'dropout': 0.25},
        {'type': 'RNN', 'units': 60, 'return_sequences': True, 'dropout': 0.25, 'activation': 'relu'},
        {'type': 'LSTM', 'units': 40, 'return_sequences': False, 'dropout': 0.25},
    ],
    2: [# Simple LSTM Model
        {'type': 'LSTM', 'units': 50, 'return_sequences': True, 'dropout': 0.2},
        {'type': 'LSTM', 'units': 50, 'return_sequences': False, 'dropout': 0.2},
    ],
    3: [# Simple GRU Model 
        {'type': 'GRU', 'units': 100, 'return_sequences': True, 'dropout': 0.2},
        {'type': 'GRU', 'units': 100, 'return_sequences': False, 'dropout': 0.2},
    ],
    4: [#Complex GRU Model
        {'type': 'GRU', 'units': 128, 'return_sequences': True, 'dropout': 0.3},
        {'type': 'GRU', 'units': 64, 'return_sequences': True, 'dropout': 0.3},
        {'type': 'GRU', 'units': 64, 'return_sequences': False, 'dropout': 0.2},
    ],
    5: [# Mixed LSTM and GRU Model
        {'type': 'LSTM', 'units': 100, 'return_sequences': True, 'dropout': 0.3},
        {'type': 'GRU', 'units': 50, 'return_sequences': False, 'dropout': 0.3},
    ],
    6: [# Deep LSTM with more Units Model
        {'type': 'LSTM', 'units': 128, 'return_sequences': True, 'dropout': 0.3},
        {'type': 'LSTM', 'units': 128, 'return_sequences': True, 'dropout': 0.3},
        {'type': 'LSTM', 'units': 64, 'return_sequences': False, 'dropout': 0.3},
    ],
    7: [#  LTSM Reduced Dropout Model
        {'type': 'LSTM', 'units': 64, 'return_sequences': True, 'dropout': 0.1},
        {'type': 'LSTM', 'units': 64, 'return_sequences': False, 'dropout': 0.1},
    ],
    8: [# Bidirectional LSTM Configuration Model
        {'type': 'Bidirectional(LSTM)', 'units': 128, 'return_sequences': True, 'dropout': 0.3},
        {'type': 'Bidirectional(LSTM)', 'units': 64, 'return_sequences': False, 'dropout': 0.3},
    ],
    9: [# Bidirectional GRU Configuration Model
        {'type': 'Bidirectional(GRU)', 'units': 128, 'return_sequences': True, 'dropout': 0.3},
        {'type': 'Bidirectional(GRU)', 'units': 64, 'return_sequences': False, 'dropout': 0.3},
    ],
    10: [ # Bidirectional GRU Configuration with more layers
        {'type': 'Bidirectional(GRU)', 'units': 128, 'return_sequences': True, 'dropout': 0.3},
        {'type': 'Bidirectional(GRU)', 'units': 128, 'return_sequences': True, 'dropout': 0.3},
        {'type': 'Bidirectional(GRU)', 'units': 64, 'return_sequences': True, 'dropout': 0.2},
        {'type': 'Bidirectional(GRU)', 'units': 64, 'return_sequences': False, 'dropout': 0.2},
    ],
    11: [  # CNN-LSTM Hybrid
        {'type': 'Conv1D', 'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
        {'type': 'LSTM', 'units': 100, 'return_sequences': False, 'dropout': 0.3},
    ],
    12: [  # GRU + Dense bridge
        {'type': 'GRU', 'units': 128, 'return_sequences': False, 'dropout': 0.3},
        {'type': 'Dense', 'units': 64, 'activation': 'relu'},
    ],
    13: [  # Minimal RNN
        {'type': 'RNN', 'units': 32, 'return_sequences': False, 'dropout': 0.1},
    ]
}


if __name__ == "__main__":
    TEST_ID = 13  # Change between 1-13 to test different configurations
    print(f"Running Test {TEST_ID}\n")

    df, train_df, test_df, scaler = load_stock_data(
        ticker="AMZN", #select company
        start_date="2020-01-01", #start date
        end_date="2025-7-31", #end date
        split_by_date=True,
        test_size=0.2,
        scale=True
    )

    def create_sequences(data, seq_len=60):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i-seq_len:i])
            y.append(data[i, 3])  # Close price
        return np.array(X), np.array(y)

    seq_len = 60
    x_train, y_train = create_sequences(train_df.values, seq_len)
    x_test, y_test = create_sequences(test_df.values, seq_len)

    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    input_shape = (x_train.shape[1], x_train.shape[2])

    model = build_model(TEST_CONFIGS[TEST_ID], input_shape=input_shape)
    model.summary()

    history = train_model(model, x_train, y_train, x_test, y_test, epochs=75, batch_size=32)

    # Predict and inverse scale
    predicted = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(
        np.hstack([np.zeros((predicted.shape[0], train_df.shape[1]-1)), predicted])
    )[:, -1]

    plot_metric(history.history['loss'], history.history['val_loss'], 'Total loss vs Total val loss')
    plot_candlestick_predicted(test_df, predicted_prices, n=4)

    print("Training complete")