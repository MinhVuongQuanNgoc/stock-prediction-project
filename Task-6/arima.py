"""
Task 6 – ARIMA Baseline
Optional traditional ARIMA model for comparison.
"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


def run_arima(df: pd.DataFrame, column: str = "Close", forecast_steps: int = 5):
    """Fit ARIMA and forecast future values."""
    data = df[column].dropna().values
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_steps)

    plt.figure(figsize=(10, 4))
    plt.plot(data, label="Historical")
    plt.plot(range(len(data), len(data) + forecast_steps), forecast, label="Forecast", linestyle="--")
    plt.title("ARIMA Forecast")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return forecast
