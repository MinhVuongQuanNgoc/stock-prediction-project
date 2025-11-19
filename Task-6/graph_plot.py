"""
Task 6 – Graph Plotting
Simple Matplotlib visualisation helper.
"""

import matplotlib.pyplot as plt


def plot_predictions(y_true, y_pred, title="Predictions vs Actual"):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true.values if hasattr(y_true, "values") else y_true, label="Actual")
    plt.plot(y_pred.values if hasattr(y_pred, "values") else y_pred, label="Predicted", linestyle="--")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
