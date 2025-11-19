import numpy as np


def predict(x: np.ndarray, m: float, b: float) -> np.ndarray:
    """
    Compute predictions for y_hat = m * x + b
    """
    return m * x + b
