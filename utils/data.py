import numpy as np
from typing import Optional

def generate_linear_data(
    n: int = 100,
    slope: float = 1.0,
    intercept: float = 0.0,
    noise_std: float = 1.0,
    random_state: Optional[int] = None,
):
    """
    Generate synthetic linear data: y = slope * x + intercept + noise.
    """
    rng = np.random.default_rng(random_state)
    x = rng.uniform(-5, 5, size=n)
    noise = rng.normal(0, noise_std, size=n)
    y = slope * x + intercept + noise
    return x, y
