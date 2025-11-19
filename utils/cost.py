import numpy as np

from .regression import predict


def mse_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error cost function.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def sweep_parameter(
    x: np.ndarray,
    y: np.ndarray,
    intercept: float,
    m_min: float = -5.0,
    m_max: float = 5.0,
    num: int = 200,
):
    """
    Sweep through slope values and compute cost for each.

    Returns:
        m_values (np.ndarray), costs (np.ndarray)
    """
    m_values = np.linspace(m_min, m_max, num=num)
    costs = []

    for m in m_values:
        y_pred = predict(x, m, intercept)
        cost = mse_cost(y, y_pred)
        costs.append(cost)

    return m_values, np.array(costs)
