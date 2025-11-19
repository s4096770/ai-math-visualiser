import numpy as np
import matplotlib.pyplot as plt

from .regression import predict


def plot_regression_with_residuals(
    x: np.ndarray,
    y: np.ndarray,
    m: float,
    b: float,
    figsize=(6, 5),
):
    """
    Create a modern-looking scatter plot with regression line and residuals.
    """
    y_pred = predict(x, m, b)

    fig, ax = plt.subplots(figsize=figsize)

    # Aesthetic: soft grid, minimal spines
    ax.grid(True, alpha=0.2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Scatter points
    ax.scatter(x, y, alpha=0.8, edgecolor="none")

    # Regression line
    x_line = np.linspace(x.min() - 1, x.max() + 1, 200)
    y_line = predict(x_line, m, b)
    ax.plot(x_line, y_line, linewidth=2)

    # Residual lines
    for xi, yi, ypi in zip(x, y, y_pred):
        ax.vlines(xi, ypi, yi, alpha=0.4, linestyle="--")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Data, regression line & residuals")

    return fig


def plot_cost_curve(
    m_values: np.ndarray,
    costs: np.ndarray,
    current_m: float,
    current_cost: float,
    figsize=(6, 5),
):
    """
    Plot cost as a function of the slope, with the current (m, cost) highlighted.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Aesthetic: soft grid & minimal spines
    ax.grid(True, alpha=0.2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Cost curve
    ax.plot(m_values, costs, linewidth=2)

    # Current point marker
    ax.scatter([current_m], [current_cost], s=60, zorder=5)
    ax.annotate(
        f"m = {current_m:.2f}\nJ = {current_cost:.3f}",
        xy=(current_m, current_cost),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.8),
        arrowprops=dict(arrowstyle="->", alpha=0.8),
    )

    ax.set_xlabel("Slope m")
    ax.set_ylabel("MSE Cost J(m, b)")
    ax.set_title("Cost function with respect to slope (b fixed)")

    return fig
