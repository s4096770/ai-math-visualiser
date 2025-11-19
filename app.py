import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from utils.data import generate_linear_data
from utils.regression import predict
from utils.cost import mse_cost, sweep_parameter
from utils.visuals import (
    plot_regression_with_residuals,
    plot_cost_curve,
)

# ---------- Page config ----------
st.set_page_config(
    page_title="AI Math Visualiser — Linear Regression & Cost",
    page_icon="📈",
    layout="wide"
)

# ---------- Sidebar controls ----------
st.sidebar.title("⚙️ Controls")

n_points = st.sidebar.slider("Number of data points", 20, 200, 60, step=5)
noise_level = st.sidebar.slider("Noise level", 0.0, 2.0, 0.6, step=0.1)

true_slope = st.sidebar.slider("True slope (for generated data)", -3.0, 3.0, 1.5, step=0.1)
true_intercept = st.sidebar.slider("True intercept (for generated data)", -5.0, 5.0, 0.5, step=0.5)

st.sidebar.markdown("---")
st.sidebar.caption("Data is regenerated whenever you change these sliders.")

# Generate synthetic data
x, y = generate_linear_data(
    n=n_points,
    slope=true_slope,
    intercept=true_intercept,
    noise_std=noise_level,
    random_state=42
)

st.title("📈 AI Math Visualiser – Linear Regression & Cost Function")
st.subheader("See how slope, intercept, residuals and cost all connect.")

st.markdown(
    """
This page helps you **build intuition** for linear regression:

- Adjust the **model parameters** (slope and intercept)
- Watch how the **regression line** and **residuals** change
- See how that changes the overall **Mean Squared Error (MSE) cost**  
"""
)

# ---------- User-controlled model parameters ----------
col_m, col_b = st.columns(2)
with col_m:
    m = st.slider("Model slope (m)", -5.0, 5.0, 1.0, step=0.1)
with col_b:
    b = st.slider("Model intercept (b)", -10.0, 10.0, 0.0, step=0.5)

y_pred = predict(x, m, b)
current_cost = mse_cost(y, y_pred)

st.markdown(
    f"""
### 🔢 Current model

- **Slope (m)** = `{m:.2f}`  
- **Intercept (b)** = `{b:.2f}`  
- **MSE Cost** = ` {current_cost:.4f} `
"""
)

# ---------- Layout: Left = regression plot, Right = cost curve ----------
left_col, right_col = st.columns(2)

with left_col:
    st.markdown("#### 1️⃣ Data, regression line & residuals")

    fig_reg = plot_regression_with_residuals(
        x=x,
        y=y,
        m=m,
        b=b,
        figsize=(6, 5)
    )
    st.pyplot(fig_reg)
    st.caption(
        "Each vertical line is a **residual** — the difference between the true data point "
        "and the prediction from your chosen line. Shorter lines = better fit."
    )

with right_col:
    st.markdown("#### 2️⃣ Cost as a function of slope")

    # Sweep slope values and compute cost
    m_values, costs = sweep_parameter(
        x=x,
        y=y,
        intercept=b,
        m_min=-5.0,
        m_max=5.0,
        num=200
    )

    fig_cost = plot_cost_curve(
        m_values=m_values,
        costs=costs,
        current_m=m,
        current_cost=current_cost,
        figsize=(6, 5)
    )
    st.pyplot(fig_cost)
    st.caption(
        "This shows the **MSE cost** for different slopes, keeping the intercept fixed. "
        "Gradient descent would move the slope downhill along this curve until it reaches the minimum."
    )

# ---------- Explanation section ----------
st.markdown("---")
st.markdown("### 🧠 What are we actually seeing?")

exp_col1, exp_col2 = st.columns(2)

with exp_col1:
    st.markdown(
        """
**Residuals & error**

For each data point:

- The model predicts \\( \\hat{y} = m x + b \\)
- The residual is \\( y - \\hat{y} \\)
- The cost we use is **Mean Squared Error (MSE)**:

\\[
J(m, b) = \\frac{1}{N} \\sum_{i=1}^{N} (y_i - (m x_i + b))^2
\\]

In the left plot, the vertical lines represent those residuals.
When you choose parameters that fit well, those lines shrink and the MSE drops.
"""
    )

with exp_col2:
    st.markdown(
        """
**Cost curve & optimisation**

In the right plot, we:

- Sweep through many possible slopes (m)
- Compute the MSE for each one (keeping b fixed)
- Plot the **cost curve** as a function of m

This curve is typically **convex** for linear regression.
That’s why algorithms like **gradient descent** work so well:  
they follow the negative gradient down this bowl-shaped curve until they reach the minimum.

In the next phase, we can:
- Animate the **gradient descent steps**
- Show how (m, b) moves along the cost landscape
"""
    )

st.markdown(
    """
> 💡 This single page already demonstrates **core AI maths**: linear models, residuals, loss functions, and the shape of the optimisation landscape.
"""
)
