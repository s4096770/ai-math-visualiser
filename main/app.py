import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🧠 AI Math Visualiser")
st.write("Explore how mathematics powers machine learning — interactively and intuitively!")

# --- Linear Regression Playground ---
st.subheader("Linear Regression Playground")

# Generate sample data
np.random.seed(42)
x = np.linspace(0, 10, 30)
y = 2.5 * x + 3 + np.random.randn(30)

# Sliders for slope and intercept
m = st.slider("Slope (m)", -5.0, 5.0, 2.5)
b = st.slider("Intercept (b)", -10.0, 10.0, 3.0)

# Predicted line
y_pred = m * x + b

# Plot
fig, ax = plt.subplots()
ax.scatter(x, y, label="Data Points", color="blue")
ax.plot(x, y_pred, color="red", label=f"y = {m:.2f}x + {b:.2f}")
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("y")
st.pyplot(fig)

st.write("Adjust the sliders above to see how slope and intercept affect the line of best fit.")
