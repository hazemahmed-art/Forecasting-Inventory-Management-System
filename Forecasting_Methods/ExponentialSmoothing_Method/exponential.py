import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --------------------- Load Data ---------------------
def load_demand_data(file_path_or_df):
    """
    Load data from Excel file or accept DataFrame directly.
    """
    if isinstance(file_path_or_df, pd.DataFrame):
        df = file_path_or_df.copy()
    else:
        df = pd.read_excel(file_path_or_df)
    df = df.reset_index(drop=True)
    return df

# --------------------- Exponential Smoothing Forecast ---------------------
def apply_exponential_smoothing(df, demand_col="Demand", forecast_col="Exp_Forecast", alpha=0.3):
    """
    Apply Simple Exponential Smoothing
    Forecast(t) = alpha * Actual(t-1) + (1-alpha) * Forecast(t-1)
    """
    df = df.copy()
    df[forecast_col] = 0.0

    # Initialize first forecast as first actual demand
    df[forecast_col].iloc[0] = df[demand_col].iloc[0]

    # Calculate forecast recursively
    for t in range(1, len(df)):
        df[forecast_col].iloc[t] = alpha * df[demand_col].iloc[t-1] + (1 - alpha) * df[forecast_col].iloc[t-1]

    return df

# --------------------- Next Period Forecast ---------------------
def next_period_exponential(df, period_col, demand_col="Demand", alpha=0.3, forecast_col="Exp_Forecast"):
    """
    Forecast next period using exponential smoothing
    """
    last_period = df[period_col].iloc[-1]
    last_demand = df[demand_col].iloc[-1]
    last_forecast = df[forecast_col].iloc[-1]

    next_forecast = alpha * last_demand + (1 - alpha) * last_forecast

    if isinstance(last_period, (int, float)):
        next_period = last_period + 1
    else:
        next_period = f"After {last_period}"

    return next_period, next_forecast

# --------------------- Plot Forecast ---------------------
def plot_exponential_forecast(df, period_col, demand_col="Demand", forecast_col="Exp_Forecast"):
    """
    Professional line chart comparing actual vs exponential forecast
    """
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df[period_col], df[demand_col], marker="o", linewidth=2, label="Actual Demand")
    ax.plot(df[period_col], df[forecast_col], marker="o", linestyle="--", linewidth=2, label="Exponential Forecast")
    ax.set_title(f"Exponential Smoothing Forecast vs Actual Demand", fontsize=14)
    ax.set_xlabel(period_col)
    ax.set_ylabel("Quantity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

# --------------------- Full Exponential Forecast Pipeline ---------------------
def run_exponential_forecasting(file_path_or_df, period_col, demand_col="Demand"):
    """
    Full pipeline:
    - Load data
    - Ask user for alpha
    - Apply Exponential Smoothing Forecast
    - Display table
    - Plot chart
    - Forecast next period
    """
    df = load_demand_data(file_path_or_df)

    # ================= User input for alpha =================
    st.sidebar.markdown("### ⚡ Exponential Smoothing α")
    alpha = st.sidebar.slider("Choose smoothing factor (α)", min_value=0.01, max_value=1.0, value=0.3, step=0.01)

    # Apply exponential smoothing
    df = apply_exponential_smoothing(df, demand_col=demand_col, alpha=alpha)

    # Show table
    st.subheader(f"📊 Exponential Smoothing Forecast Table (α = {alpha})")
    st.dataframe(df, use_container_width=True)

    # Plot forecast
    plot_exponential_forecast(df, period_col, demand_col=demand_col)

    # Forecast next period
    next_period, next_forecast = next_period_exponential(df, period_col, demand_col=demand_col, alpha=alpha)
    st.info(f"📌 Exponential Forecast for next period (**{next_period}**) = **{round(next_forecast,2)}**")

    return df
