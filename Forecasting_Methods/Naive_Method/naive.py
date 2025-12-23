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

# --------------------- Naive Forecast ---------------------
def apply_naive_forecast(df, demand_col="Demand", forecast_col="Naive_Forecast"):
    """
    Apply Naive Forecast: Forecast(t) = Actual(t-1)
    """
    df = df.copy()
    df[forecast_col] = df[demand_col].shift(1)
    return df

# --------------------- Next Period Forecast ---------------------
def next_period_naive_forecast(df, period_col, demand_col="Demand"):
    """
    Forecast for next period: Next = Last Actual
    """
    last_period = df[period_col].iloc[-1]
    last_demand = df[demand_col].iloc[-1]

    # If numeric period
    if isinstance(last_period, (int, float)):
        next_period = last_period + 1
    else:
        next_period = f"After {last_period}"

    return next_period, last_demand

# --------------------- Plot Forecast ---------------------
def plot_naive_forecast(df, period_col, demand_col="Demand", forecast_col="Naive_Forecast"):
    """
    Professional line chart comparing actual vs naive forecast
    """
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df[period_col], df[demand_col], marker="o", linewidth=2, label="Actual Demand")
    ax.plot(df[period_col], df[forecast_col], marker="o", linestyle="--", linewidth=2, label="Naive Forecast")
    ax.set_title("Naive Forecast vs Actual Demand", fontsize=14)
    ax.set_xlabel(period_col)
    ax.set_ylabel("Quantity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

# --------------------- Full Naive Forecast Pipeline ---------------------
def run_naive_forecasting(file_path_or_df, period_col, demand_col="Demand"):
    """
    Full pipeline:
    - Load data
    - Apply Naive Forecast
    - Display table
    - Plot chart
    - Forecast next period
    """
    # Load data
    df = load_demand_data(file_path_or_df)

    # Apply Naive Forecast
    df = apply_naive_forecast(df, demand_col=demand_col)

    # Display table
    st.subheader("📊 Naive Forecast Table")
    st.dataframe(df, use_container_width=True)

    # Plot
    plot_naive_forecast(df, period_col, demand_col=demand_col)

    # Next period forecast
    next_period, next_forecast = next_period_naive_forecast(df, period_col, demand_col)
    st.info(f"📌 Naive Forecast for next period (**{next_period}**) = **{next_forecast}**")

    return df
