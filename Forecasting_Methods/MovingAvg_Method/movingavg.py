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

# --------------------- Moving Average Forecast ---------------------
def apply_moving_average(df, demand_col="Demand", forecast_col="MA_Forecast", periods=3):
    """
    Apply Simple Moving Average Forecast
    Forecast(t) = Average of previous 'periods' actual demands
    """
    df = df.copy()
    df[forecast_col] = df[demand_col].shift(1).rolling(window=periods, min_periods=1).mean()
    return df

# --------------------- Next Period Forecast ---------------------
def next_period_moving_average(df, period_col, demand_col="Demand", periods=3):
    """
    Forecast next period using moving average
    """
    last_period = df[period_col].iloc[-1]
    last_values = df[demand_col].tail(periods)
    next_forecast = last_values.mean()

    if isinstance(last_period, (int, float)):
        next_period = last_period + 1
    else:
        next_period = f"After {last_period}"

    return next_period, next_forecast

# --------------------- Plot Forecast ---------------------
def plot_moving_average(df, period_col, demand_col="Demand", forecast_col="MA_Forecast"):
    """
    Professional line chart comparing actual vs moving average forecast
    """
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df[period_col], df[demand_col], marker="o", linewidth=2, label="Actual Demand")
    ax.plot(df[period_col], df[forecast_col], marker="o", linestyle="--", linewidth=2, label="Moving Avg Forecast")
    ax.set_title(f"Moving Average Forecast vs Actual Demand", fontsize=14)
    ax.set_xlabel(period_col)
    ax.set_ylabel("Quantity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

# --------------------- Full Moving Average Pipeline ---------------------
def run_moving_average_forecasting(file_path_or_df, period_col, demand_col="Demand"):
    """
    Full pipeline:
    - Load data
    - Ask user for number of periods
    - Apply Moving Average Forecast
    - Display table
    - Plot chart
    - Forecast next period
    """
    df = load_demand_data(file_path_or_df)

    # ===== User Input for Number of Periods =====
    periods = st.number_input(
        "Enter number of periods for Moving Average",
        min_value=1,
        max_value=len(df),
        value=3,
        step=1
    )

    df = apply_moving_average(df, demand_col=demand_col, periods=periods)

    st.subheader(f"📊 Moving Average Forecast Table (Periods = {periods})")
    st.dataframe(df, use_container_width=True)

    plot_moving_average(df, period_col, demand_col=demand_col)

    next_period, next_forecast = next_period_moving_average(df, period_col, demand_col=demand_col, periods=periods)
    st.info(f"📌 Moving Average Forecast for next period (**{next_period}**) = **{round(next_forecast,2)}**")

    return df
