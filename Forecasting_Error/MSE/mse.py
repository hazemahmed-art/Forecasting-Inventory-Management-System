import pandas as pd
import streamlit as st

# ====================== MSE Calculation ======================
def calculate_mse(df, actual_col="Demand", forecast_col="Forecast"):
    """
    Calculate Mean Squared Error (MSE) between actual and forecast
    MSE = mean((Actual - Forecast)^2)
    """
    if forecast_col not in df.columns:
        st.error(f"Forecast column '{forecast_col}' not found in DataFrame!")
        return None

    df = df.copy()
    df["Squared Error"] = (df[actual_col] - df[forecast_col])**2
    mse_value = df["Squared Error"].mean()
    return mse_value, df

# ====================== Example Usage in Streamlit ======================
def run_mse_analysis(file_path_or_df, forecast_col="Forecast", actual_col="Demand"):
    """
    Load data, calculate MSE, and display results
    """
    # Load data
    if isinstance(file_path_or_df, pd.DataFrame):
        df = file_path_or_df.copy()
    else:
        df = pd.read_excel(file_path_or_df)

    df.reset_index(drop=True, inplace=True)

    # Calculate MSE
    mse_value, df_with_error = calculate_mse(df, actual_col=actual_col, forecast_col=forecast_col)
    if mse_value is not None:
        st.subheader(f"📌 MSE (Mean Squared Error) for '{forecast_col}': {round(mse_value,2)}")
        st.dataframe(df_with_error, use_container_width=True)

    return mse_value, df_with_error
