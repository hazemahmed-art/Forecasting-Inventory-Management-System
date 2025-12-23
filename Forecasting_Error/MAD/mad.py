import pandas as pd
import streamlit as st

# ====================== MAD Calculation ======================
def calculate_mad(df, actual_col="Demand", forecast_col="Forecast"):
    """
    Calculate Mean Absolute Deviation (MAD) between actual and forecast
    MAD = mean(|Actual - Forecast|)
    """
    if forecast_col not in df.columns:
        st.error(f"Forecast column '{forecast_col}' not found in DataFrame!")
        return None

    df = df.copy()
    df["Absolute Error"] = (df[actual_col] - df[forecast_col]).abs()
    mad_value = df["Absolute Error"].mean()
    return mad_value, df

# ====================== Example Usage in Streamlit ======================
def run_mad_analysis(file_path_or_df, forecast_col="Forecast", actual_col="Demand"):
    """
    Load data, calculate MAD, and display results
    """
    # Load data
    if isinstance(file_path_or_df, pd.DataFrame):
        df = file_path_or_df.copy()
    else:
        df = pd.read_excel(file_path_or_df)

    df.reset_index(drop=True, inplace=True)

    # Calculate MAD
    mad_value, df_with_error = calculate_mad(df, actual_col=actual_col, forecast_col=forecast_col)
    if mad_value is not None:
        st.subheader(f"📌 MAD (Mean Absolute Deviation) for '{forecast_col}': {round(mad_value,2)}")
        st.dataframe(df_with_error, use_container_width=True)

    return mad_value, df_with_error
