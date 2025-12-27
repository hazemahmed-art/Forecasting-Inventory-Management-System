# app.py
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
# ================= External Styling =================
with open("style.css") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)
# ================= Page Config =================
st.set_page_config(page_title="Forecasting & Inventory Management System", layout="wide")
# ================= Constants =================
PERIOD_COLUMN_MAP = {
    "Weekly": "Week",
    "Monthly": "Month",
    "Quarterly": "Quarter",
    "Semi-Annual": "Half",
    "Annual": "Year"
}
# ================= Session State Initialization =================
if "page" not in st.session_state:
    st.session_state.page = 1
if "material" not in st.session_state:
    st.session_state.material = {}
if "period" not in st.session_state:
    st.session_state.period = None
if "file" not in st.session_state:
    st.session_state.file = None
if "df" not in st.session_state:
    st.session_state.df = None
if "show_table" not in st.session_state:
    st.session_state.show_table = False
if "editing" not in st.session_state:
    st.session_state.editing = False
# ================= Load Material Classification =================
try:
    df_class = pd.read_excel("Database/Classification-of-Material.xlsx")
except Exception as e:
    st.error(f"Cannot read 'Classification-of-Material.xlsx': {e}")
    st.stop()
# ================= Helper Functions =================
def select_material(df):
    st.markdown("## Select a Target Material")
    c1, c2, c3 = st.columns(3)
    with c1:
        family = st.selectbox("Material Family", sorted(df["MaterialFamily"].unique()), key="fam_sel")
    family_df = df[df["MaterialFamily"] == family]
    with c2:
        m_type = st.selectbox("Material Type", sorted(family_df["MaterialType"].unique()), key="type_sel")
    type_df = family_df[family_df["MaterialType"] == m_type]
    with c3:
        grade = st.selectbox("Material Grade", sorted(type_df["MaterialGrade"].unique()), key="grade_sel")
    st.caption(f"Selected → **{family} / {m_type} / {grade}**")
    st.divider()
    return {"family": family, "type": m_type, "grade": grade}
def select_period():
    st.markdown("## Select Data Period")
    return st.selectbox("Period", list(PERIOD_COLUMN_MAP.keys()), key="period_select")
def choose_data_source(period, family, m_type, grade):
    st.markdown("## Data Source")
    source = st.radio("Choose source", ["Upload Excel File", "Choose Existing File"], horizontal=True, key="data_source_radio")
    folder = f"Uploaded/{family}/{m_type}/{grade}/{period}"
    os.makedirs(folder, exist_ok=True)
    selected_file = None
    if source == "Upload Excel File":
        uploaded = st.file_uploader("Upload your Excel file", type=["xlsx"], key="file_uploader")
        if uploaded:
            path = os.path.join(folder, uploaded.name)
            with open(path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success("File uploaded successfully ✅")
            selected_file = path
    else:
        files = [f for f in os.listdir(folder) if f.endswith(".xlsx")]
        if files:
            name = st.selectbox("Choose existing file", files, key="existing_file_select")
            selected_file = os.path.join(folder, name)
        else:
            st.info("No uploaded files found for this period yet.")
    return selected_file
def load_table(file_path):
    try:
        df = pd.read_excel(file_path)
        df = df.fillna(0)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None
def view_table():
    st.subheader("Table Preview")
    st.dataframe(st.session_state.df, use_container_width=True)
def renumber_first_column(df, first_col):
    df[first_col] = range(1, len(df) + 1)
    return df
# ================= Forecasting Functions =================
def run_naive_forecasting(df, first_col):
    df = df.copy()
    df["Naive Forecast"] = df["Demand"].shift(1).fillna(df["Demand"].iloc[0])
    return df
def run_moving_average_forecasting(df, first_col, n=3):
    df = df.copy()
    df["Moving Avg Forecast"] = df["Demand"].rolling(n, min_periods=1).mean().shift(1).fillna(df["Demand"].iloc[0])
    return df
def run_exponential_forecasting(df, first_col, alpha=0.3):
    df = df.copy()
    fc = [df["Demand"].iloc[0]]
    for d in df["Demand"].iloc[1:]:
        fc.append(alpha * d + (1 - alpha) * fc[-1])
    df["Exponential Forecast"] = fc
    return df
# ================= Error Calculations =================
def calculate_mad(df, actual_col="Demand", forecast_col="Forecast"):
    df = df.copy()
    df["Abs Error"] = (df[actual_col] - df[forecast_col]).abs()
    return df["Abs Error"].mean()
def calculate_mse(df, actual_col="Demand", forecast_col="Forecast"):
    df = df.copy()
    df["Squared Error"] = (df[actual_col] - df[forecast_col]) ** 2
    return df["Squared Error"].mean()
# ================= Edit Table Function =================
def edit_table(file_path, period):
    st.subheader("✏ Edit / Add / Delete Data")
    if st.button("⬅ Back to View Mode"):
        st.session_state.editing = False
        st.rerun()
    if st.session_state.df is None:
        st.warning("No data loaded yet. Please upload or select a file first.")
        st.stop()
    first_col = PERIOD_COLUMN_MAP[period]
    df = st.session_state.df.copy()
    st.divider()
    # Add New Row
    st.markdown("### ➕ Add New Row")
    new_row = {}
    cols = st.columns(len(df.columns))
    for i, col in enumerate(df.columns):
        with cols[i]:
            if col == first_col:
                next_num = len(df) + 1
                new_row[col] = st.number_input(col, value=next_num, disabled=True, key=f"new_{col}")
            else:
                new_row[col] = st.number_input(col, value=0.0, step=0.01, key=f"new_input_{col}")
    if st.button("➕ Add Row", type="primary", key="add_row_btn"):
        new_data = {col: [len(df) + 1 if col == first_col else new_row[col]] for col in df.columns}
        new_df = pd.DataFrame(new_data)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_excel(file_path, index=False)
        st.session_state.df = df
        st.success("New row added successfully!")
        st.rerun()
    st.divider()
    # Edit Existing Row
    st.markdown("### ✏ Edit Existing Row")
    row_to_edit = st.selectbox("Choose row to edit", options=df[first_col].tolist(), key="edit_row_dropdown")
    row_idx = df[df[first_col] == row_to_edit].index[0]
    st.markdown(f"**Editing Row {row_to_edit}**")
    editable_columns = [col for col in df.columns if col != first_col]
    edit_cols = st.columns(len(editable_columns))
    edited_values = {}
    for i, col in enumerate(editable_columns):
        with edit_cols[i]:
            current_val = df.loc[row_idx, col]
            val = float(current_val) if pd.notna(current_val) else 0.0
            edited_values[col] = st.number_input(col, value=val, step=0.01, key=f"edit_{col}_{row_idx}")
    if st.button("💾 Save Changes", type="primary", key="save_edit_btn"):
        for col, val in edited_values.items():
            df.loc[row_idx, col] = val
        df = renumber_first_column(df, first_col)
        df.to_excel(file_path, index=False)
        st.session_state.df = df
        st.success("Changes saved!")
        st.rerun()
    st.divider()
    # Delete Row
    st.markdown("### 🗑 Delete Row")
    delete_options = ["-- Select to delete --"] + df[first_col].tolist()
    delete_key = st.selectbox("Select row", options=delete_options, index=0, key="delete_row_dropdown")
    if delete_key != "-- Select to delete --":
        st.warning(f"Delete row {delete_key}?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑 Confirm Delete", type="primary"):
                df = df[df[first_col] != delete_key].reset_index(drop=True)
                df = renumber_first_column(df, first_col)
                df.to_excel(file_path, index=False)
                st.session_state.df = df
                st.success("Row deleted!")
                st.rerun()
        with c2:
            if st.button("Cancel"):
                st.rerun()
# ================= SIDEBAR: Navigation & Current Selection Info =================
# This section creates a persistent sidebar that appears on every screen
with st.sidebar:
   
    # Show current selected material (if any)
    if st.session_state.material:
        st.subheader("Selected Material")
        st.write(f"**Family:** {st.session_state.material.get('family', '-')}")
        st.write(f"**Type:** {st.session_state.material.get('type', '-')}")
        st.write(f"**Grade:** {st.session_state.material.get('grade', '-')}")
        if st.session_state.period:
            st.write(f"**Period:** {st.session_state.period}")
   
    # Navigation buttons in sidebar
    st.subheader("Navigation")
   
    if st.button("🏠 Material Selection", use_container_width=True):
        st.session_state.page = 1
        st.session_state.df = None
        st.session_state.file = None
        st.session_state.editing = False
        st.rerun()
   
    if st.session_state.material: # Only show further navigation after material is selected
        if st.button("📁 Data & Table", use_container_width=True):
            st.session_state.page = 2
            st.rerun()
       
        if st.button("🔍 Analysis Menu", use_container_width=True):
            st.session_state.page = 3
            st.rerun()
       
        if st.button("📈 Forecasting", use_container_width=True):
            st.session_state.page = 4
            st.rerun()
       
        if st.button("📦 EOQ", use_container_width=True):
            st.session_state.page = 5
            st.rerun()
       
        if st.button("🛡️ Safety Stock", use_container_width=True):
            st.session_state.page = 6
            st.rerun()
   
    st.markdown("---")
    st.caption("Forecasting & Inventory Management System © 2025")

# ================= FLOATING HELP BUTTON (Bottom Right) =================
# Custom CSS for the floating help button
st.markdown("""
<style>
    .help-button {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 1000;
    }
    .help-button button {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background-color: #fa7328;
        color: white;
        font-size: 28px;
        border: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .help-button button:hover {
        background-color: #e0651f;
        transform: scale(1.1);
    }
</style>
""", unsafe_allow_html=True)

# Floating Help Button
with st.container():
    col1, col2, col3 = st.columns([8,1,1])
    with col3:
        if st.button("❓", key="help_button", help="Click for guidance"):
            st.session_state.show_help = True

# Help Message (appears when button is clicked)
if st.session_state.get("show_help", False):
    st.markdown("""
    <div style="
        position: fixed;
        bottom: 100px;
        right: 30px;
        width: 380px;
        background-color: #1e1e1e;
        color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.5);
        border: 2px solid #fa7328;
        z-index: 999;
    ">
        <h3 style="margin-top:0; color:#fa7328;">📋 How to Use This System</h3>
        <p><strong>1. Material Selection</strong><br>Choose Family → Type → Grade from the database.</p>
        <p><strong>2. Data & Table</strong><br>Select period, upload or choose historical demand file, view/edit data.</p>
        <p><strong>3. Analysis Menu</strong><br>Choose one of the analyses:</p>
        <ul>
            <li><strong>📈 Forecasting</strong>: Compare Naive, Moving Average & Exponential Smoothing → get best forecast.</li>
            <li><strong>📦 EOQ</strong>: Calculate optimal order quantity and reorder point.</li>
            <li><strong>🛡️ Safety Stock</strong>: Fixed or statistical based on service level.</li>
        </ul>
        <p>Use the sidebar to navigate between steps anytime.</p>
        <div style="text-align:right; margin-top:15px;">
            <strong>Click ❓ again to close</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Clicking the button again closes the help
    if st.button("❓", key="help_button_close"):
        st.session_state.show_help = False
        st.rerun()

# ================= SCREEN 1: Material Selection =================
def page_material_selection():
    st.title("Welcome to Forecasting & Inventory Management System")
    st.markdown("### Step 1: Select the Material you want to analyze")
    st.session_state.material = select_material(df_class)
    if st.button("Next ➜", type="primary"):
        st.session_state.page = 2
        st.rerun()
# ================= SCREEN 2: Data Selection & Table View/Edit =================
def page_selected_material():
    st.title("Selected Material")
    st.markdown(f"**{st.session_state.material['family']} / {st.session_state.material['type']} / {st.session_state.material['grade']}**")
    st.session_state.period = select_period()
    family = st.session_state.material['family']
    m_type = st.session_state.material['type']
    grade = st.session_state.material['grade']
    st.session_state.file = choose_data_source(st.session_state.period, family, m_type, grade)
    if st.session_state.file and st.session_state.df is None:
        st.session_state.df = load_table(st.session_state.file)
        if st.session_state.df is None:
            st.stop()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.session_state.show_table = st.checkbox("Show Table", value=st.session_state.show_table)
    with c2:
        if st.button("✏ Edit Table" if not st.session_state.editing else "✏ Editing..."):
            st.session_state.editing = not st.session_state.editing
            st.rerun()
    with c3:
        if st.button("⬅ Back"):
            st.session_state.page = 1
            st.session_state.df = st.session_state.file = None
            st.session_state.editing = False
            st.rerun()
    with c4:
        if st.button("Next ➜ Analysis", type="primary"):
            st.session_state.page = 3
            st.rerun()
    if st.session_state.show_table:
        view_table()
    if st.session_state.editing:
        edit_table(st.session_state.file, st.session_state.period)
# ================= SCREEN 3: Analysis Menu =================
def page_analysis():
    st.title("Analysis & Calculations")
    st.markdown("### Choose the analysis you want to perform")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📈 Forecasting", use_container_width=True, type="primary"):
            st.session_state.page = 4
            st.rerun()
    with col2:
        if st.button("📦 EOQ", use_container_width=True, type="primary"):
            st.session_state.page = 5
            st.rerun()
    with col3:
        if st.button("🛡️ Safety Stock", use_container_width=True, type="primary"):
            st.session_state.page = 6
            st.rerun()
    if st.button("⬅ Back to Data Editing"):
        st.session_state.page = 2
        st.rerun()
# ================= SCREEN 4: Forecasting Analysis =================
def page_forecasting():
    st.title("📈 Forecasting Analysis")
    mat = st.session_state.material
    period_name = st.session_state.period
    first_col = PERIOD_COLUMN_MAP[period_name]
    df_base = st.session_state.df.copy()
    st.markdown(
        f"""
        **Material:** {mat['family']} / {mat['type']} / {mat['grade']}   |  
        **Period:** {period_name}   |  
        **Records:** {len(df_base)}
        """
    )
    st.divider()
    if "forecast_ran" not in st.session_state:
        st.session_state.forecast_ran = False
    if not st.session_state.forecast_ran:
        st.subheader("Select Evaluation Criteria")
        criteria = st.radio("Choose the error metric:", ["MAD", "MSE"], horizontal=True)
        if st.button("RUN FORECASTING", type="primary", use_container_width=True):
            st.session_state.selected_criteria = criteria
            with st.spinner("Running forecasting models..."):
                results = {}
                errors = []
                # Naive
                df_n = run_naive_forecasting(df_base.copy(), first_col)
                mad_n = calculate_mad(df_n, forecast_col="Naive Forecast")
                mse_n = calculate_mse(df_n, forecast_col="Naive Forecast")
                results["Naive"] = df_n
                errors.append({"Method": "Naive", "MAD": mad_n, "MSE": mse_n})
                # Moving Average
                ma_n = 3
                df_ma = run_moving_average_forecasting(df_base.copy(), first_col, n=ma_n)
                mad_ma = calculate_mad(df_ma, forecast_col="Moving Avg Forecast")
                mse_ma = calculate_mse(df_ma, forecast_col="Moving Avg Forecast")
                results["Moving Average"] = df_ma
                errors.append({"Method": "Moving Average", "MAD": mad_ma, "MSE": mse_ma})
                # Exponential Smoothing
                alpha = 0.3
                df_exp = run_exponential_forecasting(df_base.copy(), first_col, alpha=alpha)
                mad_exp = calculate_mad(df_exp, forecast_col="Exponential Forecast")
                mse_exp = calculate_mse(df_exp, forecast_col="Exponential Forecast")
                results["Exponential Smoothing"] = df_exp
                errors.append({"Method": "Exponential Smoothing", "MAD": mad_exp, "MSE": mse_exp})
                error_df = pd.DataFrame(errors).round(4)
                st.session_state.all_results = results
                st.session_state.all_errors = error_df
                best_row = error_df.loc[error_df[criteria].idxmin()]
                st.session_state.best_method = best_row["Method"]
                st.session_state.best_error = best_row[criteria]
                st.session_state.forecast_ran = True
                st.rerun()
    if st.session_state.forecast_ran:
        criteria = st.session_state.selected_criteria
        best_method = st.session_state.best_method
        best_error = st.session_state.best_error
        results = st.session_state.all_results
        st.success(f"Best Method according to {criteria}: **{best_method}** ({criteria} = {best_error:.4f})")
        df_best = results[best_method]
        fc_best = [col for col in df_best.columns if "Forecast" in col][0]
        st.divider()
        st.subheader(f"📋 Forecast Table – {best_method}")
        table_best = df_base[[first_col, "Demand"]].copy()
        table_best["Forecast"] = df_best[fc_best]
        st.dataframe(table_best.style.format("{:.2f}"), use_container_width=True)
        st.subheader(f"📊 Forecast Chart – {best_method}")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(table_best[first_col], table_best["Demand"], 'o-', label="Actual Demand", color="blue")
        ax.plot(table_best[first_col], table_best["Forecast"], 's--', label="Forecast", color="red")
        ax.set_title(f"{best_method} vs Actual Demand")
        ax.set_xlabel(period_name)
        ax.set_ylabel("Demand")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.info(f"**{criteria} for {best_method}: {best_error:.4f}**")
        st.divider()
        st.subheader("🔍 View Other Forecasting Methods")
        c1, c2, c3 = st.columns(3)
        with c1:
            show_naive = st.checkbox("Naive", key="chk_naive")
        with c2:
            show_ma = st.checkbox("Moving Average", key="chk_ma")
        with c3:
            show_exp = st.checkbox("Exponential Smoothing", key="chk_exp")
        selected = []
        if show_naive:
            selected.append("Naive")
        if show_ma:
            selected.append("Moving Average")
        if show_exp:
            selected.append("Exponential Smoothing")
        for method in selected:
            if method == best_method:
                continue
            df_o = results[method]
            fc_o = [col for col in df_o.columns if "Forecast" in col][0]
            st.markdown(f"#### {method} Table")
            table_o = df_base[[first_col, "Demand"]].copy()
            table_o["Forecast"] = df_o[fc_o]
            st.dataframe(table_o.style.format("{:.2f}"), use_container_width=True)
        st.divider()
    st.divider()
    if st.button("⬅ Back to Analysis"):
        keys_to_clear = ["forecast_ran", "best_method", "best_error", "all_results", "all_errors", "selected_criteria"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.key
        st.session_state.page = 3
        st.rerun()
# ================= SCREEN 5: EOQ Calculation =================
def page_eoq():
    st.title("📦 Economic Order Quantity (EOQ) & Reorder Point")
    mat = st.session_state.material
    st.markdown(f"**Material:** {mat['family']} / {mat['type']} / {mat['grade']}")
    st.divider()
    st.subheader("EOQ Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        D = st.number_input("Annual Demand (D)", min_value=1.0, value=12000.0, step=100.0)
    with col2:
        S = st.number_input("Ordering Cost per Order (S)", min_value=0.0, value=200.0, step=10.0)
    with col3:
        H = st.number_input("Holding Cost per Unit per Year (H)", min_value=0.01, value=25.0, step=1.0)
    st.divider()
    st.subheader("Reorder Point Parameters")
    col4, col5 = st.columns(2)
    with col4:
        lead_time_days = st.number_input("Lead Time (days)", min_value=1, value=7)
    with col5:
        daily_demand = D / 365
        daily_demand = st.number_input("Average Daily Demand", min_value=0.0, value=round(daily_demand, 2), step=0.1)
    st.divider()
    if st.button("Calculate EOQ & Reorder Point", type="primary", use_container_width=True):
        if H <= 0:
            st.error("Holding cost (H) must be greater than zero.")
        else:
            EOQ = ((2 * D * S) / H) ** 0.5
            reorder_point = daily_demand * lead_time_days
            days_between_orders = EOQ / daily_demand if daily_demand > 0 else 0
            st.success("Calculation completed successfully!")
            col_eoq, col_rop, col_cycle = st.columns(3)
            with col_eoq:
                st.metric("Optimal Order Quantity (EOQ)", f"{EOQ:.2f} units")
            with col_rop:
                st.metric("Reorder Point", f"{reorder_point:.2f} units")
            with col_cycle:
                st.metric("Order Every", f"{days_between_orders:.1f} days")
            st.divider()
            st.subheader("📊 Inventory Level During Lead Time")
            days = list(range(0, int(lead_time_days + 10)))
            inventory_level = [EOQ - daily_demand * d for d in days]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(days, inventory_level, 'o-', label="Inventory Level", color="purple", linewidth=2)
            ax.axhline(y=reorder_point, color="red", linestyle="--", linewidth=2, label=f"Reorder Point ({reorder_point:.2f})")
            ax.axhline(y=0, color="black", linewidth=1)
            ax.set_title("Inventory Level During Lead Time")
            ax.set_xlabel("Days")
            ax.set_ylabel("Inventory Quantity")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    st.divider()
    if st.button("⬅ Back to Analysis"):
        st.session_state.page = 3
        st.rerun()
# ================= SCREEN 6: Safety Stock Calculation =================
def page_safety_stock():
    st.title("🛡️ Safety Stock Calculation")
    mat = st.session_state.material
    st.markdown(f"**Material:** {mat['family']} / {mat['type']} / {mat['grade']}")
    st.divider()
    st.subheader("Choose Safety Stock Method")
    ss_method = st.radio("Select method:", ["Fixed Safety Stock", "Statistical Safety Stock (Service Level)"], horizontal=True)
    st.divider()
    if ss_method == "Fixed Safety Stock":
        st.subheader("Fixed Safety Stock")
        fixed_ss = st.number_input("Enter Safety Stock Quantity (units)", min_value=0.0, value=100.0, step=10.0)
        if st.button("Calculate", type="primary"):
            st.success("Fixed Safety Stock Applied!")
            st.metric("Safety Stock", f"{fixed_ss:.2f} units")
    else:
        st.subheader("Statistical Safety Stock Parameters")
        col1, col2 = st.columns(2)
        with col1:
            service_level = st.slider("Desired Service Level (%)", min_value=80.0, max_value=99.9, value=95.0, step=0.1)
        with col2:
            std_dev_lead_demand = st.number_input("Standard Deviation of Demand During Lead Time", min_value=0.0, value=50.0, step=1.0)
        z_map = {80: 0.84, 90: 1.28, 95: 1.65, 97.5: 1.96, 99: 2.33, 99.9: 3.09}
        z_score = z_map.get(round(service_level), 1.65)
        if st.button("Calculate Safety Stock", type="primary"):
            safety_stock = z_score * std_dev_lead_demand
            st.success("Statistical Safety Stock Calculated!")
            st.metric("Safety Stock", f"{safety_stock:.2f} units")
            st.info(f"Z-Score used for {service_level}% service level: {z_score:.2f}")
    st.divider()
    if st.button("⬅ Back to Analysis"):
        st.session_state.page = 3
        st.rerun()
# ================= Main Navigation =================
if st.session_state.page == 1:
    page_material_selection()
elif st.session_state.page == 2:
    page_selected_material()
elif st.session_state.page == 3:
    page_analysis()
elif st.session_state.page == 4:
    page_forecasting()
elif st.session_state.page == 5:
    page_eoq()
elif st.session_state.page == 6:
    page_safety_stock()
