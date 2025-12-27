# app.py

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# ================= تحميل الـ CSS الخارجي =================
with open("style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

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

# ================= Load Classification =================
try:
    df_class = pd.read_excel("Database/Classification-of-Material.xlsx")
except Exception as e:
    st.error(f"Cannot read 'Classification-of-Material.xlsx': {e}")
    st.stop()

# ================= Functions =================
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

# باقي الكود زي ما هو تمامًا (جميع الدوال والصفحات)...
# (مش هكرره هنا عشان الرسالة متطولش، لكن هو نفس الكود اللي بعتناه آخر مرة بالضبط)

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

# باقي الصفحات (page_material_selection, page_selected_material, page_analysis, page_forecasting, page_eoq, page_safety_stock)
# نفس الكود اللي عندك بالضبط من غير أي تغيير

# ================= Pages =================
def page_material_selection():
    st.session_state.material = select_material(df_class)
    if st.button("Next ➜", type="primary"):
        st.session_state.page = 2
        st.rerun()

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

def page_analysis():
    st.title("Analysis & Calculations")
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

# باقي الصفحات (forecasting, eoq, safety_stock) زي ما هي

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
