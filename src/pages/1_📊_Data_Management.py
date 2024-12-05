import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Data Management", page_icon="📊", layout="wide")

def load_data():
    if "data" not in st.session_state:
        st.session_state.data = None
        st.session_state.filename = None
        st.session_state.target = None

def show_data_info(df):
    st.subheader("Dataset Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isna().sum().sum())

    # Data types information
    st.subheader("Data Types")
    dtypes_df = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Missing': df.isna().sum().values,
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(dtypes_df, hide_index=True)

def plot_distribution(df, column):
    if pd.api.types.is_numeric_dtype(df[column]):
        fig = px.histogram(df, x=column, title=f"Distribution of {column}")
    else:
        value_counts = df[column].value_counts()
        fig = px.bar(x=value_counts.index, y=value_counts.values, title=f"Distribution of {column}")
    return fig

# Main layout
st.title("📊 Data Management")

# Sidebar
with st.sidebar:
    st.header("Data Options")
    upload_type = st.radio("Select Upload Type", ["Upload File", "Sample Dataset"])

# Initialize session state
load_data()

# Main content
if upload_type == "Upload File":
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.data = df
            st.session_state.filename = uploaded_file.name
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
else:
    st.info("Sample datasets coming soon!")

if st.session_state.data is not None:
    df = st.session_state.data
    
    # Data Preview
    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    # Dataset Information
    show_data_info(df)
    
    # Data Visualization
    st.subheader("Data Visualization")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_column = st.selectbox("Select Column for Distribution", df.columns)
        if selected_column:
            st.plotly_chart(plot_distribution(df, selected_column), use_container_width=True)
    
    with col2:
        if df.select_dtypes(include=[np.number]).columns.any():
            num_cols = df.select_dtypes(include=[np.number]).columns
            corr_col = st.selectbox("Select Column for Correlation", num_cols)
            if corr_col:
                correlations = df[num_cols].corr()[corr_col].sort_values(ascending=False)
                fig = px.bar(
                    x=correlations.index,
                    y=correlations.values,
                    title=f"Correlations with {corr_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Target Variable Selection
    st.subheader("Target Variable Selection")
    target_col = st.selectbox("Select Target Variable", ["None"] + list(df.columns))
    if target_col != "None":
        st.session_state.target = target_col
        st.success(f"Target variable set to: {target_col}")
    
    # Save processed data
    if st.button("Save and Continue to Preprocessing"):
        st.session_state.data = df
        st.success("Data saved! You can now proceed to the Preprocessing page.")
