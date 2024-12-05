import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import plotly.express as px

st.set_page_config(page_title="Data Preprocessing", page_icon="🔄", layout="wide")

def load_data():
    if "data" not in st.session_state:
        st.error("Please load data first in the Data Management page!")
        st.stop()
    return st.session_state.data

def handle_missing_values(df, strategy):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    if numeric_cols.any():
        num_imputer = SimpleImputer(strategy=strategy)
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    
    if categorical_cols.any():
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    return df

def scale_features(df, scaler_type):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if scaler_type == "StandardScaler":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def encode_categorical(df, encoding_type):
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    if encoding_type == "Label Encoding":
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    elif encoding_type == "One-Hot Encoding":
        df = pd.get_dummies(df, columns=categorical_cols)
    
    return df

# Main layout
st.title("🔄 Data Preprocessing")

# Load data
df = load_data()

# Sidebar
with st.sidebar:
    st.header("Preprocessing Steps")
    
    # Missing Values
    st.subheader("1. Handle Missing Values")
    missing_strategy = st.selectbox(
        "Strategy",
        ["mean", "median", "most_frequent"],
        key="missing_strategy"
    )
    
    # Feature Scaling
    st.subheader("2. Feature Scaling")
    scaling_method = st.selectbox(
        "Scaling Method",
        ["None", "StandardScaler", "MinMaxScaler"],
        key="scaling_method"
    )
    
    # Categorical Encoding
    st.subheader("3. Categorical Encoding")
    encoding_method = st.selectbox(
        "Encoding Method",
        ["None", "Label Encoding", "One-Hot Encoding"],
        key="encoding_method"
    )

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    # Original data info
    st.metric("Missing Values", df.isna().sum().sum())
    st.metric("Categorical Columns", len(df.select_dtypes(exclude=[np.number]).columns))
    st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))

# Apply preprocessing
processed_df = df.copy()

if st.button("Apply Preprocessing"):
    with st.spinner("Processing data..."):
        # Handle missing values
        processed_df = handle_missing_values(processed_df, missing_strategy)
        
        # Scale features
        if scaling_method != "None":
            processed_df = scale_features(processed_df, scaling_method)
        
        # Encode categorical variables
        if encoding_method != "None":
            processed_df = encode_categorical(processed_df, encoding_method)
        
        with col2:
            st.subheader("Processed Data Preview")
            st.dataframe(processed_df.head(), use_container_width=True)
            
            # Processed data info
            st.metric("Missing Values", processed_df.isna().sum().sum())
            st.metric("Total Features", processed_df.shape[1])
            
            # Distribution comparison
            if scaling_method != "None":
                st.subheader("Feature Distribution After Scaling")
                numeric_col = st.selectbox("Select numeric column", processed_df.select_dtypes(include=[np.number]).columns)
                fig = px.histogram(processed_df, x=numeric_col, title=f"Distribution of {numeric_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        # Save processed data
        st.session_state.processed_data = processed_df
        st.success("Data preprocessing completed! You can now proceed to Model Training.")
