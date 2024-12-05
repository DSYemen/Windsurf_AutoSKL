import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
import os
from src.utils.rtl_utils import apply_arabic_config

# تطبيق التكوين العربي
apply_arabic_config(title="إدارة البيانات", icon="📊")

# تكوين النمط
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    </style>
""", unsafe_allow_html=True)

def load_data():
    """تهيئة حالة الجلسة"""
    if "data" not in st.session_state:
        st.session_state.data = None
        st.session_state.filename = None
        st.session_state.target = None
        st.session_state.data_info = {
            'upload_time': None,
            'last_modified': None,
            'preprocessing_steps': [],
            'data_schema': None
        }

def validate_data(df):
    """التحقق من صحة البيانات"""
    issues = []
    
    # التحقق من القيم المفقودة
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        issues.append(f"القيم المفقودة موجودة في الأعمدة: {', '.join(missing_cols)}")
    
    # التحقق من القيم المتطرفة في الأعمدة الرقمية
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col]
        if len(outliers) > 0:
            issues.append(f"القيم المتطرفة موجودة في العمود {col}: {len(outliers)} قيمة")
    
    # التحقق من تناسق أنواع البيانات
    for col in df.columns:
        unique_values = df[col].nunique()
        if unique_values == 1:
            issues.append(f"العمود {col} يحتوي على قيمة واحدة فقط")
        elif unique_values == len(df) and df[col].dtype == 'object':
            issues.append(f"العمود {col} قد يكون معرف فريد")
    
    return issues

def show_data_info(df):
    """عرض معلومات البيانات"""
    st.subheader("📋 معلومات مجموعة البيانات")
    
    # المقاييس الأساسية
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("عدد الصفوف", f"{df.shape[0]:,}")
    with col2:
        st.metric("عدد الأعمدة", df.shape[1])
    with col3:
        missing = df.isna().sum().sum()
        st.metric("القيم المفقودة", f"{missing:,}")
    with col4:
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("حجم الذاكرة", f"{memory_usage:.2f} MB")

    # معلومات أنواع البيانات
    st.subheader("📊 أنواع البيانات")
    dtypes_df = pd.DataFrame({
        'العمود': df.columns,
        'النوع': df.dtypes.values,
        'القيم المفقودة': df.isna().sum().values,
        'القيم الفريدة': [df[col].nunique() for col in df.columns],
        'العينة': [str(df[col].iloc[0]) if not df[col].empty else '' for col in df.columns]
    })
    st.dataframe(dtypes_df, hide_index=True)

    # عرض الإحصائيات الوصفية
    st.subheader("📈 الإحصائيات الوصفية")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if not numeric_cols.empty:
        stats_df = df[numeric_cols].describe()
        st.dataframe(stats_df)

def plot_distribution(df, column):
    """رسم توزيع البيانات"""
    if pd.api.types.is_numeric_dtype(df[column]):
        fig = px.histogram(
            df, x=column,
            title=f"توزيع {column}",
            template="simple_white",
            marginal="box"
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title=column,
            yaxis_title="التكرار"
        )
    else:
        value_counts = df[column].value_counts()
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"توزيع {column}",
            template="simple_white"
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title=column,
            yaxis_title="العدد"
        )
    return fig

def save_data_info():
    """حفظ معلومات البيانات"""
    if st.session_state.data is not None:
        st.session_state.data_info['upload_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.data_info['last_modified'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # إنشاء مخطط البيانات
        df = st.session_state.data
        schema = {
            'columns': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'rows': len(df),
            'missing_values': df.isna().sum().to_dict(),
            'unique_values': {col: df[col].nunique() for col in df.columns}
        }
        st.session_state.data_info['data_schema'] = schema

def generate_synthetic_data():
    """توليد بيانات اصطناعية للتدريب"""
    st.subheader("🔄 إنشاء بيانات اصطناعية")
    
    data_type = st.selectbox(
        "اختر نوع البيانات",
        ["تصنيف", "انحدار", "تجميع"]
    )
    
    n_samples = st.number_input("عدد العينات", min_value=100, max_value=10000, value=1000, step=100)
    n_features = st.number_input("عدد الخصائص", min_value=2, max_value=20, value=5, step=1)
    
    if st.button("إنشاء البيانات"):
        if data_type == "تصنيف":
            from sklearn.datasets import make_classification
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_redundant=1,
                n_informative=n_features-2,
                random_state=42,
                n_clusters_per_class=2
            )
            feature_names = [f"خاصية_{i+1}" for i in range(n_features)]
            df = pd.DataFrame(X, columns=feature_names)
            df['الهدف'] = y
            
        elif data_type == "انحدار":
            from sklearn.datasets import make_regression
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=0.1,
                random_state=42
            )
            feature_names = [f"خاصية_{i+1}" for i in range(n_features)]
            df = pd.DataFrame(X, columns=feature_names)
            df['الهدف'] = y
            
        else:  # تجميع
            from sklearn.datasets import make_blobs
            X, y = make_blobs(
                n_samples=n_samples,
                n_features=n_features,
                centers=3,
                random_state=42
            )
            feature_names = [f"خاصية_{i+1}" for i in range(n_features)]
            df = pd.DataFrame(X, columns=feature_names)
            df['المجموعة'] = y
        
        st.session_state.data = df
        st.session_state.filename = f"synthetic_data_{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        st.session_state.data_info['upload_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success("تم إنشاء البيانات الاصطناعية بنجاح!")
        show_data_info(df)

# التخطيط الرئيسي
load_data()
st.title("📊 إدارة البيانات")

# القائمة الجانبية لخيارات البيانات
with st.sidebar:
    data_option = st.radio(
        "اختر مصدر البيانات",
        ["تحميل ملف", "مجموعة بيانات نموذجية"]
    )

if data_option == "تحميل ملف":
    uploaded_file = st.file_uploader(
        "اختر ملف البيانات",
        type=['csv', 'xlsx', 'xls'],
        help="يمكنك تحميل ملفات CSV أو Excel"
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.data = df
            st.session_state.filename = uploaded_file.name
            
            # التحقق من صحة البيانات
            issues = validate_data(df)
            if issues:
                st.warning("⚠️ تم اكتشاف المشاكل التالية في البيانات:")
                for issue in issues:
                    st.write(f"- {issue}")
            
            # عرض معلومات البيانات
            show_data_info(df)
            
            # عرض البيانات
            st.subheader("🔍 استعراض البيانات")
            n_rows = st.slider("عدد الصفوف المعروضة", 5, 100, 10)
            st.dataframe(df.head(n_rows))
            
            # تحليل توزيع البيانات
            st.subheader("📊 تحليل التوزيع")
            selected_column = st.selectbox("اختر العمود للتحليل", df.columns)
            fig = plot_distribution(df, selected_column)
            st.plotly_chart(fig, use_container_width=True)
            
            # حفظ معلومات البيانات
            save_data_info()
            
        except Exception as e:
            st.error(f"حدث خطأ أثناء تحميل الملف: {str(e)}")
else:  # مجموعة بيانات نموذجية
    generate_synthetic_data()
