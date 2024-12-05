from src.utils.rtl_utils import apply_arabic_config
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import plotly.express as px
from datetime import datetime

# تطبيق التكوين العربي
apply_arabic_config(title="المعالجة المسبقة", icon="🔄")

# تكوين النمط
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .preprocessing-step {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    </style>
""", unsafe_allow_html=True)

def load_data():
    """تحميل البيانات من حالة الجلسة"""
    if "data" not in st.session_state or st.session_state.data is None:
        st.error("🚫 يرجى تحميل البيانات أولاً في صفحة إدارة البيانات!")
        st.stop()
    return st.session_state.data.copy()

def log_preprocessing_step(step_name, details):
    """تسجيل خطوة المعالجة"""
    if 'preprocessing_steps' not in st.session_state:
        st.session_state.preprocessing_steps = []
    
    step = {
        'step_name': step_name,
        'details': details,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.preprocessing_steps.append(step)

def handle_missing_values(df, numeric_strategy, categorical_strategy, custom_values=None):
    """معالجة القيم المفقودة"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    # معالجة الأعمدة الرقمية
    if len(numeric_cols) > 0:
        if numeric_strategy == 'custom':
            for col in numeric_cols:
                if col in custom_values:
                    df[col] = df[col].fillna(custom_values[col])
        else:
            num_imputer = SimpleImputer(strategy=numeric_strategy)
            df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    
    # معالجة الأعمدة الفئوية
    if len(categorical_cols) > 0:
        if categorical_strategy == 'custom':
            for col in categorical_cols:
                if col in custom_values:
                    df[col] = df[col].fillna(custom_values[col])
        else:
            cat_imputer = SimpleImputer(strategy=categorical_strategy)
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    return df

def scale_features(df, scaler_type, columns=None):
    """تطبيع البيانات"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    else:
        columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df
    
    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler()
    }
    
    scaler = scalers.get(scaler_type)
    if scaler:
        df[columns] = scaler.fit_transform(df[columns])
    
    return df

def encode_categorical(df, encoding_type, columns=None):
    """ترميز البيانات الفئوية"""
    if columns is None:
        columns = df.select_dtypes(exclude=[np.number]).columns
    else:
        columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df
    
    if encoding_type == "Label Encoding":
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            
    elif encoding_type == "One-Hot Encoding":
        df = pd.get_dummies(df, columns=columns, prefix_sep='_')
    
    return df

def remove_outliers(df, method, columns=None, threshold=1.5):
    """إزالة القيم المتطرفة"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    else:
        columns = [col for col in columns if col in df.columns]
    
    if not columns:
        return df
    
    if method == "IQR":
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
    elif method == "Z-Score":
        for col in columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores <= threshold]
    
    return df

# التخطيط الرئيسي
st.title("🔄 المعالجة المسبقة للبيانات")

# تحميل البيانات
df = load_data()
if df is not None:
    st.write("### 📊 ملخص البيانات")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("عدد الصفوف", f"{df.shape[0]:,}")
    with col2:
        st.metric("عدد الأعمدة", df.shape[1])
    with col3:
        missing = df.isna().sum().sum()
        st.metric("القيم المفقودة", f"{missing:,}")
    
    # خطوات المعالجة
    st.write("### ⚙️ خطوات المعالجة")
    
    with st.expander("📊 معالجة القيم المفقودة", expanded=True):
        st.markdown('<div class="preprocessing-step">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            numeric_strategy = st.selectbox(
                "استراتيجية معالجة القيم الرقمية المفقودة",
                ["mean", "median", "most_frequent", "constant", "custom"],
                format_func=lambda x: {
                    "mean": "المتوسط",
                    "median": "الوسيط",
                    "most_frequent": "القيمة الأكثر تكراراً",
                    "constant": "قيمة ثابتة",
                    "custom": "قيم مخصصة"
                }[x]
            )
        
        with col2:
            categorical_strategy = st.selectbox(
                "استراتيجية معالجة القيم الفئوية المفقودة",
                ["most_frequent", "constant", "custom"],
                format_func=lambda x: {
                    "most_frequent": "القيمة الأكثر تكراراً",
                    "constant": "قيمة ثابتة",
                    "custom": "قيم مخصصة"
                }[x]
            )
        
        custom_values = {}
        if numeric_strategy == 'custom' or categorical_strategy == 'custom':
            st.write("##### القيم المخصصة")
            cols_with_missing = df.columns[df.isnull().any()].tolist()
            for col in cols_with_missing:
                if df[col].dtype in [np.number] and numeric_strategy == 'custom':
                    custom_values[col] = st.number_input(f"قيمة {col}", value=0.0)
                elif categorical_strategy == 'custom':
                    custom_values[col] = st.text_input(f"قيمة {col}", "")
        
        if st.button("تطبيق معالجة القيم المفقودة"):
            df = handle_missing_values(df, numeric_strategy, categorical_strategy, custom_values)
            st.session_state.data = df
            log_preprocessing_step("معالجة القيم المفقودة", {
                "numeric_strategy": numeric_strategy,
                "categorical_strategy": categorical_strategy,
                "custom_values": custom_values
            })
            st.success("✅ تم معالجة القيم المفقودة بنجاح!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("📈 تطبيع البيانات", expanded=False):
        st.markdown('<div class="preprocessing-step">', unsafe_allow_html=True)
        
        scaler_type = st.selectbox(
            "اختر نوع التطبيع",
            ["StandardScaler", "MinMaxScaler", "RobustScaler"],
            format_func=lambda x: {
                "StandardScaler": "التطبيع المعياري",
                "MinMaxScaler": "تطبيع الحد الأدنى-الأقصى",
                "RobustScaler": "التطبيع المتين"
            }[x]
        )
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_cols = st.multiselect(
            "اختر الأعمدة للتطبيع",
            numeric_cols,
            default=list(numeric_cols)
        )
        
        if st.button("تطبيق التطبيع"):
            df = scale_features(df, scaler_type, selected_cols)
            st.session_state.data = df
            log_preprocessing_step("تطبيع البيانات", {
                "scaler_type": scaler_type,
                "columns": selected_cols
            })
            st.success("✅ تم تطبيع البيانات بنجاح!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("🏷️ ترميز البيانات الفئوية", expanded=False):
        st.markdown('<div class="preprocessing-step">', unsafe_allow_html=True)
        
        encoding_type = st.selectbox(
            "اختر نوع الترميز",
            ["Label Encoding", "One-Hot Encoding"],
            format_func=lambda x: {
                "Label Encoding": "الترميز بالتسميات",
                "One-Hot Encoding": "الترميز الأحادي"
            }[x]
        )
        
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        selected_cols = st.multiselect(
            "اختر الأعمدة للترميز",
            categorical_cols,
            default=list(categorical_cols)
        )
        
        if st.button("تطبيق الترميز"):
            df = encode_categorical(df, encoding_type, selected_cols)
            st.session_state.data = df
            log_preprocessing_step("ترميز البيانات", {
                "encoding_type": encoding_type,
                "columns": selected_cols
            })
            st.success("✅ تم ترميز البيانات بنجاح!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("🔍 معالجة القيم المتطرفة", expanded=False):
        st.markdown('<div class="preprocessing-step">', unsafe_allow_html=True)
        
        outlier_method = st.selectbox(
            "اختر طريقة اكتشاف القيم المتطرفة",
            ["IQR", "Z-Score"],
            format_func=lambda x: {
                "IQR": "نطاق الربيعات",
                "Z-Score": "درجة Z"
            }[x]
        )
        
        threshold = st.slider(
            "حد الكشف عن القيم المتطرفة",
            min_value=1.0,
            max_value=5.0,
            value=1.5,
            step=0.1
        )
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_cols = st.multiselect(
            "اختر الأعمدة لمعالجة القيم المتطرفة",
            numeric_cols,
            default=list(numeric_cols)
        )
        
        if st.button("تطبيق معالجة القيم المتطرفة"):
            df = remove_outliers(df, outlier_method, selected_cols, threshold)
            st.session_state.data = df
            log_preprocessing_step("معالجة القيم المتطرفة", {
                "method": outlier_method,
                "threshold": threshold,
                "columns": selected_cols
            })
            st.success("✅ تم معالجة القيم المتطرفة بنجاح!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # عرض سجل المعالجة
    if 'preprocessing_steps' in st.session_state and st.session_state.preprocessing_steps:
        st.write("### 📝 سجل المعالجة")
        for step in st.session_state.preprocessing_steps:
            st.markdown(f"""
            **{step['step_name']}** - {step['timestamp']}
            ```python
            {step['details']}
            ```
            """)
