import streamlit as st

def apply_rtl_style():
    """تطبيق نمط RTL على الصفحة"""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@200;300;400;600;700;900&display=swap');
        
        /* التنسيق الأساسي */
        body {
            direction: rtl;
            text-align: right;
            font-family: 'Cairo', sans-serif !important;
        }
        
        /* تنسيق العناصر الرئيسية */
        .main { 
            padding: 0rem 1rem;
            direction: rtl;
        }
        
        /* تنسيق البطاقات والأقسام */
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            text-align: right;
        }
        
        .section-card {
            border: 1px solid #e0e0e0;
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin: 1rem 0;
            background-color: #ffffff;
            text-align: right;
        }
        
        /* تنسيق الأزرار */
        .stButton > button {
            float: right;
            margin-left: 1rem;
        }
        
        /* تنسيق شريط التقدم */
        .stProgress > div > div > div {
            direction: ltr;
        }
        
        /* تنسيق المخططات */
        .plot-container {
            direction: ltr;
        }
        
        /* تنسيق الجداول */
        .dataframe {
            text-align: right !important;
        }
        
        /* تنسيق القوائم المنسدلة */
        .streamlit-expanderHeader {
            text-align: right !important;
        }
        
        /* تنسيق التنبيهات */
        .stAlert {
            text-align: right !important;
        }
        
        /* تنسيق العناوين */
        h1, h2, h3, h4, h5, h6 {
            text-align: right !important;
        }
        
        /* تنسيق المؤشرات */
        .stMetric {
            text-align: right !important;
        }
        
        /* تنسيق علامات التبويب */
        .stTabs [data-baseweb="tab-list"] {
            direction: rtl;
        }
        
        /* تنسيق الشريط الجانبي */
        section[data-testid="stSidebar"] {
            direction: rtl;
            text-align: right;
        }
        
        /* تنسيق عناصر التحكم */
        .stSlider, .stCheckbox, .stSelectbox, .stTextInput, .stNumberInput {
            direction: rtl;
            text-align: right;
        }
        
        /* تنسيق التحميل */
        .uploadedFile {
            direction: rtl;
            text-align: right;
        }
        
        /* تنسيق الرسائل */
        .element-container {
            direction: rtl;
            text-align: right;
        }
        </style>
    """, unsafe_allow_html=True)

def apply_arabic_config(title="التعلم الآلي التلقائي", icon="⚡"):
    """تطبيق التكوين العربي للصفحة"""
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    apply_rtl_style()
