from src.utils.rtl_utils import apply_arabic_config
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from pathlib import Path
import base64
from jinja2 import Template
import joblib
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

# تطبيق التكوين العربي
apply_arabic_config(title="تقارير النموذج", icon="📊")

# تكوين النمط
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .report-section {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        background-color: #ffffff;
    }
    .report-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def get_model_info(model_file):
    """الحصول على معلومات النموذج"""
    info_path = model_file.replace('.joblib', '_info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def generate_metrics_section(metrics, problem_type):
    """توليد قسم المقاييس"""
    html = "<div class='report-section'><h3>📈 مقاييس الأداء</h3>"
    
    if problem_type == "تصنيف":
        if metrics.get('accuracy'):
            html += f"<p><strong>الدقة:</strong> {metrics['accuracy']:.4f}</p>"
        if metrics.get('precision'):
            html += f"<p><strong>الضبط:</strong> {metrics['precision']:.4f}</p>"
        if metrics.get('recall'):
            html += f"<p><strong>الاسترجاع:</strong> {metrics['recall']:.4f}</p>"
        if metrics.get('f1'):
            html += f"<p><strong>F1:</strong> {metrics['f1']:.4f}</p>"
    else:
        if metrics.get('mse'):
            html += f"<p><strong>MSE:</strong> {metrics['mse']:.4f}</p>"
        if metrics.get('rmse'):
            html += f"<p><strong>RMSE:</strong> {metrics['rmse']:.4f}</p>"
        if metrics.get('mae'):
            html += f"<p><strong>MAE:</strong> {metrics['mae']:.4f}</p>"
        if metrics.get('r2'):
            html += f"<p><strong>R²:</strong> {metrics['r2']:.4f}</p>"
    
    html += "</div>"
    return html

def generate_model_info_section(info):
    """توليد قسم معلومات النموذج"""
    html = "<div class='report-section'><h3>📋 معلومات النموذج</h3>"
    html += f"<p><strong>نوع النموذج:</strong> {info['name']}</p>"
    html += f"<p><strong>نوع المشكلة:</strong> {info['type']}</p>"
    html += f"<p><strong>المتغير الهدف:</strong> {info['target']}</p>"
    html += f"<p><strong>تاريخ التدريب:</strong> {info['training_date']}</p>"
    html += f"<p><strong>عدد المتغيرات:</strong> {len(info['features'])}</p>"
    html += "</div>"
    return html

def generate_features_section(features):
    """توليد قسم المتغيرات"""
    html = "<div class='report-section'><h3>📋 المتغيرات المستخدمة</h3>"
    html += "<p>" + ", ".join(features) + "</p>"
    html += "</div>"
    return html

def generate_parameters_section(parameters):
    """توليد قسم المعاملات"""
    html = "<div class='report-section'><h3>⚙️ معاملات النموذج</h3>"
    for key, value in parameters.items():
        html += f"<p><strong>{key}:</strong> {value}</p>"
    html += "</div>"
    return html

def generate_report_html(model_info):
    """توليد التقرير بصيغة HTML"""
    html = """
    <div class='report-header'>
        <h1>تقرير النموذج</h1>
        <p>تم إنشاؤه في {date}</p>
    </div>
    """.format(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    html += generate_model_info_section(model_info)
    html += generate_metrics_section(model_info['metrics'], model_info['type'])
    html += generate_features_section(model_info['features'])
    html += generate_parameters_section(model_info['parameters'])
    
    return html

def get_html_download_link(html, filename):
    """إنشاء رابط لتحميل ملف HTML"""
    b64 = base64.b64encode(html.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">تحميل التقرير</a>'
    return href

# التخطيط الرئيسي
st.title("📊 تقارير النماذج")

# التحقق من وجود نموذج نشط
if 'active_model' not in st.session_state:
    st.warning("⚠️ الرجاء اختيار نموذج أولاً من صفحة سجل النماذج")
    if st.button("📚 الانتقال إلى سجل النماذج"):
        st.switch_page("pages/7_📚_Model_Registry.py")
    st.stop()

# الحصول على معلومات النموذج النشط
active_model = st.session_state['active_model']
model_path = active_model['path']
model_info = active_model['info']

if model_info:
    st.write("### 📋 معلومات النموذج")
    
    # عرض المعلومات الأساسية
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**نوع النموذج:**", model_info['name'])
        st.write("**نوع المشكلة:**", model_info['type'])
    with col2:
        st.write("**المتغير الهدف:**", model_info['target'])
        st.write("**تاريخ التدريب:**", model_info['training_date'])
    with col3:
        st.write("**عدد المتغيرات:**", len(model_info['features']))
    
    # عرض المقاييس
    st.write("### 📈 مقاييس الأداء")
    metrics = model_info.get('metrics', {})
    
    if model_info['type'] == "تصنيف":
        col1, col2, col3, col4 = st.columns(4)
        if metrics.get('accuracy'):
            col1.metric("الدقة", f"{metrics['accuracy']:.4f}")
        if metrics.get('precision'):
            col2.metric("الضبط", f"{metrics['precision']:.4f}")
        if metrics.get('recall'):
            col3.metric("الاسترجاع", f"{metrics['recall']:.4f}")
        if metrics.get('f1'):
            col4.metric("F1", f"{metrics['f1']:.4f}")
    else:
        col1, col2, col3, col4 = st.columns(4)
        if metrics.get('mse'):
            col1.metric("MSE", f"{metrics['mse']:.4f}")
        if metrics.get('rmse'):
            col2.metric("RMSE", f"{metrics['rmse']:.4f}")
        if metrics.get('mae'):
            col3.metric("MAE", f"{metrics['mae']:.4f}")
        if metrics.get('r2'):
            col4.metric("R²", f"{metrics['r2']:.4f}")
    
    # توليد التقرير
    if st.button("📄 إنشاء التقرير"):
        report_html = generate_report_html(model_info)
        st.markdown(
            get_html_download_link(report_html, f"model_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"),
            unsafe_allow_html=True
        )
        st.success("✅ تم إنشاء التقرير بنجاح! انقر على الرابط أعلاه للتحميل.")
else:
    st.error("❌ حدث خطأ في تحميل معلومات النموذج!")
