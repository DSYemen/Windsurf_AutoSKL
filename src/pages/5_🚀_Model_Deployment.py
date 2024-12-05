import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime
from src.utils.rtl_utils import apply_arabic_config
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix
)

# تطبيق التكوين العربي
apply_arabic_config(title="نشر النموذج", icon="🚀")

# التحقق من وجود البيانات
if "data" not in st.session_state:
    st.error("🚫 يرجى تحميل البيانات أولاً من صفحة إدارة البيانات!")
    st.stop()

# التحقق من وجود نموذج نشط
if "active_model" not in st.session_state:
    st.error("❌ لم يتم اختيار نموذج نشط!")
    st.warning("🔍 الرجاء اختيار نموذج أولاً من صفحة سجل النماذج")
    st.info("1️⃣ انتقل إلى صفحة سجل النماذج\n2️⃣ اختر النموذج المطلوب\n3️⃣ اضغط على زر 'تنشيط النموذج'")
    st.stop()

try:
    # تحميل النموذج النشط
    active_model = st.session_state.active_model
    model_path = os.path.join("models", active_model['name'])
    
    with st.spinner("جاري تحميل النموذج..."):
        model = joblib.load(model_path)
        model_info = active_model['info']
    
    st.success("✅ تم تحميل النموذج بنجاح!")
    
    # عرض معلومات النموذج
    st.write("### ℹ️ معلومات النموذج النشط")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**نوع النموذج:**", model_info['name'])
        st.write("**نوع المشكلة:**", model_info['type'])
    
    with col2:
        st.write("**المتغير الهدف:**", model_info['target'])
        st.write("**تاريخ التدريب:**", model_info['training_date'])
    
    with col3:
        st.write("**عدد المتغيرات:**", len(model_info['features']))
    
    # تحضير البيانات للتنبؤ
    st.write("### 🎯 التنبؤ")
    
    df = st.session_state.data
    features = model_info['features']
    target = model_info['target']
    
    # التحقق من وجود جميع المتغيرات المطلوبة
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        st.error(f"❌ المتغيرات التالية غير موجودة في البيانات: {', '.join(missing_features)}")
        st.stop()
    
    # إجراء التنبؤ
    X = df[features]
    
    try:
        predictions = model.predict(X)
        
        # إضافة التنبؤات إلى البيانات
        results_df = df.copy()
        results_df['التنبؤات'] = predictions
        
        # عرض النتائج
        st.write("### 📊 نتائج التنبؤ")
        st.dataframe(results_df)
        
        # حساب وعرض مقاييس الأداء إذا كان المتغير الهدف موجوداً
        if target in df.columns:
            st.write("### 📈 تقييم الأداء")
            y_true = df[target]
            
            if model_info['type'] == "تصنيف":
                accuracy = accuracy_score(y_true, predictions)
                precision = precision_score(y_true, predictions, average='weighted')
                recall = recall_score(y_true, predictions, average='weighted')
                f1 = f1_score(y_true, predictions, average='weighted')
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("الدقة", f"{accuracy:.4f}")
                col2.metric("الضبط", f"{precision:.4f}")
                col3.metric("الاسترجاع", f"{recall:.4f}")
                col4.metric("F1", f"{f1:.4f}")
                
                # مصفوفة الارتباك
                cm = confusion_matrix(y_true, predictions)
                fig = px.imshow(
                    cm,
                    labels=dict(x="التنبؤ", y="القيمة الحقيقية"),
                    title="مصفوفة الارتباك"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:  # انحدار
                mse = mean_squared_error(y_true, predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, predictions)
                r2 = r2_score(y_true, predictions)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MSE", f"{mse:.4f}")
                col2.metric("RMSE", f"{rmse:.4f}")
                col3.metric("MAE", f"{mae:.4f}")
                col4.metric("R²", f"{r2:.4f}")
                
                # رسم القيم الحقيقية مقابل المتنبأ بها
                fig = px.scatter(
                    x=y_true,
                    y=predictions,
                    labels={"x": "القيم الحقيقية", "y": "القيم المتنبأ بها"},
                    title="القيم الحقيقية مقابل المتنبأ بها"
                )
                fig.add_trace(
                    go.Scatter(
                        x=[y_true.min(), y_true.max()],
                        y=[y_true.min(), y_true.max()],
                        mode="lines",
                        name="خط التطابق المثالي"
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # تصدير النتائج
        if st.button("💾 تصدير النتائج"):
            try:
                # إنشاء مجلد للنتائج
                results_dir = "results"
                os.makedirs(results_dir, exist_ok=True)
                
                # حفظ النتائج
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_path = os.path.join(results_dir, f"predictions_{timestamp}.csv")
                results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
                
                st.success(f"✅ تم حفظ النتائج في: {results_path}")
                
            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء حفظ النتائج: {str(e)}")
        
    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء إجراء التنبؤ: {str(e)}")

except Exception as e:
    st.error(f"❌ حدث خطأ أثناء تحميل النموذج: {str(e)}")
