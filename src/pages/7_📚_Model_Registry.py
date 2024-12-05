import os
import streamlit as st
import joblib
import json
import pandas as pd
from datetime import datetime
from src.utils.rtl_utils import apply_arabic_config
import plotly.express as px

# تطبيق التكوين العربي
apply_arabic_config(title="سجل النماذج", icon="📚")

# تحديد مجلد النماذج
models_dir = "models"

# التحقق من وجود مجلد النماذج
if not os.path.exists(models_dir):
    st.error("❌ لم يتم العثور على مجلد النماذج! يرجى تدريب نموذج أولاً.")
    st.stop()

# عرض النموذج النشط إن وجد
if 'active_model' in st.session_state:
    st.sidebar.success(f"✅ النموذج النشط: {st.session_state['active_model']['name']}")
    
    # إضافة زر لإلغاء تنشيط النموذج
    if st.sidebar.button("❌ إلغاء تنشيط النموذج"):
        del st.session_state['active_model']
        st.experimental_rerun()
else:
    st.sidebar.warning("⚠️ لا يوجد نموذج نشط")

# البحث عن ملفات النماذج
model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]

if not model_files:
    st.error("❌ لم يتم العثور على أي نماذج! يرجى تدريب نموذج أولاً.")
    st.stop()

# عرض قائمة النماذج المتاحة
st.write("### 📋 النماذج المتاحة")

# تحميل معلومات جميع النماذج
models_info = []
for model_file in model_files:
    model_path = os.path.join(models_dir, model_file)
    info_path = model_path.replace('.joblib', '_info.json')
    
    try:
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                try:
                    info = json.load(f)
                    # إضافة المعلومات الأساسية إذا كانت مفقودة
                    info['file_name'] = model_file
                    info['name'] = info.get('name', 'غير معروف')
                    info['type'] = info.get('type', 'غير معروف')
                    info['target_name'] = info.get('target_name', 'غير معروف')
                    info['training_date'] = info.get('training_date', 'غير معروف')
                    models_info.append(info)
                except json.JSONDecodeError as e:
                    st.warning(f"⚠️ خطأ في قراءة ملف المعلومات {info_path}: {str(e)}")
                    continue
    except Exception as e:
        st.warning(f"⚠️ خطأ في معالجة النموذج {model_file}: {str(e)}")
        continue

# إنشاء DataFrame لعرض معلومات النماذج
if models_info:
    models_df = pd.DataFrame(models_info)
    
    # تنظيم وتحسين عرض المعلومات
    display_columns = {
        'name': 'نوع النموذج',
        'type': 'نوع المشكلة',
        'target_name': 'المتغير الهدف',
        'training_date': 'تاريخ التدريب',
        'file_name': 'اسم الملف'
    }
    
    # التحقق من وجود الأعمدة المطلوبة وإضافة القيم الافتراضية
    for col in display_columns.keys():
        if col not in models_df.columns:
            models_df[col] = 'غير متوفر'
    
    # عرض معلومات النماذج في جدول
    st.dataframe(
        models_df[display_columns.keys()].rename(columns=display_columns),
        use_container_width=True
    )
    
    # اختيار نموذج للعرض التفصيلي
    selected_model_name = st.selectbox(
        "اختر نموذجاً لعرض التفاصيل",
        models_df['file_name'].tolist(),
        format_func=lambda x: f"{x.replace('.joblib', '')}"
    )
    
    if selected_model_name:
        selected_model = models_df[models_df['file_name'] == selected_model_name].iloc[0]
        
        st.write("### 📊 تفاصيل النموذج")
        
        # عرض المعلومات الأساسية
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**نوع النموذج:**", selected_model.get('name', 'غير متوفر'))
            st.write("**نوع المشكلة:**", selected_model.get('type', 'غير معروف'))
        with col2:
            st.write("**المتغير الهدف:**", selected_model.get('target_name', 'غير متوفر'))
            st.write("**تاريخ التدريب:**", selected_model.get('training_date', 'غير معروف'))
        with col3:
            # التحقق من نوع المتغيرات وعددها
            features = selected_model.get('feature_names', [])
            if isinstance(features, (list, tuple)):
                feature_count = len(features)
            elif isinstance(features, str):
                feature_count = 1
            else:
                feature_count = 0
            st.write("**عدد المتغيرات:**", feature_count if feature_count > 0 else 'غير متوفر')
            
            # عرض معلومات إضافية
            target_stats = selected_model.get('target_statistics')
            if isinstance(target_stats, dict):
                if target_stats.get('is_numeric'):
                    st.write("**نوع البيانات:** عددية")
                else:
                    st.write("**نوع البيانات:** فئوية")
        
        # عرض المتغيرات المستخدمة
        with st.expander("📋 المتغيرات المستخدمة"):
            features = selected_model.get('feature_names', [])
            if isinstance(features, (list, tuple)) and len(features) > 0:
                # عرض المتغيرات في شكل جدول
                feature_df = pd.DataFrame({
                    'المتغير': features
                })
                st.dataframe(feature_df, use_container_width=True)
            else:
                st.info("لا توجد معلومات عن المتغيرات المستخدمة")
        
        # عرض معلومات المتغير الهدف
        with st.expander("🎯 معلومات المتغير الهدف"):
            target_stats = selected_model.get('target_statistics', {})
            if isinstance(target_stats, dict) and target_stats:
                if target_stats.get('is_numeric', False):
                    col1, col2 = st.columns(2)
                    with col1:
                        mean_val = target_stats.get('mean')
                        std_val = target_stats.get('std')
                        if mean_val is not None:
                            st.metric("المتوسط", f"{float(mean_val):.2f}")
                        if std_val is not None:
                            st.metric("الانحراف المعياري", f"{float(std_val):.2f}")
                    with col2:
                        min_val = target_stats.get('min')
                        max_val = target_stats.get('max')
                        if min_val is not None:
                            st.metric("القيمة الدنيا", f"{float(min_val):.2f}")
                        if max_val is not None:
                            st.metric("القيمة العليا", f"{float(max_val):.2f}")
                else:
                    # عرض معلومات الفئات
                    target_values = selected_model.get('target_values', [])
                    if isinstance(target_values, (list, tuple)) and target_values:
                        st.write("**الفئات المتاحة:**")
                        st.write(", ".join(map(str, target_values)))
                    
                    unique_count = target_stats.get('unique_values')
                    if unique_count is not None:
                        st.metric("عدد الفئات الفريدة", int(unique_count))
            else:
                st.info("لا توجد معلومات إحصائية عن المتغير الهدف")
        
        # عرض معاملات النموذج
        with st.expander("⚙️ معاملات النموذج"):
            parameters = selected_model.get('parameters', {})
            if parameters:
                st.json(parameters)
            else:
                st.write("لا توجد معلومات عن المعاملات")
        
        # عرض مقاييس الأداء
        st.write("### 📈 مقاييس الأداء")
        metrics = selected_model.get('metrics', {})
        
        if metrics:
            if selected_model.get('type') == "تصنيف":
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
        
        # تنشيط النموذج
        if st.button("✅ تنشيط النموذج", type="primary"):
            try:
                # التحقق من وجود الملف
                model_path = os.path.join(models_dir, selected_model_name)
                if not os.path.exists(model_path):
                    st.error("❌ ملف النموذج غير موجود!")
                    st.stop()
                
                # محاولة تحميل النموذج للتأكد من صحته
                try:
                    model = joblib.load(model_path)
                except Exception as e:
                    st.error(f"❌ فشل في تحميل النموذج: {str(e)}")
                    st.stop()
                
                # حفظ معلومات النموذج النشط في session_state
                st.session_state['active_model'] = {
                    'name': selected_model_name,
                    'path': model_path,
                    'info': selected_model.to_dict(),
                    'type': selected_model.get('type', 'غير معروف'),
                    'features': selected_model.get('features', []),
                    'target': selected_model.get('target_name', 'غير معروف')
                }
                
                st.success(f"✅ تم تنشيط النموذج: {selected_model_name}")
                
                # عرض معلومات إضافية
                st.info("""
                🔍 تم تنشيط النموذج بنجاح! يمكنك الآن:
                1. الانتقال إلى صفحة التنبؤات لاستخدام النموذج
                2. تجربة النموذج على بيانات جديدة
                3. مشاهدة نتائج التنبؤ
                """)
                
                # إضافة زر للانتقال إلى صفحة التنبؤات
                if st.button("🚀 الانتقال إلى صفحة التنبؤات"):
                    st.switch_page("pages/4_🔮_Predictions.py")
                
            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء تنشيط النموذج: {str(e)}")
        
        # حذف النموذج
        if st.button("🗑️ حذف النموذج", type="secondary"):
            try:
                # حذف ملف النموذج
                model_path = os.path.join(models_dir, selected_model_name)
                info_path = model_path.replace('.joblib', '_info.json')
                
                os.remove(model_path)
                if os.path.exists(info_path):
                    os.remove(info_path)
                
                st.success("✅ تم حذف النموذج بنجاح!")
                st.experimental_rerun()
                
            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء حذف النموذج: {str(e)}")
else:
    st.warning("⚠️ لم يتم العثور على معلومات النماذج. تأكد من وجود ملفات المعلومات JSON مع النماذج.")
