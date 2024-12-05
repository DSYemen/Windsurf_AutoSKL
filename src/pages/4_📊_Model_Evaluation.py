from src.utils.rtl_utils import apply_arabic_config
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, roc_curve, auc, silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
import joblib
import json
import os
from datetime import datetime
from sklearn.metrics import precision_recall_curve, classification_report

# دوال الرسم البياني
def plot_confusion_matrix(cm, labels=None):
    """رسم مصفوفة الارتباك"""
    if labels is None:
        labels = ['0', '1']
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='RdBu',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="مصفوفة الارتباك",
        xaxis_title="التنبؤات",
        yaxis_title="القيم الحقيقية",
        width=600,
        height=600
    )
    
    return fig

def plot_roc_curve(y_true, y_prob):
    """رسم منحنى ROC"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC (AUC = {roc_auc:.3f})'
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='منحنى ROC',
        xaxis_title='معدل الإيجابيات الخاطئة',
        yaxis_title='معدل الإيجابيات الصحيحة',
        width=700,
        height=500
    )
    
    return fig

def plot_precision_recall_curve(y_true, y_prob):
    """رسم منحنى الدقة-الاسترجاع"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        name='منحنى الدقة-الاسترجاع',
        mode='lines'
    ))

    fig.update_layout(
        title='منحنى الدقة-الاسترجاع',
        xaxis_title='الاسترجاع',
        yaxis_title='الدقة',
        width=700,
        height=500,
        hovermode='closest'
    )
    return fig

def plot_residuals(y_true, y_pred):
    """رسم البواقي"""
    residuals = y_true - y_pred
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(
            size=8,
            color='blue',
            opacity=0.6
        ),
        name='البواقي'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title='مخطط البواقي',
        xaxis_title='القيم المتنبأ بها',
        yaxis_title='البواقي',
        width=700,
        height=500,
        showlegend=True
    )
    
    return fig

def plot_actual_vs_predicted(y_true, y_pred):
    """رسم القيم الحقيقية مقابل المتنبأ بها"""
    fig = go.Figure()
    
    # إضافة نقاط البيانات
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(
            size=8,
            color='blue',
            opacity=0.6
        ),
        name='البيانات'
    ))
    
    # إضافة خط التطابق المثالي
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='التطابق المثالي'
    ))
    
    fig.update_layout(
        title='القيم الحقيقية مقابل المتنبأ بها',
        xaxis_title='القيم الحقيقية',
        yaxis_title='القيم المتنبأ بها',
        width=700,
        height=500,
        showlegend=True
    )
    
    return fig

def plot_feature_importance(model, feature_names):
    """رسم أهمية المتغيرات"""
    # الحصول على أهمية المتغيرات
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
    else:
        return None
    
    # ترتيب المتغيرات حسب الأهمية
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    feature_importance = feature_importance.sort_values('importance', ascending=True)
    
    # إنشاء الرسم البياني
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=feature_importance['feature'],
        x=feature_importance['importance'],
        orientation='h',
        marker_color='blue'
    ))
    
    fig.update_layout(
        title='أهمية المتغيرات',
        xaxis_title='الأهمية',
        yaxis_title='المتغيرات',
        width=800,
        height=max(400, len(feature_names) * 25),
        showlegend=False
    )
    
    return fig

# تطبيق التكوين العربي
apply_arabic_config(title="تقييم النموذج", icon="📊")

# التحقق من وجود النموذج والبيانات
def load_model_and_data():
    try:
        # التحقق من وجود النموذج النشط
        if 'active_model' not in st.session_state:
            st.warning("⚠️ الرجاء اختيار نموذج أولاً من صفحة سجل النماذج")
            if st.button("📚 الانتقال إلى سجل النماذج"):
                st.switch_page("pages/7_📚_Model_Registry.py")
            return None, None, None
            
        active_model = st.session_state.active_model
        model_path = active_model['path']
        
        if not os.path.exists(model_path):
            st.error("❌ النموذج المحدد غير موجود")
            return None, None, None
            
        # تحميل النموذج ومعلوماته
        try:
            model = joblib.load(model_path)
            model_info = active_model['info']
        except Exception as e:
            st.error(f"❌ فشل في تحميل النموذج: {str(e)}")
            return None, None, None
            
        # التحقق من وجود البيانات
        if 'data' not in st.session_state:
            st.warning("⚠️ الرجاء تحميل البيانات أولاً من صفحة إدارة البيانات")
            if st.button("📊 الانتقال إلى إدارة البيانات"):
                st.switch_page("pages/1_📊_Data_Management.py")
            return None, None, None
            
        df = st.session_state.data
        
        # التحقق من وجود الأعمدة المطلوبة
        required_features = model_info.get('features', [])
        if not required_features:
            required_features = model_info.get('feature_names', [])
            
        if not all(col in df.columns for col in required_features):
            st.error("❌ البيانات المحملة لا تحتوي على جميع المتغيرات المطلوبة للنموذج")
            st.write("المتغيرات المطلوبة:", ", ".join(required_features))
            st.write("المتغيرات الموجودة:", ", ".join(df.columns))
            return None, None, None
            
        return model, model_info, df
        
    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء تحميل النموذج والبيانات: {str(e)}")
        return None, None, None

# تحميل النموذج والبيانات
model, model_info, df = load_model_and_data()

# التحقق من صحة النموذج والبيانات
if model is not None and model_info is not None and df is not None and not df.empty:
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
        .stAlert {
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # التخطيط الرئيسي
    st.title("📊 تقييم النموذج")

    # تحضير البيانات للتقييم
    features = model_info.get('features', [])
    if not features:
        features = model_info.get('feature_names', [])
        
    X = df[features]
    target = model_info.get('target', '')
    if not target:
        target = model_info.get('target_name', '')
        
    if target in df.columns:
        y_true = df[target]
    else:
        st.warning("⚠️ عمود الهدف غير موجود في البيانات")
        y_true = None

    # التنبؤ
    if y_true is not None:
        y_pred = model.predict(X)

        # حساب وعرض المقاييس حسب نوع التعلم
        model_type = model_info.get('type', '').lower()
        
        if model_type == 'تصنيف' or model_type == 'classification':
            # مقاييس التصنيف
            metrics = {
                'الدقة': accuracy_score(y_true, y_pred),
                'الضبط': precision_score(y_true, y_pred, average='weighted'),
                'الاسترجاع': recall_score(y_true, y_pred, average='weighted'),
                'F1': f1_score(y_true, y_pred, average='weighted')
            }

            # عرض المقاييس
            st.write("### 📈 مقاييس الأداء")
            cols = st.columns(len(metrics))
            for col, (metric_name, value) in zip(cols, metrics.items()):
                col.metric(metric_name, f"{value:.4f}")

            # مصفوفة الارتباك
            cm = confusion_matrix(y_true, y_pred)
            st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)

            # منحنيات ROC و Precision-Recall للتصنيف الثنائي
            if len(np.unique(y_true)) == 2 and hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X)[:, 1]
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_roc_curve(y_true, y_prob), use_container_width=True)
                with col2:
                    st.plotly_chart(plot_precision_recall_curve(y_true, y_prob), use_container_width=True)

            # تقرير التصنيف
            st.write("### 📑 تقرير التصنيف التفصيلي")
            report = classification_report(y_true, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
        elif model_type == 'انحدار' or model_type == 'regression':
            # مقاييس الانحدار
            metrics = {
                'R²': r2_score(y_true, y_pred),
                'MSE': mean_squared_error(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'MAE': mean_absolute_error(y_true, y_pred)
            }

            # عرض المقاييس
            st.write("### 📈 مقاييس الأداء")
            cols = st.columns(len(metrics))
            for col, (metric_name, value) in zip(cols, metrics.items()):
                col.metric(metric_name, f"{value:.4f}")

            # رسوم بيانية للتحليل
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_actual_vs_predicted(y_true, y_pred), use_container_width=True)
            with col2:
                st.plotly_chart(plot_residuals(y_true, y_pred), use_container_width=True)
                
        elif model_type == 'تجميع' or model_type == 'clustering':
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            # مقاييس التجميع
            try:
                metrics = {
                    'معامل سيلويت': silhouette_score(X, y_pred),
                    'معامل كالينسكي-هارباز': calinski_harabasz_score(X, y_pred),
                    'معامل ديفيز-بولدن': davies_bouldin_score(X, y_pred)
                }
            except Exception as e:
                st.error(f"❌ خطأ في حساب مقاييس التجميع: {str(e)}")
                metrics = {}

            if metrics:
                # عرض المقاييس
                st.write("### 📈 مقاييس الأداء")
                cols = st.columns(len(metrics))
                for col, (metric_name, value) in zip(cols, metrics.items()):
                    col.metric(metric_name, f"{value:.4f}")

            # عرض توزيع المجموعات
            st.write("### 🎯 توزيع المجموعات")
            cluster_counts = pd.Series(y_pred).value_counts().sort_index()
            fig = go.Figure(data=[
                go.Bar(x=[f"مجموعة {i}" for i in cluster_counts.index],
                      y=cluster_counts.values)
            ])
            fig.update_layout(
                title="توزيع النقاط على المجموعات",
                xaxis_title="المجموعة",
                yaxis_title="عدد النقاط",
                width=700,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error(f"❌ نوع التعلم غير معروف: {model_type}")
            st.stop()

        # تحليل الأخطاء (للتصنيف والانحدار فقط)
        if model_type in ['تصنيف', 'classification', 'انحدار', 'regression']:
            st.write("### 🔍 تحليل الأخطاء")
            error_df = pd.DataFrame({
                'القيم الحقيقية': y_true,
                'القيم المتنبأ بها': y_pred,
                'الخطأ': np.abs(y_true - y_pred) if model_type in ['انحدار', 'regression'] else y_true != y_pred
            })
            error_df = error_df.sort_values('الخطأ', ascending=False).head(10)
            st.dataframe(error_df)

        # أهمية المتغيرات (إذا كان متاحاً)
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            st.write("### 🎯 أهمية المتغيرات")
            feature_importance_fig = plot_feature_importance(model, features)
            if feature_importance_fig:
                st.plotly_chart(feature_importance_fig, use_container_width=True)

        # تصدير النتائج
        if st.button("📥 تصدير نتائج التقييم"):
            evaluation_results = {
                'model_info': model_info,
                'metrics': metrics,
                'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'error_analysis': error_df.to_dict()
            }

            results_path = os.path.join(
                "models",
                f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=4)

            st.success(f"✅ تم حفظ نتائج التقييم في: {results_path}")
