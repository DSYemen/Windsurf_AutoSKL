import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from ...core.database import DatabaseManager

class ModelManager:
    def __init__(self):
        self.db_manager = DatabaseManager()
        
    def show_model_history(self, model_name: str):
        """عرض تاريخ تدريب النموذج"""
        history = self.db_manager.get_model_history(model_name)
        if not history:
            st.warning("لم يتم العثور على نماذج مدربة بهذا الاسم")
            return
            
        # عرض جدول تاريخ النماذج
        history_df = pd.DataFrame(history)
        history_df['created_at'] = pd.to_datetime(history_df['created_at'])
        history_df['created_at'] = history_df['created_at'].dt.strftime('%Y-%m-%d %H:%M')
        
        st.subheader("📊 تاريخ النماذج المدربة")
        st.dataframe(
            history_df[['version', 'created_at']].rename(columns={
                'version': 'الإصدار',
                'created_at': 'تاريخ الإنشاء'
            })
        )
        
        # عرض مقارنة الأداء
        st.subheader("📈 مقارنة أداء النماذج")
        metrics_df = pd.DataFrame([
            {
                'الإصدار': h['version'],
                **h['metrics']
            } for h in history
        ])
        
        # رسم بياني للمقاييس
        for metric in metrics_df.columns[1:]:
            fig = px.line(
                metrics_df,
                x='الإصدار',
                y=metric,
                title=f"تطور {metric} عبر الإصدارات",
                markers=True
            )
            st.plotly_chart(fig)
            
    def show_model_details(self, model_id: int):
        """عرض تفاصيل النموذج المحدد"""
        model, preprocessing_params, metrics = self.db_manager.load_model(model_id)
        
        # عرض المقاييس
        st.subheader("📊 مقاييس الأداء")
        metrics_df = pd.DataFrame([metrics])
        st.dataframe(metrics_df)
        
        # عرض المعلمات
        st.subheader("⚙️ معلمات النموذج")
        if hasattr(model, 'get_params'):
            params = model.get_params()
            params_df = pd.DataFrame([params])
            st.dataframe(params_df)
            
        # عرض معلمات المعالجة المسبقة
        if preprocessing_params:
            st.subheader("🔧 معلمات المعالجة المسبقة")
            st.json(preprocessing_params)
            
        # عرض أهمية المتغيرات إذا كانت متوفرة
        if hasattr(model, 'feature_importances_'):
            st.subheader("🎯 أهمية المتغيرات")
            importances = pd.DataFrame({
                'المتغير': [f"feature_{i}" for i in range(len(model.feature_importances_))],
                'الأهمية': model.feature_importances_
            }).sort_values('الأهمية', ascending=False)
            
            fig = px.bar(
                importances,
                x='المتغير',
                y='الأهمية',
                title="أهمية المتغيرات في النموذج"
            )
            st.plotly_chart(fig)
            
    def delete_model(self, model_id: int):
        """حذف نموذج محدد"""
        if st.button("حذف النموذج", key=f"delete_model_{model_id}"):
            confirm = st.checkbox("هل أنت متأكد من حذف هذا النموذج؟")
            if confirm:
                self.db_manager.delete_model(model_id)
                st.success("تم حذف النموذج بنجاح")
                
    def export_model(self, model_id: int):
        """تصدير النموذج"""
        model, preprocessing_params, _ = self.db_manager.load_model(model_id)
        if st.button("تصدير النموذج", key=f"export_model_{model_id}"):
            # تنفيذ عملية التصدير
            pass
