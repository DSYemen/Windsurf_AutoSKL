import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import json
from datetime import datetime

class ReportGenerator:
    def __init__(self):
        self.report_data = {}
        
    def add_model_performance(self, metrics: Dict[str, float], model_name: str):
        """إضافة بيانات أداء النموذج إلى التقرير"""
        self.report_data['model_performance'] = {
            'model_name': model_name,
            'metrics': metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    def add_feature_importance(self, importance_dict: Dict[str, float]):
        """إضافة بيانات أهمية المتغيرات إلى التقرير"""
        self.report_data['feature_importance'] = importance_dict
        
    def add_data_analysis(self, analysis_results: Dict[str, Any]):
        """إضافة نتائج تحليل البيانات إلى التقرير"""
        self.report_data['data_analysis'] = analysis_results
        
    def generate_report(self):
        """توليد تقرير تفاعلي"""
        st.title("📊 تقرير تحليل النموذج")
        
        # عرض معلومات النموذج
        if 'model_performance' in self.report_data:
            st.header("🎯 أداء النموذج")
            model_data = self.report_data['model_performance']
            
            st.write(f"اسم النموذج: {model_data['model_name']}")
            st.write(f"تاريخ التقرير: {model_data['timestamp']}")
            
            # عرض المقاييس في جدول
            metrics_df = pd.DataFrame([model_data['metrics']])
            st.dataframe(metrics_df)
            
            # رسم بياني للمقاييس
            fig = go.Figure(data=[
                go.Bar(
                    x=list(model_data['metrics'].keys()),
                    y=list(model_data['metrics'].values())
                )
            ])
            fig.update_layout(title="مقاييس أداء النموذج")
            st.plotly_chart(fig)
            
        # عرض أهمية المتغيرات
        if 'feature_importance' in self.report_data:
            st.header("📈 أهمية المتغيرات")
            importance_df = pd.DataFrame({
                'المتغير': list(self.report_data['feature_importance'].keys()),
                'الأهمية': list(self.report_data['feature_importance'].values())
            }).sort_values('الأهمية', ascending=False)
            
            fig = px.bar(
                importance_df,
                x='المتغير',
                y='الأهمية',
                title="ترتيب المتغيرات حسب الأهمية"
            )
            st.plotly_chart(fig)
            
        # عرض تحليل البيانات
        if 'data_analysis' in self.report_data:
            st.header("📊 تحليل البيانات")
            analysis = self.report_data['data_analysis']
            
            if 'missing_values' in analysis:
                st.subheader("القيم المفقودة")
                missing_df = pd.DataFrame(analysis['missing_values'])
                st.dataframe(missing_df)
                
            if 'correlations' in analysis:
                st.subheader("مصفوفة الارتباط")
                fig = px.imshow(
                    analysis['correlations'],
                    title="مصفوفة الارتباط بين المتغيرات"
                )
                st.plotly_chart(fig)
                
            if 'distributions' in analysis:
                st.subheader("توزيع المتغيرات")
                for var, dist in analysis['distributions'].items():
                    fig = px.histogram(
                        x=dist,
                        title=f"توزيع {var}"
                    )
                    st.plotly_chart(fig)
    
    def export_report(self) -> Dict:
        """تصدير التقرير كقاموس JSON"""
        return json.dumps(self.report_data, ensure_ascii=False)
        
    def load_report(self, report_json: str):
        """تحميل تقرير من ملف JSON"""
        self.report_data = json.loads(report_json)
