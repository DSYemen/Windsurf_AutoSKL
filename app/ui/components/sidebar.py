import streamlit as st
from typing import Tuple

class Sidebar:
    @staticmethod
    def show() -> Tuple[str, dict]:
        """عرض القائمة الجانبية وإرجاع الخيارات المحددة"""
        with st.sidebar:
            st.title("🤖 AutoSKL")
            
            # اختيار القسم الرئيسي
            main_section = st.selectbox(
                "القسم الرئيسي",
                options=[
                    "تحليل البيانات",
                    "تدريب النموذج",
                    "إدارة النماذج",
                    "المراقبة والتقارير",
                    "الإعدادات"
                ]
            )
            
            options = {}
            
            if main_section == "تحليل البيانات":
                options['analysis_type'] = st.selectbox(
                    "نوع التحليل",
                    options=[
                        "نظرة عامة",
                        "تحليل المتغيرات",
                        "الارتباطات",
                        "القيم المفقودة",
                        "الرسوم البيانية"
                    ]
                )
                
            elif main_section == "تدريب النموذج":
                options['task_type'] = st.selectbox(
                    "نوع المهمة",
                    options=[
                        "classification",
                        "regression",
                        "clustering",
                        "dimensionality_reduction"
                    ],
                    format_func=lambda x: {
                        'classification': 'تصنيف',
                        'regression': 'انحدار',
                        'clustering': 'تجميع',
                        'dimensionality_reduction': 'تقليل الأبعاد'
                    }[x]
                )
                
                options['optimization_time'] = st.slider(
                    "وقت التحسين (دقائق)",
                    min_value=1,
                    max_value=60,
                    value=10
                )
                
            elif main_section == "إدارة النماذج":
                options['model_action'] = st.selectbox(
                    "الإجراء",
                    options=[
                        "عرض النماذج",
                        "مقارنة النماذج",
                        "تصدير/استيراد",
                        "حذف النماذج"
                    ]
                )
                
            elif main_section == "المراقبة والتقارير":
                options['monitor_type'] = st.selectbox(
                    "نوع المراقبة",
                    options=[
                        "مراقبة الأداء",
                        "تحليل الانحراف",
                        "التقارير",
                        "التنبيهات"
                    ]
                )
                
                options['time_range'] = st.selectbox(
                    "النطاق الزمني",
                    options=[
                        "آخر 24 ساعة",
                        "آخر 7 أيام",
                        "آخر 30 يوم",
                        "آخر 90 يوم"
                    ]
                )
                
            elif main_section == "الإعدادات":
                options['settings_type'] = st.selectbox(
                    "نوع الإعدادات",
                    options=[
                        "إعدادات عامة",
                        "إعدادات قاعدة البيانات",
                        "إعدادات النماذج",
                        "إعدادات المراقبة"
                    ]
                )
            
            # إضافة معلومات النظام
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📊 معلومات النظام")
            
            if 'total_models' in st.session_state:
                st.sidebar.metric(
                    "عدد النماذج",
                    st.session_state.total_models
                )
                
            if 'last_training' in st.session_state:
                st.sidebar.metric(
                    "آخر تدريب",
                    st.session_state.last_training
                )
            
            return main_section, options
