import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from app.core.database import DatabaseManager
from app.services.model_trainer import AutoMLModelTrainer
from app.services.data_analyzer import DataAnalyzer
from app.services.model_monitor import ModelMonitor
from app.services.report_generator import ReportGenerator
from app.ui.components.model_manager import ModelManager
from app.ui.components.sidebar import Sidebar

class Dashboard:
    def __init__(self):
        """تهيئة لوحة التحكم"""
        self.initialize_session_state()
        self.model_manager = ModelManager()
        self.model_monitor = ModelMonitor()
        self.report_generator = ReportGenerator(output_dir="reports")
        self.data_analyzer = DataAnalyzer()
        self.model_trainer = AutoMLModelTrainer()
        self.db_manager = DatabaseManager()
        
    def initialize_session_state(self):
        """تهيئة حالة الجلسة"""
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'current_model' not in st.session_state:
            st.session_state.current_model = None
        if 'preprocessing_state' not in st.session_state:
            st.session_state.preprocessing_state = {}
        if 'target' not in st.session_state:
            st.session_state.target = None
        if 'features' not in st.session_state:
            st.session_state.features = None
        if 'task_type' not in st.session_state:
            st.session_state.task_type = None
        if 'model_results' not in st.session_state:
            st.session_state.model_results = None
        if 'saved_models' not in st.session_state:
            st.session_state.saved_models = None
        if 'monitoring_initialized' not in st.session_state:
            st.session_state.monitoring_initialized = False
            
    def _initialize_monitoring_data(self):
        """تهيئة بيانات المراقبة التجريبية"""
        if not st.session_state.monitoring_initialized:
            try:
                # Get all model IDs
                model_ids = self.db_manager.get_all_model_ids()
                if not model_ids:
                    return
                    
                # Generate sample data for the past 30 days
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                dates = pd.date_range(start=start_date, end=end_date, freq='H')
                
                for model_id in model_ids:
                    # Generate performance metrics
                    for date in dates:
                        metrics = {
                            'accuracy': np.random.uniform(0.8, 0.95),
                            'precision': np.random.uniform(0.75, 0.9),
                            'recall': np.random.uniform(0.7, 0.85),
                            'f1_score': np.random.uniform(0.75, 0.9)
                        }
                        self.db_manager.log_model_performance(model_id, metrics)
                        
                        # Generate predictions
                        features = {f'feature_{i}': np.random.random() for i in range(5)}
                        prediction = np.random.randint(0, 2)
                        actual = np.random.randint(0, 2)
                        self.db_manager.log_prediction(model_id, prediction, features, actual)
                        
                        # Generate resource usage
                        self.db_manager.log_resource_usage(
                            memory_usage=np.random.uniform(20, 80),
                            cpu_usage=np.random.uniform(10, 90),
                            disk_usage=np.random.uniform(30, 70)
                        )
                
                st.session_state.monitoring_initialized = True
                st.success("تم تهيئة بيانات المراقبة بنجاح")
            except Exception as e:
                st.error(f"خطأ في تهيئة بيانات المراقبة: {str(e)}")
                
    def load_and_process_data(self, file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """تحميل ومعالجة البيانات"""
        try:
            if file is None:
                return None, "الرجاء تحميل ملف البيانات"
            
            # قراءة البيانات
            if file.name.endswith('.csv'):
                data = pd.read_csv(file)
            elif file.name.endswith(('.xls', '.xlsx')):
                data = pd.read_excel(file)
            else:
                return None, "صيغة الملف غير مدعومة. الرجاء استخدام ملف CSV أو Excel."
            
            # التحقق من البيانات
            if data.empty:
                return None, "الملف فارغ"
            
            # معالجة القيم المفقودة
            data = self._handle_missing_values(data)
            
            return data, None
            
        except Exception as e:
            return None, f"خطأ في تحميل البيانات: {str(e)}"

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """معالجة القيم المفقودة في البيانات"""
        try:
            # نسخة من البيانات لتجنب التعديل على البيانات الأصلية
            df = data.copy()
            
            # حساب نسبة القيم المفقودة لكل عمود
            missing_percentages = df.isnull().mean() * 100
            
            for column in df.columns:
                missing_pct = missing_percentages[column]
                
                # إذا كانت نسبة القيم المفقودة أكثر من 50%، حذف العمود
                if missing_pct > 50:
                    df = df.drop(columns=[column])
                    st.warning(f"تم حذف العمود '{column}' لأن {missing_pct:.1f}% من قيمه مفقودة")
                    continue
                
                # معالجة القيم المفقودة حسب نوع البيانات
                if df[column].dtype in ['int64', 'float64']:
                    # للأعمدة الرقمية، استخدام الوسيط
                    median_value = df[column].median()
                    df[column] = df[column].fillna(median_value)
                    if missing_pct > 0:
                        st.info(f"تم ملء القيم المفقودة في العمود '{column}' باستخدام الوسيط ({median_value:.2f})")
                else:
                    # للأعمدة النصية، استخدام القيمة الأكثر تكراراً
                    mode_value = df[column].mode()[0]
                    df[column] = df[column].fillna(mode_value)
                    if missing_pct > 0:
                        st.info(f"تم ملء القيم المفقودة في العمود '{column}' باستخدام القيمة الأكثر تكراراً ('{mode_value}')")
            
            return df
            
        except Exception as e:
            st.error(f"خطأ في معالجة القيم المفقودة: {str(e)}")
            return data

    @st.cache_data
    def _load_system_info(_self) -> Dict[str, Any]:
        """تحميل معلومات النظام مع التخزين المؤقت"""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'disk_usage': psutil.disk_usage('/'),
                'python_version': platform.python_version(),
                'dependencies': {
                    'streamlit': st.__version__,
                    'pandas': pd.__version__,
                    'plotly': px.__version__,
                    'scikit-learn': sklearn.__version__
                }
            }
        except Exception as e:
            st.error(f"خطأ في تحميل معلومات النظام: {str(e)}")
            return {}

    @st.cache_data
    def _get_feature_columns(_self, data: pd.DataFrame) -> List[str]:
        """الحصول على أسماء الأعمدة المميزة مع التخزين المؤقت"""
        try:
            return [col for col in data.columns if data[col].dtype in ['int64', 'float64', 'bool']]
        except Exception as e:
            st.error(f"خطأ في تحديد المتغيرات: {str(e)}")
            return []

    def show_data_analysis_section(self, analysis_type: str):
        """عرض قسم تحليل البيانات"""
        st.header("📊 تحليل البيانات")
        
        uploaded_file = st.file_uploader(
            "قم بتحميل ملف البيانات (CSV, Excel)",
            type=['csv', 'xlsx']
        )
        
        if uploaded_file is not None:
            data, error = self.load_and_process_data(uploaded_file)
            
            if error:
                st.error(f"حدث خطأ أثناء تحميل البيانات: {error}")
                return
                
            if data is not None:
                st.session_state.data = data
                try:
                    if analysis_type == "نظرة عامة":
                        with st.spinner("جاري تحليل البيانات..."):
                            self.data_analyzer.show_overview(data)
                    elif analysis_type == "تحليل المتغيرات":
                        with st.spinner("جاري تحليل المتغيرات..."):
                            self.data_analyzer.show_variable_analysis(data)
                    elif analysis_type == "الارتباطات":
                        with st.spinner("جاري حساب الارتباطات..."):
                            self.data_analyzer.show_correlations(data)
                    elif analysis_type == "القيم المفقودة":
                        with st.spinner("جاري تحليل القيم المفقودة..."):
                            self.data_analyzer.show_missing_values(data)
                    elif analysis_type == "الرسوم البيانية":
                        with st.spinner("جاري إنشاء الرسوم البيانية..."):
                            self.data_analyzer.show_plots(data)
                except Exception as e:
                    st.error(f"حدث خطأ أثناء تحليل البيانات: {str(e)}")
                    st.info("نصيحة: تأكد من أن البيانات بالتنسيق الصحيح وتحتوي على القيم المطلوبة")
                    
    @st.cache_data
    def _get_feature_columns(_self, data: pd.DataFrame) -> List[str]:
        """الحصول على أسماء الأعمدة المتاحة للاستخدام كمتغيرات"""
        try:
            return [col for col in data.columns if data[col].dtype in ['int64', 'float64', 'bool']]
        except Exception as e:
            st.error(f"خطأ في تحديد المتغيرات: {str(e)}")
            return []

    def show_model_training_section(self, task_type: str = None):
        """عرض قسم تدريب النموذج"""
        try:
            st.header("🤖 تدريب النموذج")
            
            if 'data' not in st.session_state or st.session_state.data is None:
                st.warning("الرجاء تحميل البيانات أولاً")
                return
                
            data = st.session_state.data
            
            # اختيار نوع المهمة
            if task_type is None:
                task_type = st.selectbox(
                    "اختر نوع المهمة",
                    options=["تصنيف", "انحدار"],
                    index=0
                )
            
            # اختيار المتغير الهدف
            target_col = st.selectbox(
                "اختر المتغير الهدف",
                options=data.columns.tolist(),
                key="target_column"
            )
            
            if target_col:
                # تحديث المتغيرات المتاحة
                available_features = self._get_feature_columns(data)
                if not available_features:
                    st.error("لم يتم العثور على متغيرات رقمية مناسبة في البيانات")
                    return
                    
                # اختيار المتغيرات
                selected_features = st.multiselect(
                    "اختر المتغيرات للتدريب",
                    options=[col for col in available_features if col != target_col],
                    default=[col for col in available_features if col != target_col][:5],
                    key="selected_features"
                )
                
                if selected_features:
                    st.session_state.features = selected_features
                    st.session_state.target = target_col
                    
                    # إعدادات التدريب
                    with st.expander("⚙️ إعدادات متقدمة"):
                        optimization_time = st.slider(
                            "وقت التحسين (بالثواني)",
                            min_value=30,
                            max_value=3600,
                            value=300,
                            step=30
                        )
                        
                        n_trials = st.slider(
                            "عدد محاولات التحسين",
                            min_value=10,
                            max_value=100,
                            value=30,
                            step=5
                        )
                        
                        cv_folds = st.slider(
                            "عدد طيات التحقق المتقاطع",
                            min_value=2,
                            max_value=10,
                            value=5,
                            step=1
                        )
                    
                    # زر التدريب
                    if st.button("🚀 بدء التدريب"):
                        with st.spinner("جاري تدريب النموذج..."):
                            try:
                                X = data[selected_features]
                                y = data[target_col]
                                
                                model_results = self.model_trainer.train_model(
                                    X=X,
                                    y=y,
                                    task_type="classification" if task_type == "تصنيف" else "regression",
                                    optimization_time=optimization_time,
                                    n_trials=n_trials,
                                    cv_folds=cv_folds
                                )
                                
                                if model_results:
                                    st.session_state.model_results = model_results
                                    
                                    # حفظ النموذج في قاعدة البيانات
                                    try:
                                        model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                        model_id = self.db_manager.save_model(
                                            name=model_name,
                                            model=model_results['model'],
                                            model_type="classification" if task_type == "تصنيف" else "regression",
                                            hyperparameters=model_results['best_params'],
                                            metrics=model_results['best_metrics'],
                                            feature_importance=model_results.get('feature_importance'),
                                            preprocessing_params={
                                                'features': selected_features,
                                                'target': target_col
                                            }
                                        )
                                        st.success(f"تم حفظ النموذج بنجاح! (معرف النموذج: {model_id})")
                                    except Exception as e:
                                        st.warning(f"تم تدريب النموذج بنجاح ولكن فشل حفظه: {str(e)}")
                                    
                                    # عرض النتائج
                                    self._show_training_results(model_results)
                                else:
                                    st.error("فشل تدريب النموذج")
                                    
                            except Exception as e:
                                st.error(f"حدث خطأ أثناء تدريب النموذج: {str(e)}")
                                st.info("نصيحة: تأكد من صحة البيانات وتوافقها مع نوع المهمة المختار")
                                
        except Exception as e:
            st.error(f"حدث خطأ في قسم تدريب النموذج: {str(e)}")
            st.info("حاول إعادة تحميل الصفحة أو التحقق من صحة البيانات")

    @st.cache_data
    def _show_training_results(_self, _model_results: dict):
        """عرض نتائج التدريب مع التخزين المؤقت"""
        try:
            # عرض مقاييس الأداء
            st.subheader("📊 مقاييس الأداء")
            metrics_data = {
                'النموذج': _model_results.get('best_model_name', ''),
                'النتيجة': _model_results.get('best_score', ''),
                **_model_results.get('best_metrics', {})
            }
            metrics_df = pd.DataFrame([metrics_data])
            st.dataframe(metrics_df)
            
            # عرض أفضل المعاملات
            if 'best_params' in _model_results:
                st.subheader("⚙️ أفضل المعاملات")
                params_df = pd.DataFrame([_model_results['best_params']])
                st.dataframe(params_df)
            
            # عرض أهمية المتغيرات
            if 'feature_importance' in _model_results and _model_results['feature_importance']:
                st.subheader("🎯 أهمية المتغيرات")
                feature_importance = _model_results['feature_importance']
                if isinstance(feature_importance, dict):
                    fig = px.bar(
                        x=list(feature_importance.values()),
                        y=list(feature_importance.keys()),
                        orientation='h',
                        title="أهمية المتغيرات"
                    )
                    fig.update_layout(
                        xaxis_title="الأهمية",
                        yaxis_title="المتغير",
                        showlegend=False
                    )
                    st.plotly_chart(fig)
            
            # عرض نتائج جميع النماذج
            if 'all_models_results' in _model_results:
                st.subheader("📋 نتائج جميع النماذج")
                all_models_df = pd.DataFrame(_model_results['all_models_results'])
                st.dataframe(all_models_df)
                
        except Exception as e:
            st.error(f"خطأ في عرض نتائج التدريب: {str(e)}")
            st.exception(e)  # عرض تفاصيل الخطأ للتصحيح

    @st.cache_data
    def _load_saved_models(_self) -> List[dict]:
        """تحميل النماذج المحفوظة مع التخزين المؤقت"""
        try:
            # Get all models
            models = []
            db = _self.db_manager.SessionLocal()
            
            try:
                # Query all models with their metadata
                db_models = db.query(MLModel).all()
                
                for db_model in db_models:
                    # Load model binary and metadata
                    model_binary = io.BytesIO(db_model.model_binary)
                    model = joblib.load(model_binary)
                    
                    # Create model info dictionary
                    model_info = {
                        'id': db_model.id,
                        'name': db_model.name,
                        'type': db_model.model_type,
                        'score': db_model.metrics.get('test_score', 0.0) if db_model.metrics else 0.0,
                        'created_at': db_model.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                        'target': db_model.preprocessing_params.get('target_column') if db_model.preprocessing_params else None,
                        'features': db_model.preprocessing_params.get('feature_columns', []) if db_model.preprocessing_params else [],
                        'metrics': db_model.metrics or {},
                        'feature_importance': db_model.feature_importance or {},
                        'preprocessing_params': db_model.preprocessing_params or {}
                    }
                    models.append(model_info)
                    
            finally:
                db.close()
                    
            return models
        except Exception as e:
            st.error(f"خطأ في تحميل النماذج: {str(e)}")
            return []

    def show_model_management_section(self, action: str):
        """عرض قسم إدارة النماذج"""
        try:
            st.header("📁 إدارة النماذج")
            
            # تحميل النماذج المحفوظة
            saved_models = self._load_saved_models()
            if not saved_models:
                st.warning("لا توجد نماذج محفوظة")
                return
                
            # عرض قائمة النماذج
            st.subheader("📋 النماذج المتوفرة")
            model_table = pd.DataFrame([
                {
                    'اسم النموذج': model['name'],
                    'نوع المهمة': model['type'],
                    'النتيجة': f"{model['score']:.4f}",
                    'تاريخ الإنشاء': model['created_at']
                }
                for model in saved_models
            ])
            st.dataframe(model_table)
            
            # اختيار النموذج
            selected_model_name = st.selectbox(
                "اختر نموذجاً",
                options=[model['name'] for model in saved_models]
            )
            
            if selected_model_name:
                selected_model = next(
                    model for model in saved_models 
                    if model['name'] == selected_model_name
                )
                
                # عرض تفاصيل النموذج
                with st.expander("ℹ️ تفاصيل النموذج"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**معلومات أساسية:**")
                        st.write(f"- النوع: {selected_model['type']}")
                        st.write(f"- النتيجة: {selected_model['score']:.4f}")
                        st.write(f"- تاريخ الإنشاء: {selected_model['created_at']}")
                    
                    with col2:
                        st.write("**المتغيرات:**")
                        st.write(f"- المتغير الهدف: {selected_model['target']}")
                        st.write(f"- عدد المتغيرات: {len(selected_model['features'])}")
                
                # عرض المقاييس
                if 'metrics' in selected_model:
                    st.subheader("📊 مقاييس النموذج")
                    metrics_df = pd.DataFrame([selected_model['metrics']])
                    st.dataframe(metrics_df)
                
                # عرض أهمية المتغيرات
                if 'feature_importance' in selected_model:
                    st.subheader("📈 أهمية المتغيرات")
                    fig = self.report_generator.plot_feature_importance(
                        selected_model['feature_importance']
                    )
                    st.plotly_chart(fig)
                
                # خيارات إضافية
                st.subheader("⚙️ الإجراءات المتاحة")
                action_col1, action_col2 = st.columns(2)
                
                with action_col1:
                    if st.button("🔄 تحديث النموذج"):
                        try:
                            if 'data' not in st.session_state:
                                st.error("الرجاء تحميل البيانات أولاً")
                                return
                                
                            with st.spinner("جاري تحديث النموذج..."):
                                # تحضير البيانات
                                X = st.session_state.data[selected_model['features']]
                                y = st.session_state.data[selected_model['target']]
                                
                                # تحديث النموذج
                                updated_model = self.model_trainer.update_model(
                                    selected_model['name'],
                                    X, y
                                )
                                
                                if updated_model:
                                    st.success("تم تحديث النموذج بنجاح!")
                                    # تحديث التخزين المؤقت
                                    st.cache_data.clear()
                                else:
                                    st.error("فشل تحديث النموذج")
                        except Exception as e:
                            st.error(f"خطأ في تحديث النموذج: {str(e)}")
                
                with action_col2:
                    if st.button("🗑️ حذف النموذج"):
                        try:
                            if st.warning("هل أنت متأكد من حذف النموذج؟"):
                                self.db_manager.delete_model(selected_model['name'])
                                # تحديث التخزين المؤقت
                                st.cache_data.clear()
                                st.success("تم حذف النموذج بنجاح!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"خطأ في حذف النموذج: {str(e)}")
                
                # تصدير النموذج
                st.subheader("📤 تصدير النموذج")
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    export_format = st.selectbox(
                        "اختر صيغة التصدير",
                        options=["joblib", "pickle", "ONNX"]
                    )
                
                with export_col2:
                    if st.button("تصدير"):
                        try:
                            with st.spinner("جاري تصدير النموذج..."):
                                export_path = self.model_trainer.export_model(
                                    selected_model['name'],
                                    format=export_format
                                )
                                st.success(f"تم تصدير النموذج بنجاح إلى: {export_path}")
                        except Exception as e:
                            st.error(f"خطأ في تصدير النموذج: {str(e)}")
                            
        except Exception as e:
            st.error(f"حدث خطأ في قسم إدارة النماذج: {str(e)}")
            st.info("حاول إعادة تحميل الصفحة أو التحقق من اتصال قاعدة البيانات")

    @st.cache_data
    def _get_monitoring_data(_self, time_range: str) -> Dict[str, pd.DataFrame]:
        """جلب بيانات المراقبة مع التخزين المؤقت"""
        try:
            end_date = datetime.now()
            if time_range == "آخر 24 ساعة":
                start_date = end_date - timedelta(days=1)
            elif time_range == "آخر 7 أيام":
                start_date = end_date - timedelta(days=7)
            elif time_range == "آخر 30 يوم":
                start_date = end_date - timedelta(days=30)
            else:
                start_date = end_date - timedelta(days=90)
                
            return {
                'performance': _self.db_manager.get_model_performance_history(start_date, end_date),
                'predictions': _self.db_manager.get_prediction_history(start_date, end_date),
                'resources': _self.db_manager.get_resource_usage_history(start_date, end_date)
            }
        except Exception as e:
            st.error(f"خطأ في جلب بيانات المراقبة: {str(e)}")
            return {}

    @st.cache_data
    def _calculate_drift_metrics(_self, data: pd.DataFrame) -> Dict[str, float]:
        """حساب مقاييس انحراف البيانات"""
        try:
            return {
                'feature_drift': _self.monitor.calculate_feature_drift(data),
                'target_drift': _self.monitor.calculate_target_drift(data),
                'prediction_drift': _self.monitor.calculate_prediction_drift(data)
            }
        except Exception as e:
            st.error(f"خطأ في حساب مقاييس الانحراف: {str(e)}")
            return {}

    def show_monitoring_section(self, monitor_type: str, time_range: str):
        """عرض قسم المراقبة والتقارير"""
        try:
            st.header("📊 المراقبة والتقارير")
            
            # Initialize monitoring data if needed
            if not st.session_state.monitoring_initialized:
                self._initialize_monitoring_data()
            
            # اختيار النطاق الزمني
            time_range = st.selectbox(
                "النطاق الزمني",
                options=["آخر 24 ساعة", "آخر 7 أيام", "آخر 30 يوم", "آخر 90 يوم"],
                index=1
            )
            
            # جلب البيانات
            monitoring_data = self._get_monitoring_data(time_range)
            if not monitoring_data:
                st.warning("لا تتوفر بيانات للمراقبة في النطاق الزمني المحدد")
                return
            
            # عرض لوحة المعلومات
            st.subheader("📈 لوحة المعلومات")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'performance' in monitoring_data and not monitoring_data['performance'].empty:
                    perf_data = monitoring_data['performance']
                    # Get the first available metric for display
                    metric_cols = [col for col in perf_data.columns 
                                 if col not in ['timestamp', 'model_id']]
                    if metric_cols:
                        metric_name = metric_cols[0]
                        latest_value = perf_data[metric_name].iloc[-1]
                        value_change = latest_value - perf_data[metric_name].iloc[0]
                        st.metric(
                            f"آخر {metric_name}",
                            f"{latest_value:.4f}",
                            f"{value_change:+.4f}"
                        )
        
            with col2:
                if 'predictions' in monitoring_data and not monitoring_data['predictions'].empty:
                    pred_data = monitoring_data['predictions']
                    recent_preds = pred_data[
                        pred_data['timestamp'] > (datetime.now() - timedelta(days=1))
                    ]
                    pred_count = len(pred_data)
                    recent_count = len(recent_preds)
                    st.metric(
                        "عدد التنبؤات",
                        pred_count,
                        f"+{recent_count} في آخر 24 ساعة"
                    )
        
            with col3:
                if 'resources' in monitoring_data and not monitoring_data['resources'].empty:
                    res_data = monitoring_data['resources']
                    if 'memory_usage' in res_data.columns:
                        current_memory = res_data['memory_usage'].iloc[-1]
                        avg_memory = res_data['memory_usage'].mean()
                        st.metric(
                            "استخدام الذاكرة",
                            f"{current_memory:.1f} MB",
                            f"{current_memory - avg_memory:+.1f} MB"
                        )
        
            # عرض التفاصيل حسب نوع المراقبة
            if monitor_type == "مراقبة الأداء":
                self._show_performance_monitoring(monitoring_data.get('performance', pd.DataFrame()))
            
            elif monitor_type == "تحليل الانحراف":
                self._show_data_drift_monitoring(monitoring_data.get('predictions', pd.DataFrame()))
            
            elif monitor_type == "استخدام الموارد":
                self._show_resource_monitoring(monitoring_data.get('resources', pd.DataFrame()))
            
            elif monitor_type == "سجل التنبؤات":
                self._show_prediction_logs(monitoring_data.get('predictions', pd.DataFrame()))
            
        except Exception as e:
            st.error(f"حدث خطأ في قسم المراقبة: {str(e)}")
            st.info("حاول إعادة تحميل الصفحة أو التحقق من اتصال قاعدة البيانات")

    def _show_performance_monitoring(self, performance_data: pd.DataFrame):
        """عرض مراقبة أداء النموذج"""
        try:
            if performance_data.empty:
                st.warning("لا تتوفر بيانات أداء للعرض")
                return
            
            st.subheader("📊 مراقبة أداء النموذج")
            
            # Get metric columns (excluding timestamp and model_id)
            metric_cols = [col for col in performance_data.columns 
                          if col not in ['timestamp', 'model_id']]
            
            if not metric_cols:
                st.warning("لا توجد مقاييس أداء متاحة")
                return
            
            # Plot each metric over time
            for metric in metric_cols:
                fig = px.line(
                    performance_data,
                    x='timestamp',
                    y=metric,
                    title=f"تطور {metric} عبر الزمن",
                    markers=True
                )
                fig.update_layout(
                    xaxis_title="الوقت",
                    yaxis_title=metric,
                    showlegend=False
                )
                st.plotly_chart(fig)
            
            # تحليل الأداء
            with st.expander("📊 تحليل الأداء"):
                for metric in metric_cols:
                    st.subheader(f"تحليل {metric}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "أفضل قيمة",
                            f"{performance_data[metric].max():.4f}"
                        )
                        st.metric(
                            "أسوأ قيمة",
                            f"{performance_data[metric].min():.4f}"
                        )
                    with col2:
                        st.metric(
                            "متوسط القيم",
                            f"{performance_data[metric].mean():.4f}"
                        )
                        st.metric(
                            "انحراف القيم",
                            f"{performance_data[metric].std():.4f}"
                        )
                
        except Exception as e:
            st.error(f"خطأ في عرض مراقبة الأداء: {str(e)}")

    def _show_data_drift_monitoring(self, prediction_data: pd.DataFrame):
        """عرض مراقبة انحراف البيانات"""
        try:
            if prediction_data.empty:
                st.warning("لا تتوفر بيانات للتحليل")
                return
                
            st.subheader("🔄 تحليل انحراف البيانات")
            
            # استخراج المتغيرات من البيانات
            feature_cols = [col for col in prediction_data.columns 
                          if col not in ['timestamp', 'model_id', 'prediction', 'actual']]
            
            if not feature_cols:
                st.warning("لا توجد متغيرات متاحة للتحليل")
                return
            
            # تقسيم البيانات إلى فترتين للمقارنة
            mid_point = prediction_data['timestamp'].mean()
            period1_data = prediction_data[prediction_data['timestamp'] < mid_point]
            period2_data = prediction_data[prediction_data['timestamp'] >= mid_point]
            
            if period1_data.empty or period2_data.empty:
                st.warning("لا تتوفر بيانات كافية للمقارنة")
                return
            
            # حساب الانحراف لكل متغير
            drift_scores = {}
            for feature in feature_cols:
                drift_score = self.model_monitor.calculate_drift(
                    period1_data[[feature]], 
                    period2_data[[feature]]
                )
                drift_scores[feature] = drift_score[feature]
            
            # عرض نتائج الانحراف
            st.subheader("📊 درجات الانحراف")
            drift_df = pd.DataFrame([
                {"المتغير": feature, "درجة الانحراف": score}
                for feature, score in drift_scores.items()
            ]).sort_values("درجة الانحراف", ascending=False)
            
            # رسم بياني للانحراف
            fig = px.bar(
                drift_df,
                x="المتغير",
                y="درجة الانحراف",
                title="انحراف المتغيرات",
            )
            fig.update_layout(
                xaxis_title="المتغير",
                yaxis_title="درجة الانحراف"
            )
            st.plotly_chart(fig)
            
            # تفاصيل الانحراف
            with st.expander("📋 تفاصيل الانحراف"):
                st.dataframe(drift_df)
                
                # تحليل المتغيرات ذات الانحراف العالي
                high_drift_features = drift_df[
                    drift_df["درجة الانحراف"] > drift_df["درجة الانحراف"].mean()
                ]["المتغير"].tolist()
                
                if high_drift_features:
                    st.subheader("⚠️ متغيرات ذات انحراف عالي")
                    for feature in high_drift_features:
                        st.write(f"**{feature}**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("الفترة الأولى:")
                            st.write(period1_data[feature].describe())
                        with col2:
                            st.write("الفترة الثانية:")
                            st.write(period2_data[feature].describe())
                
        except Exception as e:
            st.error(f"خطأ في تحليل انحراف البيانات: {str(e)}")

    def _show_resource_monitoring(self, resource_data: pd.DataFrame):
        """عرض مراقبة استخدام الموارد"""
        try:
            st.subheader("⚡ مراقبة استخدام الموارد")
            
            # رسم بياني لاستخدام الموارد
            fig = self.report_generator.plot_resource_usage(
                resource_data,
                title="استخدام الموارد عبر الزمن"
            )
            st.plotly_chart(fig)
            
            # تحليل استخدام الموارد
            with st.expander("📊 تحليل الموارد"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "ذروة استخدام الذاكرة",
                        f"{resource_data['memory_usage'].max():.1f} MB"
                    )
                    st.metric(
                        "متوسط وقت الاستجابة",
                        f"{resource_data['response_time'].mean():.2f} ms"
                    )
                with col2:
                    st.metric(
                        "متوسط استخدام المعالج",
                        f"{resource_data['cpu_usage'].mean():.1f}%"
                    )
                    st.metric(
                        "عدد الطلبات",
                        len(resource_data)
                    )
                    
        except Exception as e:
            st.error(f"خطأ في عرض مراقبة الموارد: {str(e)}")

    def _show_prediction_logs(self, prediction_data: pd.DataFrame):
        """عرض سجل التنبؤات"""
        try:
            st.subheader("📝 سجل التنبؤات")
            
            # تصفية وعرض السجل
            with st.expander("🔍 خيارات التصفية"):
                col1, col2 = st.columns(2)
                with col1:
                    min_confidence = st.slider(
                        "الحد الأدنى للثقة",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.1,
                        key="min_confidence"
                    )
                with col2:
                    status_filter = st.multiselect(
                        "حالة التنبؤ",
                        options=["ناجح", "فاشل", "غير مؤكد"],
                        default=["ناجح", "فاشل", "غير مؤكد"],
                        key="status_filter"
                    )
                    
            # تطبيق التصفية
            filtered_data = prediction_data[
                (prediction_data['confidence'] >= min_confidence) &
                (prediction_data['status'].isin(status_filter))
            ]
            
            # عرض السجل
            st.dataframe(
                filtered_data[['timestamp', 'input', 'prediction', 'confidence', 'status']],
                use_container_width=True
            )
            
            # تحليل التنبؤات
            with st.expander("📊 تحليل التنبؤات"):
                col1, col2 = st.columns(2)
                with col1:
                    success_rate = (filtered_data['status'] == 'ناجح').mean()
                    st.metric(
                        "معدل النجاح",
                        f"{success_rate:.2%}"
                    )
                    st.metric(
                        "متوسط الثقة",
                        f"{filtered_data['confidence'].mean():.2%}"
                    )
                with col2:
                    st.metric(
                        "عدد التنبؤات",
                        len(filtered_data)
                    )
                    st.metric(
                        "تنبؤات غير مؤكدة",
                        len(filtered_data[filtered_data['confidence'] < 0.5])
                    )
                    
        except Exception as e:
            st.error(f"خطأ في عرض سجل التنبؤات: {str(e)}")

    @st.cache_data
    def _load_system_info(_self) -> Dict[str, Any]:
        """تحميل معلومات النظام مع التخزين المؤقت"""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'disk_usage': psutil.disk_usage('/'),
                'python_version': platform.python_version(),
                'dependencies': {
                    'streamlit': st.__version__,
                    'pandas': pd.__version__,
                    'plotly': px.__version__,
                    'scikit-learn': sklearn.__version__
                }
            }
        except Exception as e:
            st.error(f"خطأ في تحميل معلومات النظام: {str(e)}")
            return {}

    def show_settings_section(self, settings_type: str):
        """عرض قسم الإعدادات"""
        try:
            st.header("⚙️ الإعدادات")
            
            if settings_type == "إعدادات النظام":
                self._show_system_settings()
                
            elif settings_type == "إعدادات التدريب":
                self._show_training_settings()
                
            elif settings_type == "إعدادات المراقبة":
                self._show_monitoring_settings()
                
            elif settings_type == "إعدادات الواجهة":
                self._show_ui_settings()
                
        except Exception as e:
            st.error(f"حدث خطأ في قسم الإعدادات: {str(e)}")
            st.info("حاول إعادة تحميل الصفحة")

    def _show_system_settings(self):
        """عرض إعدادات النظام"""
        try:
            st.subheader("🖥️ إعدادات النظام")
            
            # معلومات النظام
            system_info = self._load_system_info()
            if system_info:
                with st.expander("📊 معلومات النظام", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "عدد المعالجات",
                            system_info['cpu_count']
                        )
                        st.metric(
                            "الذاكرة الكلية",
                            f"{system_info['memory_total'] / (1024**3):.1f} GB"
                        )
                    with col2:
                        disk = system_info['disk_usage']
                        st.metric(
                            "مساحة القرص المستخدمة",
                            f"{disk.percent}%"
                        )
                        st.metric(
                            "إصدار Python",
                            system_info['python_version']
                        )
                    
                    # إصدارات المكتبات
                    st.write("**إصدارات المكتبات:**")
                    for lib, version in system_info['dependencies'].items():
                        st.write(f"- {lib}: {version}")
        
            # إعدادات الذاكرة المؤقتة
            with st.expander("🔄 إعدادات الذاكرة المؤقتة"):
                st.number_input(
                    "مدة صلاحية الذاكرة المؤقتة (ثواني)",
                    min_value=300,
                    max_value=7200,
                    value=1800,
                    step=300,
                    key="cache_ttl"
                )
                if st.button("مسح الذاكرة المؤقتة"):
                    st.cache_data.clear()
                    st.success("تم مسح الذاكرة المؤقتة بنجاح")
                    
            # إعدادات قاعدة البيانات
            with st.expander("🗄️ إعدادات قاعدة البيانات"):
                st.text_input(
                    "عنوان قاعدة البيانات",
                    value=self.db_manager.get_connection_string(),
                    key="db_host"
                )
                if st.button("اختبار الاتصال"):
                    if self.db_manager.test_connection():
                        st.success("تم الاتصال بقاعدة البيانات بنجاح")
                    else:
                        st.error("فشل الاتصال بقاعدة البيانات")
                        
        except Exception as e:
            st.error(f"خطأ في عرض إعدادات النظام: {str(e)}")

    def _show_training_settings(self):
        """عرض إعدادات التدريب"""
        try:
            st.subheader("🤖 إعدادات التدريب")
            
            # إعدادات الموارد
            with st.expander("💻 إعدادات الموارد", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.slider(
                        "الحد الأقصى للذاكرة (GB)",
                        min_value=1,
                        max_value=32,
                        value=8,
                        key="max_memory"
                    )
                with col2:
                    st.slider(
                        "عدد المعالجات",
                        min_value=1,
                        max_value=psutil.cpu_count(),
                        value=psutil.cpu_count() // 2,
                        key="n_jobs"
                    )
        
            # إعدادات التحقق المتقاطع
            with st.expander("🔄 إعدادات التحقق المتقاطع"):
                st.number_input(
                    "عدد الطيات",
                    min_value=2,
                    max_value=10,
                    value=5,
                    key="cv_folds"
                )
                st.checkbox(
                    "استخدام التقسيم العشوائي",
                    value=True,
                    key="random_split"
                )
            
            # إعدادات التحسين
            with st.expander("🎯 إعدادات التحسين"):
                st.number_input(
                    "عدد محاولات التحسين",
                    min_value=10,
                    max_value=100,
                    value=30,
                    key="n_trials"
                )
                st.slider(
                    "وقت التحسين (دقائق)",
                    min_value=1,
                    max_value=60,
                    value=10,
                    key="optimization_time"
                )
            
        except Exception as e:
            st.error(f"خطأ في عرض إعدادات التدريب: {str(e)}")

    def _show_monitoring_settings(self):
        """عرض إعدادات المراقبة"""
        try:
            st.subheader("📊 إعدادات المراقبة")
            
            # إعدادات التنبيهات
            with st.expander("🔔 إعدادات التنبيهات", expanded=True):
                st.slider(
                    "عتبة انحراف البيانات",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.1,
                    key="drift_threshold"
                )
                st.multiselect(
                    "أنواع التنبيهات",
                    options=["انحراف البيانات", "تدهور الأداء", "استخدام الموارد"],
                    default=["انحراف البيانات"],
                    key="alert_types"
                )
            
            # إعدادات التقارير
            with st.expander("📈 إعدادات التقارير"):
                st.number_input(
                    "فترة التقارير (أيام)",
                    min_value=1,
                    max_value=90,
                    value=30,
                    key="report_period"
                )
                st.checkbox(
                    "تقارير تلقائية",
                    value=False,
                    key="auto_reports"
                )
            
            # إعدادات التخزين
            with st.expander("💾 إعدادات التخزين"):
                st.number_input(
                    "فترة الاحتفاظ بالسجلات (أيام)",
                    min_value=30,
                    max_value=365,
                    value=90,
                    key="log_retention"
                )
                st.checkbox(
                    "ضغط السجلات القديمة",
                    value=True,
                    key="compress_logs"
                )
            
        except Exception as e:
            st.error(f"خطأ في عرض إعدادات المراقبة: {str(e)}")

    def _show_ui_settings(self):
        """عرض إعدادات الواجهة"""
        try:
            st.subheader("🎨 إعدادات الواجهة")
            
            # إعدادات اللغة
            with st.expander("🌐 إعدادات اللغة", expanded=True):
                st.selectbox(
                    "اللغة",
                    options=["العربية", "English"],
                    index=0,
                    key="language"
                )
                st.checkbox(
                    "عرض الترجمة",
                    value=False,
                    key="show_translation"
                )
            
            # إعدادات العرض
            with st.expander("📱 إعدادات العرض"):
                st.selectbox(
                    "نمط العرض",
                    options=["فاتح", "داكن", "تلقائي"],
                    index=2,
                    key="theme"
                )
                st.checkbox(
                    "عرض الرموز التعبيرية",
                    value=True,
                    key="show_emoji"
                )
            
            # إعدادات الرسوم البيانية
            with st.expander("📊 إعدادات الرسوم البيانية"):
                st.selectbox(
                    "مكتبة الرسوم البيانية",
                    options=["Plotly", "Matplotlib", "Altair"],
                    index=0,
                    key="plot_library"
                )
                st.color_picker(
                    "لون الرسوم البيانية",
                    value="#1f77b4",
                    key="plot_color"
                )
            
        except Exception as e:
            st.error(f"خطأ في عرض إعدادات الواجهة: {str(e)}")

    def run(self):
        """تشغيل لوحة التحكم"""
        # عرض القائمة الجانبية
        main_section, options = Sidebar.show()
        
        # عرض القسم المحدد
        if main_section == "تحليل البيانات":
            self.show_data_analysis_section(options['analysis_type'])
            
        elif main_section == "تدريب النموذج":
            self.show_model_training_section(
                options['task_type']
            )
            
        elif main_section == "إدارة النماذج":
            self.show_model_management_section(options['model_action'])
            
        elif main_section == "المراقبة والتقارير":
            self.show_monitoring_section(
                options['monitor_type'],
                options['time_range']
            )
            
        elif main_section == "الإعدادات":
            self.show_settings_section(options['settings_type'])

def main():
    st.set_page_config(
        page_title="AutoSKL Dashboard",
        page_icon="🤖",
        layout="wide"
    )
    
    dashboard = Dashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
