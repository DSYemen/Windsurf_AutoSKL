import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import sys
import os
from typing import Optional, Dict, List, Any
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso,
    ElasticNet, SGDClassifier, SGDRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to Python path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from app.core.config import settings
from app.services.data_processor import DataProcessor
from app.services.model_trainer import ModelTrainer
from app.services.model_evaluator import ModelEvaluator

class Dashboard:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()
        self.model_evaluator = ModelEvaluator()
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        default_state = {
            'data': None,  # Raw data
            'processed_data': {  # Processed data dictionary
                'X': None,  # Feature matrix
                'y': None,  # Target vector
                'feature_names': None,  # Feature names
                'target_column': None,  # Target column name
                'categorical_features': [],  # List of categorical features
                'numerical_features': [],  # List of numerical features
                'preprocessing_info': {}  # Preprocessing parameters
            },
            'preprocessing_state': {  # Preprocessing state tracking
                'is_processed': False,
                'feature_types': None,
                'preprocessing_params': None,
                'validation_errors': []
            },
            'model_state': {  # Model state tracking
                'is_trained': False,
                'model': None,
                'model_params': None,
                'training_history': None,
                'evaluation_metrics': None
            }
        }
        
        # Initialize or update session state
        for key, default_value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                
    def show_data_analysis(self):
        """Show data analysis section with advanced preprocessing options"""
        st.header("🔍 تحليل البيانات ومعالجتها")
        
        # إضافة قسم تحميل الملفات
        uploaded_file = st.file_uploader(
            "قم بتحميل ملف البيانات (CSV, Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="يمكنك تحميل ملف CSV أو Excel"
        )
        
        if uploaded_file is not None:
            try:
                # تحميل البيانات باستخدام DataProcessor
                if uploaded_file.name.endswith('.csv'):
                    data = self.data_processor.load_csv(uploaded_file)
                else:
                    data = self.data_processor.load_excel(uploaded_file)
                
                # حفظ البيانات في session state
                st.session_state.data = data
                st.success(f"تم تحميل البيانات بنجاح! ({data.shape[0]} صف, {data.shape[1]} عمود)")
                
                # عرض البيانات
                st.subheader("📊 عرض البيانات")
                st.write(data.to_html(index=False), unsafe_allow_html=True)
                
                # معلومات عن البيانات
                st.subheader("📋 معلومات البيانات")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("أنواع البيانات:")
                    dtypes_df = pd.DataFrame({
                        'العمود': data.dtypes.index,
                        'نوع البيانات': data.dtypes.values.astype(str)
                    })
                    st.write(dtypes_df.to_html(index=False), unsafe_allow_html=True)
                with col2:
                    st.write("القيم المفقودة:")
                    missing_df = pd.DataFrame({
                        'العمود': data.columns,
                        'القيم المفقودة': data.isnull().sum().values,
                        'نسبة القيم المفقودة': (data.isnull().sum() / len(data) * 100).round(2).values,
                        'القيم الفريدة': [data[col].nunique() for col in data.columns]
                    })
                    st.write(missing_df.to_html(index=False), unsafe_allow_html=True)
                
                # خيارات معالجة البيانات
                st.subheader("⚙️ خيارات معالجة البيانات")
                
                # معالجة القيم المفقودة
                with st.expander("معالجة القيم المفقودة"):
                    # تقسيم الأعمدة حسب نوع البيانات
                    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
                    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("🔢 الأعمدة الرقمية")
                        numeric_missing_cols = st.multiselect(
                            "اختر الأعمدة الرقمية لمعالجة القيم المفقودة",
                            options=numeric_cols.tolist(),
                            default=numeric_cols[data[numeric_cols].isnull().any()].tolist()
                        )
                        numeric_strategy = st.selectbox(
                            "استراتيجية معالجة القيم المفقودة (للأعمدة الرقمية)",
                            options=['mean', 'median', 'most_frequent', 'constant'],
                            format_func=lambda x: {
                                'mean': 'المتوسط',
                                'median': 'الوسيط',
                                'most_frequent': 'القيمة الأكثر تكراراً',
                                'constant': 'قيمة ثابتة'
                            }[x]
                        )
                        if numeric_strategy == 'constant':
                            numeric_fill_value = st.text_input("القيمة الثابتة (للأعمدة الرقمية)")
                        else:
                            numeric_fill_value = None
                            
                    with col2:
                        st.write("📝 الأعمدة النصية/الفئوية")
                        categorical_missing_cols = st.multiselect(
                            "اختر الأعمدة النصية لمعالجة القيم المفقودة",
                            options=categorical_cols.tolist(),
                            default=categorical_cols[data[categorical_cols].isnull().any()].tolist()
                        )
                        categorical_strategy = st.selectbox(
                            "استراتيجية معالجة القيم المفقودة (للأعمدة النصية)",
                            options=['most_frequent', 'constant'],
                            format_func=lambda x: {
                                'most_frequent': 'القيمة الأكثر تكراراً',
                                'constant': 'قيمة ثابتة'
                            }[x]
                        )
                        if categorical_strategy == 'constant':
                            categorical_fill_value = st.text_input("القيمة الثابتة (للأعمدة النصية)")
                        else:
                            categorical_fill_value = None
                            
                    if st.button("معالجة القيم المفقودة"):
                        # معالجة الأعمدة الرقمية
                        if numeric_missing_cols:
                            data = self.data_processor.handle_missing_values(
                                data,
                                strategy=numeric_strategy,
                                columns=numeric_missing_cols,
                                fill_value=numeric_fill_value
                            )
                            
                        # معالجة الأعمدة النصية
                        if categorical_missing_cols:
                            data = self.data_processor.handle_missing_values(
                                data,
                                strategy=categorical_strategy,
                                columns=categorical_missing_cols,
                                fill_value=categorical_fill_value
                            )
                            
                        st.session_state.data = data
                        st.success("تم معالجة القيم المفقودة بنجاح!")
                        self._display_data_info(data)
                
                # معالجة القيم الشاذة
                with st.expander("معالجة القيم الشاذة"):
                    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
                    outlier_cols = st.multiselect(
                        "اختر الأعمدة لمعالجة القيم الشاذة",
                        options=numeric_cols.tolist(),
                        default=numeric_cols.tolist()
                    )
                    outlier_method = st.selectbox(
                        "اختر طريقة معالجة القيم الشاذة",
                        options=['iqr', 'zscore'],
                        format_func=lambda x: {
                            'iqr': 'نطاق الربيعات (IQR)',
                            'zscore': 'النقاط المعيارية (Z-Score)'
                        }[x]
                    )
                    threshold = st.slider(
                        "اختر حد الكشف عن القيم الشاذة",
                        min_value=1.0,
                        max_value=5.0,
                        value=1.5,
                        step=0.5
                    )
                    
                    if st.button("معالجة القيم الشاذة"):
                        data = self.data_processor.handle_outliers(
                            data,
                            method=outlier_method,
                            columns=outlier_cols,
                            threshold=threshold
                        )
                        st.session_state.data = data
                        st.success("تم معالجة القيم الشاذة بنجاح!")
                        self._display_data_info(data)
                
                # ترميز المتغيرات الفئوية
                with st.expander("ترميز المتغيرات الفئوية"):
                    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
                    encoding_cols = st.multiselect(
                        "اختر الأعمدة للترميز",
                        options=categorical_cols.tolist(),
                        default=categorical_cols.tolist()
                    )
                    encoding_method = st.selectbox(
                        "اختر طريقة الترميز",
                        options=['label', 'onehot', 'ordinal'],
                        format_func=lambda x: {
                            'label': 'ترميز التسميات',
                            'onehot': 'الترميز الأحادي',
                            'ordinal': 'الترميز الترتيبي'
                        }[x]
                    )
                    
                    if st.button("تطبيق الترميز"):
                        data = self.data_processor.encode_categorical(
                            data,
                            method=encoding_method,
                            columns=encoding_cols
                        )
                        st.session_state.data = data
                        st.success("تم ترميز المتغيرات الفئوية بنجاح!")
                        self._display_data_info(data)
                
                # تطبيع البيانات
                with st.expander("تطبيع البيانات"):
                    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
                    scaling_cols = st.multiselect(
                        "اختر الأعمدة للتطبيع",
                        options=numeric_cols.tolist(),
                        default=numeric_cols.tolist()
                    )
                    scaling_method = st.selectbox(
                        "اختر طريقة التطبيع",
                        options=['standard', 'minmax', 'robust'],
                        format_func=lambda x: {
                            'standard': 'التطبيع المعياري',
                            'minmax': 'تطبيع الحد الأدنى-الأقصى',
                            'robust': 'التطبيع المتين'
                        }[x]
                    )
                    
                    if st.button("تطبيع البيانات"):
                        data = self.data_processor.scale_features(
                            data,
                            method=scaling_method,
                            columns=scaling_cols
                        )
                        st.session_state.data = data
                        st.success("تم تطبيع البيانات بنجاح!")
                        self._display_data_info(data)
                
                # اختيار عمود الهدف للتدريب
                st.subheader("🎯 إعداد التدريب")
                target_column = st.selectbox(
                    "اختر عمود الهدف (target)",
                    options=data.columns.tolist(),
                    index=len(data.columns)-1
                )
                
                if st.button("تحضير البيانات للتدريب"):
                    # تحضير البيانات للتدريب
                    processed_data = self.data_processor.prepare_data(
                        data,
                        target_column=target_column
                    )
                    
                    # حفظ البيانات المعالجة
                    st.session_state.processed_data = {
                        'X': processed_data['X'],
                        'y': processed_data['y'],
                        'feature_names': processed_data.get('feature_names', processed_data['X'].columns.tolist()),
                        'target_column': target_column,
                        'categorical_features': processed_data.get('categorical_features', []),
                        'numerical_features': processed_data.get('numerical_features', []),
                        'preprocessing_info': processed_data.get('preprocessing_info', {})
                    }
                    
                    # التحقق من صحة البيانات المعالجة
                    if not isinstance(st.session_state.processed_data['feature_names'], (list, pd.Index)):
                        st.session_state.processed_data['feature_names'] = processed_data['X'].columns.tolist()
                    
                    st.success("تم تحضير البيانات للتدريب بنجاح!")
                    
                    # عرض البيانات النهائية
                    st.subheader("📊 البيانات النهائية")
                    st.write("المتغيرات المستقلة (X):")
                    st.write(st.session_state.processed_data['X'].to_html(index=False), unsafe_allow_html=True)
                    st.write("المتغير التابع (y):")
                    y_display = pd.DataFrame({target_column: st.session_state.processed_data['y']})
                    st.write(y_display.to_html(index=False), unsafe_allow_html=True)
                    
                self._display_operations_log()
                    
            except Exception as e:
                st.error(f"حدث خطأ أثناء تحميل الملف: {str(e)}")
                return
        
        if st.session_state.data is None:
            st.warning("الرجاء تحميل البيانات أولاً")
            return
            
    def _convert_to_arrow_compatible(self, df: pd.DataFrame) -> pd.DataFrame:
        """تحويل DataFrame إلى تنسيق متوافق مع Arrow"""
        df_converted = df.copy()
        
        # تحويل كل الأعمدة إلى أنواع متوافقة مع Arrow
        for col in df_converted.columns:
            if df_converted[col].dtype == 'object':
                # تحويل الأعمدة النصية إلى str
                df_converted[col] = df_converted[col].fillna('NA').astype(str)
            elif df_converted[col].dtype == 'datetime64[ns]':
                # تحويل التواريخ إلى نص
                df_converted[col] = df_converted[col].fillna('NA').astype(str)
            elif df_converted[col].dtype == 'category':
                # تحويل الفئات إلى نص
                df_converted[col] = df_converted[col].fillna('NA').astype(str)
            else:
                # تحويل القيم المفقودة في الأعمدة الرقمية إلى -999
                df_converted[col] = df_converted[col].fillna(-999)
                
        return df_converted

    def _prepare_data_for_training(self, data: pd.DataFrame):
        """تحضير البيانات للتدريب"""
        try:
            # Identify feature types
            categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Get target column
            target_column = st.selectbox(
                "Select Target Column",
                data.columns.tolist(),
                help="Select the column you want to predict"
            )
            st.session_state.target_column = target_column
            
            # Remove target from features
            if target_column in categorical_features:
                categorical_features.remove(target_column)
            if target_column in numerical_features:
                numerical_features.remove(target_column)
            
            # Missing value strategy
            missing_strategy = st.selectbox(
                "Missing Value Strategy",
                ["mean", "median", "most_frequent", "constant"],
                help="Choose how to handle missing values"
            )
            
            # Process data
            X_processed, y_processed = self.data_processor.fit_transform(
                data=data,
                target_column=target_column,
                categorical_features=categorical_features,
                numerical_features=numerical_features,
                missing_strategy=missing_strategy
            )
            
            # Convert to DataFrame/Series with proper feature names
            X_processed = pd.DataFrame(X_processed, columns=self.data_processor.get_feature_names())
            y_processed = pd.Series(y_processed, name=target_column)
            
            # Store processed data in the correct format
            st.session_state.processed_data = {
                'X': X_processed,
                'y': y_processed,
                'feature_names': self.data_processor.get_feature_names(),
                'target_column': target_column,
                'categorical_features': categorical_features,
                'numerical_features': numerical_features,
                'preprocessing_info': {
                    'missing_strategy': missing_strategy
                }
            }
            
            # Log operation
            self._log_operation(
                "Data Preprocessing",
                f"Processed {len(categorical_features)} categorical and {len(numerical_features)} numerical features"
            )
            
            # Display data info
            st.write("### Processed Data Information")
            st.write(f"Number of samples: {len(X_processed)}")
            st.write(f"Number of features: {X_processed.shape[1]}")
            st.write("\nFeature types:")
            st.write(f"- Categorical features: {len(categorical_features)}")
            st.write(f"- Numerical features: {len(numerical_features)}")
            
            return True
            
        except Exception as e:
            st.error(f"Error during data preparation: {str(e)}")
            st.exception(e)
            return False

    def _display_data_info(self, data: pd.DataFrame):
        """عرض معلومات البيانات"""
        st.write("### 📊 نظرة عامة على البيانات")
        
        # عرض البيانات
        st.write(data.to_html(index=False), unsafe_allow_html=True)
        
        # عرض معلومات الأعمدة
        st.write("#### معلومات الأعمدة")
        
        # تحويل معلومات الأعمدة إلى DataFrame
        dtypes_df = pd.DataFrame({
            'العمود': data.dtypes.index,
            'نوع البيانات': data.dtypes.values.astype(str)
        })
        st.write(dtypes_df.to_html(index=False), unsafe_allow_html=True)
        
        # تحويل معلومات القيم المفقودة إلى DataFrame
        missing_df = pd.DataFrame({
            'العمود': data.columns,
            'القيم المفقودة': data.isnull().sum().values,
            'نسبة القيم المفقودة': (data.isnull().sum() / len(data) * 100).round(2).values,
            'القيم الفريدة': [data[col].nunique() for col in data.columns]
        })
        st.write(missing_df.to_html(index=False), unsafe_allow_html=True)

    def _log_operation(self, operation: str, details: str):
        """تسجيل العملية في سجل العمليات"""
        if 'operations_log' not in st.session_state:
            st.session_state.operations_log = []
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {operation}: {details}"
        st.session_state.operations_log.append(log_entry)

    def _display_operations_log(self):
        """عرض سجل العمليات"""
        if 'operations_log' in st.session_state and st.session_state.operations_log:
            st.write("### 📝 سجل العمليات")
            for entry in reversed(st.session_state.operations_log):
                st.text(entry)

    def _get_processed_data(self):
        """Get processed X and y data from session state"""
        try:
            if not self._validate_processed_data():
                return None, None
                
            processed_data = st.session_state.processed_data
            X = processed_data.get('X')
            y = processed_data.get('y')
            
            if X is not None and y is not None:
                return X, y
                
            st.error("البيانات المعالجة غير متوفرة")
            return None, None
            
        except Exception as e:
            st.error(f"خطأ في استرجاع البيانات المعالجة: {str(e)}")
            return None, None

    def _validate_processed_data(self) -> bool:
        """Validate that processed data exists and is in correct format"""
        if not st.session_state.preprocessing_state.get('is_processed', False):
            st.warning("الرجاء معالجة البيانات أولاً")
            return False
            
        try:
            processed_data = st.session_state.processed_data
            
            # Check required fields
            required_fields = ['X', 'y', 'feature_names', 'target_column']
            if not all(field in processed_data for field in required_fields):
                error_msg = "بيانات معالجة غير مكتملة"
                self._update_preprocessing_state(False, error_msg)
                st.warning(error_msg)
                return False
                
            # Validate data types
            X = processed_data['X']
            y = processed_data['y']
            
            if not isinstance(X, (pd.DataFrame, np.ndarray)):
                error_msg = "تنسيق غير صحيح لمصفوفة الخصائص"
                self._update_preprocessing_state(False, error_msg)
                st.warning(error_msg)
                return False
                
            if not isinstance(y, (pd.Series, np.ndarray)):
                error_msg = "تنسيق غير صحيح لمتجه الهدف"
                self._update_preprocessing_state(False, error_msg)
                st.warning(error_msg)
                return False
                
            return True
            
        except Exception as e:
            error_msg = f"خطأ في التحقق من البيانات المعالجة: {str(e)}"
            self._update_preprocessing_state(False, error_msg)
            st.error(error_msg)
            return False

    def _update_preprocessing_state(self, success: bool, error_msg: str = None):
        """Update preprocessing state in session state"""
        if success:
            st.session_state.preprocessing_state.update({
                'is_processed': True,
                'feature_types': {
                    'categorical': st.session_state.processed_data.get('categorical_features', []),
                    'numerical': st.session_state.processed_data.get('numerical_features', [])
                },
                'preprocessing_params': st.session_state.processed_data.get('preprocessing_info', {}),
                'validation_errors': []
            })
        else:
            st.session_state.preprocessing_state.update({
                'is_processed': False,
                'validation_errors': [error_msg] if error_msg else []
            })

    def show_model_training(self):
        """Show model training section"""
        st.header("🤖 Model Training")
        
        # التحقق من وجود البيانات المعالجة
        if not self._validate_processed_data():
            st.warning("الرجاء معالجة البيانات أولاً في صفحة تحليل البيانات")
            return
            
        # عرض ملخص البيانات المعالجة
        st.write("### ملخص البيانات المعالجة")
        processed_data = st.session_state.processed_data
        st.write(f"- عدد العينات: {len(processed_data['X'])}")
        st.write(f"- عدد الخصائص: {len(processed_data['feature_names'])}")
        st.write(f"- عمود الهدف: {processed_data['target_column']}")
        
        # اختيار نوع التعلم
        learning_type = st.selectbox(
            "اختر نوع التعلم",
            ["Classification", "Regression", "Clustering", "Dimensionality Reduction"],
            help="اختر نوع مهمة التعلم الآلي التي تريد تنفيذها"
        )
        
        # تخزين نوع التعلم في حالة الجلسة
        st.session_state.learning_type = learning_type.lower()
        
        # اختيار طريقة التدريب
        training_method = st.radio(
            "اختر طريقة التدريب",
            ["AutoML", "Custom"],
            help="AutoML: اختيار وضبط النموذج تلقائياً\nCustom: اختيار وتكوين النموذج يدوياً"
        )
        
        if training_method == "AutoML":
            self.show_automl_training()
        else:
            self.show_custom_training()
            
    def show_automl_training(self):
        """Show AutoML training options"""
        st.write("### التدريب التلقائي (AutoML)")
        st.write("البحث التلقائي عن أفضل نموذج للبيانات الخاصة بك")
        
        # التحقق مرة أخرى من البيانات المعالجة
        if not self._validate_processed_data():
            return
            
        # الحصول على البيانات المعالجة
        X, y = self._get_processed_data()
        if X is None or y is None:
            st.error("خطأ في الوصول إلى البيانات المعالجة. الرجاء معالجة البيانات مرة أخرى.")
            return
            
        # عرض حجم البيانات
        st.write(f"- حجم البيانات: {X.shape[0]} عينة, {X.shape[1]} خاصية")
        
        # الخيارات الأساسية للتدريب
        col1, col2 = st.columns(2)
        with col1:
            time_limit = st.slider(
                "الحد الزمني (بالثواني)",
                min_value=10,
                max_value=3600,
                value=60,
                help="الوقت الأقصى للبحث عن أفضل نموذج"
            )
            
            cv_folds = st.slider(
                "عدد طيات التحقق المتقاطع",
                min_value=2,
                max_value=10,
                value=5,
                help="عدد الطيات للتحقق المتقاطع"
            )
            
        with col2:
            n_trials = st.slider(
                "عدد محاولات التحسين",
                min_value=10,
                max_value=100,
                value=30,
                help="عدد المحاولات لتحسين المعلمات"
            )
            
        # خيارات خاصة بنوع التعلم
        learning_type = st.session_state.learning_type
        
        if learning_type == "classification":
            metric = st.selectbox(
                "مقياس التحسين",
                ["accuracy", "precision", "recall", "f1", "roc_auc"],
                help="المقياس المستخدم لتحسين اختيار النموذج"
            )
            
            class_weight = st.selectbox(
                "وزن الفئات",
                ["balanced", "none"],
                help="معالجة عدم توازن الفئات"
            )
            
        elif learning_type == "regression":
            metric = st.selectbox(
                "مقياس التحسين",
                ["r2", "mse", "rmse", "mae", "mape"],
                help="المقياس المستخدم لتحسين اختيار النموذج"
            )
            
        # زر بدء التدريب
        if st.button("بدء التدريب التلقائي"):
            with st.spinner("جارٍ تدريب النماذج..."):
                try:
                    # تحضير معلمات التدريب
                    train_params = {
                        "time_limit": time_limit,
                        "cv_folds": cv_folds,
                        "n_trials": n_trials,
                        "metric": metric,
                        "task_type": learning_type
                    }
                    
                    # إضافة المعلمات الخاصة بنوع المهمة
                    if learning_type == "classification":
                        train_params["class_weight"] = class_weight
                    
                    # بدء التدريب
                    results = self.model_trainer.train_automl(
                        X=X,
                        y=y,
                        **train_params
                    )
                    
                    # تخزين النتائج
                    st.session_state.model_results = results
                    st.success("اكتمل التدريب بنجاح!")
                    
                    # عرض النتائج
                    self.show_model_evaluation()
                    
                except Exception as e:
                    st.error(f"خطأ أثناء التدريب: {str(e)}")
                    st.exception(e)
                    
    def show_model_evaluation(self):
        """Show model evaluation section"""
        st.header("📊 Model Evaluation")
        
        if not hasattr(st.session_state, 'model') or st.session_state.model is None:
            st.warning("Please train a model first.")
            return
            
        learning_type = st.session_state.get('learning_type')
        if learning_type is None:
            st.warning("Learning type not set. Please retrain your model.")
            return
            
        # Create tabs for different evaluation aspects
        eval_tabs = st.tabs(["Performance Metrics", "Visualizations", "Detailed Report"])
        
        with eval_tabs[0]:  # Performance Metrics
            if learning_type == "classification":
                y_pred = st.session_state.model.predict(st.session_state.processed_data['X'])
                y_true = st.session_state.processed_data['y']
                
                metrics = self.model_evaluator.evaluate_classification(y_true, y_pred)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                with col2:
                    st.metric("F1 Score", f"{metrics['f1']:.3f}")
                with col3:
                    st.metric("Precision", f"{metrics['precision']:.3f}")
                    
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Recall", f"{metrics['recall']:.3f}")
                with col2:
                    st.metric("Support", str(metrics['support']))
                    
            elif learning_type == "regression":
                y_pred = st.session_state.model.predict(st.session_state.processed_data['X'])
                y_true = st.session_state.processed_data['y']
                
                metrics = self.model_evaluator.evaluate_regression(y_true, y_pred)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("R² Score", f"{metrics['r2']:.3f}")
                    st.metric("MAE", f"{metrics['mae']:.3f}")
                with col2:
                    st.metric("MSE", f"{metrics['mse']:.3f}")
                    st.metric("RMSE", f"{metrics['rmse']:.3f}")
                    
            elif learning_type == "clustering":
                if hasattr(st.session_state.model, 'labels_'):
                    labels = st.session_state.model.labels_
                else:
                    labels = st.session_state.model.predict(st.session_state.processed_data['X'])
                    
                metrics = self.model_evaluator.evaluate_clustering(
                    st.session_state.processed_data['X'],
                    labels
                )
                
                # Display clustering metrics
                if len(metrics.keys()) > 1:  # More than just 'number_of_clusters'
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Silhouette Score", f"{metrics['silhouette']:.3f}")
                    with col2:
                        st.metric("Calinski-Harabasz Score", f"{metrics['calinski_harabasz']:.3f}")
                    st.metric("Davies-Bouldin Score", f"{metrics['davies_bouldin']:.3f}")
                else:
                    st.warning(metrics.get('note', 'Insufficient clusters for metric calculation'))
                
            else:  # dimensionality_reduction
                if hasattr(st.session_state.model, 'transform'):
                    X_transformed = st.session_state.model.transform(st.session_state.processed_data['X'])
                    
                    metrics = self.model_evaluator.evaluate_dimensionality_reduction(
                        st.session_state.processed_data['X'],
                        X_transformed
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Explained Variance", f"{metrics['explained_variance']:.3f}")
                    with col2:
                        st.metric("Reconstruction Error", f"{metrics['reconstruction_error']:.3f}")
                else:
                    st.warning("This model does not support transformation.")
                    
        with eval_tabs[1]:  # Visualizations
            if learning_type == "classification":
                # Feature importance
                if hasattr(st.session_state.model, "feature_importances_"):
                    fig_imp = self.model_evaluator.plot_feature_importance(
                        st.session_state.model,
                        st.session_state.processed_data['X'].columns
                    )
                    st.plotly_chart(fig_imp)
                    
                # Confusion Matrix
                fig_cm = self.model_evaluator.plot_confusion_matrix(
                    y_true,
                    y_pred
                )
                st.plotly_chart(fig_cm)
                
            elif learning_type == "regression":
                # Feature importance
                if hasattr(st.session_state.model, "feature_importances_"):
                    fig_imp = self.model_evaluator.plot_feature_importance(
                        st.session_state.model,
                        st.session_state.processed_data['X'].columns
                    )
                    st.plotly_chart(fig_imp)
                    
                # Actual vs Predicted
                fig_scatter = self.model_evaluator.plot_regression_scatter(
                    y_true,
                    y_pred
                )
                st.plotly_chart(fig_scatter)
                
                # Residuals
                fig_residuals = self.model_evaluator.plot_residuals(
                    y_true,
                    y_pred
                )
                st.plotly_chart(fig_residuals)
                
            elif learning_type == "clustering":
                # Cluster Visualization
                if st.session_state.processed_data['X'].shape[1] > 2:
                    # Use PCA for visualization if more than 2 dimensions
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    X_2d = pca.fit_transform(st.session_state.processed_data['X'])
                else:
                    X_2d = st.session_state.processed_data['X']
                    
                fig = px.scatter(
                    x=X_2d[:, 0],
                    y=X_2d[:, 1],
                    color=labels.astype(str),
                    title="Cluster Visualization"
                )
                st.plotly_chart(fig)
                
            else:  # dimensionality_reduction
                if hasattr(st.session_state.model, 'transform'):
                    # Plot transformed data
                    fig = px.scatter(
                        x=X_transformed[:, 0],
                        y=X_transformed[:, 1] if X_transformed.shape[1] > 1 else np.zeros(len(X_transformed)),
                        title="Transformed Data Visualization"
                    )
                    st.plotly_chart(fig)
                    
        with eval_tabs[2]:  # Detailed Report
            if learning_type == "classification":
                st.text("Classification Report:")
                report = self.model_evaluator.get_classification_report(y_true, y_pred)
                st.code(report)
                
            elif learning_type == "regression":
                st.text("Regression Report:")
                report = self.model_evaluator.get_regression_report(y_true, y_pred)
                st.code(report)
                
            elif learning_type == "clustering":
                st.text("Clustering Report:")
                report = self.model_evaluator.get_clustering_report(
                    st.session_state.processed_data['X'],
                    labels
                )
                st.code(report)
                
            else:  # dimensionality_reduction
                st.text("Dimensionality Reduction Report:")
                if hasattr(st.session_state.model, 'transform'):
                    report = self.model_evaluator.get_dimensionality_reduction_report(
                        st.session_state.processed_data['X'],
                        X_transformed
                    )
                    st.code(report)
            
    def show_predictions(self):
        """Show predictions section"""
        st.header("🎯 Predictions")
        
        if st.session_state.model is None:
            st.warning("Please train a model first.")
            return
            
        learning_type = st.session_state.get('learning_type')
        if learning_type is None:
            st.warning("Learning type not set. Please retrain your model.")
            return
            
        if learning_type not in ["classification", "regression"]:
            st.info(f"Predictions are not available for {learning_type} tasks. Try using the Model Evaluation page to analyze your results.")
            return
            
        # Prediction mode selection
        pred_mode = st.radio(
            "Select Prediction Mode",
            ["Single Prediction", "Batch Predictions"],
            help="Choose between making a single prediction or predictions for multiple samples"
        )
        
        if pred_mode == "Single Prediction":
            self._show_single_prediction()
        else:
            self._show_batch_predictions()
            
    def _show_single_prediction(self):
        """Show single prediction interface"""
        st.write("### Single Prediction")
        st.write("Enter values for each feature to get a prediction")
        
        # Get feature names and create input fields
        feature_names = st.session_state.processed_data['X'].columns
        X = st.session_state.processed_data['X']
        
        # Create columns for input fields
        col1, col2 = st.columns(2)
        input_data = {}
        
        for i, feature in enumerate(feature_names):
            # Determine min, max, and step values
            feature_min = float(X.iloc[:, i].min())
            feature_max = float(X.iloc[:, i].max())
            feature_range = feature_max - feature_min
            
            # Use appropriate input widget based on the range
            with col1 if i % 2 == 0 else col2:
                if feature_range <= 1.0:
                    input_data[feature] = st.slider(
                        feature,
                        min_value=feature_min,
                        max_value=feature_max,
                        value=float(X.iloc[:, i].mean()),
                        step=0.01
                    )
                else:
                    input_data[feature] = st.number_input(
                        feature,
                        min_value=feature_min,
                        max_value=feature_max,
                        value=float(X.iloc[:, i].mean()),
                        step=0.1
                    )
                    
        if st.button("Get Prediction", type="primary"):
            try:
                # Convert input to array
                X_pred = np.array([input_data[f] for f in feature_names]).reshape(1, -1)
                
                # Make prediction
                if st.session_state.learning_type == "classification":
                    prediction = st.session_state.model.predict(X_pred)[0]
                    
                    # Get prediction probabilities if available
                    if hasattr(st.session_state.model, "predict_proba"):
                        proba = st.session_state.model.predict_proba(X_pred)[0]
                        
                        # Create probability plot
                        fig = px.bar(
                            x=[f"Class {i}" for i in range(len(proba))],
                            y=proba,
                            title="Prediction Probabilities",
                            labels={"x": "Class", "y": "Probability"}
                        )
                        
                        # Display results
                        st.success(f"Predicted Class: {prediction}")
                        st.plotly_chart(fig)
                    else:
                        st.success(f"Predicted Class: {prediction}")
                        
                else:  # Regression
                    prediction = st.session_state.model.predict(X_pred)[0]
                    st.success(f"Predicted Value: {prediction:.3f}")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                
    def _show_batch_predictions(self):
        """Show batch predictions interface"""
        st.write("### Batch Predictions")
        st.write("Upload a CSV file with features to get predictions for multiple samples")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                
                # Check if all required features are present
                required_features = st.session_state.processed_data['X'].columns
                missing_features = [f for f in required_features if f not in df.columns]
                
                if missing_features:
                    st.error(f"Missing features in CSV: {', '.join(missing_features)}")
                    return
                    
                # Get features in correct order
                X_pred = df[required_features].values
                
                # Make predictions
                predictions = st.session_state.model.predict(X_pred)
                
                # Add predictions to dataframe
                if st.session_state.learning_type == "classification":
                    df['Predicted_Class'] = predictions
                    
                    # Add probabilities if available
                    if hasattr(st.session_state.model, "predict_proba"):
                        probas = st.session_state.model.predict_proba(X_pred)
                        for i in range(probas.shape[1]):
                            df[f'Probability_Class_{i}'] = probas[:, i]
                else:
                    df['Predicted_Value'] = predictions
                    
                # Show preview of results
                st.write("### Preview of Results")
                st.write(df.head().to_html(index=False), unsafe_allow_html=True)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download Results",
                    csv,
                    "predictions.csv",
                    "text/csv",
                    key='download-csv'
                )
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                    
def main():
    st.set_page_config(
        page_title="AutoSKL Dashboard",
        page_icon="🤖",
        layout="wide"
    )
    
    dashboard = Dashboard()
    
    # Sidebar navigation
    pages = {
        "Data Analysis": dashboard.show_data_analysis,
        "Model Training": dashboard.show_model_training,
        "Model Evaluation": dashboard.show_model_evaluation,
        "Predictions": dashboard.show_predictions
    }
    
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    # Show selected page
    pages[selection]()
    
    def _train_model(self):
        """تدريب النموذج"""
        try:
            if 'X_train' not in st.session_state or 'y_train' not in st.session_state:
                st.error("الرجاء تحضير البيانات للتدريب أولاً")
                return
                
            X = st.session_state.X_train
            y = st.session_state.y_train
            
            if X is None or y is None:
                st.error("البيانات غير متوفرة للتدريب")
                return
                
            if len(X) == 0 or len(y) == 0:
                st.error("البيانات فارغة")
                return
                
            # تحديد نوع المهمة
            task_type = 'classification' if pd.api.types.is_categorical_dtype(y) or len(np.unique(y)) < 10 else 'regression'
            metric = 'accuracy' if task_type == 'classification' else 'r2'
            
            st.info(f"نوع المهمة: {task_type}")
            st.info(f"المقياس المستخدم: {metric}")
            
            with st.spinner('جاري التدريب...'):
                try:
                    # تدريب النموذج
                    results = self.model_trainer.train_automl(
                        X=X,
                        y=y,
                        task_type=task_type,
                        time_limit=60,
                        metric=metric
                    )
                    
                    if results is None:
                        st.error("فشل التدريب. الرجاء المحاولة مرة أخرى.")
                        return
                        
                    if 'performance' not in results:
                        st.error("لم يتم العثور على مقاييس الأداء")
                        return
                        
                    # عرض نتائج التدريب
                    st.success("تم التدريب بنجاح!")
                    st.write("### 📊 نتائج التدريب")
                    
                    # عرض مقاييس الأداء
                    metrics_df = pd.DataFrame({
                        'المقياس': list(results['performance'].keys()),
                        'القيمة': [f"{v:.4f}" for v in results['performance'].values()]
                    })
                    st.write(metrics_df.to_html(index=False), unsafe_allow_html=True)
                    
                    # تخزين النموذج في session state
                    st.session_state.trained_model = results['model']
                    st.session_state.model_task_type = results['task_type']
                    st.session_state.model_performance = results['performance']
                    
                    # تسجيل العملية
                    self._log_operation(
                        "تدريب النموذج",
                        f"نوع المهمة: {results['task_type']}, "
                        f"أفضل أداء: {max(results['performance'].values()):.4f}"
                    )
                    
                except Exception as train_error:
                    st.error(f"خطأ أثناء التدريب: {str(train_error)}")
                    self.logger.error(f"Error during model training: {str(train_error)}")
                    
        except Exception as e:
            st.error(f"خطأ في إعداد التدريب: {str(e)}")
            self.logger.error(f"Error in _train_model setup: {str(e)}")

if __name__ == "__main__":
    main()
