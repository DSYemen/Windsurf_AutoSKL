from src.utils.rtl_utils import apply_arabic_config
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import optuna
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import joblib
import json
from datetime import datetime
import os

# تطبيق التكوين العربي
apply_arabic_config(title="التعلم الآلي التلقائي", icon="⚡")

def validate_data():
    """التحقق من صحة البيانات وتوفرها"""
    if "data" not in st.session_state:
        st.error("🚫 يرجى تحميل البيانات أولاً من صفحة إدارة البيانات!")
        return False
        
    df = st.session_state.data
    if df.empty:
        st.error("❌ البيانات المحملة فارغة!")
        return False
        
    if df.isnull().any().any():
        st.warning("⚠️ البيانات تحتوي على قيم مفقودة. يرجى معالجتها أولاً!")
        return False
        
    return True

def create_model_directory():
    """إنشاء مجلد للنماذج إذا لم يكن موجوداً"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        try:
            os.makedirs(models_dir)
        except Exception as e:
            st.error(f"❌ فشل في إنشاء مجلد النماذج: {str(e)}")
            return False
    return True

def create_classification_model(trial):
    """إنشاء نموذج تصنيف مع معاملات محسنة"""
    model_type = trial.suggest_categorical('model_type', ['rf', 'xgb', 'lgb', 'catboost'])
    
    try:
        if model_type == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
            return RandomForestClassifier(**params, random_state=42)
            
        elif model_type == 'xgb':
            import xgboost as xgb
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            return xgb.XGBClassifier(**params, random_state=42)
            
        elif model_type == 'lgb':
            import lightgbm as lgb
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0)
            }
            return lgb.LGBMClassifier(**params, random_state=42)
            
        else:  # catboost
            from catboost import CatBoostClassifier
            params = {
                'iterations': trial.suggest_int('iterations', 50, 300),
                'depth': trial.suggest_int('depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True)
            }
            return CatBoostClassifier(**params, random_state=42, verbose=False)
            
    except Exception as e:
        st.error(f"❌ فشل في إنشاء نموذج التصنيف: {str(e)}")
        return None

def create_regression_model(trial):
    """إنشاء نموذج انحدار مع معاملات محسنة"""
    model_type = trial.suggest_categorical('model_type', ['rf', 'xgb', 'lgb', 'catboost'])
    
    try:
        if model_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
            return RandomForestRegressor(**params, random_state=42)
            
        elif model_type == 'xgb':
            import xgboost as xgb
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            return xgb.XGBRegressor(**params, random_state=42)
            
        elif model_type == 'lgb':
            import lightgbm as lgb
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0)
            }
            return lgb.LGBMRegressor(**params, random_state=42)
            
        else:  # catboost
            from catboost import CatBoostRegressor
            params = {
                'iterations': trial.suggest_int('iterations', 50, 300),
                'depth': trial.suggest_int('depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True)
            }
            return CatBoostRegressor(**params, random_state=42, verbose=False)
            
    except Exception as e:
        st.error(f"❌ فشل في إنشاء نموذج الانحدار: {str(e)}")
        return None

def evaluate_model(model, X_test, y_test, problem_type):
    """تقييم أداء النموذج"""
    try:
        y_pred = model.predict(X_test)
        
        if problem_type == "classification":
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
        else:
            metrics = {
                'r2': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred)
            }
            
        return metrics
        
    except Exception as e:
        st.error(f"❌ فشل في تقييم النموذج: {str(e)}")
        return None

def objective(trial, X_train, X_test, y_train, y_test, problem_type):
    """دالة الهدف لتحسين المعاملات"""
    try:
        model = optimize_hyperparameters(trial, X_train, y_train, problem_type)
        if model is None:
            return float('-inf')
            
        metrics = evaluate_model(model, X_test, y_test, problem_type)
        if metrics is None:
            return float('-inf')
            
        if problem_type == "classification":
            return metrics['f1']
        else:
            return metrics['r2']
            
    except Exception as e:
        st.error(f"❌ فشل في تنفيذ دالة الهدف: {str(e)}")
        return float('-inf')

def optimize_hyperparameters(trial, X_train, y_train, problem_type):
    """تحسين المعاملات باستخدام Optuna"""
    try:
        if problem_type == "classification":
            model = create_classification_model(trial)
        else:
            model = create_regression_model(trial)
            
        model.fit(X_train, y_train)
        return model
        
    except Exception as e:
        st.error(f"❌ فشل في تحسين المعاملات: {str(e)}")
        return None

def save_model(model, model_info):
    """حفظ النموذج ومعلوماته"""
    try:
        # تحضير اسم الملف
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_info['name']}_{timestamp}"
        
        # إنشاء مجلد النماذج إذا لم يكن موجوداً
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # حفظ النموذج
        model_path = os.path.join(models_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        
        # تحضير معلومات النموذج للحفظ
        save_info = {
            'name': model_info['name'],
            'type': model_info['type'],
            'parameters': model_info.get('parameters', {}),
            'metrics': model_info.get('metrics', {}),
            'feature_names': model_info.get('feature_names', []),
            'target_name': model_info.get('target_name', 'target'),
            'target_type': model_info.get('target_type', 'unknown'),
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'file_path': model_path
        }
        
        # إضافة معلومات إضافية إذا كانت متوفرة
        if 'target_values' in model_info and model_info['target_values'] is not None:
            save_info['target_values'] = [str(v) for v in model_info['target_values']]
            
        if 'target_statistics' in model_info:
            stats = model_info['target_statistics']
            save_info['target_statistics'] = {
                'mean': float(stats['mean']) if stats.get('mean') is not None else None,
                'std': float(stats['std']) if stats.get('std') is not None else None,
                'min': float(stats['min']) if stats.get('min') is not None else None,
                'max': float(stats['max']) if stats.get('max') is not None else None,
                'unique_values': int(stats['unique_values']) if 'unique_values' in stats else None,
                'is_numeric': bool(stats.get('is_numeric', False))
            }
        
        # حفظ معلومات النموذج في ملف JSON
        info_path = model_path.replace('.joblib', '_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(save_info, f, ensure_ascii=False, indent=2)
        
        return True, model_path
        
    except Exception as e:
        st.error(f"❌ فشل في حفظ النموذج: {str(e)}")
        return False, None

def save_target_info(target_name, data, problem_type):
    """حفظ معلومات المتغير الهدف"""
    target_series = data[target_name]
    
    # تحضير الإحصائيات الأساسية
    stats = {}
    is_numeric = pd.api.types.is_numeric_dtype(target_series)
    
    if is_numeric:
        stats = {
            'mean': float(target_series.mean()) if not pd.isna(target_series.mean()) else None,
            'std': float(target_series.std()) if not pd.isna(target_series.std()) else None,
            'min': float(target_series.min()) if not pd.isna(target_series.min()) else None,
            'max': float(target_series.max()) if not pd.isna(target_series.max()) else None,
            'unique_values': int(target_series.nunique()),
            'is_numeric': True
        }
    else:
        stats = {
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'unique_values': int(target_series.nunique()),
            'is_numeric': False
        }
    
    target_info = {
        'name': target_name,
        'type': problem_type,
        'dtype': str(target_series.dtype),
        'target_statistics': stats,
        'values': list(map(str, target_series.unique())) if problem_type == 'classification' else None
    }
    
    st.session_state['target'] = target_name
    st.session_state['target_info'] = target_info
    return target_info

def load_target_info():
    """تحميل معلومات المتغير الهدف"""
    if 'target_info' in st.session_state:
        return st.session_state['target_info']
    return None

def display_target_info(target_info):
    """عرض معلومات المتغير الهدف"""
    if target_info:
        st.write("### 📊 معلومات المتغير الهدف")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**نوع النموذج:**", target_info['type'])
            st.write("**اسم المتغير الهدف:**", target_info['name'])
            st.write("**نوع البيانات:**", target_info['dtype'])
            
        with col2:
            if target_info['target_statistics']['is_numeric']:
                st.write("**إحصائيات المتغير الهدف:**")
                stats = target_info['target_statistics']
                st.write(f"- المتوسط: {stats['mean']:.2f}")
                st.write(f"- الانحراف المعياري: {stats['std']:.2f}")
                st.write(f"- القيمة الدنيا: {stats['min']:.2f}")
                st.write(f"- القيمة العليا: {stats['max']:.2f}")
            
            if target_info['values'] is not None:
                st.write("**القيم الفريدة:**", len(target_info['values']))
                st.write("**الفئات:**", ", ".join(map(str, target_info['values'])))

class AutoMLOptimizer:
    """فئة لتحسين نماذج التعلم الآلي تلقائياً"""
    
    def __init__(self, X, y, problem_type, n_trials=50, test_size=0.2):
        """تهيئة محسن التعلم الآلي التلقائي"""
        self.X = X
        self.y = y
        self.problem_type = problem_type
        self.n_trials = n_trials
        self.test_size = test_size
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.best_metrics = None
        self.feature_importance = None
        self.training_history = []
        
    def prepare_data(self):
        """تحضير البيانات للتدريب"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, 
                test_size=self.test_size, 
                random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            st.error(f"❌ فشل في تحضير البيانات: {str(e)}")
            return None, None, None, None
            
    def optimize(self):
        """تحسين النموذج"""
        try:
            # تحضير البيانات
            X_train, X_test, y_train, y_test = self.prepare_data()
            if X_train is None:
                return False
                
            # إنشاء دراسة Optuna
            study = optuna.create_study(
                direction="maximize",
                study_name="automl_optimization"
            )
            
            # تحديث شريط التقدم
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # دالة المراقب
            def callback(study, trial):
                progress = len(study.trials) / self.n_trials
                progress_bar.progress(progress)
                status_text.text(f"التقدم: {progress:.0%} - أفضل نتيجة: {study.best_value:.4f}")
                self.training_history.append({
                    'trial': len(study.trials),
                    'value': trial.value,
                    'best_value': study.best_value
                })
            
            # تحسين النموذج
            study.optimize(
                lambda trial: objective(trial, X_train, X_test, y_train, y_test, self.problem_type),
                n_trials=self.n_trials,
                callbacks=[callback],
                catch=(Exception,)
            )
            
            # حفظ أفضل نموذج ومعلوماته
            self.best_params = study.best_params
            self.best_score = study.best_value
            
            # إنشاء وتدريب أفضل نموذج
            if self.problem_type == "classification":
                self.best_model = create_classification_model(study.best_trial)
            else:
                self.best_model = create_regression_model(study.best_trial)
                
            if self.best_model is None:
                return False
                
            self.best_model.fit(X_train, y_train)
            self.best_metrics = evaluate_model(
                self.best_model, X_test, y_test, self.problem_type
            )
            
            # حساب أهمية المتغيرات
            if hasattr(self.best_model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': self.X.columns,
                    'importance': self.best_model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            progress_bar.progress(1.0)
            status_text.text("✅ اكتمل التحسين!")
            return True
            
        except Exception as e:
            st.error(f"❌ فشل في عملية التحسين: {str(e)}")
            return False
            
    def save_results(self):
        """حفظ نتائج التحسين"""
        if not self.best_model:
            st.error("❌ لا يوجد نموذج محسن للحفظ!")
            return False
            
        model_info = {
            'name': type(self.best_model).__name__,
            'type': self.problem_type,
            'parameters': self.best_params,
            'metrics': self.best_metrics,
            'feature_names': list(self.X.columns),
            'target_name': self.y.name if hasattr(self.y, 'name') else 'target',
            'target_values': list(self.y.unique()) if self.problem_type == 'classification' else None,
            'target_type': str(self.y.dtype),
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'n_trials': self.n_trials,
            'best_score': self.best_score,
            'target_statistics': {
                'mean': float(self.y.mean()) if pd.api.types.is_numeric_dtype(self.y) else None,
                'std': float(self.y.std()) if pd.api.types.is_numeric_dtype(self.y) else None,
                'min': float(self.y.min()) if pd.api.types.is_numeric_dtype(self.y) else None,
                'max': float(self.y.max()) if pd.api.types.is_numeric_dtype(self.y) else None,
                'unique_values': int(self.y.nunique()),
                'is_numeric': bool(pd.api.types.is_numeric_dtype(self.y))
            }
        }
        
        success, model_path = save_model(self.best_model, model_info)
        if success:
            st.success(f"✅ تم حفظ النموذج في: {model_path}")
            
            # حفظ معلومات المتغير الهدف في session state
            st.session_state['target_info'] = {
                'name': model_info['target_name'],
                'type': model_info['type'],
                'values': model_info['target_values'],
                'statistics': model_info['target_statistics']
            }
            return True
        return False

# التخطيط الرئيسي
st.title("⚡ التعلم الآلي التلقائي")

# التحقق من وجود البيانات في الجلسة
if 'data' not in st.session_state:
    st.error("❌ يرجى تحميل وإعداد البيانات أولاً!")
    st.stop()

data = st.session_state.data

# اختيار عمود الهدف
target = None
if 'target' in st.session_state:
    target = st.session_state.target

available_columns = list(data.columns)
selected_target = st.selectbox(
    "📊 اختر عمود الهدف",
    options=available_columns,
    index=available_columns.index(target) if target in available_columns else 0,
    help="اختر العمود الذي تريد التنبؤ به"
)

# تحديث عمود الهدف في session state
if selected_target != target:
    target = selected_target
    
    # تحديد نوع المشكلة تلقائياً
    if pd.api.types.is_numeric_dtype(data[target]):
        unique_values = data[target].nunique()
        if unique_values <= 10:
            default_problem_type = "classification"
        else:
            default_problem_type = "regression"
    else:
        default_problem_type = "classification"
    
    # حفظ معلومات المتغير الهدف
    target_info = save_target_info(target, data, default_problem_type)
    display_target_info(target_info)

# الشريط الجانبي
with st.sidebar:
    st.header("⚙️ إعدادات التحسين")
    
    # إعدادات عامة
    n_trials = st.slider("عدد المحاولات", 10, 300, 100)
    test_size = st.slider("حجم مجموعة الاختبار", 0.1, 0.9, 0.2)
    
    # تحديد نوع المشكلة تلقائياً
    if pd.api.types.is_numeric_dtype(data[target]):
        unique_values = data[target].nunique()
        if unique_values <= 10:  # تصنيف إذا كان عدد القيم الفريدة قليل
            default_problem_type = "classification"
        else:
            default_problem_type = "regression"
    else:
        default_problem_type = "classification"
    
    problem_type = st.selectbox(
        "نوع المشكلة",
        ["classification", "regression"],
        index=0 if default_problem_type == "classification" else 1
    )
    
    # إعدادات التوقف المبكر
    use_early_stopping = st.checkbox("تفعيل التوقف المبكر", True)
    if use_early_stopping:
        n_early_stopping = st.number_input("عدد المحاولات للتوقف المبكر", 5, 50, 10)
    
    # اختيار النماذج
    st.subheader("🤖 النماذج المستخدمة")
    use_rf = st.checkbox("Random Forest", True)
    use_xgb = st.checkbox("XGBoost", True)
    use_lgb = st.checkbox("LightGBM", True)
    use_catboost = st.checkbox("CatBoost", True)
    
    # إعدادات متقدمة
    with st.expander("🔧 إعدادات متقدمة"):
        pruning_enabled = st.checkbox("تفعيل التقليم", True)
        parallel_jobs = st.slider("عدد العمليات المتوازية", -1, 8, -1)
        random_seed = st.number_input("البذرة العشوائية", 0, 9999, 42)

try:
    X = data.drop(columns=[target])
    y = data[target]

    # إنشاء محسن AutoML
    optimizer = AutoMLOptimizer(
        X, y, problem_type, n_trials=n_trials, test_size=test_size
    )
    
    # بدء التحسين
    if st.button("🚀 بدء التحسين", type="primary"):
        success = optimizer.optimize()
        if success:
            # حفظ النموذج ومعلومات المتغير الهدف
            optimizer.save_results()
            
            # حفظ اسم المتغير الهدف في session state
            st.session_state['target'] = target
            st.session_state['target_column'] = target
            
            # عرض رسالة نجاح
            st.success(f"✅ تم حفظ النموذج بنجاح مع المتغير الهدف: {target}")
            
            # عرض معلومات المتغير الهدف
            if 'target_info' in st.session_state:
                st.write("### 📊 معلومات المتغير الهدف")
                target_info = st.session_state['target_info']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**نوع النموذج:**", target_info['type'])
                    st.write("**اسم المتغير الهدف:**", target_info['name'])
                
                with col2:
                    if target_info['statistics']['is_numeric']:
                        st.write("**إحصائيات المتغير الهدف:**")
                        st.write(f"- المتوسط: {target_info['statistics']['mean']:.2f}")
                        st.write(f"- الانحراف المعياري: {target_info['statistics']['std']:.2f}")
                        st.write(f"- القيمة الدنيا: {target_info['statistics']['min']:.2f}")
                        st.write(f"- القيمة العليا: {target_info['statistics']['max']:.2f}")
                    
                    if target_info['values'] is not None:
                        st.write("**القيم الفريدة:**", len(target_info['values']))
                        st.write("**الفئات:**", ", ".join(map(str, target_info['values'])))

except Exception as e:
    st.error(f"حدث خطأ أثناء تحضير البيانات: {str(e)}")
    st.stop()
