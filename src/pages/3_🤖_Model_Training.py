from src.utils.rtl_utils import apply_arabic_config
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import joblib
from datetime import datetime
import json
import os

# تطبيق التكوين العربي
apply_arabic_config(title="تدريب النموذج", icon="🤖")

# التحقق من وجود البيانات
if "data" not in st.session_state:
    st.error("🚫 يرجى تحميل البيانات أولاً من صفحة إدارة البيانات!")
    st.stop()

# تحميل البيانات
df = st.session_state.data.copy()

# اختيار المتغيرات المستقلة والتابعة
st.write("### 📊 اختيار المتغيرات")
target = st.selectbox("اختر المتغير التابع (الهدف)", df.columns)
features = st.multiselect(
    "اختر المتغيرات المستقلة",
    [col for col in df.columns if col != target],
    default=[col for col in df.columns if col != target]
)

if not features:
    st.warning("⚠️ يرجى اختيار متغير مستقل واحد على الأقل!")
    st.stop()

# تحضير البيانات
X = df[features]
y = df[target]

# تقسيم البيانات
test_size = st.slider("نسبة بيانات الاختبار", 0.1, 0.4, 0.2)
random_state = st.number_input("البذرة العشوائية", 0, 999999, 42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state
)

# اختيار نوع المشكلة والنموذج
problem_type = st.radio(
    "نوع المشكلة",
    ["تصنيف", "انحدار"],
    horizontal=True
)

model_type = st.selectbox(
    "اختر النموذج",
    ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]
)

# تكوين معاملات النموذج
st.write("### ⚙️ معاملات النموذج")

if model_type == "Random Forest":
    n_estimators = st.slider("عدد الأشجار", 10, 500, 100)
    max_depth = st.slider("أقصى عمق", 1, 50, 10)
    min_samples_split = st.slider("الحد الأدنى للانقسام", 2, 20, 2)
    min_samples_leaf = st.slider("الحد الأدنى للأوراق", 1, 10, 1)
    
    if problem_type == "تصنيف":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

elif model_type == "XGBoost":
    n_estimators = st.slider("عدد الأشجار", 10, 500, 100)
    max_depth = st.slider("أقصى عمق", 1, 20, 6)
    learning_rate = st.slider("معدل التعلم", 0.01, 0.3, 0.1)
    subsample = st.slider("نسبة العينات", 0.5, 1.0, 0.8)
    colsample_bytree = st.slider("نسبة المتغيرات", 0.5, 1.0, 0.8)
    
    if problem_type == "تصنيف":
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state
        )
    else:
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state
        )

elif model_type == "LightGBM":
    n_estimators = st.slider("عدد الأشجار", 10, 500, 100)
    max_depth = st.slider("أقصى عمق", -1, 20, -1)
    learning_rate = st.slider("معدل التعلم", 0.01, 0.3, 0.1)
    num_leaves = st.slider("عدد الأوراق", 20, 100, 31)
    
    if problem_type == "تصنيف":
        model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state
        )
    else:
        model = LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state
        )

else:  # CatBoost
    iterations = st.slider("عدد التكرارات", 10, 500, 100)
    depth = st.slider("العمق", 1, 16, 6)
    learning_rate = st.slider("معدل التعلم", 0.01, 0.3, 0.1)
    
    if problem_type == "تصنيف":
        model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=False
        )
    else:
        model = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=False
        )

# تدريب النموذج
if st.button("🚀 تدريب النموذج", type="primary"):
    try:
        with st.spinner("جاري تدريب النموذج..."):
            model.fit(X_train, y_train)
        
        st.session_state.trained_model = model
        st.session_state.model_metrics = {}
        
        st.success("✅ تم تدريب النموذج بنجاح!")
        
        # تقييم النموذج
        y_pred = model.predict(X_test)
        
        if problem_type == "تصنيف":
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            st.session_state.model_metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("الدقة", f"{accuracy:.4f}")
            col2.metric("الضبط", f"{precision:.4f}")
            col3.metric("الاسترجاع", f"{recall:.4f}")
            col4.metric("F1", f"{f1:.4f}")
            
            # مصفوفة الارتباك
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(
                cm,
                labels=dict(x="التنبؤ", y="القيمة الحقيقية"),
                title="مصفوفة الارتباك"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            st.session_state.model_metrics = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MSE", f"{mse:.4f}")
            col2.metric("RMSE", f"{rmse:.4f}")
            col3.metric("MAE", f"{mae:.4f}")
            col4.metric("R²", f"{r2:.4f}")
            
            # رسم القيم الحقيقية مقابل المتنبأ بها
            fig = px.scatter(
                x=y_test,
                y=y_pred,
                labels={"x": "القيم الحقيقية", "y": "القيم المتنبأ بها"},
                title="القيم الحقيقية مقابل المتنبأ بها"
            )
            fig.add_trace(
                go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode="lines",
                    name="خط التطابق المثالي"
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # أهمية المتغيرات
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "المتغير": features,
                "الأهمية": model.feature_importances_
            }).sort_values("الأهمية", ascending=False)
            
            fig = px.bar(
                importance_df,
                x="الأهمية",
                y="المتغير",
                orientation="h",
                title="أهمية المتغيرات"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.session_state.model_info = {
            "name": model_type,
            "type": problem_type,
            "features": features,
            "target": target,
            "parameters": model.get_params(),
        }
        
    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء تدريب النموذج: {str(e)}")

# حفظ النموذج (خارج بلوك التدريب)
if "trained_model" in st.session_state:
    if st.button("💾 حفظ النموذج"):
        try:
            # إنشاء مجلد للنماذج
            os.makedirs("models", exist_ok=True)
            
            # حفظ النموذج
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join("models", f"model_{timestamp}.joblib")
            joblib.dump(st.session_state.trained_model, model_path)
            
            # تجهيز معلومات النموذج
            model_info = st.session_state.model_info.copy()
            model_info["metrics"] = st.session_state.model_metrics
            model_info["training_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # حفظ معلومات النموذج
            info_path = os.path.join("models", f"model_{timestamp}_info.json")
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(model_info, f, ensure_ascii=False, indent=4)
            
            st.success(f"✅ تم حفظ النموذج في: {model_path}")
            st.success(f"✅ تم حفظ معلومات النموذج في: {info_path}")
            
        except Exception as e:
            st.error(f"❌ حدث خطأ أثناء حفظ النموذج: {str(e)}")
