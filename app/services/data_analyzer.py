import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from feature_engine.outliers import OutlierTrimmer
from feature_engine.selection import DropConstantFeatures, DropCorrelatedFeatures
from category_encoders import TargetEncoder, WOEEncoder, CatBoostEncoder
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import shap
import lime
import lime.lime_tabular
from yellowbrick.features import Rank1D, Rank2D
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from alibi_detect.cd import TabularDrift
import logging

class DataAnalyzer:
    def __init__(self):
        self.numerical_features = []
        self.categorical_features = []
        self.datetime_features = []
        self.target_name = None
        self.feature_importance = {}
        self.drift_detector = None
        
    def analyze_dataset(
        self,
        data: pd.DataFrame,
        target: str,
        task_type: str
    ) -> Dict[str, Any]:
        """تحليل شامل لمجموعة البيانات"""
        self.target_name = target
        analysis = {}
        
        # إحصائيات أساسية
        analysis['basic_stats'] = self._compute_basic_stats(data)
        
        # اكتشاف أنواع المتغيرات
        self._detect_feature_types(data)
        analysis['feature_types'] = {
            'numerical': self.numerical_features,
            'categorical': self.categorical_features,
            'datetime': self.datetime_features
        }
        
        # تحليل القيم المفقودة
        analysis['missing_values'] = self._analyze_missing_values(data)
        
        return analysis
        
    def show_overview(self, data: pd.DataFrame):
        """عرض نظرة عامة على البيانات"""
        st.subheader("📊 نظرة عامة على البيانات")
        
        # معلومات أساسية
        st.write(f"عدد الصفوف: {data.shape[0]}")
        st.write(f"عدد الأعمدة: {data.shape[1]}")
        
        # عرض أول بضعة صفوف
        st.subheader("🔍 عينة من البيانات")
        st.dataframe(data.head())
        
        # معلومات الأنواع
        st.subheader("📋 أنواع البيانات")
        dtypes_df = pd.DataFrame({
            'النوع': data.dtypes,
            'القيم الفريدة': data.nunique(),
            'القيم المفقودة (%)': (data.isnull().sum() / len(data) * 100).round(2)
        })
        st.dataframe(dtypes_df)
        
        # إحصائيات وصفية
        st.subheader("📈 إحصائيات وصفية")
        st.dataframe(data.describe())
        
    def show_variable_analysis(self, data: pd.DataFrame):
        """تحليل تفصيلي للمتغيرات"""
        st.subheader("🔍 تحليل المتغيرات")
        
        # اختيار العمود للتحليل
        column = st.selectbox("اختر العمود للتحليل", data.columns)
        
        if column:
            col_data = data[column]
            
            # معلومات أساسية
            st.write(f"نوع البيانات: {col_data.dtype}")
            st.write(f"عدد القيم الفريدة: {col_data.nunique()}")
            st.write(f"نسبة القيم المفقودة: {(col_data.isnull().sum() / len(col_data) * 100):.2f}%")
            
            # تحليل حسب نوع البيانات
            if np.issubdtype(col_data.dtype, np.number):
                self._analyze_numerical(col_data)
            else:
                self._analyze_categorical(col_data)

    def _analyze_numerical(self, series: pd.Series):
        """تحليل المتغيرات العددية"""
        # إحصائيات
        stats = series.describe()
        st.write("إحصائيات:")
        st.write(stats)
        
        # رسم توزيع البيانات
        fig = px.histogram(
            series,
            title=f"توزيع {series.name}",
            labels={'value': series.name, 'count': 'التكرار'}
        )
        st.plotly_chart(fig)
        
        # رسم box plot
        fig = px.box(
            series,
            title=f"Box Plot - {series.name}"
        )
        st.plotly_chart(fig)
        
    def _analyze_categorical(self, series: pd.Series):
        """تحليل المتغيرات الفئوية"""
        # توزيع القيم
        value_counts = series.value_counts()
        st.write("توزيع القيم:")
        st.write(value_counts)
        
        # رسم بياني للتوزيع
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"توزيع {series.name}",
            labels={'x': series.name, 'y': 'التكرار'}
        )
        st.plotly_chart(fig)
        
        # نسب القيم
        st.write("النسب المئوية:")
        st.write(series.value_counts(normalize=True) * 100)
        
    def show_correlations(self, data: pd.DataFrame):
        """عرض الارتباطات بين المتغيرات"""
        st.subheader("🔗 تحليل الارتباطات")
        
        # حساب مصفوفة الارتباط للمتغيرات العددية
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            corr_matrix = numeric_data.corr()
            
            # رسم خريطة حرارية
            fig = px.imshow(
                corr_matrix,
                title="مصفوفة الارتباط",
                labels=dict(color="معامل الارتباط")
            )
            st.plotly_chart(fig)
            
            # عرض أقوى الارتباطات
            st.subheader("🔝 أقوى الارتباطات")
            correlations = []
            for col1 in corr_matrix.columns:
                for col2 in corr_matrix.columns:
                    if col1 < col2:
                        correlations.append({
                            'المتغير 1': col1,
                            'المتغير 2': col2,
                            'معامل الارتباط': corr_matrix.loc[col1, col2]
                        })
            
            if correlations:
                corr_df = pd.DataFrame(correlations)
                corr_df = corr_df.sort_values('معامل الارتباط', key=abs, ascending=False)
                st.dataframe(corr_df)
        else:
            st.warning("لا توجد متغيرات عددية لحساب الارتباطات")
            
    def show_missing_values(self, data: pd.DataFrame):
        """تحليل القيم المفقودة"""
        st.subheader("❓ تحليل القيم المفقودة")
        
        # حساب القيم المفقودة
        missing = pd.DataFrame({
            'العدد': data.isnull().sum(),
            'النسبة (%)': (data.isnull().sum() / len(data) * 100).round(2)
        })
        missing = missing[missing['العدد'] > 0].sort_values('النسبة (%)', ascending=False)
        
        if not missing.empty:
            st.dataframe(missing)
            
            # رسم بياني للقيم المفقودة
            fig = px.bar(
                missing,
                y=missing.index,
                x='النسبة (%)',
                title="نسب القيم المفقودة",
                orientation='h'
            )
            st.plotly_chart(fig)
            
            # نمط القيم المفقودة
            st.subheader("🔍 نمط القيم المفقودة")
            msno_matrix = data.isnull().astype(int)
            fig = px.imshow(
                msno_matrix.sample(min(100, len(msno_matrix))),
                title="نمط القيم المفقودة (عينة)",
                labels=dict(color="مفقود")
            )
            st.plotly_chart(fig)
        else:
            st.success("لا توجد قيم مفقودة في البيانات!")
            
    def show_plots(self, data: pd.DataFrame):
        """عرض رسوم بيانية متنوعة"""
        st.subheader("📊 الرسوم البيانية")
        
        plot_type = st.selectbox(
            "نوع الرسم البياني",
            ["توزيع المتغيرات", "العلاقات الثنائية", "مخطط التشتت", "مخطط الصندوق"]
        )
        
        if plot_type == "توزيع المتغيرات":
            col = st.selectbox("اختر المتغير", data.columns)
            if col:
                if np.issubdtype(data[col].dtype, np.number):
                    fig = px.histogram(
                        data,
                        x=col,
                        title=f"توزيع {col}"
                    )
                else:
                    fig = px.bar(
                        data[col].value_counts(),
                        title=f"توزيع {col}"
                    )
                st.plotly_chart(fig)
                
        elif plot_type == "العلاقات الثنائية":
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                col1 = st.selectbox("المتغير الأول", numeric_cols)
                col2 = st.selectbox("المتغير الثاني", numeric_cols)
                
                if col1 and col2:
                    fig = px.scatter(
                        data,
                        x=col1,
                        y=col2,
                        title=f"العلاقة بين {col1} و {col2}"
                    )
                    st.plotly_chart(fig)
            else:
                st.warning("يجب وجود متغيرين عدديين على الأقل")
                
        elif plot_type == "مخطط التشتت":
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                cols = st.multiselect("اختر المتغيرات", numeric_cols)
                if len(cols) >= 2:
                    fig = px.scatter_matrix(
                        data[cols],
                        title="مصفوفة التشتت"
                    )
                    st.plotly_chart(fig)
            else:
                st.warning("يجب وجود متغيرين عدديين على الأقل")
                
        elif plot_type == "مخطط الصندوق":
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            col = st.selectbox("اختر المتغير", numeric_cols)
            if col:
                fig = px.box(
                    data,
                    y=col,
                    title=f"مخطط الصندوق - {col}"
                )
                st.plotly_chart(fig)
                
    def _compute_basic_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """حساب إحصائيات أساسية لمجموعة البيانات"""
        stats = {
            'n_samples': len(data),
            'n_features': len(data.columns),
            'memory_usage': data.memory_usage(deep=True).sum() / 1024**2,  # MB
            'dtypes': data.dtypes.value_counts().to_dict(),
            'numeric_stats': data.describe().to_dict(),
        }
        
        # إضافة إحصائيات المتغيرات الفئوية
        cat_columns = data.select_dtypes(include=['object', 'category']).columns
        if len(cat_columns) > 0:
            stats['categorical_stats'] = {
                col: {
                    'unique_values': data[col].nunique(),
                    'top_values': data[col].value_counts().head(5).to_dict()
                }
                for col in cat_columns
            }
            
        return stats
        
    def _detect_feature_types(self, data: pd.DataFrame):
        """اكتشاف أنواع المتغيرات في مجموعة البيانات"""
        for column in data.columns:
            if column == self.target_name:
                continue
                
            if pd.api.types.is_numeric_dtype(data[column]):
                self.numerical_features.append(column)
            elif pd.api.types.is_datetime64_any_dtype(data[column]):
                self.datetime_features.append(column)
            else:
                self.categorical_features.append(column)
                
    def _analyze_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """تحليل القيم المفقودة في مجموعة البيانات"""
        missing = data.isnull().sum()
        missing_pct = (missing / len(data)) * 100
        
        return {
            'total_missing': missing.sum(),
            'missing_by_feature': missing[missing > 0].to_dict(),
            'missing_percentage': missing_pct[missing_pct > 0].to_dict()
        }
        
    def setup_drift_detection(self, reference_data: pd.DataFrame):
        """إعداد كشف الانجراف للبيانات"""
        try:
            self.drift_detector = TabularDrift(
                reference_data.values,
                p_val=.05,
                categories_per_feature={
                    i: None for i in range(reference_data.shape[1])
                }
            )
        except Exception as e:
            logging.error(f"خطأ في إعداد كشف الانجراف: {str(e)}")
            
    def check_drift(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """التحقق من الانجراف في البيانات الجديدة"""
        if self.drift_detector is None:
            raise ValueError("كشف الانجراف غير مفعل. قم بإعداد كشف الانجراف أولا.")
            
        try:
            drift_prediction = self.drift_detector.predict(new_data.values)
            return {
                'drift_detected': bool(drift_prediction['data']['is_drift']),
                'p_value': float(drift_prediction['data']['p_val']),
                'threshold': 0.05,
                'feature_scores': drift_prediction['data'].get('feature_score', {})
            }
        except Exception as e:
            logging.error(f"خطأ في التحقق من الانجراف: {str(e)}")
            return {
                'drift_detected': None,
                'error': str(e)
            }
