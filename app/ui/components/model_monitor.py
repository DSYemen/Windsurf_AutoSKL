import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from ...core.database import DatabaseManager

class ModelMonitor:
    def __init__(self):
        self.db_manager = DatabaseManager()
        
    def calculate_drift(self, training_data: pd.DataFrame, new_data: pd.DataFrame) -> Dict[str, float]:
        """حساب انحراف البيانات"""
        drift_scores = {}
        
        for column in training_data.columns:
            if pd.api.types.is_numeric_dtype(training_data[column]):
                # حساب KL divergence للمتغيرات العددية
                train_hist, _ = np.histogram(training_data[column], bins=50, density=True)
                new_hist, _ = np.histogram(new_data[column], bins=50, density=True)
                
                # تجنب القسمة على صفر
                train_hist = np.clip(train_hist, 1e-10, None)
                new_hist = np.clip(new_hist, 1e-10, None)
                
                kl_div = np.sum(train_hist * np.log(train_hist / new_hist))
                drift_scores[column] = float(kl_div)
            else:
                # حساب تغير التوزيع للمتغيرات الفئوية
                train_dist = training_data[column].value_counts(normalize=True)
                new_dist = new_data[column].value_counts(normalize=True)
                
                # توحيد الفئات
                all_categories = set(train_dist.index) | set(new_dist.index)
                for cat in all_categories:
                    if cat not in train_dist:
                        train_dist[cat] = 0
                    if cat not in new_dist:
                        new_dist[cat] = 0
                
                dist_diff = np.abs(train_dist - new_dist).mean()
                drift_scores[column] = float(dist_diff)
                
        return drift_scores
        
    def monitor_predictions(self, model_id: int, predictions: np.ndarray, 
                          actual: Optional[np.ndarray] = None):
        """مراقبة تنبؤات النموذج"""
        timestamp = datetime.now()
        
        # حساب إحصائيات التنبؤات
        stats = {
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions))
        }
        
        if actual is not None:
            # حساب مقاييس الأداء
            if len(np.unique(actual)) < 10:  # تصنيف
                from sklearn.metrics import accuracy_score, f1_score
                stats['accuracy'] = float(accuracy_score(actual, predictions))
                stats['f1_score'] = float(f1_score(actual, predictions, average='weighted'))
            else:  # انحدار
                from sklearn.metrics import mean_squared_error, r2_score
                stats['mse'] = float(mean_squared_error(actual, predictions))
                stats['r2'] = float(r2_score(actual, predictions))
        
        # تخزين الإحصائيات في قاعدة البيانات
        self.db_manager.log_predictions(
            model_id=model_id,
            timestamp=timestamp,
            statistics=stats
        )
        
    def show_monitoring_dashboard(self, model_id: int, time_range: int = 30):
        """عرض لوحة مراقبة أداء النموذج"""
        st.title("📊 لوحة مراقبة النموذج")
        
        # جلب بيانات المراقبة
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_range)
        monitoring_data = self.db_manager.get_monitoring_data(
            model_id=model_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not monitoring_data:
            st.warning("لا توجد بيانات مراقبة متاحة للفترة المحددة")
            return
            
        # تحويل البيانات إلى DataFrame
        df = pd.DataFrame(monitoring_data)
        
        # عرض مقاييس الأداء عبر الزمن
        st.subheader("📈 تطور الأداء")
        metrics = [col for col in df.columns if col not in ['timestamp', 'model_id']]
        
        for metric in metrics:
            fig = px.line(
                df,
                x='timestamp',
                y=metric,
                title=f"تطور {metric} عبر الزمن",
                markers=True
            )
            st.plotly_chart(fig)
            
        # عرض إحصائيات ملخصة
        st.subheader("📊 إحصائيات الأداء")
        summary_df = df[metrics].describe()
        st.dataframe(summary_df)
        
        # تحليل الانحرافات
        st.subheader("🔍 تحليل الانحرافات")
        for metric in metrics:
            # حساب الانحراف المعياري
            mean = df[metric].mean()
            std = df[metric].std()
            threshold = 2 * std  # انحرافان معياريان
            
            anomalies = df[abs(df[metric] - mean) > threshold]
            if not anomalies.empty:
                st.warning(f"تم اكتشاف {len(anomalies)} قيم شاذة في {metric}")
                st.dataframe(anomalies[['timestamp', metric]])
