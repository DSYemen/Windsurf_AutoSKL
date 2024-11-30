from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
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
        """Comprehensive analysis of the dataset"""
        self.target_name = target
        analysis = {}
        
        # Basic statistics
        analysis['basic_stats'] = self._compute_basic_stats(data)
        
        # Feature types detection
        self._detect_feature_types(data)
        analysis['feature_types'] = {
            'numerical': self.numerical_features,
            'categorical': self.categorical_features,
            'datetime': self.datetime_features
        }
        
        # Missing values analysis
        analysis['missing_values'] = self._analyze_missing_values(data)
        
        # Outlier analysis
        analysis['outliers'] = self._detect_outliers(data)
        
        # Feature correlations
        analysis['correlations'] = self._analyze_correlations(data)
        
        # Class imbalance (for classification)
        if task_type == 'classification':
            analysis['class_balance'] = self._analyze_class_balance(data[target])
            
        # Feature importance (preliminary)
        analysis['feature_importance'] = self._analyze_feature_importance(
            data.drop(columns=[target]),
            data[target],
            task_type
        )
        
        # Data quality report
        analysis['quality_report'] = self._generate_quality_report(data)
        
        return analysis
        
    def _compute_basic_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute basic statistics of the dataset"""
        stats = {
            'n_samples': len(data),
            'n_features': len(data.columns),
            'memory_usage': data.memory_usage(deep=True).sum() / 1024**2,  # MB
            'dtypes': data.dtypes.value_counts().to_dict(),
            'numeric_stats': data.describe().to_dict(),
        }
        
        # Add categorical statistics
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
        """Detect types of features in the dataset"""
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
        """Analyze missing values in the dataset"""
        missing = data.isnull().sum()
        missing_pct = (missing / len(data)) * 100
        
        return {
            'total_missing': missing.sum(),
            'missing_by_feature': missing[missing > 0].to_dict(),
            'missing_percentage': missing_pct[missing_pct > 0].to_dict()
        }
        
    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numerical features"""
        outliers = {}
        
        for feature in self.numerical_features:
            if feature == self.target_name:
                continue
                
            Q1 = data[feature].quantile(0.25)
            Q3 = data[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers[feature] = {
                'count': len(data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]),
                'percentage': len(data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]) / len(data) * 100,
                'bounds': {'lower': lower_bound, 'upper': upper_bound}
            }
            
        return outliers
        
    def _analyze_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between features"""
        correlations = {}
        
        # Numerical correlations
        if len(self.numerical_features) > 1:
            num_corr = data[self.numerical_features].corr()
            # Get highly correlated pairs
            high_corr = np.where(np.abs(num_corr) > 0.8)
            high_corr = [(num_corr.index[x], num_corr.columns[y], num_corr.iloc[x, y])
                        for x, y in zip(*high_corr) if x != y and x < y]
            correlations['numerical'] = high_corr
            
        # Categorical correlations (Cramer's V)
        if len(self.categorical_features) > 1:
            cat_corr = []
            for i, feat1 in enumerate(self.categorical_features):
                for feat2 in self.categorical_features[i+1:]:
                    cramers_v = self._cramers_v(data[feat1], data[feat2])
                    if cramers_v > 0.5:  # Only store strong correlations
                        cat_corr.append((feat1, feat2, cramers_v))
            correlations['categorical'] = cat_corr
            
        return correlations
        
    def _cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate Cramer's V statistic between two categorical variables"""
        confusion_matrix = pd.crosstab(x, y)
        chi2 = pd.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
        
    def _analyze_class_balance(self, target: pd.Series) -> Dict[str, Any]:
        """Analyze class balance for classification tasks"""
        value_counts = target.value_counts()
        class_distribution = (value_counts / len(target) * 100).to_dict()
        
        imbalance_ratio = value_counts.max() / value_counts.min()
        is_imbalanced = imbalance_ratio > 3  # Arbitrary threshold
        
        return {
            'class_distribution': class_distribution,
            'imbalance_ratio': imbalance_ratio,
            'is_imbalanced': is_imbalanced,
            'recommended_sampling': 'SMOTE' if is_imbalanced else None
        }
        
    def _analyze_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str
    ) -> Dict[str, float]:
        """Analyze feature importance using various methods"""
        try:
            # Use SHAP for initial feature importance
            if task_type == 'classification':
                from sklearn.ensemble import RandomForestClassifier as RF
            else:
                from sklearn.ensemble import RandomForestRegressor as RF
            
            model = RF(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Get feature importance
            importance_dict = {}
            for i, col in enumerate(X.columns):
                importance_dict[col] = np.abs(shap_values[:, i]).mean()
                
            # Normalize importance values
            total = sum(importance_dict.values())
            importance_dict = {k: v/total for k, v in importance_dict.items()}
            
            self.feature_importance = importance_dict
            return importance_dict
            
        except Exception as e:
            logging.error(f"Error in feature importance analysis: {str(e)}")
            return {}
            
    def _generate_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        report = Report(metrics=[
            DataQualityPreset(),
            DataDriftPreset()
        ])
        
        try:
            report.run(reference_data=data, current_data=data)
            return report.json()
        except Exception as e:
            logging.error(f"Error generating quality report: {str(e)}")
            return {}
            
    def get_recommended_preprocessing(self) -> Dict[str, Any]:
        """Get recommended preprocessing steps based on analysis"""
        recommendations = {
            'scaling': None,
            'encoding': None,
            'imputation': None,
            'feature_selection': None,
            'sampling': None
        }
        
        # Scaling recommendation
        if self.numerical_features:
            if any(self.feature_importance.get(f, 0) > 0.1 for f in self.numerical_features):
                recommendations['scaling'] = 'StandardScaler'
            else:
                recommendations['scaling'] = 'RobustScaler'
                
        # Encoding recommendation
        if self.categorical_features:
            if len(self.categorical_features) > 10:
                recommendations['encoding'] = 'TargetEncoder'
            else:
                recommendations['encoding'] = 'OneHotEncoder'
                
        # Imputation recommendation
        missing_analysis = self._analyze_missing_values(pd.DataFrame())
        if missing_analysis['total_missing'] > 0:
            if missing_analysis['total_missing'] / len(self.numerical_features) < 0.1:
                recommendations['imputation'] = 'SimpleImputer'
            else:
                recommendations['imputation'] = 'KNNImputer'
                
        # Feature selection recommendation
        if len(self.feature_importance) > 20:
            recommendations['feature_selection'] = 'SelectFromModel'
            
        return recommendations
        
    def setup_drift_detection(self, reference_data: pd.DataFrame):
        """Setup drift detection for monitoring"""
        try:
            self.drift_detector = TabularDrift(
                reference_data.values,
                p_val=.05,
                categories_per_feature={
                    i: None for i in range(reference_data.shape[1])
                }
            )
        except Exception as e:
            logging.error(f"Error setting up drift detection: {str(e)}")
            
    def check_drift(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Check for data drift in new data"""
        if self.drift_detector is None:
            raise ValueError("Drift detector not initialized. Call setup_drift_detection first.")
            
        try:
            drift_prediction = self.drift_detector.predict(new_data.values)
            return {
                'drift_detected': bool(drift_prediction['data']['is_drift']),
                'p_value': float(drift_prediction['data']['p_val']),
                'threshold': 0.05,
                'feature_scores': drift_prediction['data'].get('feature_score', {})
            }
        except Exception as e:
            logging.error(f"Error checking drift: {str(e)}")
            return {
                'drift_detected': None,
                'error': str(e)
            }
