from typing import Dict, Any, Optional, List, Tuple, Callable, Union
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    make_scorer, accuracy_score, mean_squared_error, r2_score,
    f1_score, precision_score, recall_score, roc_auc_score,
    silhouette_score, davies_bouldin_score, mean_absolute_error
)
from sklearn import (
    ensemble, linear_model, svm, neighbors,
    neural_network, tree, naive_bayes
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import logging
import optuna
from optuna.pruners import MedianPruner
from ..core.config import Settings
import psutil
import pandas as pd
import sparse
import xgboost as xgb
import lightgbm as lgb
import numpy.typing as npt
from ..core.database import DatabaseManager
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ModelTrainer:
    def __init__(self):
        """Initialize ModelTrainer"""
        self.logger = logging.getLogger(__name__)
        self.best_model = None
        self.best_params = None
        self.task_type = None
        self.label_encoder = None
        self.settings = Settings()
        self.feature_importance = None
        self.metrics = {}
        self.db_manager = DatabaseManager()
        
        # Check available optional libraries
        try:
            import xgboost as xgb
            self.xgb_available = True
            self.logger.info("XGBoost is available")
        except ImportError:
            self.xgb_available = False
            self.logger.warning("XGBoost is not available")
            
        try:
            import lightgbm as lgb
            self.lgb_available = True
            self.logger.info("LightGBM is available")
        except ImportError:
            self.lgb_available = False
            self.logger.warning("LightGBM is not available")
        
    def _detect_task_type(self, y: np.ndarray) -> str:
        """Detect if the task is classification or regression"""
        unique_values = np.unique(y)
        if len(unique_values) < 10 or isinstance(y[0], (str, bool)):
            return 'classification'
        if np.all(np.mod(y, 1) == 0) and len(unique_values) < 100:
            return 'classification'
        return 'regression'
        
    def _get_models(self, task_type: str, data_size: int = None) -> Dict[str, Any]:
        """الحصول على قاموس النماذج بناءً على نوع المهمة وخصائص البيانات"""
        models = {}
        
        if task_type == 'regression':
            # نماذج الأشجار
            models['RandomForestRegressor'] = {
                'class': ensemble.RandomForestRegressor,
                'group': 'نماذج الأشجار',
                'arabic_name': 'الغابة العشوائية'
            }
            models['GradientBoostingRegressor'] = {
                'class': ensemble.GradientBoostingRegressor,
                'group': 'نماذج الأشجار',
                'arabic_name': 'التعزيز المتدرج'
            }
            
            # النماذج الخطية
            models['LinearRegression'] = {
                'class': linear_model.LinearRegression,
                'group': 'النماذج الخطية',
                'arabic_name': 'الانحدار الخطي'
            }
            models['Ridge'] = {
                'class': linear_model.Ridge,
                'group': 'النماذج الخطية',
                'arabic_name': 'انحدار ريدج'
            }
            models['Lasso'] = {
                'class': linear_model.Lasso,
                'group': 'النماذج الخطية',
                'arabic_name': 'انحدار لاسو'
            }
            
            # نماذج SVM
            models['SVR'] = {
                'class': svm.SVR,
                'group': 'نماذج SVM',
                'arabic_name': 'انحدار المتجهات الداعمة'
            }
            
            # نماذج الجيران
            models['KNeighborsRegressor'] = {
                'class': neighbors.KNeighborsRegressor,
                'group': 'نماذج الجيران',
                'arabic_name': 'انحدار k-جار'
            }
            
        elif task_type == 'classification':
            # نماذج الأشجار
            models['RandomForestClassifier'] = {
                'class': ensemble.RandomForestClassifier,
                'group': 'نماذج الأشجار',
                'arabic_name': 'الغابة العشوائية'
            }
            models['GradientBoostingClassifier'] = {
                'class': ensemble.GradientBoostingClassifier,
                'group': 'نماذج الأشجار',
                'arabic_name': 'التعزيز المتدرج'
            }
            
            # النماذج الخطية
            models['LogisticRegression'] = {
                'class': linear_model.LogisticRegression,
                'group': 'النماذج الخطية',
                'arabic_name': 'الانحدار اللوجستي'
            }
            
            # نماذج SVM
            models['SVC'] = {
                'class': svm.SVC,
                'group': 'نماذج SVM',
                'arabic_name': 'تصنيف المتجهات الداعمة'
            }
            
            # نماذج الجيران
            models['KNeighborsClassifier'] = {
                'class': neighbors.KNeighborsClassifier,
                'group': 'نماذج الجيران',
                'arabic_name': 'تصنيف k-جار'
            }
            
            # نماذج بايز
            models['GaussianNB'] = {
                'class': naive_bayes.GaussianNB,
                'group': 'نماذج بايز',
                'arabic_name': 'بايز البسيط'
            }
            
        return models
        
    def _get_hyperparameter_space(self, model_class: Any) -> Dict[str, Any]:
        """Define hyperparameter search space based on model type"""
        params = {}
        
        # Tree-based Models
        if model_class in [ensemble.RandomForestClassifier, ensemble.RandomForestRegressor]:
            params.update({
                'n_estimators': ('int', 50, 300),
                'max_depth': ('int', 3, 20),
                'min_samples_split': ('int', 2, 20),
                'min_samples_leaf': ('int', 1, 10),
                'max_features': ('categorical', ['sqrt', 'log2', None]),
                'bootstrap': ('categorical', [True, False])
            })
        elif model_class in [ensemble.GradientBoostingClassifier, ensemble.GradientBoostingRegressor]:
            params.update({
                'n_estimators': ('int', 50, 300),
                'max_depth': ('int', 3, 10),
                'learning_rate': ('float', 0.01, 0.3),
                'min_samples_split': ('int', 2, 20),
                'subsample': ('float', 0.6, 1.0)
            })
        elif model_class in [xgb.XGBClassifier, xgb.XGBRegressor]:
            params.update({
                'n_estimators': ('int', 50, 300),
                'max_depth': ('int', 3, 10),
                'learning_rate': ('float', 0.01, 0.3),
                'min_child_weight': ('int', 1, 7),
                'subsample': ('float', 0.6, 1.0),
                'colsample_bytree': ('float', 0.6, 1.0),
                'gamma': ('float', 1e-8, 1.0),
                'reg_alpha': ('float', 1e-8, 1.0),
                'reg_lambda': ('float', 1e-8, 1.0)
            })
        elif model_class in [lgb.LGBMClassifier, lgb.LGBMRegressor]:
            params.update({
                'n_estimators': ('int', 50, 300),
                'max_depth': ('int', 3, 10),
                'learning_rate': ('float', 0.01, 0.3),
                'num_leaves': ('int', 20, 100),
                'min_child_samples': ('int', 5, 100),
                'subsample': ('float', 0.6, 1.0),
                'colsample_bytree': ('float', 0.6, 1.0),
                'reg_alpha': ('float', 1e-8, 1.0),
                'reg_lambda': ('float', 1e-8, 1.0)
            })
            
        # Linear Models
        elif model_class in [linear_model.LogisticRegression]:
            params.update({
                'C': ('float', 0.1, 10.0),
                'penalty': ('categorical', ['l1', 'l2', 'elasticnet']),
                'solver': ('categorical', ['lbfgs', 'saga']),
                'max_iter': ('int', 100, 500)
            })
        elif model_class == linear_model.LinearRegression:
            params.update({
                'fit_intercept': ('categorical', [True, False]),
                'positive': ('categorical', [True, False])
            })
        elif model_class in [linear_model.Lasso, linear_model.Ridge]:
            params.update({
                'alpha': ('float', 0.0001, 10.0),
                'fit_intercept': ('categorical', [True, False]),
                'max_iter': ('int', 100, 1000)
            })
        elif model_class == linear_model.ElasticNet:
            params.update({
                'alpha': ('float', 0.0001, 10.0),
                'l1_ratio': ('float', 0.0, 1.0),
                'fit_intercept': ('categorical', [True, False]),
                'max_iter': ('int', 100, 1000)
            })
            
        # SVM
        elif model_class in [svm.SVC, svm.SVR]:
            params.update({
                'C': ('float', 0.1, 10.0),
                'kernel': ('categorical', ['rbf', 'linear', 'poly']),
                'gamma': ('categorical', ['scale', 'auto']),
                'degree': ('int', 2, 5)  # Only for poly kernel
            })
            
        # Neural Networks
        elif model_class in [neural_network.MLPClassifier, neural_network.MLPRegressor]:
            params.update({
                'hidden_layer_sizes': ('categorical', [(50,), (100,), (50, 50), (100, 50)]),
                'activation': ('categorical', ['relu', 'tanh']),
                'learning_rate_init': ('float', 0.0001, 0.1),
                'max_iter': ('int', 100, 500),
                'alpha': ('float', 0.0001, 0.01)
            })
            
        # Decision Trees
        elif model_class in [tree.DecisionTreeClassifier, tree.DecisionTreeRegressor]:
            params.update({
                'max_depth': ('int', 3, 20),
                'min_samples_split': ('int', 2, 20),
                'min_samples_leaf': ('int', 1, 10),
                'criterion': ('categorical', ['gini', 'entropy'] if 'Classifier' in str(model_class) else ['squared_error', 'friedman_mse'])
            })
            
        # Clustering Models
        elif model_class == KMeans:
            params.update({
                'n_clusters': ('int', 2, 20),
                'init': ('categorical', ['k-means++', 'random']),
                'max_iter': ('int', 100, 500)
            })
        elif model_class == DBSCAN:
            params.update({
                'eps': ('float', 0.1, 2.0),
                'min_samples': ('int', 2, 10),
                'metric': ('categorical', ['euclidean', 'manhattan'])
            })
        elif model_class == AgglomerativeClustering:
            params.update({
                'n_clusters': ('int', 2, 20),
                'linkage': ('categorical', ['ward', 'complete', 'average']),
                'affinity': ('categorical', ['euclidean', 'manhattan'])
            })
            
        # Dimensionality Reduction
        elif model_class == PCA:
            params.update({
                'n_components': ('float', 0.1, 0.99),
                'whiten': ('categorical', [True, False])
            })
        elif model_class == TruncatedSVD:
            params.update({
                'n_components': ('int', 2, min(100, X.shape[1] - 1)),
                'algorithm': ('categorical', ['randomized', 'arpack'])
            })
        elif model_class == KernelPCA:
            params.update({
                'n_components': ('int', 2, min(100, X.shape[1])),
                'kernel': ('categorical', ['rbf', 'poly', 'sigmoid']),
                'gamma': ('float', 0.0001, 1.0)
            })
            
        return params
        
    def _get_metric(self, task_type: str):
        """Get appropriate metrics based on task type"""
        if task_type == 'classification':
            return {
                'accuracy': make_scorer(accuracy_score),
                'f1': make_scorer(f1_score, average='weighted'),
                'precision': make_scorer(precision_score, average='weighted'),
                'recall': make_scorer(recall_score, average='weighted')
            }
        return {
            'r2': make_scorer(r2_score),
            'mse': make_scorer(mean_squared_error, greater_is_better=False)
        }
        
    def _get_scoring(self):
        if self.task_type == 'classification':
            return 'accuracy'
        return 'r2'
        
    def _optimize_hyperparameters(
        self,
        model_class: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: Optional[int] = None,
        early_stopping_rounds: Optional[int] = None,
        cv_folds: Optional[int] = None,
        timeout: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize hyperparameters using Optuna"""
        self.logger.info(f"\nStarting hyperparameter optimization for {model_class.__name__}")
        self.logger.info(f"Optimization parameters: n_trials={n_trials}, early_stopping_rounds={early_stopping_rounds}, cv_folds={cv_folds}")
        
        # Use settings from config if not provided
        n_trials = n_trials or self.settings.DEFAULT_N_TRIALS
        early_stopping_rounds = early_stopping_rounds or self.settings.DEFAULT_EARLY_STOPPING_ROUNDS
        cv_folds = cv_folds or self.settings.DEFAULT_CV_FOLDS
        timeout = timeout or (self.settings.DEFAULT_TIMEOUT_MINUTES * 60)  # Convert to seconds

        param_space = self._get_hyperparameter_space(model_class)
        
        # Set up cross-validation
        if self.task_type == 'classification':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
        # Track best score for early stopping
        best_score = float('-inf')
        no_improvement_count = 0
        
        def objective(trial):
            nonlocal best_score, no_improvement_count
            
            # Get hyperparameters for this trial
            params = {}
            for param_name, param_config in param_space.items():
                if param_config[0] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config[1],
                        param_config[2]
                    )
                elif param_config[0] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config[1],
                        param_config[2],
                        log=param_config[3] if len(param_config) > 3 else False
                    )
                elif param_space[param_name][0] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_space[param_name][1]
                    )
            
            self.logger.debug(f"Trying parameters: {params}")
            
            # Train and evaluate model
            model = model_class(**params)
            scores = cross_val_score(model, X, y, cv=cv, scoring=self._get_scoring())
            current_score = scores.mean()
            self.logger.debug(f"Trial score: {current_score:.4f} (std: {np.std(scores):.4f})")
            
            # Update best score and check for early stopping
            if current_score > best_score:
                best_score = current_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= early_stopping_rounds:
                trial.study.stop()
                
            # Call progress callback if provided
            if progress_callback:
                progress_callback(trial.number, current_score)
            
            return current_score
            
        # Create and run study
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        study = optuna.create_study(direction='maximize', pruner=pruner)
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[
                optuna.callbacks.TimeoutCallback(timeout) if timeout else None
            ]
        )
        
        return study.best_params, study.best_value
        
    def _evaluate_model_suitability(self, model_class: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate if a model is suitable for the given data"""
        try:
            # Check data size requirements
            if len(X) < self.settings.MIN_SAMPLES_FOR_MODEL:
                self.logger.info(f"Not enough samples ({len(X)} < {self.settings.MIN_SAMPLES_FOR_MODEL})")
                return 0.0
                
            if X.shape[1] > self.settings.MAX_FEATURES_FOR_MODEL:
                self.logger.info(f"Too many features ({X.shape[1]} > {self.settings.MAX_FEATURES_FOR_MODEL})")
                return 0.0
                
            # Check memory requirements (rough estimate)
            estimated_memory = X.nbytes * 3  # Rough estimate for model memory usage
            available_memory = psutil.virtual_memory().available
            if estimated_memory > available_memory * 0.5:  # Use at most 50% of available memory
                self.logger.info(f"Insufficient memory (needs {estimated_memory/1e9:.1f}GB, available {available_memory/1e9:.1f}GB)")
                return 0.0
                
            # Check if model supports the task type
            if self.task_type == 'classification':
                if not hasattr(model_class, 'predict_proba'):
                    self.logger.info("Model doesn't support probability predictions")
                    return 0.5  # Still usable but not ideal
                    
            # Check if model handles sparse data (if data is sparse)
            if sparse.issparse(X) and not hasattr(model_class, '_get_tags') or not model_class._get_tags().get('allow_sparse', False):
                self.logger.info("Model doesn't support sparse data")
                return 0.0
                
            return 1.0  # Model is fully suitable
            
        except Exception as e:
            self.logger.error(f"Error evaluating model suitability: {str(e)}")
            return 0.0
        
    def _get_clustering_models(self) -> Dict[str, Any]:
        """Get available clustering models with enhanced configurations"""
        models = {
            'kmeans': {
                'class': KMeans,
                'params': {
                    'n_clusters': ('int', 2, 20),
                    'init': ('categorical', ['k-means++', 'random']),
                    'max_iter': ('int', 100, 1000),
                    'n_init': ('int', 5, 20),
                    'tol': ('float', 1e-6, 1e-2)
                }
            },
            'dbscan': {
                'class': DBSCAN,
                'params': {
                    'eps': ('float', 0.1, 5.0),
                    'min_samples': ('int', 2, 20),
                    'metric': ('categorical', ['euclidean', 'manhattan', 'cosine']),
                    'algorithm': ('categorical', ['auto', 'ball_tree', 'kd_tree', 'brute'])
                }
            },
            'agglomerative': {
                'class': AgglomerativeClustering,
                'params': {
                    'n_clusters': ('int', 2, 20),
                    'linkage': ('categorical', ['ward', 'complete', 'average', 'single']),
                    'affinity': ('categorical', ['euclidean', 'manhattan', 'cosine']),
                    'compute_full_tree': ('categorical', ['auto', True, False])
                }
            },
            'gaussian_mixture': {
                'class': GaussianMixture,
                'params': {
                    'n_components': ('int', 2, 20),
                    'covariance_type': ('categorical', ['full', 'tied', 'diag', 'spherical']),
                    'max_iter': ('int', 100, 500),
                    'n_init': ('int', 1, 10),
                    'tol': ('float', 1e-6, 1e-2)
                }
            }
        }
        return models

    def _get_dimension_reduction_models(self) -> Dict[str, Any]:
        """Get available dimensionality reduction models with enhanced configurations"""
        models = {
            'pca': {
                'class': PCA,
                'params': {
                    'n_components': ('float', 0.5, 0.99),
                    'whiten': ('categorical', [True, False]),
                    'svd_solver': ('categorical', ['auto', 'full', 'randomized'])
                }
            },
            'kernel_pca': {
                'class': KernelPCA,
                'params': {
                    'n_components': ('int', 2, 50),
                    'kernel': ('categorical', ['linear', 'rbf', 'poly', 'sigmoid', 'cosine']),
                    'gamma': ('float', 0.0001, 10.0),
                    'degree': ('int', 2, 5),
                    'coef0': ('float', 0.0, 10.0)
                }
            },
            'truncated_svd': {
                'class': TruncatedSVD,
                'params': {
                    'n_components': ('int', 2, 100),
                    'algorithm': ('categorical', ['randomized', 'arpack']),
                    'n_iter': ('int', 5, 20),
                    'tol': ('float', 1e-6, 1e-2)
                }
            },
            'umap': {
                'class': UMAP,
                'params': {
                    'n_components': ('int', 2, 50),
                    'n_neighbors': ('int', 5, 50),
                    'min_dist': ('float', 0.0, 0.99),
                    'metric': ('categorical', ['euclidean', 'manhattan', 'cosine']),
                }
            }
        }
        return models

    def _evaluate_clustering(self, X: np.ndarray, labels: np.ndarray, metric: str = 'all') -> Dict[str, float]:
        """Evaluate clustering results using multiple metrics
        
        Args:
            X: Input data
            labels: Cluster labels
            metric: Metric to use ('silhouette', 'calinski', 'davies', 'all')
            
        Returns:
            Dictionary of scores
        """
        scores = {}
        
        if metric in ['silhouette', 'all']:
            try:
                scores['silhouette'] = silhouette_score(X, labels)
            except:
                scores['silhouette'] = None
                
        if metric in ['calinski', 'all']:
            try:
                scores['calinski_harabasz'] = calinski_harabasz_score(X, labels)
            except:
                scores['calinski_harabasz'] = None
                
        if metric in ['davies', 'all']:
            try:
                scores['davies_bouldin'] = davies_bouldin_score(X, labels)
            except:
                scores['davies_bouldin'] = None
                
        # Add cluster statistics
        unique_labels = np.unique(labels)
        scores['n_clusters'] = len(unique_labels)
        scores['cluster_sizes'] = {f"cluster_{i}": np.sum(labels == i) for i in unique_labels}
        
        return scores

    def _evaluate_dimension_reduction(
        self,
        X: np.ndarray,
        X_reduced: np.ndarray,
        model,
        reconstruction_error: bool = True,
        preserve_neighbors: bool = True,
        preserve_distances: bool = True
    ) -> Dict[str, float]:
        """Evaluate dimensionality reduction results with enhanced metrics
        
        Args:
            X: Original data
            X_reduced: Reduced data
            model: Fitted dimension reduction model
            reconstruction_error: Whether to compute reconstruction error
            preserve_neighbors: Whether to compute neighbor preservation
            preserve_distances: Whether to compute distance preservation
            
        Returns:
            Dictionary of scores
        """
        scores = {}
        
        # Reconstruction error (if applicable)
        if reconstruction_error and hasattr(model, 'inverse_transform'):
            try:
                X_reconstructed = model.inverse_transform(X_reduced)
                scores['reconstruction_error'] = -np.mean(np.square(X - X_reconstructed))
                scores['reconstruction_error_std'] = np.std(np.square(X - X_reconstructed))
            except:
                scores['reconstruction_error'] = None
                scores['reconstruction_error_std'] = None
        
        # Neighbor preservation
        if preserve_neighbors:
            try:
                k = min(20, len(X) - 1)
                nbrs_orig = NearestNeighbors(n_neighbors=k).fit(X)
                nbrs_reduced = NearestNeighbors(n_neighbors=k).fit(X_reduced)
                
                # Calculate neighbor preservation for different k values
                k_values = [5, 10, 15, 20]
                for k in k_values:
                    if k >= len(X):
                        break
                    orig_distances, orig_indices = nbrs_orig.kneighbors(n_neighbors=k)
                    reduced_distances, reduced_indices = nbrs_reduced.kneighbors(n_neighbors=k)
                    
                    preservation = np.mean([
                        len(set(orig_indices[i]) & set(reduced_indices[i])) / k
                        for i in range(len(X))
                    ])
                    scores[f'neighbor_preservation_k{k}'] = preservation
            except:
                scores['neighbor_preservation'] = None
        
        # Distance preservation (Spearman correlation between distance matrices)
        if preserve_distances:
            try:
                from scipy.spatial.distance import pdist, squareform
                from scipy.stats import spearmanr
                
                # Sample points if dataset is too large
                max_points = 1000
                if len(X) > max_points:
                    indices = np.random.choice(len(X), max_points, replace=False)
                    X_sample = X[indices]
                    X_reduced_sample = X_reduced[indices]
                else:
                    X_sample = X
                    X_reduced_sample = X_reduced
                
                # Calculate distance matrices
                dist_orig = squareform(pdist(X_sample))
                dist_reduced = squareform(pdist(X_reduced_sample))
                
                # Calculate correlation
                correlation, _ = spearmanr(dist_orig.flatten(), dist_reduced.flatten())
                scores['distance_preservation'] = correlation
            except:
                scores['distance_preservation'] = None
        
        # Add explained variance ratio if available
        if hasattr(model, 'explained_variance_ratio_'):
            scores['explained_variance_ratio'] = np.sum(model.explained_variance_ratio_)
            scores['explained_variance_ratio_per_component'] = {
                f"component_{i}": float(ratio)
                for i, ratio in enumerate(model.explained_variance_ratio_)
            }
        
        return scores
        
    def train(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        task_type: Optional[str] = None,
        time_limit: Optional[int] = None,
        model_types: Optional[List[str]] = None,
        n_trials: Optional[int] = None,
        early_stopping_rounds: Optional[int] = None,
        cv_folds: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Train and optimize multiple models, select the best one"""
        # تحويل البيانات إلى numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # تحديد نوع المهمة إذا لم يتم تحديده
        if task_type is None:
            task_type = self._detect_task_type(y)
        self.task_type = task_type
            
        # تشفير المتغير التابع إذا كان تصنيفاً
        if task_type == 'classification':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            
        # الحصول على النماذج المناسبة
        data_size = len(X)
        available_models = self._get_models(task_type, data_size)
        
        if model_types:
            models = {k: v for k, v in available_models.items() if k in model_types}
        else:
            models = available_models
            
        # تحسين كل نموذج
        best_score = float('-inf')
        best_model_name = None
        
        all_models_results = []  # لتخزين نتائج جميع النماذج
        
        for name, model_class in models.items():
            try:
                self.logger.info(f"تحسين نموذج {name}")
                
                # تحسين المعلمات
                best_trial = self._optimize_hyperparameters(
                    model_class=model_class['class'],
                    X=X,
                    y=y,
                    n_trials=n_trials,
                    early_stopping_rounds=early_stopping_rounds,
                    cv_folds=cv_folds,
                    timeout=time_limit,
                    progress_callback=progress_callback
                )
                
                if best_trial.value > best_score:
                    best_score = best_trial.value
                    best_model_name = name
                    self.best_model = model_class['class'](**best_trial.params)
                    self.best_params = best_trial.params
                    
                all_models_results.append([name, best_trial.value, best_trial.params])
                
            except Exception as e:
                self.logger.error(f"خطأ في تحسين نموذج {name}: {str(e)}")
                continue
                
        # تدريب النموذج الأفضل على كامل البيانات
        if self.best_model is not None:
            self.best_model.fit(X, y)
            self._calculate_feature_importance()
            
            # حساب المقاييس
            y_pred = self.best_model.predict(X)
            metrics = self._calculate_metrics(y, y_pred)
            
            return {
                'model_name': best_model_name,
                'model': self.best_model,
                'params': self.best_params,
                'metrics': metrics,
                'feature_importance': self.feature_importance,
                'all_models_results': all_models_results
            }
        else:
            raise ValueError("لم يتم العثور على نموذج مناسب")
            
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """حساب مقاييس الأداء"""
        metrics = {}
        
        if self.task_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
            except:
                pass
        else:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
        return metrics
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("Model has not been trained yet")
        
        predictions = self.best_model.predict(X)
        
        # Decode predictions for classification
        if self.task_type == 'classification':
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
        
    def save_trained_model(self, name: str, preprocessing_params: Dict):
        """Save trained model to database"""
        if not self.best_model:
            raise ValueError("No trained model available to save")
            
        # Calculate feature importance if available
        feature_importance = self._calculate_feature_importance()
        
        # Save to database
        model_id = self.db_manager.save_model(
            name=name,
            model=self.best_model,
            model_type=self.task_type,
            hyperparameters=self.best_params,
            metrics=self.metrics,
            feature_importance=feature_importance,
            preprocessing_params=preprocessing_params
        )
        
        return model_id
    
    def load_model(self, model_id: int) -> tuple[Any, Dict, Dict]:
        """Load model from database"""
        return self.db_manager.load_model(model_id)
    
    def _calculate_feature_importance(self) -> Optional[Dict]:
        """Calculate feature importance if model supports it"""
        if not hasattr(self.best_model, 'feature_importances_'):
            return None
            
        importances = self.best_model.feature_importances_
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}
    
    def train_automl(self, X, y, task_type='classification', metric='accuracy', time_limit=60):
        """Train an AutoML model
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series): Target variable
            task_type (str): Type of task ('classification', 'regression', or 'multiclass')
            metric (str): Metric to optimize
            time_limit (int): Time limit in seconds
            
        Returns:
            dict: Dictionary containing the trained model and performance metrics
        """
        try:
            self.logger.info(f"Starting AutoML training for {task_type} task")
            self.task_type = task_type
            
            # Validate inputs
            if not isinstance(X, (pd.DataFrame, np.ndarray)):
                raise ValueError("X must be a pandas DataFrame or numpy array")
            if not isinstance(y, (pd.Series, np.ndarray)):
                raise ValueError("y must be a pandas Series or numpy array")
                
            # Convert to numpy arrays if needed
            X = np.array(X) if isinstance(X, pd.DataFrame) else X
            y = np.array(y) if isinstance(y, pd.Series) else y
            
            # Determine available models based on task type and data size
            available_models = []
            data_size = len(X)
            
            # Add Random Forest (always available)
            available_models.append('RandomForestRegressor' if task_type == 'regression' else 'RandomForestClassifier')
            
            # Add XGBoost if available and data size is appropriate
            if self.xgb_available and data_size >= 1000:
                available_models.append('XGBRegressor' if task_type == 'regression' else 'XGBClassifier')
                
            # Add LightGBM if available and data size is appropriate
            if self.lgb_available and data_size >= 1000:
                available_models.append('LGBMRegressor' if task_type == 'regression' else 'LGBMClassifier')
                
            self.logger.info(f"Available models: {available_models}")
            
            # Create study
            study = optuna.create_study(
                direction="maximize" if metric != "mse" else "minimize",
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            
            # Add dataset info to study user attributes
            study.set_user_attrs({
                'X': X,
                'y': y,
                'task_type': task_type,
                'metric': metric,
                'available_models': available_models
            })
            
            # Define objective function
            def objective(trial):
                # Select model type
                model_type = trial.suggest_categorical('model_type', available_models)
                
                # Get model configuration
                config = self._get_model_config(trial, model_type, task_type)
                
                # Create and train model
                model = self._create_model(model_type, config, task_type)
                
                # Perform cross-validation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if task_type != 'regression' else KFold(n_splits=5, shuffle=True, random_state=42)
                
                if metric == 'accuracy':
                    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
                elif metric == 'f1':
                    scores = cross_val_score(model, X, y, scoring='f1_weighted', cv=cv, n_jobs=-1)
                elif metric == 'auc':
                    scores = cross_val_score(model, X, y, scoring='roc_auc_ovr_weighted', cv=cv, n_jobs=-1)
                elif metric == 'mse':
                    scores = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
                elif metric == 'rmse':
                    scores = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1))
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                
                return np.mean(scores)
            
            # Optimize
            study.optimize(
                objective,
                n_trials=None,
                timeout=time_limit,
                callbacks=[
                    lambda study, trial: self.logger.info(f"Trial {trial.number}: {trial.value}")
                ]
            )
            
            # Get best trial
            best_trial = study.best_trial
            
            # Train final model with best parameters
            best_config = self._get_model_config(
                best_trial,
                best_trial.params['model_type'],
                task_type
            )
            best_model = self._create_model(
                best_trial.params['model_type'],
                best_config,
                task_type
            )
            best_model.fit(X, y)
            
            # Store results
            self.best_model = best_model
            self.best_params = best_config
            
            # Calculate feature importance if available
            try:
                if hasattr(best_model, 'feature_importances_'):
                    self.feature_importance = best_model.feature_importances_
                elif hasattr(best_model, 'coef_'):
                    self.feature_importance = np.abs(best_model.coef_).mean(axis=0) if len(best_model.coef_.shape) > 1 else np.abs(best_model.coef_)
            except Exception as e:
                self.logger.warning(f"Could not calculate feature importance: {str(e)}")
                self.feature_importance = None
            
            # Store metrics
            self.metrics = {
                'best_value': best_trial.value,
                'best_model_type': best_trial.params['model_type'],
                'optimization_history': [
                    {'trial': t.number, 'value': t.value}
                    for t in study.trials
                ]
            }
            
            self.logger.info(f"AutoML training completed. Best model: {best_trial.params['model_type']} (value: {best_trial.value})")
            return {
                'model': self.best_model,
                'best_model_name': best_trial.params['model_type'],
                'best_score': best_trial.value,
                'best_params': best_trial.params,
                'feature_importance': self.feature_importance if hasattr(self, 'feature_importance') else None,
                'training_history': self.training_history if hasattr(self, 'training_history') else None
            }
            
        except Exception as e:
            self.logger.error(f"Error in train_automl: {str(e)}")
            self.logger.exception(e)
            raise
            
    def _get_model_config(self, trial, model_type, task_type):
        """Get hyperparameter configuration for a model
        
        Args:
            trial: Optuna trial object
            model_type: Type of model ('random_forest', 'xgboost', or 'lightgbm')
            task_type: Type of task ('classification', 'multiclass', or 'regression')
            
        Returns:
            dict: Model configuration
        """
        try:
            self.logger.debug(f"Getting config for {model_type} ({task_type})")
            
            # Base configuration for all models
            base_config = {
                'random_state': 42,
                'n_jobs': -1
            }
            
            if model_type == 'RandomForestRegressor':
                config = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
                }
                
            elif model_type == 'RandomForestClassifier':
                config = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
                }
                
            elif model_type == 'XGBRegressor':
                config = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
                }
                
            elif model_type == 'XGBClassifier':
                config = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
                }
                
            elif model_type == 'LGBMRegressor':
                config = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
                }
                
            elif model_type == 'LGBMClassifier':
                config = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
                }
                
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            # Add task-specific parameters
            if task_type == 'classification':
                if model_type == 'RandomForestClassifier':
                    config['class_weight'] = 'balanced'
                elif model_type == 'XGBClassifier':
                    config['objective'] = 'binary:logistic'
                    config['eval_metric'] = 'logloss'
                    config['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 0.1, 10.0, log=True)
                elif model_type == 'LGBMClassifier':
                    config['objective'] = 'binary'
                    config['metric'] = 'binary_logloss'
                    config['is_unbalance'] = True
                    
            elif task_type == 'multiclass':
                if model_type == 'RandomForestClassifier':
                    config['class_weight'] = 'balanced'
                elif model_type == 'XGBClassifier':
                    config['objective'] = 'multi:softmax'
                    config['eval_metric'] = 'mlogloss'
                    config['num_class'] = len(np.unique(trial.study.user_attrs['y']))
                elif model_type == 'LGBMClassifier':
                    config['objective'] = 'multiclass'
                    config['metric'] = 'multi_logloss'
                    config['num_class'] = len(np.unique(trial.study.user_attrs['y']))
                    
            elif task_type == 'regression':
                if model_type == 'XGBRegressor':
                    config['objective'] = 'reg:squarederror'
                    config['eval_metric'] = ['rmse', 'mae']
                elif model_type == 'LGBMRegressor':
                    config['objective'] = 'regression'
                    config['metric'] = ['rmse', 'mae']
                    
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
                
            # Merge base config with model-specific config
            config.update(base_config)
            
            self.logger.debug(f"Generated config: {config}")
            return config
            
        except Exception as e:
            self.logger.error(f"Error in _get_model_config: {str(e)}")
            self.logger.exception(e)
            raise
            
    def _create_model(self, model_type, config, task_type):
        """Create a model instance with given configuration
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost', or 'lightgbm')
            config: Model configuration dictionary
            task_type: Type of task ('classification', 'multiclass', or 'regression')
            
        Returns:
            object: Initialized model instance
        """
        try:
            self.logger.debug(f"Creating {model_type} model for {task_type}")
            self.logger.debug(f"Config: {config}")
            
            if model_type == 'RandomForestRegressor':
                model = ensemble.RandomForestRegressor(**config, random_state=42, n_jobs=-1)
            elif model_type == 'RandomForestClassifier':
                model = ensemble.RandomForestClassifier(**config, random_state=42, n_jobs=-1)
            elif model_type == 'XGBRegressor':
                model = xgb.XGBRegressor(**config, random_state=42, n_jobs=-1)
            elif model_type == 'XGBClassifier':
                model = xgb.XGBClassifier(**config, random_state=42, n_jobs=-1)
            elif model_type == 'LGBMRegressor':
                model = lgb.LGBMRegressor(**config, random_state=42, n_jobs=-1)
            elif model_type == 'LGBMClassifier':
                model = lgb.LGBMClassifier(**config, random_state=42, n_jobs=-1)
                
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            self.logger.debug(f"Successfully created {model_type} model")
            return model
            
        except Exception as e:
            self.logger.error(f"Error in _create_model: {str(e)}")
            self.logger.exception(e)
            raise
            
    def save_model(self, path: str):
        """Save the trained model to disk."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        joblib.dump(self.best_model, path)
        
    def load_model(self, path: str):
        """Load a trained model from disk."""
        self.best_model = joblib.load(path)
        return self.best_model
        
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        return self.feature_importance

class AutoMLModelTrainer(ModelTrainer):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.feature_importance = None
        self.metrics = {}
        
    def train_model(self, X, y, task_type='regression', optimization_time=60, n_trials=20, cv_folds=5, selected_models=None):
        """تدريب النموذج"""
        self.logger.info(f"بدء تدريب النموذج للمهمة {task_type}")
        
        # تحويل البيانات الفئوية
        categorical_columns = X.select_dtypes(include=['object']).columns
        numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
        
        # إنشاء نسخة من البيانات
        X_processed = X.copy()
        
        # معالجة الأعمدة الفئوية
        encoders = {}
        for col in categorical_columns:
            encoders[col] = LabelEncoder()
            X_processed[col] = encoders[col].fit_transform(X_processed[col].astype(str))
        
        # تطبيع البيانات العددية
        scaler = StandardScaler()
        if len(numeric_columns) > 0:
            X_processed[numeric_columns] = scaler.fit_transform(X_processed[numeric_columns])
        
        # الحصول على النماذج المناسبة للمهمة
        available_models = self._get_models(task_type, len(X))
        
        if selected_models:
            models = {name: model for name, model in available_models.items() if name in selected_models}
            self.logger.info(f"تم اختيار {len(models)} نموذج من قبل المستخدم")
        else:
            models = available_models
            self.logger.info(f"تم تحديد {len(models)} نموذج مناسب للمهمة {task_type}")
        
        # التحقق من وجود نماذج
        if not models:
            self.logger.warning("لم يتم العثور على نماذج مناسبة")
            raise ValueError("لم يتم العثور على نماذج مناسبة للتدريب")
        
        best_model = None
        best_score = float('inf') if task_type == 'regression' else float('-inf')
        best_model_data = None
        all_models_results = []
        
        # تدريب وتقييم كل نموذج
        for name, model_info in models.items():
            self.logger.info(f"بدء تدريب نموذج {name}")
            start_time = datetime.now()
            
            try:
                # تحديد نطاقات المعاملات للتحسين
                param_ranges = self._get_param_ranges(model_info['class'])
                if not param_ranges:
                    self.logger.warning(f"تم تخطي نموذج {name} لعدم توفر نطاقات المعاملات المناسبة")
                    continue
                
                # إنشاء دراسة Optuna جديدة
                study = optuna.create_study(
                    direction="minimize" if task_type == 'regression' else "maximize",
                    sampler=optuna.samplers.TPESampler(seed=42)
                )
                
                # تعريف دالة الهدف
                def objective(trial):
                    params = {}
                    for param, range_info in param_ranges.items():
                        if range_info['type'] == 'int':
                            params[param] = trial.suggest_int(param, range_info['low'], range_info['high'])
                        elif range_info['type'] == 'float':
                            params[param] = trial.suggest_float(param, range_info['low'], range_info['high'])
                        elif range_info['type'] == 'categorical':
                            params[param] = trial.suggest_categorical(param, range_info['choices'])
                    
                    model = model_info['class'](**params)
                    
                    # اختيار مقياس التقييم حسب نوع المهمة
                    if task_type == 'regression':
                        scoring = 'neg_mean_squared_error'
                    else:
                        scoring = 'accuracy'
                    
                    try:
                        scores = cross_val_score(model, X_processed, y, cv=cv_folds, scoring=scoring)
                        return -scores.mean() if task_type == 'regression' else scores.mean()
                    except Exception as e:
                        self.logger.error(f"خطأ في التحقق المتقاطع للنموذج {name}: {str(e)}")
                        return float('inf') if task_type == 'regression' else float('-inf')
                
                # تشغيل عملية التحسين
                remaining_time = max(1, optimization_time - (datetime.now() - start_time).seconds)
                self.logger.info(f"بدء تحسين نموذج {name} (الوقت المتبقي: {remaining_time} ثانية)")
                
                study.optimize(objective, n_trials=n_trials, timeout=remaining_time)
                
                if len(study.trials) == 0:
                    self.logger.warning(f"لم يتم إكمال أي محاولات لنموذج {name}")
                    continue
                
                # تدريب النموذج النهائي باستخدام أفضل المعاملات
                final_model = model_info['class'](**study.best_params)
                final_model.fit(X_processed, y)
                
                # حساب النتيجة النهائية والمقاييس الإضافية
                score = -study.best_value if task_type == 'regression' else study.best_value
                
                # حساب مقاييس إضافية
                y_pred = final_model.predict(X_processed)
                additional_metrics = {}
                
                if task_type == 'regression':
                    additional_metrics = {
                        'mse': mean_squared_error(y, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                        'mae': mean_absolute_error(y, y_pred),
                        'r2': r2_score(y, y_pred)
                    }
                else:
                    additional_metrics = {
                        'accuracy': accuracy_score(y, y_pred),
                        'precision': precision_score(y, y_pred, average='weighted'),
                        'recall': recall_score(y, y_pred, average='weighted'),
                        'f1': f1_score(y, y_pred, average='weighted')
                    }
                
                # حساب أهمية المتغيرات
                feature_importance = None
                if hasattr(final_model, 'feature_importances_'):
                    feature_importance = dict(zip(X.columns, final_model.feature_importances_))
                elif hasattr(final_model, 'coef_'):
                    coef = final_model.coef_.ravel() if len(final_model.coef_.shape) > 1 else final_model.coef_
                    feature_importance = dict(zip(X.columns, np.abs(coef)))
                
                training_time = datetime.now() - start_time
                
                # إضافة نتائج النموذج إلى القائمة
                model_result = {
                    'name': name,
                    'score': score,
                    'params': study.best_params,
                    'metrics': additional_metrics,
                    'feature_importance': feature_importance,
                    'training_time': training_time,
                    'n_trials_completed': len(study.trials),
                    'status': 'completed'
                }
                
                self.logger.info(f"اكتمل تدريب نموذج {name} (الوقت: {training_time}, المحاولات: {len(study.trials)})")
                
                all_models_results.append(model_result)
                
                # تحديث أفضل نموذج إذا كان أفضل من السابق
                if task_type == 'regression':
                    is_better = score < best_score
                else:
                    is_better = score > best_score
                    
                if is_better:
                    best_score = score
                    best_model = final_model
                    best_model_data = {
                        'name': name,
                        'model': final_model,
                        'score': score,
                        'params': study.best_params,
                        'metrics': additional_metrics,
                        'feature_importance': feature_importance,
                        'encoders': encoders,
                        'scaler': scaler,
                        'categorical_columns': categorical_columns,
                        'numeric_columns': numeric_columns,
                        'training_time': training_time,
                        'n_trials_completed': len(study.trials)
                    }
                    self.logger.info(f"تم تحديث أفضل نموذج: {name} (النتيجة: {score})")
                    
            except Exception as e:
                self.logger.error(f"خطأ في تدريب نموذج {name}: {str(e)}")
                all_models_results.append({
                    'name': name,
                    'status': 'failed',
                    'error': str(e),
                    'training_time': datetime.now() - start_time
                })
                continue
                
        if best_model_data:
            self.logger.info(f"اكتمل التدريب. أفضل نموذج: {best_model_data['name']} (النتيجة: {best_model_data['score']})")
            self.logger.info(f"أفضل المعاملات: {best_model_data['params']}")
            
            return {
                'best_model_name': best_model_data['name'],
                'best_score': best_model_data['score'],
                'best_params': best_model_data['params'],
                'best_metrics': best_model_data['metrics'],
                'model': best_model_data['model'],
                'feature_importance': best_model_data['feature_importance'],
                'all_models_results': all_models_results,
                'preprocessing': {
                    'encoders': best_model_data['encoders'],
                    'scaler': best_model_data['scaler'],
                    'categorical_columns': best_model_data['categorical_columns'],
                    'numeric_columns': best_model_data['numeric_columns']
                }
            }
        else:
            self.logger.warning("لم يتم العثور على نموذج مناسب")
            raise ValueError("فشل تدريب جميع النماذج. يرجى التحقق من البيانات والمعاملات المدخلة.")
        
    def _get_param_ranges(self, model_class):
        """تحديد نطاقات المعاملات للتحسين"""
        model_name = model_class.__name__
        self.logger.info(f"Getting parameter ranges for model: {model_name}")
        
        param_ranges = {
            # Tree-based Models
            'RandomForestClassifier': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10}
            },
            'RandomForestRegressor': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10}
            },
            'GradientBoostingClassifier': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10}
            },
            'GradientBoostingRegressor': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10}
            },
            
            # Linear Models
            'LinearRegression': {
                'fit_intercept': {'type': 'categorical', 'choices': [True, False]}
            },
            'LogisticRegression': {
                'C': {'type': 'float', 'low': 0.1, 'high': 10.0},
                'max_iter': {'type': 'int', 'low': 100, 'high': 500},
                'solver': {'type': 'categorical', 'choices': ['lbfgs', 'liblinear', 'saga']}
            },
            'Ridge': {
                'alpha': {'type': 'float', 'low': 0.1, 'high': 10.0},
                'fit_intercept': {'type': 'categorical', 'choices': [True, False]}
            },
            'Lasso': {
                'alpha': {'type': 'float', 'low': 0.1, 'high': 10.0},
                'fit_intercept': {'type': 'categorical', 'choices': [True, False]}
            },
            
            # SVM Models
            'SVR': {
                'C': {'type': 'float', 'low': 0.1, 'high': 10.0},
                'kernel': {'type': 'categorical', 'choices': ['linear', 'rbf', 'poly']},
                'gamma': {'type': 'float', 'low': 0.001, 'high': 1.0}
            },
            'SVC': {
                'C': {'type': 'float', 'low': 0.1, 'high': 10.0},
                'kernel': {'type': 'categorical', 'choices': ['linear', 'rbf', 'poly']},
                'gamma': {'type': 'float', 'low': 0.001, 'high': 1.0}
            },
            
            # KNN Models
            'KNeighborsRegressor': {
                'n_neighbors': {'type': 'int', 'low': 3, 'high': 20},
                'weights': {'type': 'categorical', 'choices': ['uniform', 'distance']},
                'p': {'type': 'int', 'low': 1, 'high': 2}
            },
            'KNeighborsClassifier': {
                'n_neighbors': {'type': 'int', 'low': 3, 'high': 20},
                'weights': {'type': 'categorical', 'choices': ['uniform', 'distance']},
                'p': {'type': 'int', 'low': 1, 'high': 2}
            }
        }
        
        if model_name in param_ranges:
            self.logger.info(f"Found parameter ranges for {model_name}")
            return param_ranges[model_name]
        else:
            self.logger.warning(f"No parameter ranges defined for {model_name}")
            return None

    def export_model(self, model_name: str, format: str = 'joblib') -> str:
        """تصدير النموذج إلى ملف
        
        Args:
            model_name: اسم النموذج المراد تصديره
            format: صيغة التصدير (joblib, pickle, ONNX)
            
        Returns:
            str: مسار الملف المصدر
        """
        try:
            # Load model from database
            model_data = self.db_manager.load_model(model_name)
            if not model_data:
                raise ValueError(f"لم يتم العثور على النموذج: {model_name}")
                
            # Create exports directory if it doesn't exist
            import os
            export_dir = os.path.join(os.getcwd(), 'exports')
            os.makedirs(export_dir, exist_ok=True)
            
            # Generate export filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version = model_data['version']
            base_filename = f"{model_name}_{version}_{timestamp}"
            
            if format.lower() == 'joblib':
                export_path = os.path.join(export_dir, f"{base_filename}.joblib")
                joblib.dump(model_data['model'], export_path)
                
            elif format.lower() == 'pickle':
                import pickle
                export_path = os.path.join(export_dir, f"{base_filename}.pkl")
                with open(export_path, 'wb') as f:
                    pickle.dump(model_data['model'], f)
                    
            elif format.lower() == 'onnx':
                try:
                    import skl2onnx
                    from skl2onnx.common.data_types import FloatTensorType
                    
                    # Get initial types for ONNX conversion
                    if hasattr(model_data['model'], 'n_features_in_'):
                        n_features = model_data['model'].n_features_in_
                    else:
                        # Try to get from preprocessing params
                        n_features = model_data['preprocessing_params'].get('n_features', 10)
                        
                    initial_types = [('float_input', FloatTensorType([None, n_features]))]
                    
                    # Convert to ONNX
                    onnx_model = skl2onnx.convert_sklearn(
                        model_data['model'],
                        initial_types=initial_types,
                        target_opset=13  # Use a recent opset version
                    )
                    
                    # Save ONNX model
                    export_path = os.path.join(export_dir, f"{base_filename}.onnx")
                    with open(export_path, "wb") as f:
                        f.write(onnx_model.SerializeToString())
                        
                except ImportError:
                    raise ImportError("لتصدير النموذج بصيغة ONNX، قم بتثبيت حزمة skl2onnx أولاً")
            else:
                raise ValueError(f"صيغة التصدير غير مدعومة: {format}")
                
            self.logger.info(f"تم تصدير النموذج بنجاح إلى: {export_path}")
            return export_path
            
        except Exception as e:
            self.logger.error(f"خطأ في تصدير النموذج: {str(e)}")
            raise
