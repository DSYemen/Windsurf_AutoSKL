import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
import logging
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer
import numpy.typing as npt

class DataProcessor:
    """Handles data loading, preprocessing, and feature engineering."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}
        
    def load_csv(self, file, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            file: File object or path
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            return pd.read_csv(file, **kwargs)
        except Exception as e:
            self.logger.error(f"Error loading CSV file: {str(e)}")
            raise
            
    def load_excel(self, file, **kwargs) -> pd.DataFrame:
        """
        Load data from Excel file
        
        Args:
            file: File object or path
            **kwargs: Additional arguments to pass to pd.read_excel
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            return pd.read_excel(file, **kwargs)
        except Exception as e:
            self.logger.error(f"Error loading Excel file: {str(e)}")
            raise
            
    def handle_missing_values(self, data: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """معالجة القيم المفقودة في البيانات"""
        try:
            data_processed = data.copy()
            
            # معالجة كل عمود على حدة
            for column in data.columns:
                # تحقق من وجود قيم مفقودة
                if data[column].isnull().any():
                    if pd.api.types.is_numeric_dtype(data[column]):
                        # معالجة الأعمدة الرقمية
                        if strategy == 'mean':
                            fill_value = data[column].mean()
                        elif strategy == 'median':
                            fill_value = data[column].median()
                        elif strategy == 'constant':
                            fill_value = 0
                        else:
                            fill_value = data[column].mode()[0]
                        data_processed[column] = data[column].fillna(fill_value)
                    else:
                        # معالجة الأعمدة النصية والفئوية
                        self.logger.warning(f"Column {column} is categorical/text. Using 'most_frequent' strategy instead of {strategy}")
                        fill_value = data[column].mode()[0] if not data[column].mode().empty else 'MISSING'
                        data_processed[column] = data[column].fillna(fill_value)
                        # تحويل العمود إلى نوع string
                        data_processed[column] = data_processed[column].astype(str)

            return data_processed
            
        except Exception as e:
            self.logger.error(f"Error in handle_missing_values: {str(e)}")
            raise

    def encode_categorical(self, data: pd.DataFrame, columns: List[str] = None, method: str = 'label') -> pd.DataFrame:
        """تشفير البيانات الفئوية"""
        try:
            data_encoded = data.copy()
            
            # إذا لم يتم تحديد الأعمدة، ابحث عن الأعمدة الفئوية
            if columns is None:
                columns = data.select_dtypes(include=['object', 'category']).columns
            
            for column in columns:
                if column in data.columns:
                    # تحويل القيم المفقودة إلى نص قبل التشفير
                    data_encoded[column] = data_encoded[column].fillna('MISSING').astype(str)
                    
                    if method == 'label':
                        encoder = LabelEncoder()
                        data_encoded[column] = encoder.fit_transform(data_encoded[column])
                    elif method == 'onehot':
                        # استخدام get_dummies مع تحويل العمود إلى نص أولاً
                        dummies = pd.get_dummies(data_encoded[column], prefix=column)
                        data_encoded = pd.concat([data_encoded.drop(column, axis=1), dummies], axis=1)
                    else:
                        self.logger.warning(f"Unknown encoding method: {method}. Using label encoding instead.")
                        encoder = LabelEncoder()
                        data_encoded[column] = encoder.fit_transform(data_encoded[column])
                else:
                    self.logger.warning(f"Column {column} not found in data")
            
            return data_encoded
            
        except Exception as e:
            self.logger.error(f"Error in encode_categorical: {str(e)}")
            raise

    def preprocess_data(self, data: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """معالجة البيانات الأولية"""
        try:
            # نسخ البيانات
            processed_data = data.copy()
            
            # معالجة القيم المفقودة
            processed_data = self.handle_missing_values(processed_data)
            
            # تحويل جميع الأعمدة النصية إلى نص
            for col in processed_data.select_dtypes(include=['object', 'category']).columns:
                processed_data[col] = processed_data[col].astype(str)
            
            # فصل المتغير التابع إذا تم تحديده
            if target_column and target_column in processed_data.columns:
                y = processed_data[target_column]
                X = processed_data.drop(columns=[target_column])
                
                # تشفير المتغير التابع إذا كان نصياً
                if not pd.api.types.is_numeric_dtype(y):
                    encoder = LabelEncoder()
                    y = pd.Series(encoder.fit_transform(y), name=target_column)
                
                return X, y
            else:
                return processed_data, None
                
        except Exception as e:
            self.logger.error(f"Error in preprocess_data: {str(e)}")
            raise

    def handle_outliers(self, data: pd.DataFrame, method: str = 'iqr',
                       columns: Optional[List[str]] = None,
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Handle outliers in the dataset
        
        Args:
            data: Input DataFrame
            method: Method to handle outliers ('iqr', 'zscore', 'isolation_forest')
            columns: List of columns to process (if None, process all numeric columns)
            threshold: Threshold for outlier detection
            
        Returns:
            pd.DataFrame: DataFrame with handled outliers
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                
            elif method == 'zscore':
                z_scores = (data[col] - data[col].mean()) / data[col].std()
                data[col] = data[col].mask(abs(z_scores) > threshold, data[col].mean())
                
        return data
        
    def scale_features(self, data: pd.DataFrame, method: str = 'standard',
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            data: Input DataFrame
            method: Scaling method ('standard', 'minmax', 'robust')
            columns: List of columns to scale (if None, scale all numeric columns)
            
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
            
        scaled_data = data.copy()
        
        for col in columns:
            if col not in self.scalers:
                if method == 'standard':
                    self.scalers[col] = StandardScaler()
                elif method == 'minmax':
                    self.scalers[col] = MinMaxScaler()
                elif method == 'robust':
                    self.scalers[col] = RobustScaler()
                    
            # Reshape for 1D arrays
            reshaped_data = data[col].values.reshape(-1, 1)
            scaled_data[col] = self.scalers[col].fit_transform(reshaped_data).ravel()
            
        return scaled_data
        
    def detect_data_type(self, data: pd.DataFrame) -> str:
        """
        Detect the type of machine learning task based on target variable
        
        Args:
            data: Input DataFrame with target variable
            
        Returns:
            str: Type of task ('classification', 'regression', 'clustering')
        """
        if 'target' not in data.columns:
            return 'clustering'
            
        target = data['target']
        
        if target.dtype in ['object', 'category'] or len(target.unique()) < 10:
            return 'classification'
        else:
            return 'regression'
            
    def prepare_data(self, data: pd.DataFrame, target_column: Optional[str] = None,
                    handle_missing: bool = True, handle_outliers: bool = True,
                    encode_categorical: bool = True, scale_features: bool = True) -> Dict[str, Any]:
        """
        Prepare data for machine learning
        
        Args:
            data: Input DataFrame
            target_column: Name of target column (if None, assume clustering task)
            handle_missing: Whether to handle missing values
            handle_outliers: Whether to handle outliers
            encode_categorical: Whether to encode categorical variables
            scale_features: Whether to scale features
            
        Returns:
            dict: Dictionary containing prepared X and y (if applicable)
        """
        prepared_data = data.copy()
        
        if handle_missing:
            prepared_data = self.handle_missing_values(prepared_data)
            
        if handle_outliers:
            prepared_data = self.handle_outliers(prepared_data)
            
        if encode_categorical:
            prepared_data = self.encode_categorical(prepared_data)
            
        if scale_features:
            if target_column:
                feature_columns = [col for col in prepared_data.columns if col != target_column]
                prepared_data[feature_columns] = self.scale_features(prepared_data[feature_columns])
            else:
                prepared_data = self.scale_features(prepared_data)
                
        result = {'X': prepared_data}
        
        if target_column:
            result['y'] = prepared_data[target_column]
            result['X'] = prepared_data.drop(target_column, axis=1)
            
        return result

    def prepare_data_for_training(self, data: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for model training by handling missing values and encoding categorical variables
        
        Args:
            data: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple containing features (X) and target (y)
        """
        try:
            self.logger.info("Preparing data for training...")
            self.logger.info(f"Input data shape: {data.shape}")
            
            # Separate features and target
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            # Handle missing values
            X = self.handle_missing_values(X, method='mean')
            
            # Encode categorical variables
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            if not categorical_cols.empty:
                self.logger.info(f"Encoding categorical columns: {categorical_cols.tolist()}")
                X = self.encode_categorical(X, categorical_cols, method='label')
            
            # Convert all columns to float32
            X = X.astype(np.float32)
            
            # Convert target to appropriate type
            if y.dtype == 'object' or y.dtype.name == 'category':
                self.logger.info("Converting target to numeric using label encoding")
                y = LabelEncoder().fit_transform(y)
            else:
                y = y.astype(np.float32)
            
            self.logger.info(f"Prepared data shapes - X: {X.shape}, y: {y.shape}")
            self.logger.info(f"X dtypes: {X.dtypes.value_counts().to_dict()}")
            self.logger.info(f"y dtype: {y.dtype}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error in prepare_data_for_training: {str(e)}")
            self.logger.exception(e)
            raise
