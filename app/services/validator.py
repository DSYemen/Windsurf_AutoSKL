from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel, validator
from datetime import datetime
import json
import logging
from pathlib import Path

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class DataValidationReport(BaseModel):
    """Model for data validation report"""
    timestamp: str
    dataset_name: str
    total_rows: int
    total_columns: int
    missing_values: Dict[str, int]
    data_types: Dict[str, str]
    unique_values: Dict[str, int]
    numeric_stats: Optional[Dict[str, Dict[str, float]]]
    validation_errors: List[str]
    warnings: List[str]
    
class DataValidator:
    def __init__(
        self,
        max_missing_ratio: float = 0.2,
        min_unique_ratio: float = 0.01,
        max_unique_ratio: float = 0.99
    ):
        self.max_missing_ratio = max_missing_ratio
        self.min_unique_ratio = min_unique_ratio
        self.max_unique_ratio = max_unique_ratio
        self.validation_errors = []
        self.warnings = []
        
    def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """Check for missing values in each column"""
        missing_values = df.isnull().sum().to_dict()
        
        for column, count in missing_values.items():
            ratio = count / len(df)
            if ratio > self.max_missing_ratio:
                self.validation_errors.append(
                    f"Column '{column}' has {ratio:.2%} missing values"
                    f" (threshold: {self.max_missing_ratio:.2%})"
                )
                
        return missing_values
        
    def _check_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Check data types of columns"""
        data_types = df.dtypes.astype(str).to_dict()
        
        # Check for mixed data types
        for column in df.columns:
            try:
                pd.to_numeric(df[column], errors='raise')
            except:
                unique_types = df[column].apply(type).unique()
                if len(unique_types) > 1:
                    self.warnings.append(
                        f"Column '{column}' contains mixed data types: {unique_types}"
                    )
                    
        return data_types
        
    def _check_unique_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """Check unique value ratios"""
        unique_counts = df.nunique().to_dict()
        
        for column, count in unique_counts.items():
            ratio = count / len(df)
            if ratio < self.min_unique_ratio:
                self.warnings.append(
                    f"Column '{column}' has very few unique values"
                    f" ({ratio:.2%} unique)"
                )
            elif ratio > self.max_unique_ratio:
                self.warnings.append(
                    f"Column '{column}' has too many unique values"
                    f" ({ratio:.2%} unique)"
                )
                
        return unique_counts
        
    def _compute_numeric_stats(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Compute statistics for numeric columns"""
        numeric_stats = {}
        
        for column in df.select_dtypes(include=[np.number]).columns:
            stats = df[column].describe().to_dict()
            
            # Check for outliers using IQR method
            Q1 = stats['25%']
            Q3 = stats['75%']
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[
                (df[column] < lower_bound) | (df[column] > upper_bound)
            ].shape[0]
            
            if outliers > 0:
                self.warnings.append(
                    f"Column '{column}' has {outliers} outliers"
                    f" ({outliers/len(df):.2%} of values)"
                )
                
            numeric_stats[column] = {
                **stats,
                'outliers': outliers,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
        return numeric_stats
        
    def validate_training_data(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> DataValidationReport:
        """Validate training data"""
        self.validation_errors = []
        self.warnings = []
        
        # Basic checks
        if df.empty:
            raise DataValidationError("Dataset is empty")
            
        if target_column not in df.columns:
            raise DataValidationError(f"Target column '{target_column}' not found")
            
        # Perform validation checks
        missing_values = self._check_missing_values(df)
        data_types = self._check_data_types(df)
        unique_values = self._check_unique_values(df)
        numeric_stats = self._compute_numeric_stats(df)
        
        # Additional checks for target column
        if df[target_column].isnull().any():
            self.validation_errors.append(
                "Target column contains missing values"
            )
            
        # Create validation report
        report = DataValidationReport(
            timestamp=datetime.now().isoformat(),
            dataset_name="training_data",
            total_rows=len(df),
            total_columns=len(df.columns),
            missing_values=missing_values,
            data_types=data_types,
            unique_values=unique_values,
            numeric_stats=numeric_stats,
            validation_errors=self.validation_errors,
            warnings=self.warnings
        )
        
        return report
        
    def validate_prediction_data(
        self,
        df: pd.DataFrame,
        training_columns: List[str]
    ) -> DataValidationReport:
        """Validate prediction data"""
        self.validation_errors = []
        self.warnings = []
        
        # Basic checks
        if df.empty:
            raise DataValidationError("Dataset is empty")
            
        # Check columns match training data
        missing_cols = set(training_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(training_columns)
        
        if missing_cols:
            self.validation_errors.append(
                f"Missing columns: {missing_cols}"
            )
        if extra_cols:
            self.warnings.append(
                f"Extra columns will be ignored: {extra_cols}"
            )
            
        # Perform validation checks
        missing_values = self._check_missing_values(df)
        data_types = self._check_data_types(df)
        unique_values = self._check_unique_values(df)
        numeric_stats = self._compute_numeric_stats(df)
        
        # Create validation report
        report = DataValidationReport(
            timestamp=datetime.now().isoformat(),
            dataset_name="prediction_data",
            total_rows=len(df),
            total_columns=len(df.columns),
            missing_values=missing_values,
            data_types=data_types,
            unique_values=unique_values,
            numeric_stats=numeric_stats,
            validation_errors=self.validation_errors,
            warnings=self.warnings
        )
        
        return report
        
    def save_validation_report(
        self,
        report: DataValidationReport,
        output_dir: str = "validation_reports"
    ) -> str:
        """Save validation report to file"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"validation_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report.dict(), f, indent=2)
            
        return str(report_path)
