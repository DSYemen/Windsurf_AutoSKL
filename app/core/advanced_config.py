from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from enum import Enum

class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"

class ModelPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class NotificationChannel(str, Enum):
    EMAIL = "email"
    DATABASE = "database"
    WEBHOOK = "webhook"

class AlertCondition(BaseModel):
    metric: str
    operator: str
    threshold: float
    
class ModelConfig(BaseModel):
    """إعدادات النموذج"""
    name: str
    task_type: TaskType
    priority: ModelPriority = Field(default=ModelPriority.MEDIUM)
    max_training_time: int = Field(default=3600)  # بالثواني
    optimization_metric: str
    cv_folds: int = Field(default=5)
    early_stopping_rounds: int = Field(default=10)
    enable_feature_selection: bool = Field(default=True)
    feature_selection_method: str = Field(default="mutual_info")
    
class MonitoringConfig(BaseModel):
    """إعدادات المراقبة"""
    enable_monitoring: bool = Field(default=True)
    monitoring_interval: int = Field(default=3600)  # بالثواني
    metrics_history_length: int = Field(default=30)  # بالأيام
    drift_detection_window: int = Field(default=1000)  # عدد التنبؤات
    alert_conditions: List[AlertCondition] = Field(default_factory=list)
    notification_channels: List[NotificationChannel] = Field(
        default_factory=lambda: [NotificationChannel.DATABASE]
    )
    
class DataConfig(BaseModel):
    """إعدادات البيانات"""
    max_missing_ratio: float = Field(default=0.2)
    categorical_encoding: str = Field(default="auto")
    numerical_scaling: str = Field(default="standard")
    handle_imbalance: bool = Field(default=True)
    sampling_strategy: str = Field(default="auto")
    
class SecurityConfig(BaseModel):
    """إعدادات الأمان"""
    enable_encryption: bool = Field(default=True)
    encryption_method: str = Field(default="AES")
    api_key_required: bool = Field(default=True)
    max_requests_per_minute: int = Field(default=60)
    
class AdvancedConfig(BaseModel):
    """التكوين المتقدم للنظام"""
    model: ModelConfig
    monitoring: MonitoringConfig
    data: DataConfig
    security: SecurityConfig
    
    @classmethod
    def from_file(cls, config_path: str) -> 'AdvancedConfig':
        """تحميل التكوين من ملف"""
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def save_to_file(self, config_path: str):
        """حفظ التكوين إلى ملف"""
        import yaml
        config_dict = self.dict()
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True)
            
    def update_model_config(self, **kwargs):
        """تحديث إعدادات النموذج"""
        for key, value in kwargs.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
                
    def update_monitoring_config(self, **kwargs):
        """تحديث إعدادات المراقبة"""
        for key, value in kwargs.items():
            if hasattr(self.monitoring, key):
                setattr(self.monitoring, key, value)
                
    def add_alert_condition(self, metric: str, operator: str, threshold: float):
        """إضافة شرط تنبيه جديد"""
        condition = AlertCondition(
            metric=metric,
            operator=operator,
            threshold=threshold
        )
        self.monitoring.alert_conditions.append(condition)
        
    def remove_alert_condition(self, index: int):
        """إزالة شرط تنبيه"""
        if 0 <= index < len(self.monitoring.alert_conditions):
            self.monitoring.alert_conditions.pop(index)
            
    def add_notification_channel(self, channel: NotificationChannel):
        """إضافة قناة إشعارات"""
        if channel not in self.monitoring.notification_channels:
            self.monitoring.notification_channels.append(channel)
            
    def remove_notification_channel(self, channel: NotificationChannel):
        """إزالة قناة إشعارات"""
        if channel in self.monitoring.notification_channels:
            self.monitoring.notification_channels.remove(channel)
            
    def get_optimal_settings(self, data_size: int, n_features: int) -> Dict[str, Any]:
        """الحصول على الإعدادات المثلى بناءً على حجم البيانات"""
        settings = {}
        
        # ضبط وقت التدريب
        if data_size < 1000:
            settings['max_training_time'] = 300  # 5 دقائق
        elif data_size < 10000:
            settings['max_training_time'] = 1800  # 30 دقيقة
        else:
            settings['max_training_time'] = 3600  # ساعة
            
        # ضبط التحقق المتقاطع
        if data_size < 5000:
            settings['cv_folds'] = 3
        else:
            settings['cv_folds'] = 5
            
        # ضبط اختيار المتغيرات
        if n_features > 100:
            settings['enable_feature_selection'] = True
        else:
            settings['enable_feature_selection'] = False
            
        return settings
        
    def validate_settings(self) -> List[str]:
        """التحقق من صحة الإعدادات"""
        issues = []
        
        # التحقق من إعدادات النموذج
        if self.model.max_training_time < 60:
            issues.append("وقت التدريب قصير جداً")
            
        if self.model.cv_folds < 2:
            issues.append("عدد طيات التحقق المتقاطع قليل جداً")
            
        # التحقق من إعدادات المراقبة
        if self.monitoring.enable_monitoring and self.monitoring.monitoring_interval < 60:
            issues.append("فترة المراقبة قصيرة جداً")
            
        # التحقق من إعدادات البيانات
        if not 0 <= self.data.max_missing_ratio <= 1:
            issues.append("نسبة القيم المفقودة يجب أن تكون بين 0 و 1")
            
        return issues
