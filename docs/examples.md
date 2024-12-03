# أمثلة وحالات عملية لاستخدام AutoSKL

## جدول المحتويات
1. [تحليل البيانات](#تحليل-البيانات)
2. [تدريب النماذج](#تدريب-النماذج)
3. [إدارة النماذج](#إدارة-النماذج)
4. [المراقبة والتقارير](#المراقبة-والتقارير)
5. [التنبيهات والإشعارات](#التنبيهات-والإشعارات)

## تحليل البيانات

### تحليل مجموعة بيانات للتصنيف
```python
import pandas as pd
from app.services.data_analyzer import DataAnalyzer

# تحميل البيانات
data = pd.read_csv('example_classification.csv')

# إنشاء محلل البيانات
analyzer = DataAnalyzer()

# عرض نظرة عامة
analyzer.show_overview(data)

# تحليل المتغيرات
analyzer.show_variable_analysis(data)

# عرض الارتباطات
analyzer.show_correlations(data)
```

### تحليل مجموعة بيانات للانحدار
```python
# تحميل بيانات الانحدار
data = pd.read_csv('example_regression.csv')

# تحليل القيم المفقودة
analyzer.show_missing_values(data)

# عرض الرسوم البيانية
analyzer.show_plots(data)
```

## تدريب النماذج

### تدريب نموذج تصنيف
```python
from app.services.model_trainer import ModelTrainer

# إنشاء مدرب النماذج
trainer = ModelTrainer()

# تحضير البيانات
X = data.drop('target', axis=1)
y = data['target']

# تدريب النموذج
model_info = trainer.train(
    X=X,
    y=y,
    task_type='classification',
    time_limit=600  # 10 دقائق
)

# حفظ النموذج
model_id = trainer.save_trained_model(
    name='classification_model_v1',
    preprocessing_params={'feature_names': X.columns.tolist()}
)
```

### تدريب نموذج انحدار
```python
# تدريب نموذج انحدار
model_info = trainer.train(
    X=X,
    y=y,
    task_type='regression',
    time_limit=300  # 5 دقائق
)
```

## إدارة النماذج

### عرض تاريخ النماذج
```python
from app.ui.components.model_manager import ModelManager

# إنشاء مدير النماذج
manager = ModelManager()

# عرض تاريخ نموذج معين
manager.show_model_history('classification_model_v1')

# عرض تفاصيل نموذج
manager.show_model_details(model_id)
```

### تصدير واستيراد النماذج
```python
# تصدير نموذج
manager.export_model(model_id)

# استيراد نموذج
# يتم تنفيذه عبر واجهة المستخدم
```

## المراقبة والتقارير

### مراقبة أداء النموذج
```python
from app.ui.components.model_monitor import ModelMonitor

# إنشاء مراقب النماذج
monitor = ModelMonitor()

# عرض لوحة المراقبة
monitor.show_monitoring_dashboard(model_id, days=30)

# حساب انحراف البيانات
drift_scores = monitor.calculate_drift(
    training_data=original_data,
    new_data=new_data
)
```

### إنشاء تقارير
```python
from app.ui.components.report_generator import ReportGenerator

# إنشاء مولد التقارير
generator = ReportGenerator()

# إضافة بيانات الأداء
generator.add_model_performance(metrics, model_name)

# إضافة أهمية المتغيرات
generator.add_feature_importance(importance_dict)

# توليد التقرير
generator.generate_report()
```

## التنبيهات والإشعارات

### إعداد التنبيهات
```python
from app.services.notification_service import NotificationService

# إنشاء خدمة الإشعارات
notifier = NotificationService()

# إنشاء تنبيه جديد
alert_id = notifier.create_alert(
    model_id=model_id,
    alert_type='performance_degradation',
    conditions={
        'accuracy': {
            'operator': 'less_than',
            'threshold': 0.9
        }
    },
    notification_channels=['email', 'database']
)

# التحقق من التنبيهات
notifier.check_alerts(model_id, current_metrics)

# جلب الإشعارات غير المقروءة
unread = notifier.get_unread_notifications()
```

## نصائح وأفضل الممارسات

1. **تحليل البيانات**:
   - قم دائماً بفحص البيانات قبل التدريب
   - تحقق من القيم المفقودة والقيم الشاذة
   - افحص توزيع المتغيرات والارتباطات

2. **تدريب النماذج**:
   - استخدم التحقق المتقاطع للتقييم
   - جرب مجموعة متنوعة من الخوارزميات
   - اضبط وقت التحسين بناءً على حجم البيانات

3. **إدارة النماذج**:
   - احتفظ بإصدارات مختلفة من النماذج
   - وثق التغييرات والتحسينات
   - قم بتصدير النماذج المهمة

4. **المراقبة**:
   - راقب أداء النماذج بانتظام
   - تحقق من انحراف البيانات
   - اضبط التنبيهات للكشف المبكر عن المشاكل

5. **الأمان**:
   - احمِ بيانات التدريب
   - استخدم التشفير للبيانات الحساسة
   - تحكم في الوصول إلى النماذج
