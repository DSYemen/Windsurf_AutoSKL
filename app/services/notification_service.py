import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ..core.config import settings
from ..core.database import DatabaseManager

class NotificationService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager()
        
    def create_alert(self, 
                    model_id: int,
                    alert_type: str,
                    conditions: Dict[str, Any],
                    notification_channels: List[str]) -> int:
        """إنشاء تنبيه جديد"""
        alert = {
            'model_id': model_id,
            'type': alert_type,
            'conditions': conditions,
            'channels': notification_channels,
            'created_at': datetime.now(),
            'is_active': True
        }
        
        return self.db_manager.save_alert(alert)
        
    def check_alerts(self, model_id: int, metrics: Dict[str, float]):
        """التحقق من التنبيهات وإرسال الإشعارات إذا لزم الأمر"""
        alerts = self.db_manager.get_active_alerts(model_id)
        
        for alert in alerts:
            if self._evaluate_conditions(alert['conditions'], metrics):
                self._send_notifications(
                    alert_id=alert['id'],
                    model_id=model_id,
                    alert_type=alert['type'],
                    metrics=metrics,
                    channels=alert['channels']
                )
                
    def _evaluate_conditions(self, conditions: Dict[str, Any], 
                           metrics: Dict[str, float]) -> bool:
        """تقييم شروط التنبيه"""
        for metric, condition in conditions.items():
            if metric not in metrics:
                continue
                
            value = metrics[metric]
            operator = condition['operator']
            threshold = condition['threshold']
            
            if operator == 'greater_than' and value <= threshold:
                return False
            elif operator == 'less_than' and value >= threshold:
                return False
            elif operator == 'equal' and value != threshold:
                return False
                
        return True
        
    def _send_notifications(self, alert_id: int, model_id: int,
                          alert_type: str, metrics: Dict[str, float],
                          channels: List[str]):
        """إرسال الإشعارات عبر القنوات المحددة"""
        message = self._create_notification_message(
            alert_type=alert_type,
            model_id=model_id,
            metrics=metrics
        )
        
        for channel in channels:
            try:
                if channel == 'email':
                    self._send_email_notification(message)
                elif channel == 'database':
                    self._save_notification_to_db(
                        alert_id=alert_id,
                        message=message
                    )
                # يمكن إضافة قنوات أخرى هنا
                
            except Exception as e:
                self.logger.error(
                    f"خطأ في إرسال الإشعار عبر {channel}: {str(e)}"
                )
                
    def _create_notification_message(self, alert_type: str,
                                   model_id: int,
                                   metrics: Dict[str, float]) -> str:
        """إنشاء رسالة الإشعار"""
        model_info = self.db_manager.get_model_info(model_id)
        
        message = f"""
        تنبيه: {alert_type}
        النموذج: {model_info['name']} (ID: {model_id})
        التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        المقاييس:
        """
        
        for metric, value in metrics.items():
            message += f"- {metric}: {value}\n"
            
        return message
        
    def _send_email_notification(self, message: str):
        """إرسال إشعار عبر البريد الإلكتروني"""
        if not hasattr(settings, 'EMAIL_SETTINGS'):
            self.logger.warning("لم يتم تكوين إعدادات البريد الإلكتروني")
            return
            
        email_settings = settings.EMAIL_SETTINGS
        
        msg = MIMEMultipart()
        msg['From'] = email_settings['sender']
        msg['To'] = email_settings['recipient']
        msg['Subject'] = "تنبيه من نظام AutoSKL"
        
        msg.attach(MIMEText(message, 'plain', 'utf-8'))
        
        try:
            with smtplib.SMTP(
                email_settings['smtp_server'],
                email_settings['smtp_port']
            ) as server:
                if email_settings.get('use_tls', False):
                    server.starttls()
                    
                if 'username' in email_settings:
                    server.login(
                        email_settings['username'],
                        email_settings['password']
                    )
                    
                server.send_message(msg)
                
        except Exception as e:
            self.logger.error(f"خطأ في إرسال البريد الإلكتروني: {str(e)}")
            
    def _save_notification_to_db(self, alert_id: int, message: str):
        """حفظ الإشعار في قاعدة البيانات"""
        notification = {
            'alert_id': alert_id,
            'message': message,
            'created_at': datetime.now(),
            'is_read': False
        }
        
        self.db_manager.save_notification(notification)
        
    def get_unread_notifications(self) -> List[Dict[str, Any]]:
        """جلب الإشعارات غير المقروءة"""
        return self.db_manager.get_unread_notifications()
        
    def mark_notification_as_read(self, notification_id: int):
        """تحديد الإشعار كمقروء"""
        self.db_manager.update_notification_status(
            notification_id=notification_id,
            is_read=True
        )
