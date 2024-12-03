import streamlit as st
from typing import Optional, Dict, Any
import time
from datetime import datetime
import json
from pathlib import Path

class NotificationSystem:
    def __init__(self):
        self.notification_types = {
            'success': '✅',
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'model': '🤖',
            'data': '📊',
            'system': '⚙️'
        }
        
        # Initialize session state for notifications
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
            
        if 'notification_history' not in st.session_state:
            st.session_state.notification_history = []
            
    def notify(
        self,
        message: str,
        type: str = 'info',
        duration: Optional[int] = None,
        persist: bool = False
    ):
        """Show a notification"""
        notification = {
            'id': int(time.time() * 1000),
            'message': message,
            'type': type,
            'icon': self.notification_types.get(type, 'ℹ️'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'persist': persist
        }
        
        # Add to current notifications
        st.session_state.notifications.append(notification)
        
        # Add to history
        st.session_state.notification_history.append(notification)
        
        # Show the notification
        self._display_notification(notification)
        
        # Auto-dismiss if duration is set
        if duration:
            time.sleep(duration)
            self.dismiss(notification['id'])
            
    def dismiss(self, notification_id: int):
        """Dismiss a notification"""
        st.session_state.notifications = [
            n for n in st.session_state.notifications
            if n['id'] != notification_id
        ]
        
    def clear_all(self):
        """Clear all notifications"""
        st.session_state.notifications = []
        
    def show_history(self, max_items: int = 10):
        """Show notification history"""
        with st.expander("📋 Notification History", expanded=False):
            for notification in list(reversed(
                st.session_state.notification_history
            ))[:max_items]:
                self._display_notification(notification, in_history=True)
                
    def _display_notification(
        self,
        notification: Dict[str, Any],
        in_history: bool = False
    ):
        """Display a single notification"""
        type_colors = {
            'success': '#4CAF50',
            'info': '#2196F3',
            'warning': '#ff9800',
            'error': '#f44336',
            'model': '#9c27b0',
            'data': '#00bcd4',
            'system': '#607d8b'
        }
        
        color = type_colors.get(notification['type'], '#2196F3')
        
        notification_html = f"""
        <div style="
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 0.5rem;
            background-color: {color}15;
            border-left: 4px solid {color};
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 1.2em; margin-right: 0.5rem;">
                        {notification['icon']}
                    </span>
                    <span style="color: {color};">
                        {notification['message']}
                    </span>
                </div>
                <div style="color: gray; font-size: 0.8em;">
                    {notification['timestamp']}
                </div>
            </div>
        </div>
        """
        
        st.markdown(notification_html, unsafe_allow_html=True)
        
        if not in_history and not notification['persist']:
            # Add dismiss button
            if st.button(
                "Dismiss",
                key=f"dismiss_{notification['id']}"
            ):
                self.dismiss(notification['id'])
                
    def show_success(self, message: str, persist: bool = False):
        """Show a success notification"""
        self.notify(message, type='success', persist=persist)
        
    def show_info(self, message: str, persist: bool = False):
        """Show an info notification"""
        self.notify(message, type='info', persist=persist)
        
    def show_warning(self, message: str, persist: bool = False):
        """Show a warning notification"""
        self.notify(message, type='warning', persist=persist)
        
    def show_error(self, message: str, persist: bool = True):
        """Show an error notification"""
        self.notify(message, type='error', persist=persist)
        
    def show_model_update(self, message: str, persist: bool = False):
        """Show a model update notification"""
        self.notify(message, type='model', persist=persist)
        
    def show_data_update(self, message: str, persist: bool = False):
        """Show a data update notification"""
        self.notify(message, type='data', persist=persist)
        
    def show_system_update(self, message: str, persist: bool = False):
        """Show a system update notification"""
        self.notify(message, type='system', persist=persist)
        
    def save_history(self, file_path: str):
        """Save notification history to file"""
        history = {
            'notifications': st.session_state.notification_history,
            'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        Path(file_path).write_text(
            json.dumps(history, indent=2),
            encoding='utf-8'
        )
        
    def load_history(self, file_path: str):
        """Load notification history from file"""
        if Path(file_path).exists():
            history = json.loads(
                Path(file_path).read_text(encoding='utf-8')
            )
            st.session_state.notification_history = history['notifications']
            
    def create_toast(
        self,
        message: str,
        type: str = 'info',
        duration: int = 3
    ):
        """Create a toast notification"""
        st.toast(
            f"{self.notification_types.get(type, 'ℹ️')} {message}",
            icon=self.notification_types.get(type, 'ℹ️')
        )
        
    def create_progress(
        self,
        message: str,
        total_steps: int
    ) -> 'ProgressNotification':
        """Create a progress notification"""
        return ProgressNotification(self, message, total_steps)
        
class ProgressNotification:
    def __init__(
        self,
        notification_system: NotificationSystem,
        message: str,
        total_steps: int
    ):
        self.ns = notification_system
        self.message = message
        self.total_steps = total_steps
        self.current_step = 0
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        
    def update(
        self,
        step: Optional[int] = None,
        message: Optional[str] = None
    ):
        """Update progress"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        if message is not None:
            self.message = message
            
        progress = min(self.current_step / self.total_steps, 1.0)
        self.progress_bar.progress(progress)
        self.status_text.text(
            f"{self.message} ({self.current_step}/{self.total_steps})"
        )
        
    def complete(self, message: Optional[str] = None):
        """Complete the progress"""
        self.progress_bar.progress(1.0)
        if message:
            self.status_text.text(message)
        else:
            self.status_text.text("✅ Complete!")
            
    def error(self, error_message: str):
        """Show error in progress"""
        self.status_text.error(f"❌ Error: {error_message}")
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.complete()
        else:
            self.error(str(exc_val))
