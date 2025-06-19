import streamlit as st
import os
import time
import json
import hashlib
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd

class UnifiedAnalytics:
    """Comprehensive analytics system with secure access"""
    
    def __init__(self, log_file_path: str = "analytics_data.json"):
        self.log_file = log_file_path
        self.session_data = {}
        self.setup_security()
        self.init_session()
        
        # Auto-save every 5 minutes
        self.auto_save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self.auto_save_thread.start()
    
    def setup_security(self):
        """Setup secure access methods"""
        # Method 1: Streamlit Secrets (for Streamlit Cloud)
        try:
            self.admin_key = st.secrets.get("ADMIN_KEY", None)
            self.analytics_password = st.secrets.get("ANALYTICS_PASSWORD", None)
        except:
            self.admin_key = None
            self.analytics_password = None
        
        # Method 2: Environment Variables (for other platforms)
        if not self.admin_key:
            self.admin_key = os.environ.get("ADMIN_KEY")
            self.analytics_password = os.environ.get("ANALYTICS_PASSWORD")
        
        # Method 3: Generate demo credentials for development
        if not self.admin_key:
            self.admin_key = "demo_analytics_access"
            self.analytics_password = "analytics_123s_password"
    
    def init_session(self):
        """Initialize session tracking"""
        if 'analytics_session_id' not in st.session_state:
            timestamp = str(time.time())
            session_id = hashlib.md5(f"{timestamp}_{id(st.session_state)}".encode()).hexdigest()[:12]
            st.session_state.analytics_session_id = session_id
            
            self.session_data = {
                'session_id': session_id,
                'start_time': datetime.now().isoformat(),
                'user_agent': self._get_user_agent(),
                'events': [],
                'performance_metrics': [],
                'errors': []
            }
            
            self._log_event('session_start')
    
    def _get_user_agent(self) -> str:
        """Try to get user agent info"""
        try:
            return "Streamlit_App"
        except:
            return "Unknown"
    
    def _log_event(self, event_type: str, data: Dict[Any, Any] = None):
        """Log an event to the backend"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'data': data or {},
            'memory_usage_mb': self._get_memory_usage()
        }
        
        self.session_data['events'].append(event)
        
        # Keep only last 100 events per session
        if len(self.session_data['events']) > 100:
            self.session_data['events'] = self.session_data['events'][-100:]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        try:
            return psutil.Process().memory_info().rss / 1024 / 1024
        except:
            return 0
    
    def _auto_save_loop(self):
        """Auto-save analytics data every 5 minutes"""
        while True:
            try:
                time.sleep(300)  # 5 minutes
                self.save_to_disk()
            except Exception as e:
                print(f"Analytics auto-save error: {e}")
    
    def save_to_disk(self):
        """Save analytics data to disk"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = {'sessions': []}
            
            session_copy = self.session_data.copy()
            session_copy['end_time'] = datetime.now().isoformat()
            session_copy['total_events'] = len(session_copy['events'])
            
            # Check if session already exists (update) or add new
            session_id = session_copy['session_id']
            session_exists = False
            
            for i, session in enumerate(existing_data['sessions']):
                if session['session_id'] == session_id:
                    existing_data['sessions'][i] = session_copy
                    session_exists = True
                    break
            
            if not session_exists:
                existing_data['sessions'].append(session_copy)
            
            # Keep only last 1000 sessions
            if len(existing_data['sessions']) > 1000:
                existing_data['sessions'] = existing_data['sessions'][-1000:]
            
            with open(self.log_file, 'w') as f:
                json.dump(existing_data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error saving analytics: {e}")
    
    # Event tracking methods
    def track_page_load(self):
        """Track page load"""
        self._log_event('page_load')
    
    def track_images_uploaded(self, count: int, total_size_mb: float):
        """Track image uploads"""
        self._log_event('images_uploaded', {
            'image_count': count,
            'total_size_mb': total_size_mb
        })
    
    def track_layout_generation_start(self, num_images: int, settings: Dict):
        """Track when layout generation starts"""
        self._log_event('layout_generation_start', {
            'num_images': num_images,
            'settings': settings
        })
    
    def track_layout_generation_complete(self, num_layouts: int, processing_time: float):
        """Track successful layout generation"""
        self._log_event('layout_generation_complete', {
            'num_layouts': num_layouts,
            'processing_time_seconds': processing_time,
            'success': True
        })
    
    def track_layout_download(self, layout_index: int, layout_name: str):
        """Track layout downloads"""
        self._log_event('layout_download', {
            'layout_index': layout_index,
            'layout_name': layout_name
        })
    
    def track_performance_metric(self, operation: str, duration: float, memory_delta: float):
        """Track performance metrics"""
        metric = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'duration_seconds': duration,
            'memory_delta_mb': memory_delta
        }
        
        self.session_data['performance_metrics'].append(metric)
        
        if len(self.session_data['performance_metrics']) > 50:
            self.session_data['performance_metrics'] = self.session_data['performance_metrics'][-50:]
    
    def check_admin_access(self) -> bool:
        """Check if user has admin access via URL parameter"""
        url_key = st.query_params.get("admin")
        return url_key == self.admin_key
    
    def check_password_access(self) -> bool:
        """Check admin access via password input"""
        if "analytics_authenticated" in st.session_state:
            return st.session_state.analytics_authenticated
        
        st.title("🔒 Analytics Access")
        st.markdown("Enter the analytics password to view usage data:")
        
        password = st.text_input("Password", type="password", key="analytics_password_input")
        
        if st.button("Access Analytics"):
            if password == self.analytics_password:
                st.session_state.analytics_authenticated = True
                st.success("Access granted! Refreshing page...")
                st.rerun()
            else:
                st.error("Invalid password")
        
        with st.expander("📋 For Resume Reviewers"):
            st.markdown("""
            **Demo Access Information:**
            - Password: `demo123`
            - This demonstrates the analytics system I built
            - In production, this would use secure environment variables
            """)
        
        return False
    
    def show_analytics_dashboard(self):
        """Show the analytics dashboard"""
        st.title("📊 Analytics Dashboard")
        st.markdown("*This is a demonstration of the analytics system.*")
        
        try:
            reader = AnalyticsReader(self.log_file)
            
            days = st.selectbox("Time Period", [1, 7, 14, 30], index=1)
            summary = reader.get_summary_stats(days=days)
            
            if not summary:
                st.warning("No analytics data available yet.")
                self.show_demo_analytics()
                return
            
            # Summary metrics
            st.subheader("📈 Usage Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Sessions", summary.get('total_sessions', 0))
            with col2:
                st.metric("Layout Generations", summary.get('layout_generations', 0))
            with col3:
                st.metric("Downloads", summary.get('layout_downloads', 0))
            with col4:
                conversion = summary.get('conversion_rate', 0)
                st.metric("Conversion Rate", f"{conversion:.1f}%")
            
            # Performance data
            performance = reader.get_performance_summary()
            if performance:
                st.subheader("⚡ Performance Metrics")
                
                perf_data = []
                for operation, stats in performance.items():
                    perf_data.append({
                        'Operation': operation.replace('_', ' ').title(),
                        'Executions': stats['count'],
                        'Avg Duration': f"{stats['avg_duration']:.3f}s",
                        'Max Duration': f"{stats['max_duration']:.3f}s"
                    })
                
                df = pd.DataFrame(perf_data)
                st.dataframe(df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error loading analytics: {str(e)}")
            self.show_demo_analytics()
    
    def show_demo_analytics(self):
        """Show demo analytics for resume reviewers"""
        st.markdown("*Sample data to demonstrate analytics capabilities:*")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sessions", 156)
        with col2:
            st.metric("Layout Generations", 89)
        with col3:
            st.metric("Downloads", 67)
        with col4:
            st.metric("Conversion Rate", "75.3%")
        
        st.subheader("⚡ Performance Metrics (Sample)")
        sample_perf = pd.DataFrame([
            {'Operation': 'Image Processing', 'Executions': 89, 'Avg Duration': '2.34s', 'Max Duration': '5.67s'},
            {'Operation': 'Layout Generation', 'Executions': 89, 'Avg Duration': '1.89s', 'Max Duration': '4.12s'},
            {'Operation': 'Color Analysis', 'Executions': 89, 'Avg Duration': '0.45s', 'Max Duration': '0.89s'},
        ])
        st.dataframe(sample_perf, use_container_width=True)

class AnalyticsReader:
    """Read and analyze stored analytics data"""
    
    def __init__(self, log_file_path: str = "analytics_data.json"):
        self.log_file = log_file_path
    
    def get_summary_stats(self, days: int = 7) -> Dict:
        """Get summary statistics for the last N days"""
        if not os.path.exists(self.log_file):
            return {}
        
        with open(self.log_file, 'r') as f:
            data = json.load(f)
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_sessions = []
        
        for session in data.get('sessions', []):
            session_start = datetime.fromisoformat(session['start_time'])
            if session_start >= cutoff_date:
                recent_sessions.append(session)
        
        if not recent_sessions:
            return {}
        
        total_sessions = len(recent_sessions)
        total_events = sum(session['total_events'] for session in recent_sessions)
        
        # Count specific events
        layout_generations = 0
        layout_downloads = 0
        image_uploads = 0
        errors = 0
        
        for session in recent_sessions:
            for event in session.get('events', []):
                event_type = event['event_type']
                if event_type == 'layout_generation_complete':
                    layout_generations += 1
                elif event_type == 'layout_download':
                    layout_downloads += 1
                elif event_type == 'images_uploaded':
                    image_uploads += 1
                elif event_type == 'error_occurred':
                    errors += 1
        
        return {
            'period_days': days,
            'total_sessions': total_sessions,
            'total_events': total_events,
            'layout_generations': layout_generations,
            'layout_downloads': layout_downloads,
            'image_uploads': image_uploads,
            'errors': errors,
            'conversion_rate': (layout_downloads / max(layout_generations, 1)) * 100,
            'avg_events_per_session': total_events / max(total_sessions, 1),
            'most_recent_session': recent_sessions[-1]['start_time'] if recent_sessions else None
        }
    
    def get_performance_summary(self) -> Dict:
        """Get performance statistics"""
        if not os.path.exists(self.log_file):
            return {}
        
        with open(self.log_file, 'r') as f:
            data = json.load(f)
        
        all_metrics = []
        for session in data.get('sessions', []):
            all_metrics.extend(session.get('performance_metrics', []))
        
        if not all_metrics:
            return {}
        
        operations = {}
        for metric in all_metrics:
            op = metric['operation']
            if op not in operations:
                operations[op] = []
            operations[op].append({
                'duration': metric['duration_seconds'],
                'memory_delta': metric['memory_delta_mb']
            })
        
        summary = {}
        for op, metrics in operations.items():
            durations = [m['duration'] for m in metrics]
            memory_deltas = [m['memory_delta'] for m in metrics]
            
            summary[op] = {
                'count': len(metrics),
                'avg_duration': sum(durations) / len(durations),
                'max_duration': max(durations),
                'avg_memory_delta': sum(memory_deltas) / len(memory_deltas)
            }
        
        return summary

# Global analytics instance
_analytics_instance = None

def get_analytics() -> UnifiedAnalytics:
    """Get global analytics instance"""
    global _analytics_instance
    if _analytics_instance is None:
        _analytics_instance = UnifiedAnalytics()
    return _analytics_instance

def setup_secure_analytics():
    """Main function to setup secure analytics access"""
    analytics = get_analytics()
    
    # Check for URL-based admin access
    if analytics.check_admin_access():
        analytics.show_analytics_dashboard()
        return True
    
    # Check for password-based access
    if analytics.check_password_access():
        analytics.show_analytics_dashboard()
        return True
    
    return False

def integrate_analytics():
    """Integration function for main app"""
    return setup_secure_analytics()

# Alias for backward compatibility  
def integrate_secure_analytics():
    """Alias for integrate_analytics"""
    return integrate_analytics()

# Performance tracking decorator for compatibility
def track_performance(operation_name: str):
    """Decorator to automatically track function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            analytics = get_analytics()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                print(f"Error in {operation_name}: {e}")
                raise
            finally:
                end_time = time.time()
                analytics.track_performance_metric(
                    operation=operation_name,
                    duration=end_time - start_time,
                    memory_delta=0  # Simplified for compatibility
                )
        
        return wrapper
    return decorator