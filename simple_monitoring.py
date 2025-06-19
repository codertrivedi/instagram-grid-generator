import streamlit as st
import time
import psutil
import functools
from datetime import datetime
from typing import Dict, Any

class SimplePerformanceMonitor:
    
    def __init__(self):
        self.metrics = []
        
    def track_operation(self, operation_name: str):
        """Context manager for tracking operations"""
        return OperationTracker(operation_name, self)
    
    def log_metric(self, operation: str, duration: float, memory_mb: float):
        """Log a performance metric"""
        metric = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'duration': duration,
            'memory_mb': memory_mb
        }
        self.metrics.append(metric)
        
        # Keep only last 100 metrics
        if len(self.metrics) > 100:
            self.metrics = self.metrics[-100:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics:
            return {}
        
        # Group by operation
        operations = {}
        for metric in self.metrics:
            op = metric['operation']
            if op not in operations:
                operations[op] = []
            operations[op].append(metric)
        
        summary = {}
        for op, metrics in operations.items():
            durations = [m['duration'] for m in metrics]
            summary[op] = {
                'count': len(metrics),
                'avg_duration': sum(durations) / len(durations),
                'max_duration': max(durations),
                'total_duration': sum(durations)
            }
        
        return summary

class OperationTracker:
    """Context manager for tracking operation performance"""
    
    def __init__(self, operation_name: str, monitor: SimplePerformanceMonitor):
        self.operation_name = operation_name
        self.monitor = monitor
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        self.start_time = time.time()
        try:
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        except:
            self.start_memory = 0
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        try:
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        except:
            current_memory = self.start_memory
        
        self.monitor.log_metric(self.operation_name, duration, current_memory)

def performance_monitor(operation_name: str):
    """Decorator for monitoring function performance"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if 'performance_monitor' not in st.session_state:
                st.session_state.performance_monitor = SimplePerformanceMonitor()
            
            with st.session_state.performance_monitor.track_operation(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Simple tracking functions for compatibility
def track_request(endpoint_name: str):
    """Simple request tracking decorator"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Initialize simple monitor
def start_performance_monitoring():
    """Initialize performance monitoring"""
    if 'performance_monitor' not in st.session_state:
        st.session_state.performance_monitor = SimplePerformanceMonitor()
    return True

# Compatibility classes
class PerformanceDashboard:
    def __init__(self, monitor):
        self.monitor = monitor
    
    def render(self):
        st.header("📊 Simple Performance Dashboard")
        st.info("Simplified performance monitoring active")

# Mock metrics for compatibility
class MockMetric:
    def time(self):
        return MockTimer()

class MockTimer:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

IMAGE_PROCESSING_TIME = MockMetric()
LAYOUT_GENERATION_TIME = MockMetric()