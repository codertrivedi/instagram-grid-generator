import time
import functools
import threading
import gc
from typing import Dict, Any, Callable
import streamlit as st
from PIL import Image
import numpy as np
import io
import psutil
import tracemalloc
import hashlib
import weakref

class PerformanceProfiler:
    """Context manager for profiling performance"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.start_memory = None
        
    def __enter__(self):
        self.start_time = time.time()
        try:
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            tracemalloc.start()
        except:
            self.start_memory = 0
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        
        try:
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_delta = end_memory - self.start_memory
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak / 1024 / 1024
        except:
            memory_delta = 0
            peak_mb = 0
        
        # Store in session state for monitoring
        if 'performance_logs' not in st.session_state:
            st.session_state.performance_logs = []
        
        st.session_state.performance_logs.append({
            'operation': self.operation_name,
            'duration': duration,
            'memory_delta': memory_delta,
            'peak_memory': peak_mb,
            'timestamp': time.time()
        })
        
        # Keep only last 50 logs
        if len(st.session_state.performance_logs) > 50:
            st.session_state.performance_logs = st.session_state.performance_logs[-50:]

def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with PerformanceProfiler(func.__name__):
            return func(*args, **kwargs)
    return wrapper

class StreamlitCache:
    """Simple caching implementation with TTL and LRU eviction"""
    
    def __init__(self, maxsize: int = 128, ttl: int = 3600):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self.lock = threading.RLock()
        self._hits = 0
        self._total = 0
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments"""
        key_data = f"{func_name}_{str(args)}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _evict_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, creation_time in self.creation_times.items()
            if current_time - creation_time > self.ttl
        ]
        for key in expired_keys:
            self._remove_entry(key)
    
    def _evict_lru(self):
        """Remove least recently used entries if cache is full"""
        while len(self.cache) >= self.maxsize:
            if not self.access_times:
                break
            lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            self._remove_entry(lru_key)
    
    def _remove_entry(self, key: str):
        """Remove entry from all tracking dictionaries"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.creation_times.pop(key, None)
    
    def get(self, func_name: str, args: tuple, kwargs: dict):
        """Get cached result"""
        with self.lock:
            self._total += 1
            key = self._generate_key(func_name, args, kwargs)
            
            # Clean up expired entries
            self._evict_expired()
            
            if key in self.cache:
                self.access_times[key] = time.time()
                self._hits += 1
                return self.cache[key]
            
            return None
    
    def set(self, func_name: str, args: tuple, kwargs: dict, result):
        """Set cached result"""
        with self.lock:
            key = self._generate_key(func_name, args, kwargs)
            
            # Evict if necessary
            self._evict_expired()
            self._evict_lru()
            
            # Store result
            current_time = time.time()
            self.cache[key] = result
            self.access_times[key] = current_time
            self.creation_times[key] = current_time
    
    def clear(self):
        """Clear all cached entries"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.creation_times.clear()
            self._hits = 0
            self._total = 0
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        with self.lock:
            hit_rate = self._hits / max(self._total, 1)
            return {
                'size': len(self.cache),
                'maxsize': self.maxsize,
                'hit_rate': hit_rate,
                'hits': self._hits,
                'total': self._total,
                'memory_entries': len(self.cache)
            }

# Global cache instance
_global_cache = StreamlitCache()

def cached_function(ttl: int = 3600, maxsize: int = 128):
    """Decorator for caching function results with TTL"""
    def decorator(func):
        # Create a local cache for this function if using different settings
        if ttl != 3600 or maxsize != 128:
            local_cache = StreamlitCache(maxsize=maxsize, ttl=ttl)
        else:
            local_cache = _global_cache
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Convert unhashable types to hashable ones
            hashable_args = []
            for arg in args:
                if isinstance(arg, (list, dict, set)):
                    hashable_args.append(str(arg))
                elif hasattr(arg, '__dict__'):
                    hashable_args.append(str(arg.__dict__))
                else:
                    hashable_args.append(arg)
            
            hashable_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, (list, dict, set)):
                    hashable_kwargs[key] = str(value)
                elif hasattr(value, '__dict__'):
                    hashable_kwargs[key] = str(value.__dict__)
                else:
                    hashable_kwargs[key] = value
            
            # Try to get from cache
            result = local_cache.get(func.__name__, tuple(hashable_args), hashable_kwargs)
            if result is not None:
                return result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache (only if result is not too large)
            try:
                # Rough size check - avoid caching very large objects
                if hasattr(result, '__len__') and len(str(result)) < 1024 * 1024:  # 1MB
                    local_cache.set(func.__name__, tuple(hashable_args), hashable_kwargs, result)
                elif not hasattr(result, '__len__'):
                    local_cache.set(func.__name__, tuple(hashable_args), hashable_kwargs, result)
            except Exception:
                # If we can't cache it, just return the result
                pass
            
            return result
        
        # Add cache management methods to function
        wrapper.cache_stats = lambda: local_cache.get_stats()
        wrapper.cache_clear = lambda: local_cache.clear()
        
        return wrapper
    return decorator

class ImageOptimizer:
    """Image optimization utilities"""
    
    @staticmethod
    def smart_resize(image: Image.Image, target_size: tuple, maintain_aspect: bool = True) -> Image.Image:
        """Smart resize that maintains quality while reducing size"""
        if maintain_aspect:
            # Calculate aspect ratio preserving size
            current_ratio = image.width / image.height
            target_ratio = target_size[0] / target_size[1]
            
            if current_ratio > target_ratio:
                # Image is wider, fit to width
                new_width = target_size[0]
                new_height = int(target_size[0] / current_ratio)
            else:
                # Image is taller, fit to height
                new_height = target_size[1]
                new_width = int(target_size[1] * current_ratio)
            
            new_size = (new_width, new_height)
        else:
            new_size = target_size
        
        # Use high-quality resampling
        return image.resize(new_size, Image.Resampling.LANCZOS)
    
    @staticmethod
    def progressive_jpeg(image: Image.Image, quality: int = 85) -> bytes:
        """Create progressive JPEG for faster web loading"""
        output = io.BytesIO()
        
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        
        image.save(
            output,
            format='JPEG',
            quality=quality,
            optimize=True,
            progressive=True,
            subsampling=0,  # No subsampling for better quality
        )
        
        output.seek(0)
        return output.getvalue()
    
    @staticmethod
    def create_webp_variant(image: Image.Image, quality: int = 80) -> bytes:
        """Create WebP variant with optimal settings"""
        output = io.BytesIO()
        
        try:
            # Use advanced WebP encoding options
            image.save(
                output,
                format='WEBP',
                quality=quality,
                method=6,  # Highest compression effort
                lossless=False,
                exact=False
            )
        except Exception:
            # Fallback to JPEG if WebP fails
            return ImageOptimizer.progressive_jpeg(image, quality)
        
        output.seek(0)
        return output.getvalue()
    
    @staticmethod
    def create_thumbnail(image: Image.Image, size: tuple = (300, 300)) -> Image.Image:
        """Create optimized thumbnail"""
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        return thumbnail

class MemoryProfiler:
    """Memory profiling utilities"""
    
    @staticmethod
    def get_object_size(obj) -> int:
        """Get size of object in bytes"""
        import sys
        
        if isinstance(obj, Image.Image):
            # PIL Image size calculation
            return obj.width * obj.height * len(obj.getbands()) * 4  # Assuming 32-bit
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, (str, bytes)):
            return sys.getsizeof(obj)
        elif isinstance(obj, (list, tuple, dict, set)):
            size = sys.getsizeof(obj)
            if isinstance(obj, dict):
                size += sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in obj.items())
            else:
                size += sum(sys.getsizeof(item) for item in obj)
            return size
        else:
            return sys.getsizeof(obj)
    
    @staticmethod
    def profile_session_state() -> dict:
        """Profile current session state memory usage"""
        total_size = 0
        object_counts = {}
        large_objects = []
        
        for key, value in st.session_state.items():
            try:
                size = MemoryProfiler.get_object_size(value)
                total_size += size
                
                obj_type = type(value).__name__
                object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
                
                if size > 1024 * 1024:  # Larger than 1MB
                    large_objects.append({
                        'key': key,
                        'type': obj_type,
                        'size_mb': size / (1024 * 1024)
                    })
            except Exception:
                # Skip objects that can't be sized
                continue
        
        return {
            'total_size_mb': total_size / (1024 * 1024),
            'object_counts': object_counts,
            'large_objects': sorted(large_objects, key=lambda x: x['size_mb'], reverse=True)
        }

class StreamlitOptimizer:
    """Collection of Streamlit-specific optimizations"""
    
    @staticmethod
    def clear_large_objects(threshold_mb: float = 100):
        """Clear large objects from session state"""
        profile = MemoryProfiler.profile_session_state()
        
        keys_to_clear = []
        for obj in profile['large_objects']:
            if obj['size_mb'] > threshold_mb:
                keys_to_clear.append(obj['key'])
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Force garbage collection
        gc.collect()
        
        return len(keys_to_clear)
    
    @staticmethod
    def optimize_dataframes(df_dict: dict) -> dict:
        """Optimize pandas DataFrames in session state"""
        try:
            import pandas as pd
            
            optimized = {}
            for key, df in df_dict.items():
                if isinstance(df, pd.DataFrame):
                    # Optimize dtypes
                    for col in df.select_dtypes(include=['object']):
                        if df[col].nunique() / len(df) < 0.5:  # Many duplicates
                            df[col] = df[col].astype('category')
                    
                    # Optimize numeric columns
                    for col in df.select_dtypes(include=['int64']):
                        if df[col].min() >= 0 and df[col].max() <= 255:
                            df[col] = df[col].astype('uint8')
                        elif df[col].min() >= -128 and df[col].max() <= 127:
                            df[col] = df[col].astype('int8')
                    
                    optimized[key] = df
                else:
                    optimized[key] = df
            
            return optimized
        except ImportError:
            # pandas not available
            return df_dict
    
    @staticmethod
    def setup_performance_monitoring():
        """Setup performance monitoring dashboard"""
        if 'perf_data' not in st.session_state:
            st.session_state.perf_data = {
                'memory_usage': [],
                'cpu_usage': [],
                'cache_stats': [],
                'timestamps': []
            }
        
        # Collect current metrics
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
        except:
            memory_mb = 0
            cpu_percent = 0
        
        cache_stats = _global_cache.get_stats()
        
        # Store metrics
        st.session_state.perf_data['memory_usage'].append(memory_mb)
        st.session_state.perf_data['cpu_usage'].append(cpu_percent)
        st.session_state.perf_data['cache_stats'].append(cache_stats['hit_rate'])
        st.session_state.perf_data['timestamps'].append(time.time())
        
        # Keep only last 100 entries
        for key in st.session_state.perf_data:
            if len(st.session_state.perf_data[key]) > 100:
                st.session_state.perf_data[key] = st.session_state.perf_data[key][-100:]

# Context manager for batch operations
class BatchOperationContext:
    """Context manager for optimizing batch operations"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.operations_count = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if self.operations_count > 0:
            ops_per_sec = self.operations_count / duration
            print(f"Batch operation '{self.operation_name}': "
                  f"{self.operations_count} operations in {duration:.3f}s "
                  f"({ops_per_sec:.1f} ops/sec)")
    
    def add_operation(self):
        """Increment operation counter"""
        self.operations_count += 1

def get_cache_manager():
    """Get the global cache manager"""
    return _global_cache

def cleanup_memory():
    """Force garbage collection and memory cleanup"""
    try:
        # Clear matplotlib figures if available
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except ImportError:
            pass
        
        # Force garbage collection
        gc.collect()
        
        # Get current memory usage
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            return memory_usage
        except:
            return 0
        
    except Exception:
        return 0

# Export frequently used functions
__all__ = [
    'PerformanceProfiler',
    'performance_monitor',
    'StreamlitCache',
    'cached_function',
    'ImageOptimizer',
    'MemoryProfiler',
    'StreamlitOptimizer',
    'BatchOperationContext',
    'get_cache_manager',
    'cleanup_memory'
]