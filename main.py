import streamlit as st
import os
import time
import copy
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
import json
import psutil
import base64
from io import BytesIO
import tempfile
import matplotlib.pyplot as plt
import torch
import threading
import queue
import asyncio
import functools
import weakref
import gc
from concurrent.futures import ThreadPoolExecutor
import hashlib
from typing import Optional
import atexit

# Debug mode flag
DEBUG_MODE = False

# Constants for performance optimization
MAX_IMAGE_SIZE = 1024  # Max dimension for processing
THUMBNAIL_SIZE = 300   # Size for thumbnails
JPEG_QUALITY = 85     # Quality for JPEG compression
CACHE_TTL = 3600      # Cache time-to-live in seconds
MAX_WORKERS = 4       # Maximum worker threads

# Import components (with fallback for missing modules)
try:
    from enhanced_instagram_grid import EnhancedInstagramGridLayoutGenerator
except ImportError:
    st.error("⚠️ Enhanced Instagram Grid module not found. Please ensure all dependencies are installed.")
    EnhancedInstagramGridLayoutGenerator = None

try:
    from simple_monitoring import (
        start_performance_monitoring, 
        PerformanceDashboard, 
        performance_monitor,
        track_request,
        IMAGE_PROCESSING_TIME,
        LAYOUT_GENERATION_TIME
    )
    monitoring_available = True
except ImportError:
    monitoring_available = False

try:
    from performance_utils import (
        PerformanceProfiler,
        performance_monitor as perf_decorator,
        StreamlitCache,
        cached_function,
        ImageOptimizer,
        MemoryProfiler,
        StreamlitOptimizer,
        BatchOperationContext
    )
    performance_utils_available = True
except ImportError:
    performance_utils_available = False
    # Create fallback classes
    class PerformanceProfiler:
        def __init__(self, name): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    class BatchOperationContext:
        def __init__(self, name): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    def cached_function(ttl=3600):
        def decorator(func):
            return func
        return decorator
    
    def perf_decorator(func):
        return func


# Initialize performance monitoring on app start
@st.cache_resource
def initialize_monitoring():
    """Initialize performance monitoring system"""
    try:
        if monitoring_available:
            start_performance_monitoring()
        return True
    except Exception as e:
        print(f"Failed to initialize monitoring: {e}")
        return False

# Initialize monitoring
monitoring_initialized = initialize_monitoring()

def initialize_session_state():
    """Initialize all session state variables"""
    if 'cache' not in st.session_state:
        st.session_state.cache = {}

    if 'processing_queue' not in st.session_state:
        st.session_state.processing_queue = queue.Queue()
        st.session_state.processing_results = {}
        st.session_state.processing_status = {}

    if 'request_metrics' not in st.session_state:
        st.session_state.request_metrics = {
            'total_requests': 0,
            'total_response_time': 0,
            'error_count': 0
        }

    if 'user_sessions' not in st.session_state:
        st.session_state.user_sessions = {}

    if 'cache_manager' not in st.session_state:
        if performance_utils_available:
            st.session_state.cache_manager = StreamlitCache()
        else:
            st.session_state.cache_manager = {}

    if 'layouts' not in st.session_state:
        st.session_state.layouts = None
        st.session_state.generator = None
        st.session_state.layouts_ready = False
        st.session_state.processing_steps = {
            "images_loaded": False,
            "colors_extracted": False,
            "features_extracted": False,
            "layouts_generated": False
        }

    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()

    if 'bg_processor' not in st.session_state:
        try:
            from simple_monitoring import MonitoredBackgroundProcessor
            st.session_state.bg_processor = MonitoredBackgroundProcessor()
        except ImportError:
            st.session_state.bg_processor = None
        
    # Initialize app_settings
    if 'app_settings' not in st.session_state:
        st.session_state.app_settings = {
            'num_colors': 3,
            'num_layouts': 5,
            'use_dinov2': True,
            'use_resized_images': True,
            'enable_caching': True,
            'lazy_loading': True
        }

def track_user_session():
    """Track current user session for monitoring"""
    session_id = st.session_state.get('session_id')
    if not session_id:
        session_id = hashlib.md5(f"{time.time()}_{id(st.session_state)}".encode()).hexdigest()
        st.session_state.session_id = session_id
    
    st.session_state.user_sessions[session_id] = {
        'last_activity': time.time(),
        'page_views': st.session_state.user_sessions.get(session_id, {}).get('page_views', 0) + 1
    }

def apply_instagram_styling():
    """CSS for Instagram-like styling with performance optimizations"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Force white background and black text EVERYWHERE */
    *, *::before, *::after {
        color: #000000 !important;
    }
    
    .stApp, .stApp > div, .main, .block-container, 
    .main .block-container, .css-1d391kg, .css-1lcbmhc,
    .sidebar .sidebar-content, [data-testid="stSidebar"],
    [data-testid="stSidebarNav"], .css-17eq0hr {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Force ALL text elements to be black with high specificity */
    h1, h2, h3, h4, h5, h6, p, span, div, label, a, li, ul, ol,
    .stMarkdown, .stMarkdown *, .stText, .stText *, 
    .css-10trblm, .css-16idsys, .css-1cpxqw2, .css-1inwz65,
    .stSelectbox label, .stSlider label, .stCheckbox label,
    .stFileUploader label, .stButton label, .stDownloadButton label,
    .stSubheader, .stSubheader *, .stTitle, .stTitle *,
    .css-1ekf893, .css-1ekf893 *, .css-1v0mbdj, .css-1v0mbdj *,
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
    [data-testid="stText"], [data-testid="stText"] * {
        color: #000000 !important;
    }
    
    /* Main content area - ensure black text */
    .main .block-container * {
        color: #000000 !important;
    }
    
    /* Root variables for consistent theming */
    :root {
        --primary-color: #ff4b4b;
        --secondary-color: #ff6b6b;
        --accent-color: #405DE6;
        --text-primary: #000000;
        --text-secondary: #333333;
        --background-primary: #ffffff;
        --background-secondary: #f8f9fa;
        --background-tertiary: #e9ecef;
        --border-color: #dee2e6;
        --shadow-light: 0 1px 3px rgba(0,0,0,0.1);
        --shadow-medium: 0 4px 12px rgba(0,0,0,0.15);
        --border-radius: 8px;
    }
    
    /* Main app styling */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main container */
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 1200px;
        margin: 0 auto;
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Custom header styling */
    .app-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        padding: 3rem 2rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: var(--shadow-medium);
    }
    
    .app-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
        color: white !important;
    }
    
    .app-header p {
        font-size: 1.1rem;
        font-weight: 400;
        opacity: 0.9;
        margin-bottom: 0.5rem;
        color: white !important;
    }
    
    /* Sidebar styling with clear partition */
    .css-1d391kg, .css-1lcbmhc, .sidebar .sidebar-content,
    [data-testid="stSidebar"], [data-testid="stSidebarNav"] {
        background: #f8f9fa !important;
        color: #000000 !important;
        border-right: 2px solid #dee2e6 !important;
        box-shadow: 2px 0 4px rgba(0,0,0,0.08) !important;
    }
    
    /* Card components */
    .card {
        background: #ffffff !important;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-light);
        border: 1px solid var(--border-color);
    }
    
    .layout-card {
        background: #ffffff !important;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-light);
        border: 1px solid var(--border-color);
        transition: all 0.2s ease;
    }
    
    .layout-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #000000 !important;
        margin-bottom: 0.5rem;
    }
    
    .layout-description {
        font-size: 0.9rem;
        color: #333333 !important;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
        color: white !important;
        border: none;
        border-radius: var(--border-radius);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-light);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-medium);
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #28a745, #20c997) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        width: 100% !important;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #218838, #1fa184) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
    }
    
    /* Enhanced scrollable tabs for all devices */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #ffffff !important;
        padding: 0.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
        overflow-x: auto !important;
        white-space: nowrap !important;
        scrollbar-width: thin;
        scrollbar-color: #dee2e6 #ffffff;
    }
    
    /* Webkit scrollbar styling */
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        height: 6px;
    }
    
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 3px;
    }
    
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
        background: #dee2e6;
        border-radius: 3px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        background: #ffffff !important;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        font-weight: 500;
        color: #000000 !important;
        border: 1px solid #dee2e6 !important;
        transition: all 0.2s ease;
        white-space: nowrap !important;
        flex-shrink: 0 !important;
        margin-right: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f8f9fa !important;
        color: #000000 !important;
        border-color: #adb5bd !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #ff4b4b !important;
        color: white !important;
        border-color: #ff4b4b !important;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: #ffffff !important;
        border: 2px dashed var(--border-color);
        border-radius: var(--border-radius);
        padding: 2rem;
        text-align: center;
        transition: all 0.2s ease;
        color: #000000 !important;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--primary-color);
        background: var(--background-secondary) !important;
    }
    
    .stFileUploader label {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* MOBILE GRID FIXES */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem 0.5rem !important;
            max-width: 100% !important;
        }
        
        .app-header {
            padding: 2rem 1rem !important;
            margin-bottom: 1rem !important;
        }
        
        .app-header h1 {
            font-size: 1.8rem !important;
            line-height: 1.2 !important;
        }
        
        .app-header p {
            font-size: 0.9rem !important;
            line-height: 1.4 !important;
        }
        
        /* Force horizontal layout for columns */
        .row-widget.stHorizontal {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            gap: 2px !important;
            width: 100% !important;
        }
        
        /* Force equal width columns */
        .row-widget.stHorizontal > div {
            flex: 1 1 33.333% !important;
            width: 33.333% !important;
            min-width: 0 !important;
            padding: 1px !important;
            margin: 0 !important;
        }
        
        /* Make images square and fit properly */
        .stImage {
            width: 100% !important;
        }
        
        .stImage > img {
            width: 100% !important;
            height: auto !important;
            aspect-ratio: 1 !important;
            object-fit: cover !important;
            border-radius: 3px !important;
        }
        
        .stImage figcaption {
            font-size: 0.6rem !important;
            text-align: center !important;
            margin-top: 2px !important;
            color: #000000 !important;
        }
    }
    
    /* Desktop styling */
    @media (min-width: 769px) {
        .row-widget.stHorizontal {
            gap: 8px !important;
        }
        
        .row-widget.stHorizontal > div {
            padding: 4px !important;
        }
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .card, .layout-card {
        animation: fadeIn 0.3s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

@cached_function(ttl=CACHE_TTL)
def get_img_as_base64(img, quality=JPEG_QUALITY):
    """Convert PIL Image to base64 with caching and optimization"""
    try:
        # Copy to avoid modifying the original
        img_copy = img.copy()
        
        # Optimize image before conversion
        if performance_utils_available:
            optimized_img = ImageOptimizer.smart_resize(img_copy, (800, 800), maintain_aspect=True)
            img_bytes = ImageOptimizer.progressive_jpeg(optimized_img, quality=quality)
            img_str = base64.b64encode(img_bytes).decode()
        else:
            # Fallback optimization
            if img_copy.width > 800 or img_copy.height > 800:
                img_copy.thumbnail((800, 800), Image.Resampling.LANCZOS)
            buffered = BytesIO()
            img_copy.save(buffered, format="JPEG", quality=quality)
            img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        # Simple fallback
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=quality)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

def render_optimized_grid(images, layout_indices, lazy_load=True):
    """Render Instagram-like grid with monitoring and optimizations"""
    with PerformanceProfiler("render_grid"):
        grid_html = '<div class="instagram-grid">'
        
        for i, idx in enumerate(layout_indices[:9]):
            if idx < len(images):
                img = images[idx]
                
                if lazy_load and i > 3:  # Lazy load images after the first row
                    grid_html += f'''
                    <div class="image-placeholder instagram-post" 
                         data-src="placeholder_{idx}" 
                         data-index="{idx}">
                    </div>
                    '''
                else:
                    # Convert to optimized format
                    img_b64 = get_img_as_base64(img)
                    
                    grid_html += f'''
                    <img class="instagram-post optimized-image" 
                         src="data:image/jpeg;base64,{img_b64}" 
                         loading="lazy"
                         decoding="async" />
                    '''
        
        grid_html += '</div>'
        
        return grid_html

def progress_indicator(steps, current_step):
    """Enhanced progress indicator with monitoring"""
    progress_percent = (current_step / len(steps)) * 100 if len(steps) > 0 else 0
    
    # Use native Streamlit components
    st.progress(progress_percent / 100)
    
    # Display current step
    if current_step < len(steps):
        current_task = steps[current_step]
        st.info(f"Current step: {current_task} ({progress_percent:.0f}% complete)")
    
    # List all steps with status indicators
    for i, step in enumerate(steps):
        if i < current_step:
            st.success(f"✅ {step} - Completed")
        elif i == current_step:
            st.info(f"⏳ {step} - In Progress")
        else:
            st.text(f"○ {step} - Pending")
    
    return ""

@perf_decorator
def cleanup_memory():
    """Force garbage collection and memory cleanup with monitoring"""
    try:
        # Clear matplotlib figures
        plt.close('all')
        
        # Force garbage collection
        gc.collect()
        
        # Clear large objects from session state if too much memory used
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        if memory_usage > 1000:  # More than 1GB
            if performance_utils_available:
                cleared_count = StreamlitOptimizer.clear_large_objects(threshold_mb=100)
                if cleared_count > 0:
                    st.warning(f"Cleared {cleared_count} large objects due to high memory usage ({memory_usage:.1f}MB)")
        
        return memory_usage
    except Exception as e:
        print(f"Error during memory cleanup: {e}")
        return 0

def track_page_load():
    """Track page load performance"""
    if 'page_load_start' not in st.session_state:
        st.session_state.page_load_start = time.time()
    
    # Track page load time
    load_time = time.time() - st.session_state.page_load_start
    
    # Update request metrics
    if 'request_metrics' in st.session_state:
        st.session_state.request_metrics['total_requests'] += 1
        st.session_state.request_metrics['total_response_time'] += load_time

def preserve_layout_state():
    """Ensure layout state is preserved across interactions"""
    if ('layouts' in st.session_state and st.session_state.layouts and
        'generator' in st.session_state and st.session_state.generator):
        st.session_state.layouts_ready = True
        return True
    return False

# Main function with comprehensive monitoring integration
@track_request("main_app")
def main_content():
    """Main application content"""
    
    # Track user session
    track_user_session()
    
    # Apply custom styling
    apply_instagram_styling()
    
    # Preserve layout state
    preserve_layout_state()
    
    # Simple, clean app header
    st.markdown("""
    <div class="app-header">
        <h1>📷 Instagram Grid Layout Generator</h1>
        <p>Create aesthetically pleasing Instagram grid layouts based on image content and colors.</p>
        <p>Upload your images and get AI-powered layout suggestions that will make your profile stand out!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for controls
    render_sidebar()
    
    # Main content area
    render_main_content()

def render_sidebar():
    """Render the sidebar with settings"""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        try:
            import transformers
            transformers_available = True
        except ImportError:
            transformers_available = False
            
        if not transformers_available:
            st.warning("""
            ⚠️ **DINOv2 Not Available**
            
            For better image analysis, install the required packages:
            ```
            pip install transformers torch torchvision
            ```
            
            Without these packages, the app will use color-based analysis only.
            """)
        
        # Application settings
        render_app_settings()
        
        # System info
        render_system_info()

def render_app_settings():
    """Render application settings section"""
    # Color analysis settings
    st.subheader("🎨 Color Analysis")
    num_colors = st.slider("Number of dominant colors per image", 1, 5, 3)
    
    # Layout generation settings
    st.subheader("📐 Layout Generation")
    num_layouts = st.slider("Number of layout options", 3, 10, 5)
    
    # Performance settings - only show DINOv2 option
    st.subheader("⚡ Performance Settings")

    # Check if transformers is available
    try:
        import transformers
        transformers_available = True
    except ImportError:
        transformers_available = False

    use_dinov2 = st.checkbox("Use DINOv2 for image analysis", 
                            value=True if transformers_available else False,
                            disabled=not transformers_available, 
                            help="Better results but requires more processing time. Install with 'pip install transformers'")
    
    # Set default values for other performance settings (not displayed)
    use_resized_images = True  # Default: enabled
    enable_caching = True      # Default: enabled
    lazy_loading = True        # Default: enabled
    
    # Store settings in session state
    st.session_state.app_settings = {
        'num_colors': num_colors,
        'num_layouts': num_layouts,
        'use_dinov2': use_dinov2,
        'use_resized_images': use_resized_images,
        'enable_caching': enable_caching,
        'lazy_loading': lazy_loading
    }
    
    # Add app info
    st.markdown("---")
    st.info("💡 **Tip:** For best results, upload at least 9 images of similar dimensions.")

def render_system_info():
    """Render system information section"""
    with st.expander("ℹ️ System Info"):
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        st.text(f"System Memory: {memory.percent:.1f}% used")
        st.text(f"Available: {memory.available / (1024**3):.1f}GB")
        st.text(f"Total: {memory.total / (1024**3):.1f}GB")
        st.text(f"Disk Free: {disk.free / (1024**3):.1f}GB")

def render_main_content():
    """Render the main content area with PERSISTENT layout display"""
    # File upload section
    st.subheader("📁 Upload Your Images")
    uploaded_files = st.file_uploader(
        "Choose your images (JPG, JPEG, PNG)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        help="Upload at least 9 images for a complete Instagram grid"
    )
    
    # Check if layouts are already generated and should be displayed
    if (hasattr(st.session_state, 'layouts_ready') and st.session_state.layouts_ready and
        'layouts' in st.session_state and st.session_state.layouts and
        'generator' in st.session_state and st.session_state.generator):
        
        render_generated_layouts()
    
    # Process uploaded files only if no layouts are ready
    if uploaded_files and (not hasattr(st.session_state, 'layouts_ready') or not st.session_state.layouts_ready):
        process_uploaded_files(uploaded_files)

def process_uploaded_files(uploaded_files):
    """Process uploaded files with monitoring"""
    if not EnhancedInstagramGridLayoutGenerator:
        st.error("❌ Enhanced Instagram Grid module not available. Please check your installation.")
        return
        
    settings = st.session_state.app_settings
    
    with PerformanceProfiler("file_upload_processing"):
        # Generate cache key
        upload_filenames = [f.name for f in uploaded_files]
        upload_key = "_".join(sorted(upload_filenames))
        settings_key = f"{settings['num_colors']}_{settings['num_layouts']}_{settings['use_dinov2']}_{settings['use_resized_images']}"
        cache_key = f"{upload_key}_{settings_key}"
        
        # Check cache first (IMPORTANT for avoiding regeneration)
        if settings['enable_caching'] and cache_key in st.session_state.cache:
            cached_data = st.session_state.cache[cache_key]
            if cached_data.get("processed") and cached_data.get("layouts"):
                st.success("✅ Loading cached results...")
                st.session_state.layouts = cached_data["layouts"]
                st.session_state.generator = cached_data["generator"]
                st.session_state.layouts_ready = True
                st.rerun()
                return
        
        # Handle file saving and caching
        image_paths = handle_file_caching(uploaded_files, cache_key, settings['enable_caching'])
        
        # If cache hit, return early
        if not image_paths:
            return
        
        # Validate image count
        if not validate_image_count(len(image_paths)):
            return

        # Show preview
        show_image_preview_fixed(image_paths, settings)
        
        # Generate layouts button
        st.markdown("---")
        if st.button("🚀 Generate Layout Options", type="primary", help="This will analyze your images and create optimized layout suggestions"):
            generate_layouts(image_paths, cache_key, settings)

def handle_file_caching(uploaded_files, cache_key, enable_caching):
    """Handle file saving and caching"""
    # Check cache first
    if enable_caching and cache_key in st.session_state.cache:
        if st.session_state.cache[cache_key].get("processed"):
            st.success("✅ Loading cached results...")
            st.session_state.layouts = st.session_state.cache[cache_key]["layouts"]
            st.session_state.generator = st.session_state.cache[cache_key]["generator"]
            st.session_state.layouts_ready = True
            st.rerun()
            return []
        else:
            st.info("🔄 Found cached files, processing...")
            return st.session_state.cache[cache_key]["image_paths"]
    
    # Save files
    image_paths = []
    with st.spinner("💾 Saving uploaded files..."):
        for uploaded_file in uploaded_files:
            file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_paths.append(file_path)
    
    # Store in cache
    if enable_caching:
        st.session_state.cache[cache_key] = {
            "image_paths": image_paths,
            "processed": False
        }
    
    st.success("✅ Files saved successfully!")
    return image_paths

def validate_image_count(count):
    """Validate the number of uploaded images"""
    if count < 1:
        st.error("❌ No valid images found. Please upload at least one image.")
        return False
    elif count < 9:
        st.warning(f"⚠️ You've uploaded {count} images. For a complete Instagram grid, 9 images are recommended.")
    else:
        st.success(f"✅ {count} images uploaded successfully!")
    return True

def show_image_preview_fixed(image_paths, settings):
    """Show preview using HTML grid instead of Streamlit columns"""
    with st.expander("👀 Preview Uploaded Images", expanded=True):
        
        # Load and preprocess images for preview
        with PerformanceProfiler("image_preview_generation"):
            preview_images = []
            failed_images = []
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            for i, path in enumerate(image_paths[:9]):
                try:
                    progress_text.text(f"Loading image {i+1} of {min(9, len(image_paths))}...")
                    progress_bar.progress((i + 1) / min(9, len(image_paths)))
                    
                    img = safe_image_load(path)
                    if img is not None:
                        if settings.get('use_resized_images', True):
                            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                        preview_images.append(img)
                    else:
                        failed_images.append(os.path.basename(path))
                except Exception as e:
                    st.error(f"❌ Error loading {os.path.basename(path)}: {str(e)}")
                    failed_images.append(os.path.basename(path))
            
            progress_bar.empty()
            progress_text.empty()
            
            if failed_images:
                st.warning(f"⚠️ Failed to load {len(failed_images)} images: {', '.join(failed_images)}")
        
        if preview_images:
            # Use HTML grid instead of Streamlit columns
            preview_indices = list(range(len(preview_images)))
            render_html_grid(preview_images, preview_indices, "📸 Image Grid Preview")
            
            # Show statistics
            show_image_statistics_fixed(preview_images)

def safe_image_load(file_path):
    """Safely load an image with comprehensive error handling and optimization"""
    try:
        # Open and verify the image
        img = Image.open(file_path)
        img.verify()  # Verify it's a valid image
        
        # Reopen after verify (verify closes the file)
        img = Image.open(file_path)
        
        # Convert to RGB if necessary (handles RGBA, P, etc.)
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        
        # Check for extremely large images and resize if needed
        max_dimension = 2048  # Maximum dimension to allow
        if img.width > max_dimension or img.height > max_dimension:
            # Calculate new size preserving aspect ratio
            if img.width > img.height:
                new_width = max_dimension
                new_height = int(img.height * (max_dimension / img.width))
            else:
                new_height = max_dimension
                new_width = int(img.width * (max_dimension / img.height))
            
            # Resize using high-quality resampling
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Log that we resized the image
            print(f"Resized image {os.path.basename(file_path)} from original size to {new_width}x{new_height}")
        
        return img
    except Exception as e:
        st.error(f"Failed to load image {os.path.basename(file_path)}: {str(e)}")
        return None

def crop_to_square(img):
    """Crop image to square from center"""
    width, height = img.size
    
    # Already square
    if width == height:
        return img.copy()
    
    # Calculate dimensions for square crop
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    
    # Return cropped copy
    return img.crop((left, top, right, bottom))

def optimize_image_for_web(img, max_size=(600, 600), quality=85):
    """Optimize image for web display"""
    # Create a copy to avoid modifying original
    img_copy = img.copy()
    
    # Convert to RGB if needed
    if img_copy.mode != 'RGB':
        img_copy = img_copy.convert('RGB')
    
    # Resize if larger than max_size
    if img_copy.width > max_size[0] or img_copy.height > max_size[1]:
        img_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    return img_copy

def show_image_statistics_fixed(images):
    """Display image statistics in a clean format"""
    st.markdown("### 📊 Upload Statistics")
    
    if not images:
        st.warning("No images to analyze")
        return
    
    # Calculate comprehensive statistics
    total_images = len(images)
    dimensions = [img.size for img in images]
    total_pixels = sum(w * h for w, h in dimensions)
    avg_resolution = total_pixels / total_images
    
    # Analyze image formats and modes
    formats = []
    modes = []
    for img in images:
        # Get format (if available)
        if hasattr(img, 'format') and img.format:
            formats.append(img.format)
        else:
            # Infer format from mode
            if img.mode == 'RGB':
                formats.append('JPEG')
            elif img.mode in ['RGBA', 'P']:
                formats.append('PNG')
            else:
                formats.append('Unknown')
        
        modes.append(img.mode)
    
    # Count occurrences
    format_counts = {}
    for fmt in formats:
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", total_images)
    
    with col2:
        st.metric("Avg Resolution", f"{avg_resolution/1000000:.1f}MP")
    
    with col3:
        most_common_format = max(format_counts.items(), key=lambda x: x[1])[0] if format_counts else "Unknown"
        st.metric("Primary Format", most_common_format)
    
    with col4:
        # Calculate estimated file size (rough approximation)
        avg_size_mb = (avg_resolution * 3) / (1024 * 1024 * 10)
        st.metric("Est. Avg Size", f"{avg_size_mb:.1f}MB")

def generate_layouts(image_paths, cache_key, settings):
    """Generate layout options with comprehensive monitoring"""
    with PerformanceProfiler("layout_generation_full_process"):
        # Initialize generator
        generator = EnhancedInstagramGridLayoutGenerator(st.session_state.temp_dir)
        st.session_state.generator = generator
        
        # Set up processing steps
        processing_steps = [
            "Loading and preparing images",
            "Extracting color information", 
            "Analyzing image content" if settings['use_dinov2'] else "Analyzing image features",
            "Generating layout options"
        ]
        
        # Display progress
        progress_container = st.empty()
        status_container = st.empty()
        
        # Use batch operation context for optimization
        with BatchOperationContext("image_processing"):
            try:
                # Step 1: Load images
                status_container.info("⏳ Step 1/4: Loading and preparing images...")
                
                with PerformanceProfiler("load_images_step"):
                    load_success = generator.load_images()
                    if not load_success:
                        status_container.error("❌ Failed to load images!")
                        return False
                    
                    if len(generator.images) == 0:
                        status_container.error("❌ No images were loaded!")
                        return False
                        
                    status_container.success(f"✅ Successfully loaded {len(generator.images)} images")
                
                # Step 2: Extract colors
                status_container.info("🎨 Step 2/4: Extracting color information from images...")
                
                with PerformanceProfiler("extract_colors_step"):
                    generator.extract_dominant_colors(num_colors=settings['num_colors'])
                    status_container.success("✅ Color extraction complete")
                
                # Step 3: Extract features
                if settings['use_dinov2']:
                    status_container.info("🧠 Step 3/4: Analyzing image content...")
                else:
                    status_container.info("🔍 Step 3/4: Analyzing image features...")
                
                # Feature extraction
                generator._extract_image_features()
                
                # Step 4: Generate layouts
                status_container.info("🎯 Step 4/4: Generating layout options...")
                
                with PerformanceProfiler("generate_layouts_step"):
                    layouts = generator.generate_layouts(num_layouts=settings['num_layouts'])
                    st.session_state.layouts = layouts
                    status_container.success(f"✅ Successfully generated {len(layouts)} layout options!")
                
                # Store objects in session state for later use
                st.session_state.generator = generator
                st.session_state.layouts = layouts
                st.session_state.layouts_ready = True
                
                # Clear the progress containers
                progress_container.empty()
                status_container.empty()
                
                # Cache results if needed
                if settings['enable_caching']:
                    st.session_state.cache[cache_key] = {
                        "image_paths": image_paths,
                        "processed": True,
                        "layouts": layouts,
                        "generator": copy.deepcopy(generator)
                    }
                
                # Success message
                st.success("🎉 Layout generation complete! Scroll down to see your options.")
                
                # Force a rerun to display layouts immediately
                st.rerun()
                
                return True
                
            except Exception as e:
                import traceback
                status_container.error(f"❌ Error during processing: {str(e)}")
                st.error(traceback.format_exc())
                return False

def render_generated_layouts():
    """Render the generated layouts with PERSISTENT state"""
    if not ('layouts' in st.session_state and st.session_state.layouts and 
            'generator' in st.session_state and st.session_state.generator):
        return
    
    # Set a flag to track that layouts are being displayed
    st.session_state.layouts_displayed = True
    
    with PerformanceProfiler("layout_display"):
        st.subheader("🎨 Layout Options")
        st.markdown("---")
        
        # Add helpful info
        st.info("💡 **Tip**: Click any download button to save the layout. The page won't refresh!")
        
        # Create tab names (shortened for better display)
        tab_names = []
        for i, layout in enumerate(st.session_state.layouts):
            # Shorten the layout name for tabs
            short_name = layout['name']
            if len(short_name) > 20:
                short_name = short_name[:17] + "..."
            tab_names.append(f"Layout {i+1}: {short_name}")
        
        # Add scroll hint if there are many tabs
        if len(st.session_state.layouts) > 5:
            st.info("💡 **Tip:** You can scroll horizontally through the layout tabs above to see all options.")
        
        # Create tabs for different layouts
        tabs = st.tabs(tab_names)
        
        # Render each layout
        for i, tab in enumerate(tabs):
            with tab:
                render_single_layout(i, st.session_state.layouts[i])
        
        # Additional sections
        render_color_analysis()
        render_similarity_analysis()

def render_html_grid(images, layout_indices, grid_title="Grid Preview", max_images=9):
    """Render 3x3 grid with minimal white space below"""
    
    # Build the complete HTML document
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                margin: 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: #ffffff;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
                align-items: center;
                overflow: hidden;
            }}
            .grid-title {{
                color: #000000;
                font-weight: 600;
                text-align: center;
            }}
            .grid-container {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                background: #ffffff;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                border: 1px solid #dee2e6;
            }}
            .grid-item {{
                aspect-ratio: 1;
                overflow: hidden;
                border-radius: 4px;
                position: relative;
                background: #f8f9fa;
                border: 1px solid #dee2e6;
            }}
            .grid-item img {{
                width: 100%;
                height: 100%;
                object-fit: cover;
                display: block;
            }}
            .position-label {{
                position: absolute;
                bottom: 4px;
                right: 4px;
                background: rgba(0,0,0,0.7);
                color: white;
                border-radius: 3px;
                font-weight: 500;
            }}
            .empty-cell {{
                aspect-ratio: 1;
                background: #f8f9fa;
                border: 2px dashed #dee2e6;
                border-radius: 4px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #6c757d;
            }}
            
            /* Mobile: Minimal white space */
            @media (max-width: 768px) {{
                body {{
                    padding: 8px;
                }}
                .grid-container {{
                    width: 320px;
                    gap: 2px;
                    padding: 6px;
                }}
                .position-label {{
                    font-size: 0.6rem;
                    padding: 1px 4px;
                }}
                .grid-title {{
                    font-size: 1rem;
                    margin-bottom: 8px;
                }}
                .empty-cell {{
                    font-size: 0.7rem;
                }}
            }}
            
            /* Desktop: Larger size with minimal bottom space */
            @media (min-width: 769px) {{
                body {{
                    padding: 15px;
                }}
                .grid-container {{
                    width: 1000px;
                    gap: 15px;
                    padding: 25px;
                }}
                .position-label {{
                    font-size: 1.3rem;
                    padding: 8px 12px;
                    bottom: 12px;
                    right: 12px;
                }}
                .grid-title {{
                    font-size: 1.5rem;
                    margin-bottom: 15px;
                }}
                .empty-cell {{
                    font-size: 1.2rem;
                }}
                .grid-item {{
                    border-radius: 10px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="grid-title">{grid_title}</div>
        <div class="grid-container">
    """
    
    # Add each image to the grid
    for i in range(9):
        if i < len(layout_indices) and layout_indices[i] < len(images):
            img = images[layout_indices[i]]
            img_square = crop_to_square(img)
            buffered = BytesIO()
            img_square.save(buffered, format="JPEG", quality=90)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            html_content += f"""
            <div class="grid-item">
                <img src="data:image/jpeg;base64,{img_str}" alt="Grid image {i+1}" />
                <div class="position-label">{i+1}</div>
            </div>
            """
        else:
            html_content += '<div class="empty-cell">Empty</div>'
    
    # Close the HTML
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Use precise heights to minimize white space
    import streamlit.components.v1 as components
    components.html(html_content, height=400, scrolling=False)

def render_single_layout(index, layout):
    """Render layout using HTML grid for both mobile and desktop"""
    with PerformanceProfiler(f"render_layout_{index}"):
        # Layout info
        st.markdown(f"""
        <div class="layout-card">
            <div class="layout-title">{layout['name']}</div>
            <div class="layout-description">{layout['description']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Use the same HTML grid approach as the preview
        layout_indices = layout['layout']
        render_html_grid(
            st.session_state.generator.images, 
            layout_indices, 
            "📱 Instagram Grid Preview"
        )
        
        # Download button
        create_download_button(index, layout)

@cached_function(ttl=CACHE_TTL)
def create_layout_visualization(layout_indices, images, layout_name):
    """Create layout visualization with caching"""
    with PerformanceProfiler(f"create_visualization_{layout_name}"):
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        
        # Disable axes
        for ax in axes.flatten():
            ax.axis('off')
        
        # Place images according to layout
        for j, idx in enumerate(layout_indices):
            if j >= 9:  # Only show first 9 images
                break
                
            row, col = j // 3, j % 3
            if idx < len(images):
                axes[row, col].imshow(np.array(images[idx]))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save to BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf

def create_download_button(index, layout):
    """Create download button for layout WITHOUT causing page refresh"""
    try:
        # Create the layout visualization in memory
        layout_viz = create_layout_visualization(
            tuple(layout['layout']), 
            tuple(st.session_state.generator.images), 
            layout['name']
        )
        
        # Create filename
        filename = f"instagram_layout_{index+1}_{layout['name'].replace(' ', '_').replace('/', '_')}.png"
        
        # Use a unique key that includes hash to avoid conflicts
        unique_key = f"download_layout_{index}_{hash(str(layout['layout']))}"
        
        # Create the download button with proper configuration
        st.download_button(
            label=f"⬇️ Download Layout {index+1}",
            data=layout_viz.getvalue(),
            file_name=filename,
            mime="image/png",
            key=unique_key,
            help=f"Download {layout['name']} as PNG image"
        )
        
    except Exception as e:
        st.error(f"Error creating download button for layout {index+1}: {str(e)}")
        st.info(f"Layout {index+1} ready for download (refresh to try again)")

def render_color_analysis():
    """Render color analysis section"""
    st.subheader("🎨 Color Analysis")
    with st.expander("Show Color Analysis", expanded=False):
        with PerformanceProfiler("color_analysis_display"):
            color_profiles = st.session_state.generator.color_profiles
            
            st.markdown("### 🌈 Image Color Profiles")
            st.markdown("Each image's dominant colors are shown below. These colors influence the layout suggestions.")
            
            # Create color grid using native Streamlit
            render_color_analysis_native(st.session_state.generator.images[:9], color_profiles[:9])

def render_color_analysis_native(images, color_profiles):
    """Render color analysis with accurate color display"""
    # Display in 3-column grid
    for row in range((len(images) + 2) // 3):
        cols = st.columns(3)
        for col_idx in range(3):
            img_idx = row * 3 + col_idx
            if img_idx < len(images) and img_idx < len(color_profiles):
                with cols[col_idx]:
                    # Display image
                    st.image(images[img_idx], caption=f"Image {img_idx + 1}", use_container_width=True)
                    
                    # Display colors with Streamlit native approach
                    hex_colors = color_profiles[img_idx]['hex']
                    rgb_colors = color_profiles[img_idx]['rgb']
                    
                    st.write("**Dominant Colors:**")
                    
                    # Use Streamlit columns for color display instead of HTML
                    color_cols = st.columns(len(hex_colors))
                    
                    for i, (hex_color, rgb_color) in enumerate(zip(hex_colors, rgb_colors)):
                        with color_cols[i]:
                            # Create a simple colored square using CSS in markdown
                            r, g, b = rgb_color
                            
                            # Use a simpler approach with minimal HTML
                            color_html = f"""
                            <div style="text-align: center;">
                                <div style="
                                    width: 40px;
                                    height: 40px;
                                    background-color: rgb({int(r)}, {int(g)}, {int(b)});
                                    border-radius: 4px;
                                    border: 1px solid #ccc;
                                    margin: 0 auto 8px auto;
                                "></div>
                                <small style="font-size: 10px; color: #000;">{hex_color}</small>
                            </div>
                            """
                            
                            st.markdown(color_html, unsafe_allow_html=True)

def render_similarity_analysis():
    """Render similarity analysis section"""
    if not (hasattr(st.session_state.generator, 'image_features') and st.session_state.generator.image_features):
        return
    
    st.subheader("🔗 Content Similarity Analysis")
    with st.expander("Show Content Similarity Analysis", expanded=False):
        with PerformanceProfiler("similarity_analysis_display"):
            st.markdown("""
            This visualization shows how similar images are based on their content, 
            as analyzed by DINOv2. Connected images have similar visual characteristics.
            """)
            
            # Create similarity visualization
            num_images = min(len(st.session_state.generator.images), 9)
            similarity_fig = create_similarity_visualization(
                st.session_state.generator.image_features[:num_images], 
                num_images
            )
            st.pyplot(similarity_fig)

@cached_function(ttl=CACHE_TTL)
def create_similarity_visualization(image_features, num_images):
    """Create similarity visualization with caching"""
    with PerformanceProfiler("create_similarity_heatmap"):
        import matplotlib.pyplot as plt
        
        # Calculate similarity scores
        similarity_matrix = np.zeros((num_images, num_images))
        for i in range(num_images):
            for j in range(num_images):
                if i != j:
                    # Calculate cosine similarity
                    feat1 = np.array(image_features[i])
                    feat2 = np.array(image_features[j])
                    
                    norm1 = np.linalg.norm(feat1)
                    norm2 = np.linalg.norm(feat2)
                    
                    if norm1 > 0 and norm2 > 0:
                        similarity_matrix[i][j] = np.dot(feat1, feat2) / (norm1 * norm2)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(similarity_matrix, cmap='viridis')
        
        # Add labels
        ax.set_xticks(np.arange(num_images))
        ax.set_yticks(np.arange(num_images))
        ax.set_xticklabels([f"Image {i+1}" for i in range(num_images)])
        ax.set_yticklabels([f"Image {i+1}" for i in range(num_images)])
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Similarity Score", rotation=-90, va="bottom")
        
        ax.set_title("Content Similarity Between Images")
        fig.tight_layout()
        
        return fig

def clear_session_and_restart():
    """Clear session data and restart"""
    with PerformanceProfiler("clear_session_data"):
        # Reset layout-related session state only
        keys_to_clear = ['layouts', 'generator', 'layouts_ready', 'layouts_displayed', 'processing_layouts']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Keep cache and other settings intact
        # Force garbage collection
        cleanup_memory()
        
        st.success("✅ Session cleared! You can now upload new images.")
        st.rerun()

def handle_upload_errors():
    """Handle various upload error scenarios"""
    try:
        # Check for common issues
        if 'temp_dir' not in st.session_state:
            st.error("❌ Temporary directory not initialized. Please refresh the page.")
            return False
        
        if not os.path.exists(st.session_state.temp_dir):
            os.makedirs(st.session_state.temp_dir)
        
        return True
    except Exception as e:
        st.error(f"❌ Upload error: {str(e)}")
        return False

def validate_app_state():
    """Validate that the app is in a consistent state"""
    required_keys = ['app_settings', 'temp_dir', 'cache']
    for key in required_keys:
        if key not in st.session_state:
            st.warning(f"⚠️ Missing session state: {key}. Reinitializing...")
            initialize_session_state()
            break

def show_app_info():
    """Show helpful information about the app"""
    with st.expander("ℹ️ How to Use This App", expanded=False):
        st.markdown("""
        ### 📱 Instagram Grid Layout Generator
        
        **How it works:**
        1. **Upload Images**: Select 9+ images for best results
        2. **AI Analysis**: The app analyzes colors and content using DINOv2
        3. **Layout Generation**: Get multiple layout options optimized for visual harmony
        4. **Download**: Save your preferred layout as a high-quality image
        
        **Tips for Best Results:**
        - Use high-quality images (minimum 1080x1080px recommended)
        - Upload at least 9 images for a complete grid
        - Similar lighting and style work best together
        - Enable DINOv2 analysis for content-aware layouts
        
        **Supported Formats:**
        - JPEG (.jpg, .jpeg)
        - PNG (.png)
        - Maximum file size: 10MB per image
        
        **Performance Notes:**
        - First run may take longer due to AI model loading
        - Results are cached for faster subsequent processing
        - Use the compact grid view for quick previews
        """)

def cleanup_on_exit():
    """Cleanup resources when app exits"""
    try:
        if 'bg_processor' in st.session_state and st.session_state.bg_processor:
            st.session_state.bg_processor.cleanup()
        
        # Stop performance monitoring
        if monitoring_available:
            performance_monitor.stop_monitoring()
        
        # Clean up temporary files
        if 'temp_dir' in st.session_state and os.path.exists(st.session_state.temp_dir):
            import shutil
            shutil.rmtree(st.session_state.temp_dir)
    except Exception as e:
        print(f"Error during cleanup: {e}")

def setup_monitoring_endpoints():
    """Setup monitoring endpoints for external tools"""

    # Analytics endpoint (separate from main app)
    if st.query_params.get("analytics") == "admin":
        try:
            from unified_analytics import integrate_secure_analytics
            integrate_secure_analytics()
            st.stop()
        except ImportError:
            st.error("Analytics module not available")
            st.stop()
    
    # Health check endpoint
    if st.query_params.get("health"):
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "monitoring_active": monitoring_available
        }
        st.json(health_status)
        st.stop()

def main_with_error_handling():
    """Main function with comprehensive error handling"""
    try:
        # Validate app state
        validate_app_state()
        
        # Check for upload errors
        if not handle_upload_errors():
            return
        
        # Show app info
        show_app_info()
        
        # Run main content
        main_content()
        
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        st.markdown("Please try refreshing the page. If the problem persists, please report this issue.")
        
        # Log the error
        if hasattr(st.session_state, 'performance_logs'):
            st.session_state.performance_logs.append({
                'operation': 'main_app_error',
                'duration': 0,
                'memory_delta': 0,
                'peak_memory': 0,
                'timestamp': time.time(),
                'error': str(e)
            })

def debug_session_state():
    """Debug function to print session state variables"""
    # Only in debug mode
    if st.query_params.get("debug") == "true":
        st.write("Session State Debug:")
        st.write(f"layouts_ready: {st.session_state.get('layouts_ready', False)}")
        st.write(f"Has layouts: {st.session_state.get('layouts') is not None}")
        if st.session_state.get('layouts') is not None:
            st.write(f"Number of layouts: {len(st.session_state.get('layouts'))}")
        st.write(f"Has generator: {st.session_state.get('generator') is not None}")

# Main function (for backward compatibility)
def main():
    """Main entry point"""
    main_content()

# Register cleanup function
atexit.register(cleanup_on_exit)

# Setup monitoring endpoints
setup_monitoring_endpoints()

# Track this page load
track_page_load()

# Entry point
if __name__ == "__main__":
    # Initialize session state
    initialize_session_state()
    
    # Use the error-handling wrapper
    main_with_error_handling()
    
    # Debug if requested
    debug_session_state()