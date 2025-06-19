# 📷 Instagram Grid Layout Generator

> An AI-powered web application that creates aesthetically pleasing Instagram grid layouts using advanced computer vision and color analysis.

## 🌟 Features

- **AI-Powered Analysis**: Uses DINOv2 (Vision Transformer) for intelligent image content analysis
- **Smart Color Harmony**: Advanced color extraction and harmony analysis for visually cohesive layouts
- **Multiple Layout Options**: Generates 5+ different layout strategies based on image content and aesthetics
- **Real-Time Processing**: Optimized performance with caching and background processing
- **Mobile-Responsive Design**: Instagram-style interface that works on all devices
- **High-Quality Export**: Download layouts as high-resolution PNG files
- **Performance Analytics**: Built-in monitoring and performance tracking

## 🚀 Live Demo

**[Try the app here →](https://your-app-name.streamlit.app)**

## 📸 Screenshots

### Main Interface
*Upload your images and get instant layout suggestions*

### Generated Layouts
*Multiple AI-generated layout options with visual flow optimization*

### Color Analysis
*Detailed color harmony analysis for each image*

## 🛠️ Technology Stack

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web application framework |
| **DINOv2** | Vision transformer for image analysis |
| **PyTorch** | Deep learning backend |
| **Pillow (PIL)** | Image processing and manipulation |
| **scikit-learn** | Clustering and ML algorithms |
| **matplotlib** | Visualization and chart generation |
| **Redis** | Caching and performance optimization |

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │    │  Image Processor │    │  AI Analyzer    │
│   (Streamlit)   │◄──►│   (PIL/OpenCV)   │◄──►│   (DINOv2)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Layout Generator│    │  Color Analyzer  │    │ Performance     │
│  (Custom Logic) │    │  (K-Means/HSV)   │    │ Monitor         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Option 1: Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/instagram-grid-generator.git
   cd instagram-grid-generator
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run main.py
   ```

5. **Open your browser**
   ```
   http://localhost:8501
   ```

### Option 2: Docker

1. **Build and run with Docker**
   ```bash
   docker build -t instagram-grid-generator .
   docker run -p 8501:8501 instagram-grid-generator
   ```

### Option 3: One-Click Deploy

[![Deploy to Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/yourusername/instagram-grid-generator/main/main.py)

## 📋 Usage Guide

### Step 1: Upload Images
- Select 9+ images (JPG, PNG formats)
- Optimal size: 1080x1080px or larger
- For best results, use images with similar lighting and style

### Step 2: Configure Settings
- **Number of colors**: Adjust dominant color analysis (1-5)
- **Layout options**: Choose how many layout variations to generate (3-10)
- **AI Analysis**: Enable/disable DINOv2 content analysis
- **Performance**: Toggle caching and optimization features

### Step 3: Generate Layouts
- Click "Generate Layout Options"
- Wait for AI analysis to complete
- Browse through generated layout tabs

### Step 4: Download
- Preview each layout option
- Click download button for your preferred layout
- Get high-resolution PNG file ready for Instagram

## 🎯 Layout Algorithms

### 1. Visual Similarity Flow
Groups images with similar visual content using DINOv2 embeddings and cosine similarity.

### 2. Content Grouping
Clusters images by content type using K-means clustering on image features.

### 3. Color Harmony
Arranges images based on color wheel relationships and HSV color space analysis.

### 4. Brightness Flow
Creates smooth transitions from darker to brighter images for visual continuity.

### 5. Creative Mix
Randomized arrangements with aesthetic scoring for unexpected yet pleasing layouts.

## ⚙️ Configuration

### Environment Variables
```env
# Optional Redis caching
REDIS_URL=redis://localhost:6379

# Analytics (for production)
ADMIN_KEY=your_secure_admin_key
ANALYTICS_PASSWORD=your_analytics_password

# Performance tuning
MAX_UPLOAD_SIZE=200
CACHE_TTL=3600
```

### Streamlit Configuration
Create `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 200
enableXsrfProtection = false

[theme]
primaryColor = "#ff4b4b"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

## 🔧 Advanced Features

### Performance Monitoring
- Real-time memory usage tracking
- Operation timing and bottleneck analysis
- Cache hit/miss ratios
- Session analytics dashboard

### Caching System
- **Image caching**: Processed images stored in memory/Redis
- **Feature caching**: AI embeddings cached for repeated use
- **Layout caching**: Generated layouts saved for instant loading

### Batch Processing
- Optimized for multiple image processing
- Memory-efficient image handling
- Background task processing

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Image Processing** | ~2-4 seconds per image |
| **Layout Generation** | ~1-3 seconds for 5 layouts |
| **Memory Usage** | <500MB for 9 images |
| **Supported Image Size** | Up to 4K resolution |
| **Concurrent Users** | 10+ (depends on hosting) |

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit changes** (`git commit -m 'Add amazing feature'`)
4. **Push to branch** (`git push origin feature/amazing-feature`)
5. **Open Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black .
isort .

# Linting
flake8 .
```

## 📝 API Reference

### Core Classes

#### `EnhancedInstagramGridLayoutGenerator`
Main class for generating Instagram grid layouts.

```python
generator = EnhancedInstagramGridLayoutGenerator(image_folder="./images")
generator.load_images()
generator.extract_dominant_colors(num_colors=3)
layouts = generator.generate_layouts(num_layouts=5)
```

#### `PerformanceProfiler`
Context manager for tracking operation performance.

```python
with PerformanceProfiler("image_processing"):
    # Your code here
    pass
```

## 🐛 Troubleshooting

### Common Issues

#### DINOv2 Model Loading Error
```bash
# Install transformers with specific version
pip install transformers==4.21.0 torch==2.0.0
```

#### Memory Issues
```python
# Reduce image size in settings
MAX_IMAGE_SIZE = 512  # Instead of 1024
```

#### Upload Failures
```bash
# Check file formats (JPG, PNG only)
# Ensure file size < 10MB per image
```

### Debug Mode
Enable debug mode by adding `?debug=true` to the URL for detailed logging.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Meta AI** for the DINOv2 vision transformer
- **Streamlit** for the amazing web framework
- **PyTorch** team for the deep learning tools
- **Instagram** for layout inspiration

## 📞 Contact

- **Author**: Siddharth Trivedi
- **LinkedIn**: [LinkedIn Profile](https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/siddharth-trivedi-dev/))

## 🚀 Roadmap

- [ ] **Video Support**: Generate layouts from video thumbnails
- [ ] **Batch Upload**: Process multiple image sets simultaneously
- [ ] **Style Transfer**: Apply artistic filters to enhance coherence
- [ ] **Social Integration**: Direct posting to Instagram
- [ ] **API Endpoints**: RESTful API for developers

---

<div align="center">

**⭐ Star this repo if it helped you create amazing Instagram layouts!**

[Report Bug](https://github.com/yourusername/instagram-grid-generator/issues) • [Request Feature](https://github.com/yourusername/instagram-grid-generator/issues) • [Documentation](https://github.com/yourusername/instagram-grid-generator/wiki)

</div>
