# People Detection and Analytics System using Ultralytics YOLOv11L

A comprehensive computer vision pipeline for real-time people detection, gender classification, movement tracking, and analytical visualization using Ultralytics YOLOv11L. This web-based application processes camera feeds, photographs, and videos to provide detailed analytics with an intuitive user interface.

![People Detection Demo](https://img.shields.io/badge/YOLO-v11L-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green) ![Python](https://img.shields.io/badge/Python-3.8+-yellow)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Model Training](#-model-training)
- [Model Dependencies](#-model-dependencies)
- [Installation](#-installation)
- [Usage](#-usage)
- [Screenshots](#-screenshots)
- [Documentation](#-documentation)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

This project develops a state-of-the-art computer vision pipeline for analyzing camera feeds, photographs, and videos. The system leverages **Ultralytics YOLO** for robust people detection, implements gender classification, tracks movement patterns across frames, and generates comprehensive analytical visualizations including heatmaps and statistics.

### Key Capabilities:
- **Real-time Detection**: Process live webcam feeds or uploaded videos
- **Gender Classification**: Identify and count male/female individuals
- **Movement Tracking**: Track unique individuals across frames using IOU-based tracking
- **Heatmap Generation**: Visualize movement patterns and high-traffic areas
- **Statistical Analytics**: Real-time statistics including entry/exit counts, current occupancy, and gender distribution
- **Web Interface**: User-friendly FastAPI-based web application

---

## ✨ Features

### Core Detection Features
- ✅ **YOLO-based People Detection**: Utilizes Ultralytics YOLOv11L for accurate person detection
- ✅ **Gender Classification**: Classifies detected individuals as male or female with confidence scoring
- ✅ **Multi-Object Tracking**: Tracks individuals across frames with unique IDs using SimpleIOU tracker
- ✅ **Bounding Box Visualization**: Color-coded boxes (green for male, pink for female)

### Analytics & Visualization
- 📊 **Real-time Statistics Dashboard**:
  - Current frame count
  - Current people count
  - Total detected individuals
  - Male/Female count with unique IDs
  - Entry/Exit tracking
  
- 🔥 **Dynamic Heatmap Generation**:
  - Visualizes movement patterns
  - Decay-based heatmap for temporal analysis
  - Color-coded intensity (blue → yellow → red)

### Video Processing
- 🎥 **Multi-format Support**: Process MP4, AVI, and other common video formats
- 🎬 **Frame Skip Optimization**: Configurable frame skipping for faster processing
- 📹 **Output Video Generation**: Annotated videos with detection boxes and statistics
- 🌐 **HTTP Streaming**: Optimized MP4 output with faststart for web streaming

### Web Interface
- 🖥️ **Modern UI**: Clean, responsive web interface with pink/blue gradient theme
- 📤 **Drag & Drop Upload**: Easy video upload functionality
- 👁️ **Live Preview**: Preview uploaded videos before processing
- 📷 **Webcam Support**: Real-time webcam detection capability
- 📥 **Download Results**: Download processed videos with analytics

---

## 📁 Project Structure

```
people_detection_app/
│
├── app/                          # Main application package
│   ├── __init__.py              # Package initializer
│   ├── main.py                  # FastAPI application entry point
│   ├── pipeline.py              # Core detection and processing pipeline
│   ├── tracker.py               # SimpleIOU tracker implementation
│   ├── gender_detect.py         # Gender classification logic
│   ├── count.py                 # People counting and statistics
│   ├── heatmap.py               # Heatmap generation utilities
│   ├── utils.py                 # Utility functions and configurations
│   └── video_processor.py       # Video processing helpers
│
├── models/                       # YOLO model files
│   └── best (1).pt              # Trained YOLO model (51MB)
│
├── templates/                    # HTML templates
│   ├── index.html               # Main upload page
│   ├── preview.html             # Video preview and detection page
│   └── webcam.html              # Webcam detection page
│
├── static/                       # Static assets (CSS, JS, images)
│   ├── css/
│   └── js/
│
├── uploads/                      # Uploaded video storage
├── outputs/                      # Processed video output
│
├── requirements.txt              # Python dependencies
├── check_classes.py             # Model class verification script
├── check_model_classes_temp.py  # Temporary model checker
└── README.md                    # This file
```

### Key Components

#### `app/main.py`
FastAPI application with endpoints for:
- `/` - Home page with upload interface
- `/upload` - Video upload endpoint
- `/preview/{filename}` - Preview uploaded video
- `/process/{filename}` - Process video with detection
- `/video/{filename}` - Stream processed video
- `/webcam` - Webcam detection interface

#### `app/pipeline.py`
Core processing pipeline containing:
- YOLO model initialization
- Frame-by-frame detection
- Gender classification integration
- Tracking and statistics
- Heatmap generation
- Video encoding with FFmpeg

#### `app/tracker.py`
SimpleIOU tracker for multi-object tracking:
- Intersection over Union (IOU) calculation
- Track assignment and management
- Unique ID generation for individuals

#### `app/gender_detect.py`
Gender classification module:
- Extracts gender predictions from YOLO model
- Confidence-based filtering
- Gender assignment to tracked individuals

---

## ⚙️ Configuration

### Processing Parameters

The system can be configured by modifying constants in `app/pipeline.py`:

```python
# Detection Confidence
CONF_THRESHOLD = 0.3              # Minimum confidence for person detection

# Heatmap Settings
HEATMAP_DECAY = 0.985             # Decay rate for heatmap (0-1)
HEATMAP_INTENSITY = 50            # Intensity of heatmap points
HEATMAP_RADIUS = 80               # Radius of heatmap influence

# Gender Classification
GENDER_CONF_TH = 0.55             # Minimum confidence for gender classification

# Performance Optimization
FRAME_SKIP = 3                    # Process every Nth frame (1 = no skip)
```

### Model Configuration

Model path is configured in `app/utils.py`:
```python
MODEL_PATH = "models/best (1).pt"
```

### Directory Structure

Upload and output directories are automatically created:
- `uploads/` - Stores uploaded videos
- `outputs/` - Stores processed videos with analytics

---

## 🎓 Model Training

The YOLO model used in this project has been custom-trained for people detection and gender classification.

### Training Resources

Access the complete training resources, datasets, and model checkpoints:

🔗 **[Model Training Files on Google Drive](https://drive.google.com/drive/folders/1WaUXiC5rTVupRFZKTrFS_fdzuloRMRBa?usp=drive_link)**

This includes:
- Training datasets
- Validation datasets
- Model checkpoints
- Training configuration files
- Performance metrics and logs

### Model Specifications

- **Architecture**: YOLOv11L (Ultralytics)
- **Classes**: Person, Male, Female
- **Input Size**: 640x640
- **Model Size**: ~51MB
- **Format**: PyTorch (.pt)

---

## 📦 Model Dependencies

The system relies on the following key dependencies:

### Core Libraries
- **ultralytics** - YOLO implementation and model inference
- **opencv-python** - Video processing and computer vision operations
- **numpy** - Numerical computations and array operations

### Web Framework
- **fastapi** - Modern web framework for API endpoints
- **uvicorn** - ASGI server for FastAPI
- **python-multipart** - File upload handling
- **jinja2** - HTML template rendering

### Additional Tools
- **FFmpeg** - Video encoding and optimization (system dependency)

All Python dependencies are listed in `requirements.txt`:
```
fastapi
uvicorn
python-multipart
jinja2
opencv-python
numpy
ultralytics
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.8+** installed on your system
- **FFmpeg** installed (for video processing)
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - Linux: `sudo apt-get install ffmpeg`
  - macOS: `brew install ffmpeg`

### Clone the Repository

```bash
git clone https://github.com/yourusername/people_detection_app.git
cd people_detection_app
```

### Set Up Virtual Environment

Create and activate a Python virtual environment:

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Model

Ensure the trained YOLO model is placed in the `models/` directory:
- Download from the [Model Training Drive](https://drive.google.com/drive/folders/1WaUXiC5rTVupRFZKTrFS_fdzuloRMRBa?usp=drive_link)
- Place `best (1).pt` in `models/` folder

---

## 💻 Usage

### Start the Application

From the project root directory (`D:\people_detection_app`), run:

```powershell
uvicorn app.main:app --reload
```

The application will start on `http://127.0.0.1:8000`

### Access the Web Interface

1. Open your browser and navigate to: **http://127.0.0.1:8000**
2. You'll see the main upload interface

### Process a Video

1. **Upload**: Click "Upload" or drag and drop a video file (MP4, AVI, etc.)
2. **Preview**: Click "Preview & Detect" to view the uploaded video
3. **Detect**: Click "Run Detection" to start processing
4. **View Results**: Watch the processed video with:
   - Bounding boxes around detected people
   - Gender labels (Male/Female)
   - Real-time statistics panel
   - Unique tracking IDs
5. **Download**: Download the processed video using the download button

### Use Webcam Detection

1. Navigate to the "Live Webcam" option
2. Grant camera permissions
3. Real-time detection will begin automatically

### Terminal Commands Summary

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn app.main:app --reload
```

---

## 📸 Screenshots

Experience the application interface through these screenshots:


### 1. Upload Interface
![Upload Interface](./SS%20&%20video/56.PNG)

<!-- slide -->

### 2. File Selection
![File Selection](./SS%20&%20video/44.PNG)


<!-- slide -->

### 3. Processing Status
![Processing Status](./SS%20&%20video/43.PNG)


<!-- slide -->

### 4. Detection Results
![Detection Results](./SS%20&%20video/66.PNG)

## 5. Final Result
<iframe width="720" height="405"
src="https://www.youtube.com/embed/kbzv0LaFjsc"
title="Final Result Video"
frameborder="0"
allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
allowfullscreen>
</iframe>

**Processed video** with complete analytics:
- Green bounding boxes for males
- Pink bounding boxes for females
- Statistics panel with real-time counts
- Unique tracking IDs for each person
- Current frame information and gender distribution
````

---

## 📚 Documentation

For comprehensive project documentation, development phases, and detailed findings, please refer to:

🔗 **[Complete Project Documentation](https://docs.google.com/document/d/13g2G4WhUwOdepc1mslRuffs_tn9IRXHh/edit?usp=drive_link&ouid=106075921329979584868&rtpof=true&sd=true)**

This documentation includes:
- Detailed system architecture
- Algorithm explanations
- Development methodology
- Testing procedures
- Performance benchmarks
- Research findings
- Technical specifications

---

## ⚠️ Limitations

### Current Limitations

1. **Processing Speed**
   - Real-time processing depends on hardware capabilities
   - Large videos may take significant time to process
   - Frame skipping is used to optimize performance

2. **Detection Accuracy**
   - Accuracy depends on video quality and lighting conditions
   - Occlusion can affect detection and tracking
   - Small or distant individuals may not be detected

3. **Gender Classification**
   - Gender classification is based on visual appearance
   - May have reduced accuracy with certain clothing or angles
   - Confidence threshold filtering may miss some classifications

4. **Tracking Limitations**
   - SimpleIOU tracker may lose tracks during heavy occlusion
   - ID switches can occur in crowded scenes
   - No re-identification after track loss

5. **Video Format Support**
   - Primarily optimized for MP4 format
   - Some codecs may require FFmpeg conversion
   - Very high-resolution videos may cause memory issues

6. **Webcam Support**
   - Browser compatibility varies
   - Requires HTTPS for production deployment
   - Limited to single camera source

---

## 🔮 Future Improvements

### Planned Enhancements

#### Performance Optimization
- [ ] GPU acceleration support (CUDA)
- [ ] Multi-threading for parallel frame processing
- [ ] Adaptive frame skipping based on scene complexity
- [ ] Video compression optimization
- [ ] Caching mechanism for repeated processing

#### Detection & Tracking
- [ ] Integration of more advanced trackers (DeepSORT, ByteTrack)
- [ ] Re-identification capabilities for lost tracks
- [ ] Age estimation alongside gender classification
- [ ] Pose estimation for activity recognition
- [ ] Crowd density estimation

#### Analytics & Visualization
- [ ] Advanced analytics dashboard with charts
- [ ] Historical data storage and analysis
- [ ] Zone-based analytics (entry/exit zones)
- [ ] Dwell time calculation
- [ ] Path prediction and trajectory analysis
- [ ] Export analytics to CSV/JSON

#### User Interface
- [ ] Batch video processing
- [ ] Progress bar with ETA
- [ ] Video trimming and region selection
- [ ] Real-time parameter adjustment
- [ ] Mobile-responsive design improvements
- [ ] Dark/light theme toggle

#### Model Improvements
- [ ] Fine-tuning on domain-specific datasets
- [ ] Support for multiple camera angles
- [ ] Custom class training interface
- [ ] Model versioning and A/B testing
- [ ] Ensemble model support

#### Integration & Deployment
- [ ] REST API documentation (Swagger/OpenAPI)
- [ ] Docker containerization
- [ ] Cloud deployment guides (AWS, Azure, GCP)
- [ ] Database integration for analytics storage
- [ ] Webhook support for event notifications
- [ ] Multi-camera synchronization

#### Security & Privacy
- [ ] User authentication and authorization
- [ ] Video encryption at rest
- [ ] Privacy mode (face blurring)
- [ ] GDPR compliance features
- [ ] Audit logging

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide for Python code
- Add docstrings to all functions and classes
- Write unit tests for new features
- Update documentation for significant changes
- Ensure all tests pass before submitting PR

---



## 👥 Authors

- **Adiba Rahman Joarder** - *Initial work* - [adibajoarder](https://github.com/adibajoarder)

---

##  Acknowledgments

- **Ultralytics** for the YOLO implementation
- **FastAPI** team for the excellent web framework
- **OpenCV** community for computer vision tools
- All contributors and testers

---

## 📞 Contact 01708046272

For questions, issues, or suggestions:

- **GitHub Issues**: [Create an issue](https://github.com/adibajoarder)
- **Email**: adibarahmanjoarder@gmail.com
- **Documentation**: [Project Docs](https://docs.google.com/document/d/13g2G4WhUwOdepc1mslRuffs_tn9IRXHh/edit?usp=drive_link&ouid=106075921329979584868&rtpof=true&sd=true)

---



**Made with ❤️ using YOLO, FastAPI, and OpenCV**
