"""
Configuration module for People Detection Application
Handles automatic environment setup and dependency management
"""
import os
import sys
import subprocess
import importlib.util
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory paths
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"

# Model configuration
MODEL_PATH = MODELS_DIR / "best (1).pt"
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IOU = 0.45

# Video processing configuration
FRAME_SKIP = 1  # Process every nth frame (1 = process all frames)
MAX_VIDEO_SIZE_MB = 500  # Maximum video file size in MB

# Application configuration
HOST = "0.0.0.0"
PORT = 8000
DEBUG = True

# Required packages
REQUIRED_PACKAGES = [
    "fastapi",
    "uvicorn",
    "python-multipart",
    "jinja2",
    "opencv-python",
    "numpy",
    "ultralytics",
]


def check_package_installed(package_name):
    """Check if a package is installed"""
    # Handle package name variations
    import_name = package_name
    if package_name == "opencv-python":
        import_name = "cv2"
    elif package_name == "python-multipart":
        import_name = "multipart"
    
    spec = importlib.util.find_spec(import_name)
    return spec is not None


def install_package(package_name):
    """Install a package using pip"""
    try:
        print(f"Installing {package_name}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"✓ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package_name}")
        return False


def setup_environment():
    """
    Automatically check and install required dependencies
    Creates necessary directories
    """
    print("=" * 60)
    print("People Detection Application - Environment Setup")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"\nPython Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("⚠ Warning: Python 3.8 or higher is recommended")
    
    # Check and install missing packages
    print("\n" + "-" * 60)
    print("Checking dependencies...")
    print("-" * 60)
    
    missing_packages = []
    for package in REQUIRED_PACKAGES:
        if not check_package_installed(package):
            missing_packages.append(package)
            print(f"✗ {package} - NOT INSTALLED")
        else:
            print(f"✓ {package} - installed")
    
    # Install missing packages
    if missing_packages:
        print(f"\n{len(missing_packages)} package(s) need to be installed.")
        response = input("Install missing packages? (y/n): ").lower().strip()
        
        if response == 'y':
            print("\nInstalling packages...")
            failed_packages = []
            for package in missing_packages:
                if not install_package(package):
                    failed_packages.append(package)
            
            if failed_packages:
                print(f"\n⚠ Failed to install: {', '.join(failed_packages)}")
                print("Please install them manually using:")
                print(f"pip install {' '.join(failed_packages)}")
                return False
        else:
            print("\nSkipping package installation.")
            print("Please install required packages manually:")
            print(f"pip install -r requirements.txt")
            return False
    
    # Create necessary directories
    print("\n" + "-" * 60)
    print("Setting up directories...")
    print("-" * 60)
    
    directories = [UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR, MODELS_DIR, TEMPLATES_DIR]
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created: {directory.name}/")
        else:
            print(f"✓ Exists: {directory.name}/")
    
    # Check if model file exists
    print("\n" + "-" * 60)
    print("Checking model files...")
    print("-" * 60)
    
    if MODEL_PATH.exists():
        print(f"✓ Model found: {MODEL_PATH.name}")
    else:
        print(f"⚠ Model not found: {MODEL_PATH.name}")
        print("  Please ensure the YOLO model file is placed in the models/ directory")
    
    print("\n" + "=" * 60)
    print("Environment setup completed!")
    print("=" * 60)
    print(f"\nTo start the application, run:")
    print(f"  uvicorn app.main:app --host {HOST} --port {PORT} --reload")
    print(f"\nOr use Python:")
    print(f"  python -m uvicorn app.main:app --host {HOST} --port {PORT} --reload")
    print("\n" + "=" * 60)
    
    return True


def verify_environment():
    """Quick verification that all dependencies are available (silent check)"""
    all_installed = all(check_package_installed(pkg) for pkg in REQUIRED_PACKAGES)
    all_dirs_exist = all(d.exists() for d in [UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR])
    model_exists = MODEL_PATH.exists()
    
    return all_installed and all_dirs_exist and model_exists


def get_config():
    """Return configuration dictionary"""
    return {
        "base_dir": str(BASE_DIR),
        "upload_dir": str(UPLOAD_DIR),
        "output_dir": str(OUTPUT_DIR),
        "static_dir": str(STATIC_DIR),
        "models_dir": str(MODELS_DIR),
        "templates_dir": str(TEMPLATES_DIR),
        "model_path": str(MODEL_PATH),
        "host": HOST,
        "port": PORT,
        "debug": DEBUG,
        "confidence": DEFAULT_CONFIDENCE,
        "iou": DEFAULT_IOU,
        "frame_skip": FRAME_SKIP,
        "max_video_size_mb": MAX_VIDEO_SIZE_MB,
    }


if __name__ == "__main__":
    # Run setup when executed directly
    setup_environment()
