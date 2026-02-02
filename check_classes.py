
import sys
import os

# Add the parent directory to sys.path to ensure imports work if needed
sys.path.append(os.path.join(os.getcwd(), 'app'))

try:
    from ultralytics import YOLO
    model_path = r"d:\people_detection_app\best(1).pt"
    model = YOLO(model_path)
    print("Model Classes:", model.names)
except Exception as e:
    print(f"Error loading model: {e}")
