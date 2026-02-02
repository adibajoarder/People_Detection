from ultralytics import YOLO
try:
    model = YOLO(r"d:\people_detection_app\models\best (1).pt")
    print("Model Classes:", model.names)
except Exception as e:
    print(e)
