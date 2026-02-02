import cv2
from ultralytics import YOLO

def yolo_detect_and_track(input_path: str, output_path: str, model_path: str):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # tracking with persist=True
        results = model(frame, conf=0.4, classes=[0])

        # draw boxes + IDs
        annotated = results[0].plot()
        out.write(annotated)

    cap.release()
    out.release()
