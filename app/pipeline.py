import cv2
import os
import numpy as np
import shutil
import subprocess
from ultralytics import YOLO

from app.utils import MODEL_PATH
from app.tracker import SimpleIOUTracker
# from app.gender_detect import apply_gender_to_tracks

# ----------------------------
# SETTINGS (from your mentor)
# ----------------------------
SIDEBAR_WIDTH = 320

# Counting
MIN_FRAMES_TO_COUNT = 8
EXIT_TIMEOUT = 20

# Heatmap
HEATMAP_DECAY = 0.985
HEATMAP_INTENSITY = 50
HEATMAP_RADIUS = 80

# Gender confidence threshold
GENDER_CONF_TH = 0.55

VALID_GENDERS = {"male", "female"}

# Processing Speed Settings
FRAME_SKIP = 3  # Process 1 frame, skip 2 (repeat visualization)


def _draw_sidebar(frame, stats, frame_idx):
    h, w = frame.shape[:2]
    out = cv2.copyMakeBorder(frame, 0, 0, 0, SIDEBAR_WIDTH, cv2.BORDER_CONSTANT, value=(40, 40, 40))

    cv2.putText(out, "STATISTICS", (w + 20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)

    lines = [
        ("Frame:", frame_idx, (255, 255, 255)),
        ("Current Count:", stats["current_count"], (255, 255, 255)),
        ("Total Entered:", stats["total_entered"], (0, 255, 0)),
        ("Total Exited:", stats["total_exited"], (0, 0, 255)),
        ("Males:", stats["males"], (0, 255, 0)),
        ("Females:", stats["females"], (255, 100, 255)),
    ]

    y = 95
    for label, value, color in lines:
        cv2.putText(out, f"{label} {value}", (w + 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2)
        y += 40

    return out


# Load model once at module level
model = YOLO(MODEL_PATH)


def _run_ffmpeg_faststart(src_path: str, dst_path: str) -> bool:
    """Prepare MP4 for HTTP streaming (faststart, H.264 compatible)."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, "-c", "copy", "-movflags", "faststart", dst_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                src_path,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "faststart",
                "-an",
                dst_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def run_full_pipeline_single(input_path: str, output_path: str):
    # model is now global


    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    # Read first frame to ensure valid dimensions
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise RuntimeError(f"No frames in video: {input_path}")

    h, w = first_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # Output video size includes sidebar
    out_w = w + SIDEBAR_WIDTH
    out_h = h

    # Try using H.264 codec directly if available, fallback to mp4v
    try:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
    except:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        
    temp_output_path = f"{output_path}.tmp.mp4" # Ensure extension
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (out_w, out_h))
    if not out.isOpened():
        # Fallback to mp4v if avc1 failed to open
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (out_w, out_h))
        if not out.isOpened():
             cap.release()
             raise RuntimeError(f"Cannot write video: {temp_output_path}")

    # Tracker (no lap)
    tracker = SimpleIOUTracker(iou_threshold=0.35, max_lost=25)

    # Heatmap accum
    heatmap_accum = np.zeros((h, w), dtype=np.float32)

    # Gaussian blob (thick heatmap like notebook)
    x = np.arange(0, 2 * HEATMAP_RADIUS + 1)
    y = np.arange(0, 2 * HEATMAP_RADIUS + 1)
    xx, yy = np.meshgrid(x, y)
    cx0, cy0 = HEATMAP_RADIUS, HEATMAP_RADIUS
    gaussian = np.exp(-((xx - cx0) ** 2 + (yy - cy0) ** 2) / (2 * (HEATMAP_RADIUS / 2.2) ** 2))
    gaussian = gaussian / gaussian.max()

    # Counting states (from your logic)
    frame_idx = 0
    first_seen = {}
    last_seen = {}
    counted_entry = set()
    counted_exit = set()

    total_entered = 0
    total_exited = 0
    males = 0
    females = 0

    # helper for gender read
    id_gender_map = {}  # tid -> male/female/unknown

    # Counting line (you can move this if needed)
    line_y = int(h * 0.55)

    frame = first_frame
    last_processed_frame = None  # To hold the last fully processed frame for skipping
    
    # Pre-process the first frame to have something to show if we skip initially (unlikely with logic)
    # But loop structure handles it
    
    while True:
        if frame is None:
            break

        frame_idx += 1
        
        # Frame Skipping Logic:
        # If not (frame_idx % FRAME_SKIP == 1), just write the last processed frame
        # We process frame 1, 1+SKIP, etc.
        # Actually safer to process frame 1.
        
        should_process = (frame_idx % FRAME_SKIP == 1) or (FRAME_SKIP == 1)

        if not should_process and last_processed_frame is not None:
             out.write(last_processed_frame)
             ret, frame = cap.read()
             if not ret:
                 break
             continue

        # ----------------------------
        # YOLO PERSON DETECTION
        # ----------------------------
        # Enable both classes 0 (female) and 1 (male)
        # Lower confidence to catch more people
        results = model(frame, conf=0.25, classes=[0, 1], verbose=False)
        detections = []

        if results and results[0].boxes is not None:
            for b in results[0].boxes:
                coords = b.xyxy[0].tolist()  # [x1, y1, x2, y2]
                cls_id = int(b.cls[0].item())
                
                # Map class to gender
                # Based on check_classes.py: {0: 'female', 1: 'male'}
                if cls_id == 0:
                    g_label = "female"
                elif cls_id == 1:
                    g_label = "male"
                else:
                    g_label = "unknown"
                    
                detections.append((coords, g_label))

        # ----------------------------
        # TRACK IDs
        # ----------------------------
        tracked = tracker.update(detections)

        # ----------------------------
        # 1) GENDER (Handled in Tracker now)
        # ----------------------------
        # apply_gender_to_tracks(frame, tracked, tracker) <--- REMOVED

        # Pull gender from tracker into map
        for tid, _ in tracked:
            g = tracker.get_gender(tid)
            if g not in VALID_GENDERS:
                g = "unknown"
            id_gender_map[tid] = g

        # ----------------------------
        # 2) COUNTING ENTRY/EXIT
        # ----------------------------
        # update seen times
        for tid, _ in tracked:
            if tid not in first_seen:
                first_seen[tid] = frame_idx
            last_seen[tid] = frame_idx

            # Enter rule: must exist for MIN_FRAMES_TO_COUNT frames
            if tid not in counted_entry and (frame_idx - first_seen[tid] >= MIN_FRAMES_TO_COUNT):
                counted_entry.add(tid)
                total_entered += 1

                g = id_gender_map.get(tid, "unknown")
                if g == "male":
                    males += 1
                elif g == "female":
                    females += 1

        # Exit rule: if missing more than EXIT_TIMEOUT frames
        # We count exit only if it was already counted as entry
        # active_ids = set([tid for tid, _ in tracked]) # Remove: optimize
        current_tracked_ids = set(t[0] for t in tracked)
        
        for tid in list(last_seen.keys()):
            if tid in counted_entry and tid not in current_tracked_ids:
                if (frame_idx - last_seen[tid] > EXIT_TIMEOUT) and (tid not in counted_exit):
                    counted_exit.add(tid)
                    total_exited += 1

        current_count = len(tracked)

        # ----------------------------
        # 3) HEATMAP UPDATE
        # ----------------------------
        heatmap_accum *= HEATMAP_DECAY

        for tid, box in tracked:
            x1, y1, x2, y2 = [int(v) for v in box]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # thick gaussian blob add
            x_start = cx - HEATMAP_RADIUS
            y_start = cy - HEATMAP_RADIUS
            x_end = cx + HEATMAP_RADIUS + 1
            y_end = cy + HEATMAP_RADIUS + 1

            # clip region
            gx1, gy1 = 0, 0
            gx2, gy2 = gaussian.shape[1], gaussian.shape[0]

            if x_start < 0:
                gx1 = -x_start
                x_start = 0
            if y_start < 0:
                gy1 = -y_start
                y_start = 0
            if x_end > w:
                gx2 = gx2 - (x_end - w)
                x_end = w
            if y_end > h:
                gy2 = gy2 - (y_end - h)
                y_end = h

            if x_start < x_end and y_start < y_end:
                heatmap_accum[y_start:y_end, x_start:x_end] += gaussian[gy1:gy2, gx1:gx2] * HEATMAP_INTENSITY

        # Render heatmap box
        heat_norm = cv2.normalize(heatmap_accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

        heat_box = cv2.resize(heat_color, (240, 240))

        # merge heatmap into bottom-right
        box_h, box_w = heat_box.shape[:2]
        x0 = w - box_w - 10
        y0 = h - box_h - 10

        cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (0, 255, 255), 2)
        frame[y0:y0 + box_h, x0:x0 + box_w] = heat_box

        # draw counting line
        cv2.line(frame, (0, line_y), (w, line_y), (255, 255, 255), 2)

        # ----------------------------
        # DRAW BOXES + LABELS
        # ----------------------------
        for tid, box in tracked:
            x1, y1, x2, y2 = [int(v) for v in box]
            gender_final = id_gender_map.get(tid, "unknown")

            if gender_final == "male":
                color = (0, 255, 0)
            elif gender_final == "female":
                color = (255, 100, 255)
            else:
                color = (200, 200, 200)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{tid} {gender_final.capitalize()}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # ----------------------------
        # Sidebar stats
        # ----------------------------
        stats = {
            "current_count": current_count,
            "total_entered": total_entered,
            "total_exited": total_exited,
            "males": males,
            "females": females
        }

        final_frame = _draw_sidebar(frame, stats, frame_idx)
        last_processed_frame = final_frame.copy() # Save for skipping
        out.write(final_frame)

        ret, frame = cap.read()
        if not ret:
            break

    cap.release()
    out.release()

    # Faststart for better HTTP playback
    if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) == 0:
        os.remove(temp_output_path)
        raise RuntimeError(f"Empty output video: {temp_output_path}")
    
    # We used avc1 if possible, but let's run ffmpeg faststart anyway to ensure web compatibility
    # and to potentially fix any encoding issues if mp4v was used.
    if not _run_ffmpeg_faststart(temp_output_path, output_path):
        # If ffmpeg fails, just move the file
        if os.path.exists(output_path):
             os.remove(output_path)
        shutil.move(temp_output_path, output_path)
    else:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
