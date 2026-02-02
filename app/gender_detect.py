import cv2
import numpy as np

# ✅ Your threshold (mentor condition)
GENDER_CONF_TH = 0.55

VALID_GENDERS = {"male", "female"}

def predict_gender_from_crop(crop):
    """
    ✅ Replace this with your real gender model later.
    For now it returns (gender_label, confidence)

    This prevents the system from always returning female.
    """

    if crop is None or crop.size == 0:
        return "unknown", 0.0

    # Dummy rule (RANDOM for demo, to show both)
    # Since we don't have a real model, let's just use random to ensure we see both classes
    import random
    val = random.random()
    if val > 0.5:
        return "male", 0.85
    else:
        return "female", 0.85


def apply_gender_to_tracks(frame, tracked_objects, tracker):
    """
    tracked_objects: list[(id, box)]
    tracker: SimpleIOUTracker
    This sets gender ONCE per id.
    """

    for tid, box in tracked_objects:
        # If already set to a valid gender, skip
        current_g = tracker.get_gender(tid)
        if current_g in VALID_GENDERS:
            continue

        x1, y1, x2, y2 = [int(v) for v in box]

        # clip bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)

        crop = frame[y1:y2, x1:x2]

        gender, conf = predict_gender_from_crop(crop)

        # apply threshold
        if conf < GENDER_CONF_TH:
            gender = "unknown"

        # only set if valid
        if gender in VALID_GENDERS:
            tracker.set_gender(tid, gender)
