import os
import uuid

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # app/

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best (1).pt")

def ensure_dirs():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def unique_filename(original_name: str):
    ext = os.path.splitext(original_name)[1].lower()
    if ext == "":
        ext = ".mp4"
    return f"{uuid.uuid4().hex}{ext}"
