import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
DB_PATH = os.path.join(DATA_DIR, "ppe_monitor.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "ppe_model.pt")

# Ensure required directories exist
for path in [DATA_DIR, FRAMES_DIR, VIDEOS_DIR, os.path.dirname(MODEL_PATH)]:
    os.makedirs(path, exist_ok=True)

# YOLO Classes
CLASS_PERSON = 0
CLASS_GLASSES = 1
CLASS_SHOES = 2
CLASS_NAMES = {
    0: "person",
    1: "safety_glasses",
    2: "safety_shoes"
}

# Detection Thresholds
PERSON_CONF_THRESHOLD = 0.25
GLASSES_CONF_THRESHOLD = 0.10
SHOES_CONF_THRESHOLD = 0.25

# Application Settings
RECORD_VIDEO = False
DEBUG_MODE = False
FULLSCREEN = False



# Temporal & Debounce Settings
SMOOTHING_WINDOW_SIZE = 5
VIOLATION_CONSECUTIVE_FRAMES = 5
VIOLATION_COOLDOWN_SECONDS = 30.0

# Camera Index (0 = Built-in webcam, 1 = USB camera)
CAMERA_INDEX = 1

