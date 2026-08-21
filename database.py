import os
import sqlite3
import cv2
from datetime import datetime
import config

def get_connection(db_path=config.DB_PATH):
    """Establishes SQLite connection and creates tables if missing."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    return conn

def init_db(db_path=config.DB_PATH):
    """Initializes SQLite schema for detection_records and compliance_events."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detection_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        person_id TEXT NOT NULL,
        class_name TEXT NOT NULL,
        confidence REAL NOT NULL,
        bbox_x1 REAL NOT NULL,
        bbox_y1 REAL NOT NULL,
        bbox_x2 REAL NOT NULL,
        bbox_y2 REAL NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS compliance_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        person_id TEXT NOT NULL,
        status TEXT NOT NULL,
        has_glasses INTEGER NOT NULL,
        has_shoes INTEGER NOT NULL,
        feet_visible INTEGER NOT NULL,
        missing_ppe TEXT,
        frame_path TEXT
    )
    """)

    conn.commit()
    conn.close()

def log_detection(person_id, class_name, confidence, bbox, db_path=config.DB_PATH):
    """Logs raw or associated detection record."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now_str = datetime.now().isoformat()
    x1, y1, x2, y2 = bbox
    cursor.execute("""
    INSERT INTO detection_records (timestamp, person_id, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (now_str, str(person_id), str(class_name), float(confidence), float(x1), float(y1), float(x2), float(y2)))
    conn.commit()
    conn.close()

def log_compliance_event(person_id, status, has_glasses, has_shoes, feet_visible, missing_ppe, frame_path=None, db_path=config.DB_PATH):
    """Logs a compliance event (e.g. violation or status change)."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now_str = datetime.now().isoformat()
    missing_str = ", ".join(missing_ppe) if isinstance(missing_ppe, list) else str(missing_ppe or "")
    cursor.execute("""
    INSERT INTO compliance_events (timestamp, person_id, status, has_glasses, has_shoes, feet_visible, missing_ppe, frame_path)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (now_str, str(person_id), str(status), int(has_glasses), int(has_shoes), int(feet_visible), missing_str, frame_path))
    conn.commit()
    event_id = cursor.lastrowid
    conn.close()
    return event_id

def save_violation_frame(frame, person_id, frames_dir=config.FRAMES_DIR):
    """Saves the current annotated frame to disk under data/frames/."""
    os.makedirs(frames_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"violation_{timestamp_str}_{person_id}.jpg"
    file_path = os.path.join(frames_dir, filename)
    cv2.imwrite(file_path, frame)
    return file_path
