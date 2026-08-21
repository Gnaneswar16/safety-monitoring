import os
import sqlite3
# pyrefly: ignore [missing-import]
import numpy as np
import pytest
import cv2
import time

import config
import database
from core import Detection, PPEDetector, PPEAssociator, ComplianceEngine, TemporalSmoother, ComplianceResult

@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test_ppe_monitor.db")
    database.init_db(db_file)
    return db_file

@pytest.fixture
def temp_frames_dir(tmp_path):
    frames_path = str(tmp_path / "frames")
    os.makedirs(frames_path, exist_ok=True)
    return frames_path

def test_person_detection_structure():
    det = Detection(class_id=0, class_name="person", confidence=0.92, bbox=(100, 50, 300, 450))
    assert det.class_id == 0
    assert det.class_name == "person"
    assert det.confidence == 0.92
    assert det.bbox == (100, 50, 300, 450)

def test_glasses_association_valid():
    associator = PPEAssociator()
    # Person box: width=200, height=400 (x1=100, y1=100, x2=300, y2=500)
    person_bbox = (100.0, 100.0, 300.0, 500.0)

    # Valid glasses in eye level: y1 = 100 + 0.15*400 = 160, height = 30
    glasses_valid = (170.0, 150.0, 230.0, 180.0)
    valid, reason, score = associator.validate_glasses_candidate(person_bbox, glasses_valid)
    assert valid is True
    assert reason == "ACCEPTED"

def test_glasses_association_forehead_rejection():
    associator = PPEAssociator()
    person_bbox = (100.0, 100.0, 300.0, 500.0)
    # Glasses on forehead (y = 110, top 2.5% of body)
    glasses_forehead = (170.0, 105.0, 230.0, 115.0)
    valid, reason, score = associator.validate_glasses_candidate(person_bbox, glasses_forehead)
    assert valid is False
    assert "forehead" in reason

def test_glasses_association_chest_rejection():
    associator = PPEAssociator()
    person_bbox = (100.0, 100.0, 300.0, 500.0)
    # Glasses on chest (y = 300, 50% down body)
    glasses_chest = (170.0, 290.0, 230.0, 320.0)
    valid, reason, score = associator.validate_glasses_candidate(person_bbox, glasses_chest)
    assert valid is False

def test_shoes_association_valid():
    associator = PPEAssociator()
    person_bbox = (100.0, 100.0, 300.0, 500.0)
    # Valid shoes in feet region (y1 = 100 + 0.75*400 = 400)
    shoes_valid = (160.0, 420.0, 240.0, 480.0)
    valid, reason, score = associator.validate_shoes_candidate(person_bbox, shoes_valid)
    assert valid is True
    assert reason == "ACCEPTED"

def test_shoes_association_waist_rejection():
    associator = PPEAssociator()
    person_bbox = (100.0, 100.0, 300.0, 500.0)
    # Shoes at waist level (y = 250)
    shoes_waist = (160.0, 240.0, 240.0, 280.0)
    valid, reason, score = associator.validate_shoes_candidate(person_bbox, shoes_waist)
    assert valid is False

def test_single_person_ppe_association():
    associator = PPEAssociator()
    person = Detection(class_id=0, class_name="person", confidence=0.90, bbox=(100, 50, 300, 400))
    glasses = Detection(class_id=1, class_name="safety_glasses", confidence=0.85, bbox=(170, 95, 230, 125))
    shoes = Detection(class_id=2, class_name="safety_shoes", confidence=0.80, bbox=(160, 330, 240, 380))

    associations = associator.associate([person, glasses, shoes], frame_shape=(480, 640, 3))
    assert len(associations) == 1
    assoc = associations[0]
    assert len(assoc.glasses) == 1
    assert len(assoc.shoes) == 1

def test_multiple_person_ppe_association():
    associator = PPEAssociator()
    person1 = Detection(class_id=0, class_name="person", confidence=0.90, bbox=(50, 50, 200, 400))
    person2 = Detection(class_id=0, class_name="person", confidence=0.88, bbox=(350, 50, 500, 400))

    glasses1 = Detection(class_id=1, class_name="safety_glasses", confidence=0.85, bbox=(100, 95, 150, 125))
    shoes2 = Detection(class_id=2, class_name="safety_shoes", confidence=0.82, bbox=(400, 330, 460, 380))

    associations = associator.associate([person1, person2, glasses1, shoes2], frame_shape=(480, 640, 3))
    assert len(associations) == 2

    # Map by person_id
    p1_assoc = next(a for a in associations if a.person_detection == person1)
    p2_assoc = next(a for a in associations if a.person_detection == person2)

    assert len(p1_assoc.glasses) == 1
    assert len(p1_assoc.shoes) == 0

    assert len(p2_assoc.glasses) == 0
    assert len(p2_assoc.shoes) == 1

def test_missing_glasses_compliance():
    engine = ComplianceEngine()
    associator = PPEAssociator()
    person = Detection(class_id=0, class_name="person", confidence=0.90, bbox=(100, 50, 300, 400))
    shoes = Detection(class_id=2, class_name="safety_shoes", confidence=0.80, bbox=(160, 330, 240, 380))

    assocs = associator.associate([person, shoes], frame_shape=(480, 640, 3))
    result = engine.evaluate(assocs[0])

    assert result.status == "VIOLATION"
    assert "safety_glasses" in result.missing_ppe

def test_missing_shoes_compliance():
    engine = ComplianceEngine()
    associator = PPEAssociator()
    person = Detection(class_id=0, class_name="person", confidence=0.90, bbox=(100, 50, 300, 400))
    glasses = Detection(class_id=1, class_name="safety_glasses", confidence=0.85, bbox=(170, 95, 230, 125))

    assocs = associator.associate([person, glasses], frame_shape=(480, 640, 3))
    result = engine.evaluate(assocs[0])

    assert result.status == "VIOLATION"
    assert "safety_shoes" in result.missing_ppe

def test_missing_both_compliance():
    engine = ComplianceEngine()
    associator = PPEAssociator()
    person = Detection(class_id=0, class_name="person", confidence=0.90, bbox=(100, 50, 300, 400))

    assocs = associator.associate([person], frame_shape=(480, 640, 3))
    result = engine.evaluate(assocs[0])

    assert result.status == "VIOLATION"
    assert "safety_glasses" in result.missing_ppe
    assert "safety_shoes" in result.missing_ppe

def test_feet_not_visible_insufficient_evidence():
    engine = ComplianceEngine()
    associator = PPEAssociator()
    # Person box y2 = 475 in frame_height = 480 (feet cut off near bottom)
    person = Detection(class_id=0, class_name="person", confidence=0.90, bbox=(100, 50, 300, 475))
    glasses = Detection(class_id=1, class_name="safety_glasses", confidence=0.85, bbox=(170, 95, 230, 125))

    assocs = associator.associate([person, glasses], frame_shape=(480, 640, 3))
    result = engine.evaluate(assocs[0])

    assert result.status == "INSUFFICIENT EVIDENCE"
    assert result.feet_visible is False

def test_temporal_smoothing():
    smoother = TemporalSmoother(window_size=5)
    person_id = "P001"

    res_compliant = ComplianceResult(person_id, "COMPLIANT", True, True, True, True, 0.9, 0.9)
    res_violation = ComplianceResult(person_id, "VIOLATION", False, True, True, True, None, 0.9, ["safety_glasses"])

    # Initial frame compliant
    s_res1, trigger1 = smoother.process(person_id, res_compliant)
    assert s_res1.status == "COMPLIANT"

    # Single violation frame should be smoothed out to COMPLIANT
    s_res2, trigger2 = smoother.process(person_id, res_violation)
    assert s_res2.status == "COMPLIANT"
    assert trigger2 is False

def test_violation_debounce_and_cooldown():
    smoother = TemporalSmoother(window_size=5, violation_frames=5, cooldown_seconds=30.0)
    person_id = "P001"
    res_violation = ComplianceResult(person_id, "VIOLATION", False, False, True, True, None, None, ["safety_glasses", "safety_shoes"])

    start_time = 1000.0

    # 4 frames violation -> no trigger yet
    for i in range(4):
        s_res, trigger = smoother.process(person_id, res_violation, current_time=start_time + i)
        assert trigger is False

    # 5th frame violation -> trigger event!
    s_res5, trigger5 = smoother.process(person_id, res_violation, current_time=start_time + 4)
    assert s_res5.status == "VIOLATION"
    assert trigger5 is True

    # Immediate 6th frame violation (within 30s cooldown) -> no trigger
    s_res6, trigger6 = smoother.process(person_id, res_violation, current_time=start_time + 5)
    assert s_res6.status == "VIOLATION"
    assert trigger6 is False

    # After 31 seconds -> trigger event again!
    s_res31, trigger31 = smoother.process(person_id, res_violation, current_time=start_time + 36)
    assert trigger31 is True

def test_sqlite_storage(temp_db):
    database.log_detection("P001", "person", 0.95, (100, 50, 300, 400), db_path=temp_db)
    database.log_compliance_event("P001", "VIOLATION", 0, 1, 1, ["safety_glasses"], "data/frames/test.jpg", db_path=temp_db)

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    cursor.execute("SELECT person_id, class_name, confidence FROM detection_records")
    row_det = cursor.fetchone()
    assert row_det == ("P001", "person", 0.95)

    cursor.execute("SELECT person_id, status, missing_ppe, frame_path FROM compliance_events")
    row_event = cursor.fetchone()
    assert row_event == ("P001", "VIOLATION", "safety_glasses", "data/frames/test.jpg")
    conn.close()

def test_violation_frame_creation(temp_frames_dir):
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    saved_path = database.save_violation_frame(dummy_frame, "P001", frames_dir=temp_frames_dir)

    assert os.path.exists(saved_path)
    assert "violation_" in saved_path
    assert saved_path.endswith(".jpg")
