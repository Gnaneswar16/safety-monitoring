import cv2
import time
import os
import sys

# Ensure root directory is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core import PPEDetector, PPEAssociator, ComplianceEngine, TemporalSmoother
from main import draw_overlay

def run_webcam_test(camera_index=config.CAMERA_INDEX):
    print(f"[TEST] Opening physical webcam at index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {camera_index}.")
        sys.exit(1)

    print("[TEST] Webcam opened successfully.")
    print("[TEST] Initializing YOLO model and pipeline...")

    detector = PPEDetector()
    associator = PPEAssociator()
    compliance_engine = ComplianceEngine()
    smoother = TemporalSmoother()

    prev_time = time.time()
    last_report_time = time.time()
    fps = 0.0

    window_name = "Webcam Test Feed"
    fullscreen = config.FULLSCREEN
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

    print("[TEST] Live feed starting. Controls: Press 'Q' to Exit | Press 'F' to toggle Fullscreen.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARNING] Could not read frame from webcam.")
            time.sleep(0.1)
            continue

        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
        prev_time = curr_time

        # Pipeline execution
        detections = detector.detect(frame)
        associations = associator.associate(detections, frame_shape=frame.shape)

        smoothed_results = {}
        compliant_count = 0
        violation_count = 0
        insufficient_count = 0

        for assoc in associations:
            raw_result = compliance_engine.evaluate(assoc)
            smoothed_result, _ = smoother.process(assoc.person_id, raw_result, current_time=curr_time)
            smoothed_results[assoc.person_id] = smoothed_result

            if smoothed_result.status == "COMPLIANT":
                compliant_count += 1
            elif smoothed_result.status == "VIOLATION":
                violation_count += 1
            else:
                insufficient_count += 1

        # Render display
        display_frame = draw_overlay(frame, associations, smoothed_results, debug_mode=True)

        cv2.putText(display_frame, f"TEST MODE | FPS: {fps:.1f} | Persons: {len(associations)}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Compliant: {compliant_count} | Violation: {violation_count} | Insufficient: {insufficient_count}",
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow(window_name, display_frame)

        # Print periodic statistics every 3 seconds
        if curr_time - last_report_time >= 3.0:
            print(f"[STATUS] FPS: {fps:.1f} | Persons: {len(associations)} | "
                  f"Compliant: {compliant_count} | Violation: {violation_count} | "
                  f"Insufficient: {insufficient_count}")
            last_report_time = curr_time

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("[TEST] Q pressed. Stopping webcam test.")
            break
        elif key == ord('f') or key == ord('F'):
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()
    print("[TEST] Webcam test completed successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Webcam Test Diagnostics")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Camera index (e.g. 0 or 1)")
    args = parser.parse_args()
    run_webcam_test(camera_index=args.camera)
