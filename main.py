# pyrefly: ignore [missing-import]
import cv2
import time
import os
import argparse
from datetime import datetime
import config
import database
from core import PPEDetector, PPEAssociator, ComplianceEngine, TemporalSmoother

# Color Palette (BGR)
COLOR_BLUE = (255, 0, 0)       # Person BBox
COLOR_CYAN = (255, 255, 0)     # Safety Glasses BBox
COLOR_MAGENTA = (255, 0, 255)  # Safety Shoes BBox

STATUS_COLORS = {
    "COMPLIANT": (0, 255, 0),             # Green
    "VIOLATION": (0, 0, 255),             # Red
    "INSUFFICIENT EVIDENCE": (0, 255, 255) # Yellow
}

def draw_overlay(frame, associations, smoothed_results, debug_mode=False):
    """Draws bounding boxes and structured compliance overlays on the frame."""
    annotated = frame.copy()

    for assoc in associations:
        p_det = assoc.person_detection
        p_id = assoc.person_id
        px1, py1, px2, py2 = map(int, p_det.bbox)
        result = smoothed_results.get(p_id)

        status = result.status if result else "INSUFFICIENT EVIDENCE"
        status_color = STATUS_COLORS.get(status, (0, 255, 255))

        # 1. Draw Person Bounding Box (BLUE)
        cv2.rectangle(annotated, (px1, py1), (px2, py2), COLOR_BLUE, 2)

        # 2. Draw Associated Glasses (CYAN)
        for g_det in assoc.glasses:
            gx1, gy1, gx2, gy2 = map(int, g_det.bbox)
            cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), COLOR_CYAN, 2)
            cv2.putText(annotated, f"Glasses {g_det.confidence:.2f}", (gx1, max(gy1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_CYAN, 1)

        # 3. Draw Associated Shoes (MAGENTA)
        for s_det in assoc.shoes:
            sx1, sy1, sx2, sy2 = map(int, s_det.bbox)
            cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2), COLOR_MAGENTA, 2)
            cv2.putText(annotated, f"Shoes {s_det.confidence:.2f}", (sx1, max(sy1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_MAGENTA, 1)

        # 4. Debug Mode Drawings
        if debug_mode:
            # Eye ROI
            ex1, ey1, ex2, ey2 = map(int, assoc.eye_roi)
            cv2.rectangle(annotated, (ex1, ey1), (ex2, ey2), (0, 255, 255), 1, cv2.LINE_AA)
            # Feet ROI
            fx1, fy1, fx2, fy2 = map(int, assoc.feet_roi)
            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (255, 0, 255), 1, cv2.LINE_AA)

        # 5. Formatted Status Card Above Person
        card_x1 = px1
        card_y2 = max(py1 - 10, 10)

        lines = [f"Person {p_id}", f"Status: {status}"]
        if status == "COMPLIANT":
            g_conf_str = f"{result.glasses_conf:.2f}" if result and result.glasses_conf else "N/A"
            s_conf_str = f"{result.shoes_conf:.2f}" if result and result.shoes_conf else "N/A"
            lines.append(f"Glasses: YES {g_conf_str}")
            lines.append(f"Shoes: YES {s_conf_str}")
        elif status == "VIOLATION":
            missing_str = ", ".join(result.missing_ppe) if result else "N/A"
            lines.append(f"Missing: {missing_str}")
        else:
            reason_str = result.reason if result else "Feet not visible"
            lines.append(reason_str)

        # Background box for text clarity
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        line_height = 18
        box_h = len(lines) * line_height + 10
        box_w = 220
        box_y1 = max(card_y2 - box_h, 5)

        # Semi-transparent backdrop
        overlay_sub = annotated[box_y1:box_y1+box_h, card_x1:card_x1+box_w]
        if overlay_sub.shape[0] > 0 and overlay_sub.shape[1] > 0:
            bg_rect = cv2.rectangle(overlay_sub.copy(), (0, 0), (box_w, box_h), (0, 0, 0), -1)
            cv2.addWeighted(bg_rect, 0.7, overlay_sub, 0.3, 0, overlay_sub)
            cv2.rectangle(annotated, (card_x1, box_y1), (card_x1+box_w, box_y1+box_h), status_color, 2)

        # Render status text lines
        for idx, line in enumerate(lines):
            txt_color = status_color if idx < 2 else (255, 255, 255)
            cv2.putText(annotated, line, (card_x1 + 8, box_y1 + 16 + (idx * line_height)),
                        font, font_scale, txt_color, 1, cv2.LINE_AA)

    return annotated

def main():
    parser = argparse.ArgumentParser(description="Real-Time PPE Detection Application")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Camera index (default 0)")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG_MODE")
    parser.add_argument("--fullscreen", action="store_true", default=config.FULLSCREEN, help="Start in fullscreen mode")
    parser.add_argument("--windowed", action="store_true", help="Start in windowed mode")
    args = parser.parse_args()

    debug_mode = args.debug or config.DEBUG_MODE
    fullscreen = False if args.windowed else (args.fullscreen or config.FULLSCREEN)

    # Initialize SQLite Database
    database.init_db()

    # Initialize Core Components
    detector = PPEDetector()
    associator = PPEAssociator()
    compliance_engine = ComplianceEngine()
    smoother = TemporalSmoother()

    # Open Physical Webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {args.camera}.")
        return

    window_name = "Real-Time Safety PPE Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    print(f"[INFO] Real-time PPE Detection running on physical webcam (index {args.camera})...")
    print("[INFO] Controls: Press 'Q' to quit | Press 'F' to toggle Fullscreen | Press 'D' to toggle DEBUG_MODE")

    # Video Writer Initialization
    video_writer = None
    if config.RECORD_VIDEO:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(config.VIDEOS_DIR, f"webcam_{timestamp_str}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps_out = 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        video_writer = cv2.VideoWriter(video_path, fourcc, fps_out, (width, height))
        print(f"[INFO] Recording video to {video_path}")

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Empty frame received from webcam. Retrying...")
                time.sleep(0.1)
                continue

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
            prev_time = curr_time

            # 1. Run YOLO Detection
            detections = detector.detect(frame)

            # 2. Run Spatial Association
            associations = associator.associate(detections, frame_shape=frame.shape)

            # 3. Evaluate Compliance & Apply 5-frame Temporal Smoothing
            smoothed_results = {}
            for assoc in associations:
                raw_result = compliance_engine.evaluate(assoc)
                smoothed_result, trigger_event = smoother.process(assoc.person_id, raw_result, current_time=curr_time)
                smoothed_results[assoc.person_id] = smoothed_result

                # Log detection records into DB
                database.log_detection(
                    person_id=assoc.person_id,
                    class_name="person",
                    confidence=assoc.person_detection.confidence,
                    bbox=assoc.person_detection.bbox
                )

                # Log violation event & save frame snapshot if triggered
                if trigger_event:
                    annotated_frame = draw_overlay(frame, associations, smoothed_results, debug_mode)
                    frame_path = database.save_violation_frame(annotated_frame, assoc.person_id)
                    database.log_compliance_event(
                        person_id=assoc.person_id,
                        status=smoothed_result.status,
                        has_glasses=int(smoothed_result.has_glasses),
                        has_shoes=int(smoothed_result.has_shoes),
                        feet_visible=int(smoothed_result.feet_visible),
                        missing_ppe=smoothed_result.missing_ppe,
                        frame_path=frame_path
                    )
                    print(f"[EVENT] Saved violation snapshot for {assoc.person_id} at {frame_path}")

            # 4. Draw Overlay & Diagnostic Info
            display_frame = draw_overlay(frame, associations, smoothed_results, debug_mode)

            # FPS Counter Header
            cv2.putText(display_frame, f"FPS: {fps:.1f} | Persons: {len(associations)} | Q: Quit | F: Fullscreen | D: Debug",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            if debug_mode:
                cv2.putText(display_frame, "DEBUG MODE ON", (display_frame.shape[1] - 180, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                print(f"[DEBUG] Raw YOLO Detections ({len(detections)}): " +
                      ", ".join([f"{d.class_name}({d.confidence:.2f})" for d in detections]))
                for assoc in associations:
                    g_logs = assoc.debug_info.get("glasses_logs", [])
                    print(f"[DEBUG] {assoc.person_id} Box={[round(x) for x in assoc.person_detection.bbox]} EyeROI={[round(x) for x in assoc.eye_roi]}")
                    for log in g_logs:
                        print(f"  └─ Glasses candidate conf={log['confidence']:.2f} -> Accepted={log['accepted']} ({log['reason']})")

            # 5. Write Frame if Video Recording is Enabled
            if video_writer is not None:
                video_writer.write(display_frame)

            # 6. Show OpenCV Window
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("[INFO] Clean exit requested by user (Q pressed).")
                break
            elif key == ord('f') or key == ord('F'):
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print(f"[INFO] Fullscreen mode set to {fullscreen}")
            elif key == ord('d') or key == ord('D'):
                debug_mode = not debug_mode
                print(f"[INFO] DEBUG_MODE set to {debug_mode}")

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam pipeline stopped cleanly.")

if __name__ == "__main__":
    main()
