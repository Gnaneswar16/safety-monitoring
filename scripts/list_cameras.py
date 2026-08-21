import cv2

def list_available_cameras(max_check=5):
    print("[INFO] Scanning connected cameras...")
    found = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  -> Camera Index {i}: Available ({w}x{h})")
                found.append((i, w, h))
            else:
                print(f"  -> Camera Index {i}: Opened but failed to capture frame")
            cap.release()
        else:
            print(f"  -> Camera Index {i}: Not available")
    return found

if __name__ == "__main__":
    cams = list_available_cameras()
    if cams:
        print(f"\n[SUMMARY] Found {len(cams)} camera(s).")
        print(f"To use USB camera (e.g. index 1), run: python main.py --camera 1")
    else:
        print("\n[WARNING] No active cameras detected.")
