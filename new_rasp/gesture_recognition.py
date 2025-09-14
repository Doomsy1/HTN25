import os
import sys
import time
import urllib.request
import cv2

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    print("mediapipe is not installed. Install with: pip install mediapipe opencv-contrib-python")
    sys.exit(1)

# Stream helpers (used for URL sources)
try:
    from streamcam import start_camera, read_frame_bgr
except Exception:
    start_camera = None
    read_frame_bgr = None

# Model download info
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "gesture_recognizer.task")


def ensure_model(path=MODEL_PATH, url=MODEL_URL):
    if os.path.exists(path):
        return path
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading Gesture Recognizer model to {path} ...")
        urllib.request.urlretrieve(url, path)
    except Exception as exc:
        print(f"Failed to download model to {path}: {exc}")
        sys.exit(1)
    return path


def is_url(s: str) -> bool:
    return "://" in s


def build_input_source(argv):
    # Prioritize webcam by default.
    # Forms:
    # - no args -> webcam index 0
    # - "webcam" or integer -> webcam
    # - full URL or ip port -> URL
    if not argv:
        return ("webcam", 0)
    if len(argv) == 1:
        token = argv[0].strip()
        if token.lower() == "webcam":
            return ("webcam", 0)
        if token.isdigit():
            return ("webcam", int(token))
        if is_url(token):
            return ("url", token)
        # Fallback: treat as URL host expecting default RTSP port path
        return ("url", token)
    if len(argv) >= 2:
        ip = argv[0].strip()
        port = argv[1].strip()
        return ("url", f"rtsp://{ip}:{port}")


def draw_overlay(frame_bgr, top_gestures):
    # top_gestures: list of tuples (label, score)
    y = 24
    for label, score in top_gestures:
        txt = f"{label}: {score:.2f}"
        cv2.putText(frame_bgr, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        y += 24


def top_gesture_labels(result):
    """
    Return all available gesture categories with scores for each detected hand.

    The MediaPipe Gesture Recognizer returns a list of Category objects per hand.
    This function aggregates all categories (not just the top one) for each hand,
    sorted by descending score.
    """
    labels = []
    if result is None:
        return labels
    if getattr(result, "gestures", None):
        for hand_idx, categories in enumerate(result.gestures):
            if not categories:
                continue
            # Sort categories by score (desc) and include them all
            sorted_categories = sorted(
                categories, key=lambda c: getattr(c, "score", 0.0), reverse=True
            )
            for cat in sorted_categories:
                label = getattr(cat, "category_name", "?")
                score = float(getattr(cat, "score", 0.0))
                labels.append((f"hand{hand_idx+1}-{label}", score))
    return labels


def open_webcam(index=0):
    # Prefer DSHOW on Windows for stability; fallback to default backend
    cap = cv2.VideoCapture(int(index), cv2.CAP_DSHOW)
    if not cap or not cap.isOpened():
        cap = cv2.VideoCapture(int(index))
    # Optional: set a sane resolution
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    except Exception:
        pass
    return cap


def main(argv=None):
    argv = argv or sys.argv[1:]
    source_kind, source_value = build_input_source(argv)

    model_file = ensure_model(MODEL_PATH, MODEL_URL)

    base_options = mp_python.BaseOptions(model_asset_path=model_file)
    options = mp_vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    recognizer = mp_vision.GestureRecognizer.create_from_options(options)

    window_name = "Gestures"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280 // 2, 720 // 2)

    # Select frame source
    cap = None
    stream = None
    if source_kind == "webcam":
        cap = open_webcam(source_value)
        if not cap or not cap.isOpened():
            print(f"Failed to open webcam at index {source_value}")
            return 1
    else:
        if start_camera is None or read_frame_bgr is None:
            print("Stream helpers unavailable and non-webcam source provided.")
            return 1
        stream = start_camera(source_value)

    try:
        while True:
            # Get frame
            frame_bgr = None
            if cap is not None:
                ok, frame_bgr = cap.read()
                if not ok:
                    time.sleep(0.01)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    continue
            else:
                frame_bgr = read_frame_bgr(stream)
                if frame_bgr is None:
                    time.sleep(0.01)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(time.time() * 1000)

            result = recognizer.recognize_for_video(mp_image, timestamp_ms)
            labels = top_gesture_labels(result)

            if labels:
                draw_overlay(frame_bgr, labels)
                printed = ", ".join([f"{l}:{s:.2f}" for l, s in labels])
                print(f"gestures: {printed}")

            cv2.imshow(window_name, frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
