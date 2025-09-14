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
    labels = []
    if result is None:
        return labels
    if getattr(result, "gestures", None):
        for hand_idx, categories in enumerate(result.gestures):
            if not categories:
                continue
            # categories is a list of Category objects
            best = max(categories, key=lambda c: getattr(c, "score", 0.0))
            label = getattr(best, "category_name", "?")
            score = float(getattr(best, "score", 0.0))
            labels.append((f"hand{hand_idx+1}-{label}", score))
    return labels


def _canonical_thumb_label(raw: str) -> str:
    s = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    if s in {"thumb_up", "thumbs_up", "thumbup", "thumbsup"}:
        return "thumbs_up"
    if s in {"thumb_down", "thumbs_down", "thumbdown", "thumbsdown"}:
        return "thumbs_down"
    return ""


def filter_model_labels(model_labels):
    # model_labels: list of ("handX-<label>", score)
    filtered = []
    for label, score in model_labels:
        # Extract the raw model label after the 'handX-' prefix
        raw = label.split("-", 1)[-1] if "-" in label else label
        canon = _canonical_thumb_label(raw)
        if canon:
            # Preserve hand prefix if present
            prefix = label.split("-", 1)[0] if "-" in label else "hand1"
            filtered.append((f"{prefix}-{canon}", score))
    return filtered


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
        # Pinch detection setup (MediaPipe Hands)
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
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

            # Pinch gesture: thumb tip (4) to index tip (8) distance
            pinch_labels = []
            index_only_labels = []
            try:
                hand_res = hands.process(frame_rgb)
                if hand_res and hand_res.multi_hand_landmarks:
                    h, w = frame_bgr.shape[:2]
                    for idx, hls in enumerate(hand_res.multi_hand_landmarks):
                        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hls.landmark]
                        x_vals = [p[0] for p in pts]
                        y_vals = [p[1] for p in pts]
                        bbox_w = max(1, max(x_vals) - min(x_vals))
                        bbox_h = max(1, max(y_vals) - min(y_vals))
                        scale = float(min(bbox_w, bbox_h))
                        thumb = pts[4]
                        index_tip = pts[8]
                        dx = float(thumb[0] - index_tip[0])
                        dy = float(thumb[1] - index_tip[1])
                        dist = (dx * dx + dy * dy) ** 0.5
                        norm = dist / scale
                        # Scored pinch: higher score when closer than pinch_hi, saturates at pinch_lo
                        pinch_lo = 0.13
                        pinch_hi = 0.25
                        pinch_score = max(0.0, min(1.0, (pinch_hi - float(norm)) / max(1e-6, (pinch_hi - pinch_lo))))
                        is_pinch = pinch_score >= 0.5
                        # Draw helpers
                        cv2.circle(frame_bgr, thumb, 5, (255, 0, 0), -1)
                        cv2.circle(frame_bgr, index_tip, 5, (0, 0, 255), -1)
                        cv2.line(frame_bgr, thumb, index_tip, (0, 255, 255), 2)
                        cv2.putText(
                            frame_bgr,
                            f"hand{idx+1}-pinch:{'Y' if is_pinch else 'N'} d={norm:.2f} s={pinch_score:.2f}",
                            (12, 100 + 24 * idx),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        if pinch_score > 0:
                            pinch_labels.append((f"hand{idx+1}-pinch", float(pinch_score)))

                        # Index-only-open (orientation invariant): index straight and long; others bent/short
                        wrist = pts[0]
                        def vec(a, b):
                            return (float(a[0]-b[0]), float(a[1]-b[1]))
                        def vlen(v):
                            return (v[0]*v[0] + v[1]*v[1]) ** 0.5
                        def norm_dist(a, b):
                            return vlen(vec(a, b)) / max(1.0, scale)

                        ids_tip = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
                        ids_pip = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
                        ids_mcp = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}

                        cos_thr = 0.86  # ~30 degrees alignment threshold
                        len_thr = 0.25  # normalized distance delta threshold

                        def extension_score(name: str) -> float:
                            tip = pts[ids_tip[name]]
                            pip = pts[ids_pip[name]]
                            mcp = pts[ids_mcp[name]]
                            v1 = vec(tip, pip)
                            v2 = vec(pip, mcp)
                            l1 = max(1e-6, vlen(v1))
                            l2 = max(1e-6, vlen(v2))
                            cosang = (v1[0]*v2[0] + v1[1]*v2[1]) / (l1*l2)
                            cos_term = max(0.0, min(1.0, (cosang - cos_thr) / max(1e-6, 1.0 - cos_thr)))
                            d_tip = norm_dist(tip, wrist)
                            d_pip = norm_dist(pip, wrist)
                            len_term = max(0.0, min(1.0, (d_tip - d_pip) / max(1e-6, len_thr)))
                            return float(max(0.0, min(1.0, 0.5*cos_term + 0.5*len_term)))

                        idx_ext = extension_score("index")
                        other_ext = [extension_score(n) for n in ["thumb", "middle", "ring", "pinky"]]
                        others_fold = [max(0.0, 1.0 - e) for e in other_ext]
                        index_only_score = float(max(0.0, min(1.0, idx_ext * min(others_fold) if others_fold else 0.0)))
                        is_index_only = index_only_score >= 0.5

                        cv2.putText(
                            frame_bgr,
                            f"hand{idx+1}-index_only:{'Y' if is_index_only else 'N'} s={index_only_score:.2f}",
                            (12, 100 + 24 * idx + 16),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 200, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        if index_only_score > 0:
                            index_only_labels.append((f"hand{idx+1}-index_only", float(index_only_score)))
            except Exception:
                pass

            # Keep only: thumbs_up, thumbs_down (from model) + custom pinch + index_only
            allowed_model = filter_model_labels(labels)
            final_labels = []
            if allowed_model:
                final_labels.extend(allowed_model)
            if pinch_labels:
                final_labels.extend(pinch_labels)
            if index_only_labels:
                final_labels.extend(index_only_labels)

            if not final_labels:
                final_labels = [("none", 1.0)]

            draw_overlay(frame_bgr, final_labels)
            printed = ", ".join([f"{l}:{s:.2f}" for l, s in final_labels])
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
        try:
            hands.close()
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