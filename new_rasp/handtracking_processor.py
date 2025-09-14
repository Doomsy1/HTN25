import os, time, threading, urllib.request
import cv2
try:
    from .handtracking_utils import HAND_CONNECTIONS, render_hands  # type: ignore
except Exception:
    try:
        from new_rasp.handtracking_utils import HAND_CONNECTIONS, render_hands  # type: ignore
    except Exception:
        from handtracking_utils import HAND_CONNECTIONS, render_hands  # type: ignore

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    mp = None
    mp_python = None
    mp_vision = None


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

 

# Resolve optional helper from streamcam in both module and script contexts
_READ_FRAME_BGR = None
try:
    # When run as a package module: py -m new_rasp.run_handtracking_stream
    from new_rasp.streamcam import read_frame_bgr as _READ_FRAME_BGR  # type: ignore
except Exception:
    try:
        # When running the script directly: py new_rasp\\run_handtracking_stream.py
        from streamcam import read_frame_bgr as _READ_FRAME_BGR  # type: ignore
    except Exception:
        _READ_FRAME_BGR = None


def ensure_model(path=MODEL_PATH, url=MODEL_URL):
    if os.path.exists(path):
        return path
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
    except Exception as exc:
        raise RuntimeError(f"Failed to download model to {path}: {exc}")
    return path


class HandTrackingProcessor:
    def __init__(
        self,
        stream,
        model_path=None,
        num_hands=1,
        min_hand_detection_confidence=0.2,
        min_hand_presence_confidence=0.2,
        min_tracking_confidence=0.2,
        use_streamcam_flip=True,
    ):
        self.stream = stream
        self.model_path = model_path or MODEL_PATH
        self.num_hands = int(num_hands)
        self.min_hand_detection_confidence = float(min_hand_detection_confidence)
        self.min_hand_presence_confidence = float(min_hand_presence_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.use_streamcam_flip = bool(use_streamcam_flip)

        self._landmarker = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_result = None
        self._latest_frame = None

    def start(self):
        if self._thread is not None:
            return

        if mp is None or mp_python is None or mp_vision is None:
            raise RuntimeError("mediapipe is not installed. Install it with: pip install mediapipe opencv-contrib-python")

        model_file = ensure_model(self.model_path)
        # Use in-memory buffer to avoid Windows absolute-path resolution issues inside MediaPipe
        try:
            with open(model_file, "rb") as f:
                model_bytes = f.read()
        except Exception as exc:
            raise RuntimeError(f"Failed to read model file at {model_file}: {exc}")

        base_options = mp_python.BaseOptions(model_asset_buffer=model_bytes)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_hand_detection_confidence,
            min_hand_presence_confidence=self.min_hand_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _get_frame(self):
        frame_bgr = None
        if self.stream is None:
            return None
        try:
            if self.use_streamcam_flip and _READ_FRAME_BGR is not None:
                frame_bgr = _READ_FRAME_BGR(self.stream)
            else:
                # Directly use the latest frame from stream
                frame_bgr = self.stream.read_latest()
        except Exception:
            frame_bgr = None
        return frame_bgr

    def _run(self):
        while not self._stop.is_set():
            frame_bgr = self._get_frame()
            if frame_bgr is None:
                time.sleep(0.01)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(time.time() * 1000)

            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

            with self._lock:
                self._latest_result = result
                self._latest_frame = frame_bgr

    def read_latest_frame(self):
        with self._lock:
            return self._latest_frame

    def read_latest_result(self):
        with self._lock:
            return self._latest_result

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
            self._thread = None
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None
 


