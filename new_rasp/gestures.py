import os
import time
import urllib.request
from typing import List, Tuple, Optional, Set

import cv2

try:
    import mediapipe as mp  # type: ignore
    from mediapipe.tasks import python as mp_python  # type: ignore
    from mediapipe.tasks.python import vision as mp_vision  # type: ignore
except Exception as exc:  # pragma: no cover - surfaced at runtime via RuntimeError
    raise RuntimeError(
        "mediapipe is not installed. Install with: pip install mediapipe opencv-contrib-python"
    ) from exc


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "gesture_recognizer.task")


def _ensure_model(path: str = MODEL_PATH, url: str = MODEL_URL) -> str:
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


def _canonical_thumb_label(raw: str) -> str:
    s = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    if s in {"thumb_up", "thumbs_up", "thumbup", "thumbsup"}:
        return "thumbs_up"
    if s in {"thumb_down", "thumbs_down", "thumbdown", "thumbsdown"}:
        return "thumbs_down"
    return ""


def _filter_model_labels(model_labels: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    filtered: List[Tuple[str, float]] = []
    for label, score in model_labels:
        raw = label.split("-", 1)[-1] if "-" in label else label
        canon = _canonical_thumb_label(raw)
        if canon:
            prefix = label.split("-", 1)[0] if "-" in label else "hand1"
            filtered.append((f"{prefix}-{canon}", float(score)))
    return filtered


class GesturesEngine:
    """
    Class-based gesture recognizer that fuses MediaPipe's GestureRecognizer with
    custom Hand landmarks heuristics to produce normalized gesture names:
    - thumb_up, thumb_down (from model)
    - pinch (custom)
    - point (custom, index-only extended)

    Usage:
        engine = GesturesEngine()
        labels, names = engine.recognize(frame_bgr)
        ...
        engine.close()
    """

    def __init__(
        self,
        *,
        min_confidence: float = 0.3,
        max_num_hands: int = 2,
    ) -> None:
        model_file = _ensure_model(MODEL_PATH, MODEL_URL)
        base_options = mp_python.BaseOptions(model_asset_path=model_file)
        options = mp_vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._recognizer = mp_vision.GestureRecognizer.create_from_options(options)

        mp_hands = mp.solutions.hands
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=0,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )

        # Tunables for pinch and index-only scoring (copied from ges.py)
        self._pinch_lo = 0.13
        self._pinch_hi = 0.25
        self._index_cos_thr = 0.86
        self._index_len_thr = 0.25

    def close(self) -> None:
        try:
            self._hands.close()
        except Exception:
            pass

    def _top_gesture_labels(self, result) -> List[Tuple[str, float]]:
        labels: List[Tuple[str, float]] = []
        if result is None:
            return labels
        if getattr(result, "gestures", None):
            for hand_idx, categories in enumerate(result.gestures):
                if not categories:
                    continue
                best = max(categories, key=lambda c: getattr(c, "score", 0.0))
                label = getattr(best, "category_name", "?")
                score = float(getattr(best, "score", 0.0))
                labels.append((f"hand{hand_idx+1}-{label}", score))
        return labels

    def _compute_custom_labels(self, frame_bgr) -> List[Tuple[str, float]]:
        labels: List[Tuple[str, float]] = []
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            hand_res = self._hands.process(frame_rgb)
            if hand_res and hand_res.multi_hand_landmarks:
                h, w = frame_bgr.shape[:2]
                for idx, hls in enumerate(hand_res.multi_hand_landmarks):
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hls.landmark]
                    x_vals = [p[0] for p in pts]
                    y_vals = [p[1] for p in pts]
                    bbox_w = max(1, max(x_vals) - min(x_vals))
                    bbox_h = max(1, max(y_vals) - min(y_vals))
                    scale = float(min(bbox_w, bbox_h))

                    # Pinch score
                    thumb = pts[4]
                    index_tip = pts[8]
                    dx = float(thumb[0] - index_tip[0])
                    dy = float(thumb[1] - index_tip[1])
                    dist = (dx * dx + dy * dy) ** 0.5
                    norm = dist / max(1.0, scale)
                    pinch_score = max(
                        0.0,
                        min(
                            1.0,
                            (self._pinch_hi - float(norm)) / max(1e-6, (self._pinch_hi - self._pinch_lo)),
                        ),
                    )
                    if pinch_score > 0:
                        labels.append((f"hand{idx+1}-pinch", float(pinch_score)))

                    # Index-only score (orientation invariant): index extended, others folded
                    wrist = pts[0]

                    def vec(a, b):
                        return (float(a[0] - b[0]), float(a[1] - b[1]))

                    def vlen(v):
                        return (v[0] * v[0] + v[1] * v[1]) ** 0.5

                    def norm_dist(a, b):
                        return vlen(vec(a, b)) / max(1.0, scale)

                    ids_tip = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
                    ids_pip = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
                    ids_mcp = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}

                    def extension_score(name: str) -> float:
                        tip = pts[ids_tip[name]]
                        pip = pts[ids_pip[name]]
                        mcp = pts[ids_mcp[name]]
                        v1 = vec(tip, pip)
                        v2 = vec(pip, mcp)
                        l1 = max(1e-6, vlen(v1))
                        l2 = max(1e-6, vlen(v2))
                        cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
                        cos_term = max(0.0, min(1.0, (cosang - self._index_cos_thr) / max(1e-6, 1.0 - self._index_cos_thr)))
                        d_tip = norm_dist(tip, wrist)
                        d_pip = norm_dist(pip, wrist)
                        len_term = max(0.0, min(1.0, (d_tip - d_pip) / max(1e-6, self._index_len_thr)))
                        return float(max(0.0, min(1.0, 0.5 * cos_term + 0.5 * len_term)))

                    idx_ext = extension_score("index")
                    other_ext = [extension_score(n) for n in ["thumb", "middle", "ring", "pinky"]]
                    others_fold = [max(0.0, 1.0 - e) for e in other_ext]
                    index_only_score = float(max(0.0, min(1.0, idx_ext * min(others_fold) if others_fold else 0.0)))
                    if index_only_score > 0:
                        labels.append((f"hand{idx+1}-index_only", float(index_only_score)))
        except Exception:
            pass
        return labels

    def recognize(self, frame_bgr) -> Tuple[List[Tuple[str, float]], Set[str]]:
        """
        Returns:
            labels: list of tuples ("handX-<label>", score)
            names: normalized set {"thumb_up", "thumb_down", "pinch", "point"}
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(time.time() * 1000)
        labels: List[Tuple[str, float]] = []
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._recognizer.recognize_for_video(mp_image, timestamp_ms)
            model_labels = self._top_gesture_labels(result)
            labels.extend(_filter_model_labels(model_labels))
        except Exception:
            pass

        labels.extend(self._compute_custom_labels(frame_bgr))

        names: Set[str] = set()
        for raw, _score in labels:
            base = raw.split("-", 1)[-1]
            if base == "thumbs_up":
                names.add("thumb_up")
            elif base == "thumbs_down":
                names.add("thumb_down")
            elif base == "pinch":
                names.add("pinch")
            elif base == "index_only":
                names.add("point")

        return labels, names


