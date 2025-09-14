import os
import sys
import json
import threading
import time
import cv2  # type: ignore

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, Tuple, List


# Ensure we can import modules from new_rasp/ by adding it to sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
NEW_RASP_DIR = os.path.join(HERE, "new_rasp")
PROJECTOR_JSON_PATH = os.path.join(NEW_RASP_DIR, "screen_corners_3d.json")
BALL_JSON_PATH = os.path.join(NEW_RASP_DIR, "ball_calibration.json")
if NEW_RASP_DIR not in sys.path:
    sys.path.insert(0, NEW_RASP_DIR)


# Import calibration entrypoint from existing code
try:
    from calibrate_projector_corners_stereo import calibrate_projector_corners_stereo  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Failed to import calibrator from new_rasp/") from exc

# Optional stream helpers to read RTSP frames for ball calibration
try:
    from streamcam import start_stream as _start_stream, read_frame_bgr as _read_frame_bgr  # type: ignore
except Exception:
    try:
        from new_rasp.streamcam import start_stream as _start_stream, read_frame_bgr as _read_frame_bgr  # type: ignore
    except Exception:
        _start_stream = None  # type: ignore
        _read_frame_bgr = None  # type: ignore


APP = FastAPI(title="HTN25 Stereo API", version="0.1.0")


# Calibration state
_calibration_lock = threading.Lock()
_is_calibrating = False
_last_calibration_time_iso = None
_last_ball_calibration_time_iso = None

# Cached calibration data (avoid reading files repeatedly)
_projector_calib_data = None
_ball_calib_data = None


def _try_load_json(path):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


# Load cached data at startup (best-effort)
_projector_calib_data = _try_load_json(PROJECTOR_JSON_PATH)
_ball_calib_data = _try_load_json(BALL_JSON_PATH)


class SingleBallStore:
    """Threaded single-ball buffer with velocity and expiry.

    - Holds only the latest ball sample.
    - If a new sample arrives within `vel_window_ms`, compute velocity.
    - Ball auto-expires after `expire_ms` and is cleared.
    - `get_and_clear()` pops and clears the current ball if not expired.
    """

    def __init__(self, expire_ms: int = 250, vel_window_ms: int = 150) -> None:
        self.expire_ms = int(expire_ms)
        self.vel_window_ms = int(vel_window_ms)
        self._lock = threading.Lock()
        self._current: Optional[Dict[str, Any]] = None
        self._last_monotonic: Optional[float] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._thread.start()

    def _watchdog_loop(self) -> None:
        # Periodically clear expired state so it doesn't linger
        while not self._stop.is_set():
            now_m = time.monotonic()
            with self._lock:
                if self._current is not None and self._last_monotonic is not None:
                    age_ms = (now_m - self._last_monotonic) * 1000.0
                    if age_ms > float(self.expire_ms):
                        self._current = None
                        self._last_monotonic = None
            self._stop.wait(0.02)  # 20ms cadence

    def stop(self) -> None:
        self._stop.set()
        try:
            self._thread.join(timeout=0.5)
        except Exception:
            pass

    def offer(self, position_m: Tuple[float, float, float], t_epoch: Optional[float] = None) -> None:
        now_epoch = float(time.time() if t_epoch is None else t_epoch)
        now_mono = time.monotonic()
        px, py, pz = float(position_m[0]), float(position_m[1]), float(position_m[2])

        with self._lock:
            vx = vy = vz = 0.0
            if self._current is not None and self._last_monotonic is not None:
                prev_pos = self._current.get("position_m")
                dt_ms = (now_mono - self._last_monotonic) * 1000.0
                if prev_pos is not None and dt_ms > 0.0 and dt_ms <= float(self.vel_window_ms):
                    vx = (px - float(prev_pos[0])) / (dt_ms / 1000.0)
                    vy = (py - float(prev_pos[1])) / (dt_ms / 1000.0)
                    vz = (pz - float(prev_pos[2])) / (dt_ms / 1000.0)

            self._current = {
                "t": now_epoch,
                "position_m": [px, py, pz],
                "velocity_mps": [vx, vy, vz],
            }
            self._last_monotonic = now_mono

    def get_and_clear(self) -> Optional[Dict[str, Any]]:
        now_mono = time.monotonic()
        with self._lock:
            if self._current is None or self._last_monotonic is None:
                return None
            age_ms = (now_mono - self._last_monotonic) * 1000.0
            if age_ms > float(self.expire_ms):
                # Expired; clear and return nothing
                self._current = None
                self._last_monotonic = None
                return None
            # Pop current sample
            event = self._current
            self._current = None
            self._last_monotonic = None
            return event


_ball_store = SingleBallStore(expire_ms=250, vel_window_ms=150)


def _run_calibration_background() -> None:
    global _is_calibrating, _last_calibration_time_iso, _projector_calib_data, _ball_calib_data
    try:
        out_data, out_path = calibrate_projector_corners_stereo(show=False)
        if out_data is not None:
            _last_calibration_time_iso = out_data.get("timestamp")
            _projector_calib_data = out_data
            # Also perform ball-tracker calibration and persist it for server use
            try:
                _calibrate_ball_tracker(out_data)
                # Refresh cache from disk in case writer normalizes
                _ball_calib_data = _try_load_json(BALL_JSON_PATH)
            except Exception:
                # Ball calibration is best-effort; ignore failures
                pass
    finally:
        # Mark as done
        with _calibration_lock:
            _is_calibrating = False


def _calibrate_ball_tracker(projector_calib: Dict[str, Any]) -> None:
    """
    Derive and persist ball-tracker calibration from the latest projector calibration.

    This stores simple per-camera brightness baselines and processing params so the
    server can perform more reliable ball tracking without an explicit API call.
    """
    global _last_ball_calibration_time_iso
    # Prefer sources from the projector calibration JSON
    sources: List[str] = projector_calib.get("sources", [])
    if not sources or len(sources) < 2 or _start_stream is None or _read_frame_bgr is None:
        # Not enough info to sample streams; persist defaults tied to projector calib
        data = {
            "timestamp": projector_calib.get("timestamp", time.time()),
            "sources": sources,
            "delta_thr": -20,
            "min_area": 400,
            "blur_k": 5,
            "close_k": 5,
            "calib_min": [0, 0],
        }
        _write_ball_calibration(data)
        _last_ball_calibration_time_iso = projector_calib.get("timestamp")
        return

    # Sample a few frames to compute darkest pixel baseline per camera
    try:
        stream1 = _start_stream(sources[0], width=None)
        stream2 = _start_stream(sources[1], width=None)
    except Exception:
        # If starting streams fails, persist defaults
        data = {
            "timestamp": projector_calib.get("timestamp", time.time()),
            "sources": sources,
            "delta_thr": -20,
            "min_area": 400,
            "blur_k": 5,
            "close_k": 5,
            "calib_min": [0, 0],
        }
        _write_ball_calibration(data)
        _last_ball_calibration_time_iso = projector_calib.get("timestamp")
        return

    try:
        mins = [255, 255]
        samples = 10
        for _ in range(samples):
            f1 = _read_frame_bgr(stream1)
            f2 = _read_frame_bgr(stream2)
            if f1 is not None:
                g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
                mins[0] = min(mins[0], int(g1.min()))
            if f2 is not None:
                g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
                mins[1] = min(mins[1], int(g2.min()))
            time.sleep(0.02)
        data = {
            "timestamp": projector_calib.get("timestamp", time.time()),
            "sources": sources,
            "delta_thr": -20,
            "min_area": 400,
            "blur_k": 5,
            "close_k": 5,
            "calib_min": mins,
        }
        _write_ball_calibration(data)
        _last_ball_calibration_time_iso = projector_calib.get("timestamp")
    finally:
        try:
            stream1.stop()
        except Exception:
            pass
        try:
            stream2.stop()
        except Exception:
            pass


def _write_ball_calibration(data: Dict[str, Any]) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "new_rasp", "ball_calibration.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[server] Saved ball calibration to {out_path}")
    except Exception as exc:
        print(f"[server] Failed to save ball calibration: {exc}")


@APP.post("/calibrate")
def start_calibration():
    global _is_calibrating
    with _calibration_lock:
        if _is_calibrating:
            raise HTTPException(status_code=409, detail="Calibration already in progress")
        _is_calibrating = True
    t = threading.Thread(target=_run_calibration_background, daemon=True)
    t.start()
    return JSONResponse(status_code=202, content={"started": True})


@APP.get("/is_calibrated")
def is_calibrated() -> bool:
    # Per spec: return True if calibrate() is NOT currently running
    with _calibration_lock:
        return not _is_calibrating


@APP.get("/get_corners")
def get_corners():
    # Serve cached projector calibration if available; fallback to file once
    global _projector_calib_data
    if _projector_calib_data is None:
        _projector_calib_data = _try_load_json(PROJECTOR_JSON_PATH)
    if _projector_calib_data is None:
        raise HTTPException(status_code=404, detail="Calibration not found")
    return _projector_calib_data


@APP.get("/get_ball")
def get_ball():
    event = _ball_store.get_and_clear()
    if event is None:
        return JSONResponse(status_code=204, content=None)
    return event


def offer_ball(position_m, t: Optional[float] = None) -> None:
    # Publish a new ball sample to the single-ball store.
    _ball_store.offer((float(position_m[0]), float(position_m[1]), float(position_m[2])), t_epoch=float(time.time() if t is None else t))


if __name__ == "__main__":
    # Run with: py server.py
    import uvicorn  # type: ignore

    uvicorn.run(
        "server:APP",
        host="127.0.0.1",
        port=8000,
        reload=False,
        factory=False,
    )


