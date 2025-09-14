import time
import threading
import logging

from fastapi import FastAPI, HTTPException, Response, Body


# Minimal logger
_log = logging.getLogger("HTN25")
if not _log.handlers:
    _log.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
    _log.addHandler(_h)


APP = FastAPI(title="HTN25 Minimal Ball Server", version="0.2.0")


# Single in-memory event (pop-once)
_lock = threading.Lock()
_current_event = None  # dict or None


def _set_event(evt):
    global _current_event
    with _lock:
        _current_event = evt


def _pop_event():
    global _current_event
    with _lock:
        evt = _current_event
        _current_event = None
        return evt


@APP.get("/health")
def health():
    return {"ok": True}


@APP.get("/get_ball")
def get_ball():
    evt = _pop_event()
    if evt is None:
        return Response(status_code=204)
    return evt


@APP.post("/offer_ball")
def offer_ball(payload: dict = Body(...)):
    # Accept either projector pixel space or meters
    pos_px = payload.get("position_px")
    pos_m = payload.get("position_m") or payload.get("position") or payload.get("p")
    t = payload.get("t")

    ts = float(time.time() if t is None else t)

    if pos_px is not None:
        if not isinstance(pos_px, (list, tuple)) or len(pos_px) != 2:
            raise HTTPException(status_code=400, detail="position_px must be [x,y]")
        x = float(pos_px[0]); y = float(pos_px[1])
        evt = {"t": ts, "space": "projector_px", "position_px": [x, y]}
        _set_event(evt)
        _log.info(f"offer_ball px=({x:.1f},{y:.1f})")
        return {"ok": True}

    if pos_m is not None:
        if not isinstance(pos_m, (list, tuple)) or len(pos_m) != 3:
            raise HTTPException(status_code=400, detail="position_m must be [x,y,z]")
        px = float(pos_m[0]); py = float(pos_m[1]); pz = float(pos_m[2])
        evt = {"t": ts, "space": "meters", "position_m": [px, py, pz]}
        _set_event(evt)
        _log.info(f"offer_ball m=({px:.3f},{py:.3f},{pz:.3f})")
        return {"ok": True}

    raise HTTPException(status_code=400, detail="payload must include position_px or position_m")


if __name__ == "__main__":
    # Run with: py server.py
    import uvicorn  # type: ignore

    uvicorn.run(
        "server:APP",
        host="0.0.0.0",
        port=8000,
        reload=False,
        factory=False,
    )

import os
import sys
import json
import threading
import time
import logging
import cv2  # type: ignore
import logging

from fastapi import FastAPI, HTTPException, Response, Body
from fastapi.responses import JSONResponse


# Ensure we can import modules from new_rasp/ by adding it to sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
NEW_RASP_DIR = os.path.join(HERE, "new_rasp")
PROJECTOR_JSON_PATH = os.path.join(NEW_RASP_DIR, "screen_corners_3d.json")
BALL_JSON_PATH = os.path.join(NEW_RASP_DIR, "ball_calibration.json")
if NEW_RASP_DIR not in sys.path:
    sys.path.insert(0, NEW_RASP_DIR)


# Import calibration entrypoint and stream helpers from new_rasp
from calibrate_projector_corners_stereo import calibrate_projector_corners_stereo  # type: ignore
from streamcam import start_stream as _start_stream, read_frame_bgr as _read_frame_bgr  # type: ignore


APP = FastAPI(title="HTN25 Stereo API", version="0.1.0")


# Calibration state
_calibration_lock = threading.Lock()
_is_calibrating_projector = False
_is_calibrating_ball = False
_last_calibration_time_iso = None
_last_ball_calibration_time_iso = None

# Cached calibration data (avoid reading files repeatedly)
_projector_calib_data = None
_ball_calib_data = None


# Minimal logging setup
_logger = logging.getLogger("HTN25")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
    _logger.addHandler(_handler)


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

    def __init__(self, expire_ms=250, vel_window_ms=150):
        self.expire_ms = int(expire_ms)
        self.vel_window_ms = int(vel_window_ms)
        self._lock = threading.Lock()
        self._current = None
        self._last_monotonic = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._thread.start()

    def _watchdog_loop(self):
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

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=0.5)

    def offer(self, position_m, t_epoch=None, space=None, raw_px=None):
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
            if space is not None:
                self._current["space"] = str(space)
            if raw_px is not None and isinstance(raw_px, (list, tuple)) and len(raw_px) == 2:
                self._current["position_px"] = [float(raw_px[0]), float(raw_px[1])]
            self._last_monotonic = now_mono

    def get_and_clear(self):
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


def _run_projector_calibration_background():
    global _is_calibrating_projector, _last_calibration_time_iso, _projector_calib_data
    out_data, out_path = calibrate_projector_corners_stereo(show=False)
    if out_data is not None:
        _last_calibration_time_iso = out_data.get("timestamp")
        _projector_calib_data = out_data
    # Mark as done
    with _calibration_lock:
        _is_calibrating_projector = False


def _run_ball_calibration_background():
    global _is_calibrating_ball, _ball_calib_data
    proj = _projector_calib_data or _try_load_json(PROJECTOR_JSON_PATH)
    if proj is None:
        # Nothing to do; projector calibration missing
        with _calibration_lock:
            _is_calibrating_ball = False
        return
    _calibrate_ball_tracker(proj)
    _ball_calib_data = _try_load_json(BALL_JSON_PATH)
    with _calibration_lock:
        _is_calibrating_ball = False


def _calibrate_ball_tracker(projector_calib):
    """
    Derive and persist ball-tracker calibration from the latest projector calibration.

    This stores simple per-camera brightness baselines and processing params so the
    server can perform more reliable ball tracking without an explicit API call.
    """
    global _last_ball_calibration_time_iso
    # Prefer sources from the projector calibration JSON
    sources = projector_calib.get("sources", [])
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
    stream1 = _start_stream(sources[0], width=None)
    stream2 = _start_stream(sources[1], width=None)

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
    stream1.stop()
    stream2.stop()


def _write_ball_calibration(data):
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "new_rasp", "ball_calibration.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[server] Saved ball calibration to {out_path}")


@APP.post("/calibrate_projector")
def start_calibration_projector():
    global _is_calibrating_projector
    with _calibration_lock:
        if _is_calibrating_projector:
            raise HTTPException(status_code=409, detail="Projector calibration already in progress")
        _is_calibrating_projector = True
    t = threading.Thread(target=_run_projector_calibration_background, daemon=True)
    t.start()
    return JSONResponse(status_code=202, content={"started": True})


@APP.post("/calibrate_ball")
def start_calibration_ball():
    global _is_calibrating_ball
    with _calibration_lock:
        if _is_calibrating_ball:
            raise HTTPException(status_code=409, detail="Ball calibration already in progress")
        _is_calibrating_ball = True
    t = threading.Thread(target=_run_ball_calibration_background, daemon=True)
    t.start()
    return JSONResponse(status_code=202, content={"started": True})


@APP.get("/is_calibrated")
def is_calibrated():
    # True if neither projector nor ball calibration is currently running
    with _calibration_lock:
        return (not _is_calibrating_projector) and (not _is_calibrating_ball)


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
        # For 204, return an empty body to avoid Content-Length mismatch
        return Response(status_code=204)
    return event


def offer_ball(position_m=None, position_px=None, t=None):
    # Publish a new ball sample to the single-ball store.
    ts = float(time.time() if t is None else t)
    if position_px is not None:
        x = float(position_px[0]); y = float(position_px[1])
        logging.info(f"[server] ball positioned (proj px)=({x:.1f},{y:.1f}) t={ts:.3f}")
        _ball_store.offer((x, y, 0.0), t_epoch=ts, space="projector_px", raw_px=[x, y])
        try:
            _logger.info(f"ball position_px: x={x:.1f}, y={y:.1f}")
        except Exception:
            pass
        return
    if position_m is not None:
        px = float(position_m[0]); py = float(position_m[1]); pz = float(position_m[2])
        logging.info(f"[server] ball positioned pos=({px:.3f},{py:.3f},{pz:.3f}) t={ts:.3f}")
        _ball_store.offer((px, py, pz), t_epoch=ts, space="meters")
        try:
            _logger.info(f"ball position_m: x={px:.3f}, y={py:.3f}, z={pz:.3f}")
        except Exception:
            pass


@APP.post("/offer_ball")
def api_offer_ball(payload: dict = Body(...)):
    # Accepts JSON like {"position_px": [x,y]} or {"position_m": [x,y,z]}, plus optional "t"
    pos_px = payload.get("position_px")
    pos_m = payload.get("position_m") or payload.get("position") or payload.get("p")
    t = payload.get("t")
    if pos_px is not None:
        if not isinstance(pos_px, (list, tuple)) or len(pos_px) != 2:
            raise HTTPException(status_code=400, detail="position_px must be [x,y]")
        try:
            _logger.info(f"offer_ball POST received (px): x={float(pos_px[0]):.1f}, y={float(pos_px[1]):.1f}")
        except Exception:
            pass
        offer_ball(position_px=pos_px, t=t)
        return {"ok": True}
    if pos_m is not None:
        if not isinstance(pos_m, (list, tuple)) or len(pos_m) != 3:
            raise HTTPException(status_code=400, detail="position_m must be [x,y,z]")
        try:
            _logger.info(f"offer_ball POST received (m): x={float(pos_m[0]):.3f}, y={float(pos_m[1]):.3f}, z={float(pos_m[2]):.3f}")
        except Exception:
            pass
        offer_ball(position_m=pos_m, t=t)
        return {"ok": True}
    raise HTTPException(status_code=400, detail="payload must include position_px or position_m")


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


