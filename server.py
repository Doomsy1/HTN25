import os
import sys
import json
import threading
import time
from typing import Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse


# Ensure we can import modules from new_rasp/ by adding it to sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
NEW_RASP_DIR = os.path.join(HERE, "new_rasp")
if NEW_RASP_DIR not in sys.path:
    sys.path.insert(0, NEW_RASP_DIR)


# Import calibration entrypoint from existing code
try:
    from calibrate_projector_corners_stereo import calibrate_projector_corners_stereo  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Failed to import calibrator from new_rasp/") from exc


APP = FastAPI(title="HTN25 Stereo API", version="0.1.0")


# Calibration state
_calibration_lock = threading.Lock()
_is_calibrating = False
_last_calibration_time_iso: Optional[str] = None


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
    global _is_calibrating, _last_calibration_time_iso
    try:
        out_data, out_path = calibrate_projector_corners_stereo(show=False)
        if out_data is not None:
            _last_calibration_time_iso = out_data.get("timestamp")
    finally:
        # Mark as done
        with _calibration_lock:
            _is_calibrating = False


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
    # Read the latest calibration JSON written by the calibrator
    json_path = os.path.join(NEW_RASP_DIR, "screen_corners_3d.json")
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Calibration not found")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read calibration JSON: {exc}")


@APP.get("/get_ball")
def get_ball():
    event = _ball_store.get_and_clear()
    if event is None:
        return JSONResponse(status_code=204, content=None)
    return event


def offer_ball(position_m, t: Optional[float] = None) -> None:
    """Publish a new ball sample to the single-ball store.

    position_m: (x, y, z) in meters
    t: epoch seconds or None (defaults to time.time())
    Velocity is computed automatically if the sample arrives soon after the last one.
    """
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


