import os
import sys
import threading
import time
from typing import Optional, Tuple

import cv2


def _here(*paths: str) -> str:
    return os.path.join(os.path.dirname(__file__), *paths)


def _generate_or_load_grid() -> str:
    try:
        # Prefer generating to ensure it matches current config
        try:
            from create_aruco_png import generate_aruco_corners_image  # when run from within new_rasp/
        except Exception:
            from new_rasp.create_aruco_png import generate_aruco_corners_image  # when run from repo root

        return generate_aruco_corners_image()
    except Exception:
        # Fallback to an existing image in the same directory
        fallback = _here("aruco_grid.png")
        if not os.path.exists(fallback):
            raise
        return fallback


def _run_calibration_in_thread(outcome: dict):
    # Run headless to avoid extra OpenCV windows; keep only the fullscreen grid visible
    try:
        from calibrate_projector_corners_stereo import calibrate_projector_corners_stereo  # when run from within new_rasp/
    except Exception:
        from new_rasp.calibrate_projector_corners_stereo import calibrate_projector_corners_stereo  # when run from repo root

    try:
        out_data, out_path = calibrate_projector_corners_stereo(show=False)
        outcome["data"] = out_data
        outcome["path"] = out_path
    except Exception as exc:
        outcome["error"] = exc


def _show_fullscreen(image_path: str, should_close: callable) -> None:
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    win = "Projector Grid"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        # Older OpenCV builds may not support this; best effort
        pass

    while True:
        cv2.imshow(win, img)
        key = cv2.waitKey(16) & 0xFF
        if key == ord('q'):
            break
        if should_close():
            break

    cv2.destroyWindow(win)


def main() -> int:
    # 1) Ensure grid exists
    grid_path = _generate_or_load_grid()

    # 2) Start calibration in background
    outcome: dict = {"data": None, "path": None, "error": None}
    th = threading.Thread(target=_run_calibration_in_thread, args=(outcome,), daemon=True)
    th.start()

    # 3) Keep grid fullscreen until calibration completes (or user presses 'q')
    _show_fullscreen(grid_path, should_close=lambda: not th.is_alive())

    # Ensure calibration thread is finished
    th.join(timeout=0.1)

    if outcome.get("error") is not None:
        print(f"Calibration failed: {outcome['error']}")
    else:
        print(f"Calibration complete. Saved: {outcome.get('path')}")

    # 4) Launch stereo projection demo using the new reusable entrypoint
    calib_path = _here("screen_corners_3d.json")
    if not os.path.exists(calib_path):
        print(f"Calibration file not found at {calib_path}. Exiting.")
        return 1

    try:
        from stereo_projection_demo import run_stereo_projection  # when run from within new_rasp/
    except Exception:
        from new_rasp.stereo_projection_demo import run_stereo_projection  # when run from repo root

    return run_stereo_projection(
        calib_path,
        show_debug_windows=True,
    )


if __name__ == "__main__":
    sys.exit(main())


