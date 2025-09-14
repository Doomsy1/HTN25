import os
import sys
import json
import time
import math
import threading
from typing import Optional, Tuple

import numpy as np
import cv2
import ctypes
import urllib.request

# Reuse stream and handtracking utilities
try:
    from new_rasp.streamcam import start_camera
except Exception:
    from streamcam import start_camera  # type: ignore

try:
    from new_rasp.handtracking_processor import HandTrackingProcessor
    from new_rasp.handtracking_utils import render_hands, draw_hands_with_index
except Exception:
    from handtracking_processor import HandTrackingProcessor  # type: ignore
    from handtracking_utils import render_hands, draw_hands_with_index  # type: ignore

try:
    # Reuse drawing helper and import core stereo utils from central module
    from new_rasp.stereo_utils import (
        triangulate_index,
        recommended_hfov_deg,
        load_calibration,
        project_point_to_plane,
        signed_distance_to_plane,
        calc_plane_basis,
        solve_plane_uv,
        clamp01,
    )
except Exception:
    from stereo_utils import triangulate_index, recommended_hfov_deg, load_calibration, project_point_to_plane, signed_distance_to_plane, calc_plane_basis, solve_plane_uv, clamp01  # type: ignore

# Centralized gestures engine
try:
    from new_rasp.gestures import GesturesEngine  # type: ignore
except Exception:
    try:
        from gestures import GesturesEngine  # type: ignore
    except Exception:
        GesturesEngine = None  # type: ignore

## Legacy MediaPipe helpers removed in favor of centralized gestures engine

# =============================
# Configuration (all in one place)
#
# Edit these values to tune behavior. Each parameter is documented.
# =============================

# Display/visualization
# - SHOW_DEBUG_WINDOWS_DEFAULT: Whether to show per-camera annotated preview windows for debugging.
# - DEBUG_WINDOW_SIZE_PX: Size (width, height) of each debug window in pixels.
SHOW_DEBUG_WINDOWS_DEFAULT = True
DEBUG_WINDOW_SIZE_PX = (640, 360)

# Cursor smoothing (projector pixel space)
# - EMA_ALPHA: Exponential moving average factor for cursor smoothing.
#   0.0 = no smoothing (instant). Higher values weight new positions more (less smoothing).
EMA_ALPHA = 0.25

# Projector/desktop alignment
# - SCREEN_OFFSET_PX: OS desktop offset (x, y) for the projector top-left corner, in pixels.
#   Adjust if your projector output is not at (0, 0) on the Windows desktop.
SCREEN_OFFSET_PX = (0, 0)

# Depth-based interaction gate
# - PRESS_THRESHOLD_M: 3D distance to the screen plane considered "touch/near" for gating actions.
# - PRESS_DEBOUNCE_SEC: Time required inside threshold before considering it a press candidate.
# - RELEASE_DEBOUNCE_SEC: Time required outside threshold to release a press in non-gesture mode.
PRESS_THRESHOLD_M = 0.03
PRESS_DEBOUNCE_SEC = 0.050
RELEASE_DEBOUNCE_SEC = 0.100

# Point (index-only) click timing
# - POINT_DOWN_DEBOUNCE_SEC: Hold time with "point" inside threshold before click-down.
# - POINT_CLICK_DURATION_SEC: Duration to hold the mouse down to generate a click.
# - POINT_UP_COOLDOWN_SEC: Cooldown after a point click to avoid repeated clicks.
POINT_DOWN_DEBOUNCE_SEC = 0.050
POINT_CLICK_DURATION_SEC = 0.050
POINT_UP_COOLDOWN_SEC = 0.500

# Pinch (drag/hold) timing
# - PINCH_DOWN_DEBOUNCE_SEC: Hold time with "pinch" inside threshold before starting a drag.
# - PINCH_UP_DEBOUNCE_SEC: Time without pinch before releasing the drag.
PINCH_DOWN_DEBOUNCE_SEC = 0.050
PINCH_UP_DEBOUNCE_SEC = 0.200

# Scroll behavior (thumbs up/down)
# - SCROLL_DEBOUNCE_SEC: Minimum interval between scroll ticks.
# - SCROLL_TICK_WHEEL: Magnitude of one scroll event (Windows wheel delta units; 120 = one notch).
SCROLL_DEBOUNCE_SEC = 0.025
SCROLL_TICK_WHEEL = 20

# Lightweight mouse control using Windows ctypes (fallback to pynput if not on Windows)
IS_WINDOWS = os.name == "nt"

if IS_WINDOWS:
    user32 = ctypes.WinDLL("user32", use_last_error=True)

    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_WHEEL = 0x0800

    def mouse_move_to(x: int, y: int) -> None:
        user32.SetCursorPos(int(x), int(y))

    def mouse_left_down() -> None:
        user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

    def mouse_left_up() -> None:
        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def mouse_scroll(dy: int) -> None:
        # dy is in wheel ticks; Windows uses multiples of 120
        user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(dy), 0)

    def _disable_console_quick_edit() -> None:
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            STD_INPUT_HANDLE = -10
            h_stdin = kernel32.GetStdHandle(STD_INPUT_HANDLE)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(h_stdin, ctypes.byref(mode)):
                ENABLE_QUICK_EDIT_MODE = 0x0040
                ENABLE_EXTENDED_FLAGS = 0x0080
                new_mode = (mode.value | ENABLE_EXTENDED_FLAGS) & (~ENABLE_QUICK_EDIT_MODE)
                kernel32.SetConsoleMode(h_stdin, new_mode)
        except Exception:
            # Non-fatal; continue without disabling QuickEdit
            pass
else:
    try:
        from pynput.mouse import Button as _Btn, Controller as _Ctl  # type: ignore
    except Exception as exc:  # pragma: no cover - non-Windows minimal fallback
        raise RuntimeError("Mouse control requires Windows ctypes or pynput installed") from exc

    _mouse = _Ctl()

    def mouse_move_to(x: int, y: int) -> None:
        _mouse.position = (int(x), int(y))

    def mouse_left_down() -> None:
        _mouse.press(_Btn.left)

    def mouse_left_up() -> None:
        _mouse.release(_Btn.left)

    def mouse_scroll(dy: int) -> None:
        # Convert Windows-style ticks to lines: positive up, negative down
        _mouse.scroll(0, int(dy / 120))





def _create_gestures_engine():
    try:
        if GesturesEngine is not None:
            return GesturesEngine()
    except Exception:
        return None
    return None


def run_stereo_projection(
    calib_path: str,
    *,
    show_debug_windows: Optional[bool] = None,
    ema_alpha: Optional[float] = None,
    screen_offset_px: Optional[Tuple[int, int]] = None,
    press_threshold_m: Optional[float] = None,
    release_debounce_sec: Optional[float] = None,
    press_debounce_sec: Optional[float] = None,
) -> int:
    # Avoid console freeze when clicking inside PowerShell/Terminal on Windows
    if IS_WINDOWS:
        _disable_console_quick_edit()

    calib = load_calibration(calib_path)
    sources = calib["sources"]
    fx = calib["fx"]
    fy = calib["fy"]
    cx = calib["cx"]
    cy = calib["cy"]
    baseline_m = calib["baseline_m"]
    proj_corners_px = calib["projector_corners_px"]
    screen_corners_3d = calib["screen_corners_3d"]

    if len(sources) < 2:
        print("Expected two sources in calibration JSON; falling back to default rtsp URLs")
        sources = [
            "rtsp://10.37.111.247:8554/cam1",
            "rtsp://10.37.111.247:8554/cam2",
        ]

    # Prepare plane and basis
    U, V, N = calc_plane_basis(screen_corners_3d)
    P0 = screen_corners_3d[0]
    # Precompute homography from plane UV to projector pixels
    plane_quad = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    H_plane_to_proj = cv2.getPerspectiveTransform(plane_quad, proj_corners_px.astype(np.float32))

    # Initialize gestures engine (only once) to run on camera 1 frames
    gesture_engine = _create_gestures_engine()
    gesture_mode = gesture_engine is not None

    # Start streams and processors
    cams = [start_camera(src) for src in sources[:2]]
    processors = [HandTrackingProcessor(cam) for cam in cams]
    for p in processors:
        p.start()

    windows = [f"Stereo Cam{i+1}" for i in range(len(processors))]
    # Resolve effective settings (fall back to top-level config if None was provided)
    effective_show_debug = SHOW_DEBUG_WINDOWS_DEFAULT if show_debug_windows is None else bool(show_debug_windows)
    effective_win_w, effective_win_h = DEBUG_WINDOW_SIZE_PX
    effective_ema_alpha = EMA_ALPHA if ema_alpha is None else float(ema_alpha)
    effective_screen_offset = SCREEN_OFFSET_PX if screen_offset_px is None else tuple(screen_offset_px)
    effective_press_threshold_m = PRESS_THRESHOLD_M if press_threshold_m is None else float(press_threshold_m)
    effective_release_debounce = RELEASE_DEBOUNCE_SEC if release_debounce_sec is None else float(release_debounce_sec)
    effective_press_debounce = PRESS_DEBOUNCE_SEC if press_debounce_sec is None else float(press_debounce_sec)

    if effective_show_debug:
        for title in windows:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, effective_win_w, effective_win_h)

    smoothed_xy: Optional[Tuple[float, float]] = None
    is_mouse_down: bool = False
    release_candidate_since: Optional[float] = None
    press_candidate_since: Optional[float] = None
    # Gesture timing specs (from configuration)
    point_down_debounce = POINT_DOWN_DEBOUNCE_SEC
    point_up_cooldown = POINT_UP_COOLDOWN_SEC
    point_click_duration = POINT_CLICK_DURATION_SEC
    pinch_down_debounce = PINCH_DOWN_DEBOUNCE_SEC
    pinch_up_debounce = PINCH_UP_DEBOUNCE_SEC

    # Gesture state
    point_candidate_since: Optional[float] = None
    point_clicking: bool = False
    point_click_down_time: Optional[float] = None
    gesture_cooldown_until: float = 0.0
    pinch_candidate_since: Optional[float] = None
    pinch_release_candidate_since: Optional[float] = None
    pinch_active: bool = False
    # Scroll behavior
    scroll_debounce = SCROLL_DEBOUNCE_SEC
    scroll_tick = SCROLL_TICK_WHEEL  # smaller than 120 for slower scrolling
    last_scroll_time: float = 0.0

    try:
        while True:
            now = time.monotonic()
            frames = [p.read_latest_frame() for p in processors]
            results = [p.read_latest_result() for p in processors]

            if any(f is None for f in frames):
                time.sleep(0.005)
                continue

            # Run gesture recognition on the first camera only
            gesture_text = "none"
            current_gestures: set = set()
            if gesture_engine is not None and frames and frames[0] is not None:
                try:
                    labels, names = gesture_engine.recognize(frames[0])
                    if labels:
                        gesture_text = ", ".join([f"{l}:{s:.2f}" for l, s in labels])
                    # Choose highest-confidence gesture per frame, with tie-break priority
                    # priority: pinch > point > thumb_up > thumb_down
                    priority = {"pinch": 0, "point": 1, "thumb_up": 2, "thumb_down": 3}
                    scores = {}
                    for raw, sc in labels:
                        base = raw.split("-", 1)[-1]
                        if base == "thumbs_up":
                            key = "thumb_up"
                        elif base == "thumbs_down":
                            key = "thumb_down"
                        elif base == "pinch":
                            key = "pinch"
                        elif base == "index_only":
                            key = "point"
                        else:
                            continue
                        prev = scores.get(key)
                        if (prev is None) or (sc > prev):
                            scores[key] = float(sc)
                    chosen_gesture = None
                    if scores:
                        # max by score, then by priority
                        chosen_gesture = max(
                            scores.items(),
                            key=lambda kv: (kv[1], -priority.get(kv[0], 9999)),
                        )[0]
                    current_gestures = set([chosen_gesture]) if chosen_gesture else set()
                except Exception:
                    pass

            # Draw and get index tip per view
            index_pts = []
            for idx, (frame, result) in enumerate(zip(frames, results)):
                annotated = frame.copy()
                idx_pt = draw_hands_with_index(annotated, result)
                index_pts.append(idx_pt)
                if effective_show_debug:
                    cv2.imshow(windows[idx], annotated)

            # Need both views to triangulate
            had_valid_point = False
            # Defaults for logging when no valid 3D point
            X = Y = Z = float("nan")
            dist_m = float("nan")
            a = b = float("nan")
            mx = my = -1
            if len(index_pts) >= 2 and index_pts[0] is not None and index_pts[1] is not None:
                (uL, vL) = index_pts[0]
                (uR, vR) = index_pts[1]

                X, Y, Z = triangulate_index(
                    float(uL), float(vL), float(uR), float(vR),
                    fx, fy, cx, cy, baseline_m
                )

                if not (math.isfinite(X) and math.isfinite(Y) and math.isfinite(Z)):
                    # Skip invalid triangulations
                    pass
                else:
                    had_valid_point = True
                    P = np.array([X, Y, Z], dtype=np.float32)
                    # Distance to screen plane (absolute)
                    dist_m = abs(signed_distance_to_plane(P, P0, N))
                    # Project to screen plane and solve UV
                    P_proj = project_point_to_plane(P, P0, N)
                    a, b = solve_plane_uv(P_proj, P0, U, V)

                    # Optionally clamp within screen for cursor confinement
                    a_clamped = clamp01(a)
                    b_clamped = clamp01(b)
                    src_pt = np.array([[[a_clamped, b_clamped]]], dtype=np.float32)
                    dst_pt = cv2.perspectiveTransform(src_pt, H_plane_to_proj)
                    target_x, target_y = float(dst_pt[0, 0, 0]), float(dst_pt[0, 0, 1])

                    # Smooth
                    if smoothed_xy is None:
                        smoothed_xy = (target_x, target_y)
                    else:
                        sx, sy = smoothed_xy
                        smoothed_xy = (
                            sx * (1.0 - effective_ema_alpha) + target_x * effective_ema_alpha,
                            sy * (1.0 - effective_ema_alpha) + target_y * effective_ema_alpha,
                        )

                    # Move mouse (apply desktop offset for projector)
                    mx = int(smoothed_xy[0] + effective_screen_offset[0])
                    my = int(smoothed_xy[1] + effective_screen_offset[1])
                    mouse_move_to(mx, my)

                    # Default distance-based click logic only if gesture mode is disabled
                    if not gesture_mode:
                        if dist_m <= effective_press_threshold_m:
                            if not is_mouse_down:
                                if press_candidate_since is None:
                                    press_candidate_since = now
                                elif (now - press_candidate_since) >= effective_press_debounce:
                                    try:
                                        mouse_left_down()
                                        is_mouse_down = True
                                    except Exception:
                                        pass
                            release_candidate_since = None
                        else:
                            if is_mouse_down:
                                if release_candidate_since is None:
                                    release_candidate_since = now
                                elif (now - release_candidate_since) >= effective_release_debounce:
                                    try:
                                        mouse_left_up()
                                        is_mouse_down = False
                                    except Exception:
                                        pass
                                    release_candidate_since = None
                            press_candidate_since = None

            # If no valid point this frame, handle potential release with debounce (only for non-gesture mode)
            if not gesture_mode:
                if not had_valid_point and is_mouse_down:
                    if release_candidate_since is None:
                        release_candidate_since = now
                    elif (now - release_candidate_since) >= effective_release_debounce:
                        try:
                            mouse_left_up()
                            is_mouse_down = False
                        except Exception:
                            pass
                        release_candidate_since = None
                if not had_valid_point:
                    press_candidate_since = None

            # Gesture-driven actions (point, palm, thumbs up/down)
            action = "none"
            if gesture_mode:
                in_cooldown = now < gesture_cooldown_until

                # POINT click: enter threshold with point gesture -> 50ms debounce -> click 25ms -> cooldown 250ms
                point_condition = ("point" in current_gestures) and had_valid_point and math.isfinite(dist_m) and (dist_m <= effective_press_threshold_m)
                if not in_cooldown and not pinch_active and not point_clicking and point_condition:
                    if point_candidate_since is None:
                        point_candidate_since = now
                    elif (now - point_candidate_since) >= point_down_debounce:
                        try:
                            mouse_left_down()
                            is_mouse_down = True
                        except Exception:
                            pass
                        point_clicking = True
                        point_click_down_time = now
                        action = "point_click_down"
                elif not point_condition and not point_clicking:
                    point_candidate_since = None

                if (not pinch_active) and point_clicking and point_click_down_time is not None:
                    if (now - point_click_down_time) >= point_click_duration:
                        try:
                            mouse_left_up()
                            is_mouse_down = False
                        except Exception:
                            pass
                        point_clicking = False
                        point_candidate_since = None
                        point_click_down_time = None
                        gesture_cooldown_until = now + point_up_cooldown
                        action = "point_click_up"

                # PINCH hold: start after 50ms debounce inside threshold; continue hold even if exit threshold while pinching; release when pinch absent for 250ms
                pinch_condition = ("pinch" in current_gestures) and had_valid_point and math.isfinite(dist_m) and (dist_m <= effective_press_threshold_m)
                if not pinch_active:
                    if pinch_condition:
                        if pinch_candidate_since is None:
                            pinch_candidate_since = now
                        elif (now - pinch_candidate_since) >= pinch_down_debounce:
                            try:
                                mouse_left_down()
                                is_mouse_down = True
                            except Exception:
                                pass
                            pinch_active = True
                            # Cancel any pending point click state to avoid override
                            point_clicking = False
                            point_candidate_since = None
                            point_click_down_time = None
                            pinch_release_candidate_since = None
                            action = "pinch_hold_start"
                    else:
                        pinch_candidate_since = None
                else:
                    if pinch_active:
                        if "pinch" in current_gestures:
                            pinch_release_candidate_since = None
                        else:
                            if pinch_release_candidate_since is None:
                                pinch_release_candidate_since = now
                            elif (now - pinch_release_candidate_since) >= pinch_up_debounce:
                                try:
                                    mouse_left_up()
                                    is_mouse_down = False
                                except Exception:
                                    pass
                                pinch_active = False
                                pinch_candidate_since = None
                                pinch_release_candidate_since = None
                                action = "pinch_hold_end"

                # THUMBS scroll: no debounce, no threshold; skip if holding palm/point-clicking or in cooldown
                if (not pinch_active) and (not point_clicking) and (not in_cooldown):
                    can_scroll = (now - last_scroll_time) >= scroll_debounce
                    if can_scroll and ("thumb_up" in current_gestures):
                        try:
                            mouse_scroll(+scroll_tick)
                            last_scroll_time = now
                            action = "scroll_up"
                        except Exception:
                            pass
                    elif can_scroll and ("thumb_down" in current_gestures):
                        try:
                            mouse_scroll(-scroll_tick)
                            last_scroll_time = now
                            action = "scroll_down"
                        except Exception:
                            pass

            # Print exactly one line per frame, including action summary
            if had_valid_point:
                print(
                    f"gesture={gesture_text}  action={action}  3D(m)=({X:.3f},{Y:.3f},{Z:.3f})  dist={dist_m:.3f}m  uv=({a:.3f},{b:.3f})  mouse=({mx},{my})  down={is_mouse_down}"
                )
            else:
                print(
                    f"gesture={gesture_text}  action={action}  3D(m)=(N/A,N/A,N/A)  dist=N/A  uv=(N/A,N/A)  mouse=(N/A,N/A)  down={is_mouse_down}"
                )

            if effective_show_debug:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        for p in processors:
            p.stop()
        for c in cams:
            try:
                c.stop()
            except Exception:
                pass
        # Ensure mouse is released on exit
        try:
            if 'is_mouse_down' in locals() and is_mouse_down:
                mouse_left_up()
        except Exception:
            pass
        if show_debug_windows:
            cv2.destroyAllWindows()

    return 0


def main() -> int:
    # Fixed defaults (CLI modifiers removed). Use top-level configuration values.
    calib_path = os.path.join(os.path.dirname(__file__), "screen_corners_3d.json")
    return run_stereo_projection(
        calib_path,
        show_debug_windows=SHOW_DEBUG_WINDOWS_DEFAULT,
        ema_alpha=EMA_ALPHA,
        screen_offset_px=SCREEN_OFFSET_PX,
        press_threshold_m=PRESS_THRESHOLD_M,
        release_debounce_sec=RELEASE_DEBOUNCE_SEC,
        press_debounce_sec=PRESS_DEBOUNCE_SEC,
    )


if __name__ == "__main__":
    sys.exit(main())

 