import os
import json
from collections import deque
import time

import cv2
import numpy as np

# Stream helpers
from streamcam import start_stream, read_frame_bgr  # type: ignore

# Stereo utilities for triangulation and calibration
from stereo_utils import triangulate_index, load_calibration, project_point_to_plane, signed_distance_to_plane, calc_plane_basis, solve_plane_uv, clamp01  # type: ignore


CALIB_MIN_BY_CAM = {}
CALIB_MSG_UNTIL_BY_CAM = {}


def resize_with_aspect(image, width):
    if width is None:
        return image
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return image
    scale = width / float(w)
    return cv2.resize(image, (int(width), int(h * scale)), interpolation=cv2.INTER_LINEAR)


def compute_fps(prev_time, frame_count, interval_s=1.0):
    now = time.time()
    elapsed = now - prev_time
    if elapsed >= interval_s:
        fps = frame_count / elapsed
        return now, fps, 0
    return prev_time, -1.0, frame_count


def calibrate_darkest(cam_id, gray_frame):
    darkest = int(np.min(gray_frame))
    CALIB_MIN_BY_CAM[cam_id] = darkest
    CALIB_MSG_UNTIL_BY_CAM[cam_id] = time.time() + 1.5
    print(f"[Calibrated cam{cam_id}] darkest={darkest}")


def derive_hfov_from_intrinsics(fx, image_width):
    # Horizontal FOV in radians from fx and image width
    return 2.0 * float(np.arctan((image_width * 0.5) / float(fx)))


def main():
    # Fixed defaults (no CLI args)
    calib_path = os.path.join(os.path.dirname(__file__), "screen_corners_3d.json")
    width = 960
    buffer_len = 64
    delta_thr = -20
    min_area = 400
    blur_k = 5
    close_k = 5
    show_mask = False
    flip_view = False
    ball_diam_cm = 6.5
    size_max_ratio = 2.5
    match_row_px = 25

    # Server publish config (optional)
    SERVER_OFFER_URL = os.environ.get("HTN25_BALL_URL", "http://127.0.0.1:8000/offer_ball")
    SEND_MIN_INTERVAL_S = 0.05  # at most 20 Hz
    last_sent_mono = 0.0
    last_post_ok = None
    last_post_ts = 0.0

    def _post_json(url, payload):
        # Minimal stdlib JSON POST to avoid extra deps
        try:
            import urllib.request
        except Exception:
            return False
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                code = getattr(resp, 'status', 200)
                return 200 <= int(code) < 300
        except Exception as exc:
            try:
                print(f"[POST FAIL] {exc}")
            except Exception:
                pass
            return False

    # Load calibration (intrinsics, baseline, default sources)
    calib = load_calibration(calib_path)
    sources = calib.get("sources", [])
    fx = float(calib["fx"])  # pixels
    fy = float(calib["fy"])  # pixels
    cx = float(calib["cx"])  # pixels
    cy = float(calib["cy"])  # pixels
    baseline_m = float(calib["baseline_m"])  # meters
    image_size = calib.get("image_size", None)
    proj_corners_px = np.array(calib.get("projector_corners_px", [[0.0,0.0],[1280.0,0.0],[1280.0,720.0],[0.0,720.0]]), dtype=np.float32)
    screen_corners_3d = np.array(calib.get("screen_corners_3d", calib.get("screen_corners_3d_m", [])), dtype=np.float32)

    # Select sources: from calibration, fallback to defaults
    src1 = sources[0] if len(sources) >= 1 else None
    src2 = sources[1] if len(sources) >= 2 else None
    if not src1 or not src2:
        # Final fallback to project defaults
        src1 = src1 or "rtsp://10.37.111.247:8554/cam1"
        src2 = src2 or "rtsp://10.37.111.247:8554/cam2"

    if not str(src1).lower().startswith("rtsp://") or not str(src2).lower().startswith("rtsp://"):
        print("[ERROR] Please supply RTSP URLs (via calibration JSON)")
        return

    print("[INFO] Using RTSP sources:")
    print("  cam1:", src1)
    print("  cam2:", src2)

    stream1 = start_stream(src1, width=None)
    stream2 = start_stream(src2, width=None)

    points1 = deque(maxlen=buffer_len)

    if show_mask:
        cv2.namedWindow("Mask 1", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Mask 1", 480, 360)
        cv2.namedWindow("Mask 2", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Mask 2", 480, 360)

    cv2.namedWindow("Ball Tracker 1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Ball Tracker 1", max(320, width//2), int(max(320, width//2) * 9 / 16))
    cv2.namedWindow("Ball Tracker 2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Ball Tracker 2", max(320, width//2), int(max(320, width//2) * 9 / 16))

    prev_time = time.time()
    fps_value = -1.0
    frame_counter = 0
    wait1 = 0.0
    wait2 = 0.0

    # Determine calibration image size for pixel mapping and projector transform
    calib_w = None
    calib_h = None
    if image_size is not None:
        calib_w = float(image_size[0])
        calib_h = float(image_size[1])

    # Precompute plane basis and homography to projector pixel space
    H_plane_to_proj = None
    U = V = N = None
    P0 = None
    try:
        if screen_corners_3d is not None and len(screen_corners_3d) >= 4 and proj_corners_px is not None and proj_corners_px.shape == (4, 2):
            U, V, N = calc_plane_basis(screen_corners_3d)
            P0 = screen_corners_3d[0]
            plane_quad = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
            H_plane_to_proj = cv2.getPerspectiveTransform(plane_quad, proj_corners_px.astype(np.float32))
    except Exception:
        H_plane_to_proj = None

    def process_one(cam_id, frame):
        if flip_view:
            frame = cv2.flip(frame, 1)
        orig_h, orig_w = frame.shape[:2]
        frame_res = resize_with_aspect(frame, width)
        res_h, res_w = frame_res.shape[:2]

        # Grayscale and blur
        if blur_k <= 1:
            blurred = frame_res
        else:
            k = int(blur_k)
            if k % 2 == 0:
                k += 1
            blurred = cv2.GaussianBlur(frame_res, (k, k), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # Initialize baseline on first frame if missing
        if cam_id not in CALIB_MIN_BY_CAM:
            calibrate_darkest(cam_id, gray)

        # Threshold: pixels darker than (baseline + delta) using inverse binary
        base = int(CALIB_MIN_BY_CAM.get(cam_id, 0))
        thr = min(255, base + int(delta_thr))
        _ret, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)

        # Morphological cleanup
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)
        if close_k >= 3:
            ck = int(close_k)
            if ck % 2 == 0:
                ck += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Contour detection and selection
        contours, _hier = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        if len(contours) > 0:
            for c in contours:
                if cv2.contourArea(c) < float(min_area):
                    continue
                (x, y), radius = cv2.minEnclosingCircle(c)
                if radius <= 0:
                    continue
                m = cv2.moments(c)
                if m["m00"] <= 0:
                    continue
                center = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
                contour_mask = np.zeros((res_h, res_w), dtype=np.uint8)
                cv2.drawContours(contour_mask, [c], -1, 255, -1)
                mean_val = float(cv2.mean(gray, mask=contour_mask)[0])
                detections.append({
                    "center": center,
                    "radius": float(radius),
                    "mean": mean_val,
                    "orig_size": (orig_w, orig_h),
                    "res_size": (res_w, res_h),
                })

        # Overlays
        if fps_value >= 0:
            cv2.putText(frame_res, f"FPS:{fps_value:.1f} thr:{thr}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        # Show last post status
        if last_post_ok is not None:
            status_txt = "POST:OK" if last_post_ok else "POST:FAIL"
            color = (60, 220, 60) if last_post_ok else (40, 40, 220)
            cv2.putText(frame_res, status_txt, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        until = CALIB_MSG_UNTIL_BY_CAM.get(cam_id, 0.0)
        if time.time() < until:
            cv2.putText(frame_res, "Calibrated", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        return frame_res, mask, detections

    while True:
        f1 = read_frame_bgr(stream1)
        f2 = read_frame_bgr(stream2)
        if f1 is None:
            now = time.time()
            if now - wait1 > 2.0:
                print("[INFO] Waiting for cam1...")
                wait1 = now
        if f2 is None:
            now = time.time()
            if now - wait2 > 2.0:
                print("[INFO] Waiting for cam2...")
                wait2 = now
        if f1 is None and f2 is None:
            time.sleep(0.01)
            continue

        frame_out1 = mask1 = None
        frame_out2 = mask2 = None
        dets1 = []
        dets2 = []
        if f1 is not None:
            frame_out1, mask1, dets1 = process_one(1, f1)
        if f2 is not None:
            frame_out2, mask2, dets2 = process_one(2, f2)

        # Multi-object matching and triangulation
        accepted_cam1 = []
        if len(dets1) > 0 and len(dets2) > 0:
            used2 = set()
            for i, d1 in enumerate(dets1):
                # match by nearest row within threshold
                best_j, best_cost = -1, 1e9
                for j, d2 in enumerate(dets2):
                    if j in used2:
                        continue
                    cost = abs(float(d1["center"][1]) - float(d2["center"][1]))
                    if cost < best_cost:
                        best_cost, best_j = cost, j
                if best_j == -1 or best_cost > float(match_row_px):
                    continue
                used2.add(best_j)
                d2 = dets2[best_j]

                # brightness gating (reject too bright)
                # Reconstruct thresholds used when processing
                # Note: We do not re-threshold here; we simply require the mean to be below the local threshold stored during processing.
                # frame-specific thresholds were rendered into the overlay; conservatively gate against the local mean vs base+delta
                base1 = int(CALIB_MIN_BY_CAM.get(1, 0))
                base2 = int(CALIB_MIN_BY_CAM.get(2, 0))
                thr1 = min(255, base1 + int(delta_thr))
                thr2 = min(255, base2 + int(delta_thr))
                if d1["mean"] > float(thr1) or d2["mean"] > float(thr2):
                    continue

                # Map centers from resized back to original frame pixels
                (o1w, o1h) = d1["orig_size"]
                (r1w, r1h) = d1["res_size"]
                (o2w, o2h) = d2["orig_size"]
                (r2w, r2h) = d2["res_size"]
                u1_orig = float(d1["center"][0]) * (float(o1w) / float(max(1, r1w)))
                v1_orig = float(d1["center"][1]) * (float(o1h) / float(max(1, r1h)))
                u2_orig = float(d2["center"][0]) * (float(o2w) / float(max(1, r2w)))
                v2_orig = float(d2["center"][1]) * (float(o2h) / float(max(1, r2h)))

                # If calibration image size known, scale to that coordinate system
                if calib_w is not None and calib_h is not None and o1w > 0 and o2w > 0 and o1h > 0 and o2h > 0:
                    uL = u1_orig * (float(calib_w) / float(o1w))
                    vL = v1_orig * (float(calib_h) / float(o1h))
                    uR = u2_orig * (float(calib_w) / float(o2w))
                    vR = v2_orig * (float(calib_h) / float(o2h))
                else:
                    uL, vL, uR, vR = u1_orig, v1_orig, u2_orig, v2_orig

                Xcm, Ycm, Zcm = triangulate_index(float(uL), float(vL), float(uR), float(vR), fx, fy, cx, cy, baseline_m)

                # Project 3D point onto the screen plane, then map to projector pixels
                px_proj = py_proj = None
                if H_plane_to_proj is not None and U is not None and V is not None and P0 is not None:
                    P = np.array([Xcm, Ycm, Zcm], dtype=np.float32)
                    P_proj = project_point_to_plane(P, P0, N)
                    a, b = solve_plane_uv(P_proj, P0, U, V)
                    a_c = clamp01(float(a))
                    b_c = clamp01(float(b))
                    src_pt = np.array([[[a_c, b_c]]], dtype=np.float32)
                    dst_pt = cv2.perspectiveTransform(src_pt, H_plane_to_proj)
                    px_proj = float(dst_pt[0, 0, 0])
                    py_proj = float(dst_pt[0, 0, 1])

                # expected radius in resized cam1 frame at depth Z
                # Use intrinsics-derived hFOV rather than a hard-coded value
                hFOV = derive_hfov_from_intrinsics(fx, float(calib_w if calib_w is not None else o1w))
                ball_d = float(max(0.001, ball_diam_cm))
                Z = max(0.001, float(Zcm))
                ang_d = 2.0 * np.arctan((ball_d / 2.0) / Z)
                # Pixels in resized frame: (angular span / hFOV) * resized_width / 2
                exp_rad_resized = (ang_d / hFOV) * (float(r1w) / 2.0)
                too_big = float(d1["radius"]) > (exp_rad_resized * float(max(1.0, size_max_ratio)))

                color = (0, 255, 0) if not too_big else (0, 0, 255)
                if frame_out1 is not None:
                    cv2.circle(frame_out1, d1["center"], int(d1["radius"]), color, 2)
                    cv2.circle(frame_out1, d1["center"], 3, (0, 0, 255), -1)
                    cv2.putText(
                        frame_out1,
                        f"Z:{Zcm:.1f} r:{d1['radius']:.0f}/{exp_rad_resized:.0f}",
                        (int(d1["center"][0] + 8), int(d1["center"][1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                if frame_out2 is not None:
                    cv2.circle(frame_out2, d2["center"], int(d2["radius"]), color, 2)
                    cv2.circle(frame_out2, d2["center"], 3, (0, 0, 255), -1)
                if not too_big:
                    accepted_cam1.append(d1["center"])

                # Publish first valid ball position to server (non-blocking best-effort)
                # Send 2D projected pixel if available; otherwise skip publish
                if SERVER_OFFER_URL and (px_proj is not None) and (py_proj is not None):
                    nowm = time.monotonic()
                    if (nowm - last_sent_mono) >= SEND_MIN_INTERVAL_S:
                        payload = {"position_px": [float(px_proj), float(py_proj)], "t": float(time.time())}
                        ok = _post_json(SERVER_OFFER_URL, payload)
                        last_sent_mono = nowm
                        last_post_ok = bool(ok)
                        # Also print once per second to avoid spam
                        if (time.time() - last_post_ts) >= 1.0:
                            print(f"[POST {'OK' if ok else 'FAIL'}] to {SERVER_OFFER_URL} pos={payload.get('position_px')}")
                            last_post_ts = time.time()

        # draw a simple trail for first accepted detection on cam1
        if len(accepted_cam1) > 0:
            points1.appendleft(accepted_cam1[0])
        else:
            points1.appendleft(None)
        for i in range(1, len(points1)):
            if points1[i - 1] is None or points1[i] is None:
                continue
            thickness = int(np.sqrt(buffer_len / float(i + 1)) * 2)
            if frame_out1 is not None:
                cv2.line(frame_out1, points1[i - 1], points1[i], (0, 0, 255), thickness)

        # FPS update
        frame_counter += 1
        prev_time, maybe_fps, maybe_reset = compute_fps(prev_time, frame_counter)
        if maybe_fps >= 0:
            fps_value = maybe_fps
            frame_counter = maybe_reset

        if frame_out1 is not None:
            cv2.imshow("Ball Tracker 1", frame_out1)
        if frame_out2 is not None:
            cv2.imshow("Ball Tracker 2", frame_out2)
        if show_mask:
            if mask1 is not None:
                cv2.imshow("Mask 1", mask1)
            if mask2 is not None:
                cv2.imshow("Mask 2", mask2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            points1.clear()
        elif key == ord('c'):
            # Calibrate on current frames (if available)
            if f1 is not None:
                g1 = cv2.cvtColor(resize_with_aspect(f1, width), cv2.COLOR_BGR2GRAY)
                calibrate_darkest(1, g1)
            if f2 is not None:
                g2 = cv2.cvtColor(resize_with_aspect(f2, width), cv2.COLOR_BGR2GRAY)
                calibrate_darkest(2, g2)
        elif key == ord('1'):
            if f1 is not None:
                g1 = cv2.cvtColor(resize_with_aspect(f1, width), cv2.COLOR_BGR2GRAY)
                calibrate_darkest(1, g1)
        elif key == ord('2'):
            if f2 is not None:
                g2 = cv2.cvtColor(resize_with_aspect(f2, width), cv2.COLOR_BGR2GRAY)
                calibrate_darkest(2, g2)

    stream1.stop()
    stream2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


