import os
import sys
import json
import time
import datetime
import numpy as np
import cv2

try:
    from new_rasp.aruco_config import get_aruco_dictionary, get_aruco_detector, PROJ_W, PROJ_H  # type: ignore
except Exception:
    from aruco_config import get_aruco_dictionary, get_aruco_detector, PROJ_W, PROJ_H  # type: ignore

try:
    from new_rasp.streamcam import start_camera, read_frame_bgr  # type: ignore
except Exception:
    from streamcam import start_camera, read_frame_bgr  # type: ignore


try:
    from new_rasp.stereo_utils import get_intrinsics, triangulate_pairs, recommended_hfov_deg  # type: ignore
except Exception:
    from stereo_utils import get_intrinsics, triangulate_pairs, recommended_hfov_deg  # type: ignore


# Easily editable default calibration parameters
DEFAULT_URL1 = "rtsp://10.37.111.247:8554/cam1"
DEFAULT_URL2 = "rtsp://10.37.111.247:8554/cam2"
DEFAULT_BASELINE_M = 0.10
DEFAULT_HFOV_DEG = None  # set a float (e.g., 66.0) to override camera-recommended
DEFAULT_REQUIRED_STABLE = 5
DEFAULT_TOLERANCE_PCT = 0.5
DEFAULT_OUTPUT_JSON = "screen_corners_3d.json"
 


def _quad_scale_length(pts3d):
    """Return a robust scale for a quad: max of its two diagonals.

    This avoids division-by-near-zero issues when some coordinates are ~0.
    """
    try:
        d1 = float(np.linalg.norm(pts3d[2] - pts3d[0]))
        d2 = float(np.linalg.norm(pts3d[3] - pts3d[1]))
        scale = max(d1, d2)
        return scale if np.isfinite(scale) else 1.0
    except Exception:
        return 1.0


def detect_markers(gray, dictionary):
    detect_fn = get_aruco_detector(dictionary)
    corners, ids, _ = detect_fn(gray)
    return corners, ids


def collect_correspondences(detected_corners, detected_ids, projector_pts_fn):
    proj_pts = []
    cam_pts = []
    if detected_ids is None or len(detected_ids) == 0:
        return None, None
    for i, mid in enumerate(detected_ids.flatten()):
        proj_quad = projector_pts_fn(int(mid))
        if proj_quad is None:
            continue
        cam_quad = np.squeeze(detected_corners[i], axis=0).astype(np.float32)
        for j in range(4):
            proj_pts.append(proj_quad[j])
            cam_pts.append(cam_quad[j])
    if len(proj_pts) < 4:
        return None, None
    return np.array(proj_pts, dtype=np.float32), np.array(cam_pts, dtype=np.float32)


def projector_corners():
    # Screen corners (no margin): TL, TR, BR, BL in projector pixel space
    return np.array([
        [0.0, 0.0],
        [float(PROJ_W), 0.0],
        [float(PROJ_W), float(PROJ_H)],
        [0.0, float(PROJ_H)],
    ], dtype=np.float32).reshape(-1, 1, 2)


def warp_projector_corners(H):
    if H is None:
        return None
    pts = projector_corners()
    dst = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return dst


def draw_outline(frame, corners_2d):
    if corners_2d is None:
        return frame
    pts = corners_2d.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
    for (x, y) in corners_2d:
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    return frame


def calibrate_projector_corners_stereo(url1="rtsp://10.37.111.247:8554/cam1",
                                       url2="rtsp://10.37.111.247:8554/cam2",
                                       baseline=0.10,
                                       hfov=None,
                                       required=DEFAULT_REQUIRED_STABLE,
                                       tolerance=DEFAULT_TOLERANCE_PCT,
                                       out="screen_corners_3d.json",
                                       show=False):
    """Run stereo calibration to estimate 3D projector screen corners.

    Parameters
    - url1, url2: camera stream URLs
    - baseline: stereo baseline in meters
    - hfov: horizontal FOV in degrees (None uses camera-recommended)
    - required: number of stable detections before saving
    - tolerance: stability tolerance in percent (e.g., 1.0 for ±1%)
    - out: output JSON filename (relative to this file's directory)
    - show: if True, opens display windows; if False, runs headless

    Returns (out_data, out_path):
    - out_data: dict with calibration results, or None if aborted
    - out_path: absolute path to saved JSON, or None if not saved
    """

    cam1 = start_camera(url1)
    cam2 = start_camera(url2)

    dictionary = get_aruco_dictionary()
    # Create CLAHE once for contrast boosting on grayscale frames
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    if show:
        print("Calibrating projector corners with ArUco. Waiting for detections... Press 'q' to abort.")
        cv2.namedWindow("Calib Cam1", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Calib Cam2", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calib Cam1", 1280//2, 720//2)
        cv2.resizeWindow("Calib Cam2", 1280//2, 720//2)

    hfov_deg = float(hfov) if hfov is not None else recommended_hfov_deg()
    fx = fy = cx = cy = None
    baseline_m = float(baseline)

    corners_cam1 = None
    corners_cam2 = None
    stable_ref = None
    stable_count = 0

    out_data = None
    out_path_abs = None

    try:
        while True:
            f1 = read_frame_bgr(cam1)
            f2 = read_frame_bgr(cam2)
            if f1 is None or f2 is None:
                time.sleep(0.01)
                continue

            if fx is None:
                h, w = f1.shape[:2]
                fx, fy, cx, cy = get_intrinsics(w, h, hfov_deg)

            g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
            # Try raw grayscale first; fall back to CLAHE only if needed
            corners1, ids1 = detect_markers(g1, dictionary)
            if ids1 is None or len(ids1) == 0:
                g1c = clahe.apply(g1)
                corners1, ids1 = detect_markers(g1c, dictionary)
            corners2, ids2 = detect_markers(g2, dictionary)
            if ids2 is None or len(ids2) == 0:
                g2c = clahe.apply(g2)
                corners2, ids2 = detect_markers(g2c, dictionary)

            if show:
                if ids1 is not None and len(ids1) > 0:
                    cv2.aruco.drawDetectedMarkers(f1, corners1, ids1)
                if ids2 is not None and len(ids2) > 0:
                    cv2.aruco.drawDetectedMarkers(f2, corners2, ids2)

            # Projector->camera homographies
            from aruco_config import projector_marker_corners as proj_pts_fn
            proj_pts_1, cam_pts_1 = collect_correspondences(corners1, ids1, proj_pts_fn)
            proj_pts_2, cam_pts_2 = collect_correspondences(corners2, ids2, proj_pts_fn)

            H1 = H2 = None
            if proj_pts_1 is not None:
                H1, _ = cv2.findHomography(proj_pts_1, cam_pts_1, cv2.RANSAC, 5.0)
            if proj_pts_2 is not None:
                H2, _ = cv2.findHomography(proj_pts_2, cam_pts_2, cv2.RANSAC, 5.0)

            # Only use fresh detections from this frame; clear if missing
            corners_cam1 = warp_projector_corners(H1) if H1 is not None else None
            corners_cam2 = warp_projector_corners(H2) if H2 is not None else None

            if show:
                if corners_cam1 is not None:
                    f1 = draw_outline(f1, corners_cam1)
                if corners_cam2 is not None:
                    f2 = draw_outline(f2, corners_cam2)

            if show:
                cv2.imshow("Calib Cam1", f1)
                cv2.imshow("Calib Cam2", f2)

            # Require both cameras to detect this frame; enforce consecutive stability
            both_detected = (corners_cam1 is not None) and (corners_cam2 is not None)
            if not both_detected:
                stable_ref = None
                stable_count = 0
            else:
                ptsL = corners_cam1.astype(np.float64)
                ptsR = corners_cam2.astype(np.float64)
                pts3d = triangulate_pairs(ptsL, ptsR, fx, fy, cx, cy, baseline_m)

                if stable_ref is None:
                    stable_ref = pts3d.copy()
                    stable_count = 1
                else:
                    tol = float(tolerance) / 100.0
                    scale = _quad_scale_length(stable_ref)
                    if scale <= 1e-12:
                        scale = 1.0
                    max_err = float(np.max(np.linalg.norm(pts3d - stable_ref, axis=1)))
                    if max_err <= tol * scale:
                        stable_count += 1
                    else:
                        stable_ref = pts3d.copy()
                        stable_count = 1

                if show:
                    cv2.putText(f1, f"Stable: {stable_count}/{int(required)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2, cv2.LINE_AA)
                    cv2.putText(f2, f"Stable: {stable_count}/{int(required)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2, cv2.LINE_AA)

                if stable_count >= int(required):
                    out_data = {
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "source_urls": [url1, url2],
                        "image_size": [int(w), int(h)],
                        "intrinsics": {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy), "hfov_deg": float(hfov_deg), "baseline_m": float(baseline_m)},
                        "projector_corners_px": [[0.0, 0.0], [910.0, 0.0], [910.0, 605.0], [0.0, 605.0]], #TODO
                        "screen_corners_2d": {
                            "cam1": [[float(x), float(y)] for (x, y) in corners_cam1.tolist()],
                            "cam2": [[float(x), float(y)] for (x, y) in corners_cam2.tolist()],
                        },
                        "screen_corners_3d_m": [[float(v[0]), float(v[1]), float(v[2])] for v in pts3d.tolist()],
                    }

                    here = os.path.dirname(__file__)
                    out_path_abs = os.path.join(here, str(out))
                    with open(out_path_abs, "w", encoding="utf-8") as f:
                        json.dump(out_data, f, indent=2)
                    print(f"Saved 3D screen corners to: {out_path_abs}")
                    time.sleep(0.3)
                    break

            if show:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
    finally:
        cam1.stop()
        cam2.stop()
        if show:
            cv2.destroyAllWindows()

    return out_data, out_path_abs


def main():
    calibrate_projector_corners_stereo(
        url1=DEFAULT_URL1,
        url2=DEFAULT_URL2,
        baseline=DEFAULT_BASELINE_M,
        hfov=DEFAULT_HFOV_DEG,
        required=DEFAULT_REQUIRED_STABLE,
        tolerance=DEFAULT_TOLERANCE_PCT,
        out=DEFAULT_OUTPUT_JSON,
        show=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


