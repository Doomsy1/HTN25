import sys
import time
import cv2
import numpy as np

from streamcam import start_camera, read_frame_bgr
from handtracking_processor import HandTrackingProcessor
from handtracking_utils import HAND_CONNECTIONS, render_hands, draw_hands_with_index


from stereo_utils import get_intrinsics, triangulate_index, recommended_hfov_deg


 



def main():
    # Fixed RTSP sources (CLI overrides removed)
    sources = [
        "rtsp://10.37.111.247:8554/cam1",
        "rtsp://10.37.111.247:8554/cam2",
    ]

    cams = [start_camera(src) for src in sources]
    processors = [HandTrackingProcessor(cam) for cam in cams]
    for p in processors:
        p.start()

    windows = [f"Stereo Cam{i+1}" for i in range(len(processors))]
    for title in windows:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 1280//2, 720//2)

    hfov_deg = recommended_hfov_deg()
    fx = fy = cx = cy = None
    baseline_m = 0.10

    try:
        while True:
            frames = [p.read_latest_frame() for p in processors]
            results = [p.read_latest_result() for p in processors]

            if any(f is None for f in frames):
                time.sleep(0.005)
                continue

            # On first good frames, compute intrinsics based on resolution
            if fx is None:
                h, w = frames[0].shape[:2]
                fx, fy, cx, cy = get_intrinsics(w, h, hfov_deg)

            index_pts = []
            for idx, (frame, result) in enumerate(zip(frames, results)):
                annotated = frame.copy()
                idx_pt = draw_hands_with_index(annotated, result)
                index_pts.append(idx_pt)
                cv2.imshow(windows[idx], annotated)

            # Triangulate if both present
            line = []
            if len(index_pts) >= 2 and index_pts[0] is not None and index_pts[1] is not None:
                (uL, vL) = index_pts[0]
                (uR, vR) = index_pts[1]
                X, Y, Z = triangulate_index(float(uL), float(vL), float(uR), float(vR), fx, fy, cx, cy, baseline_m)
                line.append(f"Cam1: {uL},{vL}")
                line.append(f"Cam2: {uR},{vR}")
                line.append(f"3D(m): {X:.3f},{Y:.3f},{Z:.3f}")
            else:
                for i, pt in enumerate(index_pts):
                    if pt is None:
                        line.append(f"Cam{i+1}: -")
                    else:
                        line.append(f"Cam{i+1}: {pt[0]},{pt[1]}")

            if line:
                print("  |  ".join(line))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        for p in processors:
            p.stop()
        for c in cams:
            try:
                c.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())


