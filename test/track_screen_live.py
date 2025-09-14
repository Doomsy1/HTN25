import numpy as np
import cv2

from aruco_config import get_aruco_dictionary, get_aruco_detector, PROJ_W, PROJ_H, projector_marker_corners, MARGIN
from streamcam import start_camera, read_frame_bgr


def detect_markers(gray, dictionary):
    detect_fn = get_aruco_detector(dictionary)
    corners, ids, _ = detect_fn(gray)
    return corners, ids


def collect_correspondences(detected_corners, detected_ids):
    proj_pts = []
    cam_pts = []
    if detected_ids is None:
        return None, None
    for i, mid in enumerate(detected_ids.flatten()):
        proj_quad = projector_marker_corners(int(mid))
        if proj_quad is None:
            continue
        cam_quad = np.squeeze(detected_corners[i], axis=0).astype(np.float32)
        for j in range(4):
            proj_pts.append(proj_quad[j])
            cam_pts.append(cam_quad[j])
    if len(proj_pts) < 4:
        return None, None
    return np.array(proj_pts, dtype=np.float32), np.array(cam_pts, dtype=np.float32)


def draw_projector_outline(frame, H, draw_outline):
    if not draw_outline or H is None:
        return frame
    rect = np.array([[-MARGIN, -MARGIN], [PROJ_W + MARGIN, -MARGIN], [PROJ_W + MARGIN, PROJ_H + MARGIN], [-MARGIN, PROJ_H + MARGIN]], dtype=np.float32).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(rect, H).reshape(-1, 2)
    pts = dst.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(frame, [pts], True, (0, 0, 255), 3)
    return frame


def main():
    URL1 = "rtsp://10.37.111.247:8554/cam1"
    URL2 = "rtsp://10.37.111.247:8554/cam2"
    cam1 = start_camera(URL1)
    cam2 = start_camera(URL2)

    dictionary = get_aruco_dictionary()
    print("Tracking live with ArUco grid. Press q to quit. 'o' outline on/off.")
    draw_outline = True
    cv2.namedWindow("ArUco Track Live (Cam1)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("ArUco Track Live (Cam2)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ArUco Track Live (Cam1)", 1280//2, 720//2)
    cv2.resizeWindow("ArUco Track Live (Cam2)", 1280//2, 720//2)

    try:
        while True:
            frame1 = read_frame_bgr(cam1)
            frame2 = read_frame_bgr(cam2)
            if frame1 is None or frame2 is None:
                continue

            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            corners1, ids1 = detect_markers(gray1, dictionary)
            if ids1 is not None and len(ids1) > 0:
                cv2.aruco.drawDetectedMarkers(frame1, corners1, ids1)
                proj_pts1, cam_pts1 = collect_correspondences(corners1, ids1)
                if proj_pts1 is not None:
                    H1, _ = cv2.findHomography(proj_pts1, cam_pts1, cv2.RANSAC, 5.0)
                    frame1 = draw_projector_outline(frame1, H1, draw_outline)

            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            corners2, ids2 = detect_markers(gray2, dictionary)
            if ids2 is not None and len(ids2) > 0:
                cv2.aruco.drawDetectedMarkers(frame2, corners2, ids2)
                proj_pts2, cam_pts2 = collect_correspondences(corners2, ids2)
                if proj_pts2 is not None:
                    H2, _ = cv2.findHomography(proj_pts2, cam_pts2, cv2.RANSAC, 5.0)
                    frame2 = draw_projector_outline(frame2, H2, draw_outline)

            cv2.imshow("ArUco Track Live (Cam1)", frame1)
            cv2.imshow("ArUco Track Live (Cam2)", frame2)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('o'):
                draw_outline = not draw_outline
    finally:
        try:
            cam1.stop()
        except Exception:
            pass
        try:
            cam2.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


