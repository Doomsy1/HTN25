import sys
import cv2

try:
    from new_rasp.streamcam import start_camera  # type: ignore
except Exception:
    from streamcam import start_camera  # type: ignore

try:
    from new_rasp.handtracking_processor import HandTrackingProcessor  # type: ignore
except Exception:
    from handtracking_processor import HandTrackingProcessor  # type: ignore

try:
    from new_rasp.handtracking_utils import HAND_CONNECTIONS, render_hands, draw_hands_with_index  # type: ignore
except Exception:
    from handtracking_utils import HAND_CONNECTIONS, render_hands, draw_hands_with_index  # type: ignore


 
def draw_hands(frame_bgr, result):
    draw_hands_with_index(frame_bgr, result)


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

    windows = [f"Cam{i+1} Hand" for i in range(len(processors))]
    for title in windows:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 1280//2, 720//2)

    try:
        while True:
            line_parts = []
            for idx, p in enumerate(processors):
                frame = p.read_latest_frame()
                result = p.read_latest_result()
                if frame is None:
                    line_parts.append(f"Cam{idx+1}: -")
                    continue
                annotated = frame.copy()
                draw_hands(annotated, result)
                cv2.imshow(windows[idx], annotated)
                try:
                    if result is not None and getattr(result, "hand_landmarks", None):
                        h, w = frame.shape[:2]
                        hand = result.hand_landmarks[0]
                        ix = int(hand[8].x * w)
                        iy = int(hand[8].y * h)
                        line_parts.append(f"Cam{idx+1}: {ix},{iy}")
                    else:
                        line_parts.append(f"Cam{idx+1}: -")
                except Exception:
                    line_parts.append(f"Cam{idx+1}: -")
            if line_parts:
                print("  |  ".join(line_parts))
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


