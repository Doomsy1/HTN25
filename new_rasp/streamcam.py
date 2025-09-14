"""Minimal stream helpers using OpenCV (low-noise, no typing/docs)."""

import threading, time
import cv2

class LatestFrameStream:
    def __init__(self, input_url, width=None, height=None, fps=None,
                 backend=cv2.CAP_FFMPEG, reconnect=True, reconnect_interval_s=1.0):
        self.input_url = input_url
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self.backend = backend
        self.reconnect = reconnect
        self.reconnect_interval_s = max(0.05, float(reconnect_interval_s))

        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._open_capture()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _open_capture(self):
        # Release any prior instance first
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        cap = cv2.VideoCapture(self.input_url, self.backend)
        if cap is not None:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            if self.requested_width is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.requested_width))
            if self.requested_height is not None:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.requested_height))
            if self.requested_fps is not None:
                cap.set(cv2.CAP_PROP_FPS, int(self.requested_fps))

        self.cap = cap

    def _run(self):
        consecutive_failures = 0
        while not self._stop.is_set():
            cap = self.cap
            if cap is None or not cap.isOpened():
                if not self.reconnect:
                    time.sleep(0.05)
                    continue
                time.sleep(self.reconnect_interval_s)
                self._open_capture()
                consecutive_failures = 0
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                consecutive_failures += 1
                # If we fail a few times, try to re-open to clear stuck pipelines
                if consecutive_failures >= 15 and self.reconnect:
                    self._open_capture()
                    consecutive_failures = 0
                else:
                    time.sleep(0.01)
                continue

            consecutive_failures = 0
            with self.lock:
                self.frame = frame

    def read_latest(self):
        with self.lock:
            return self.frame

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


def start_stream(input_url, width=None, height=None, fps=None):
    stream = LatestFrameStream(input_url, width=width, height=height, fps=fps)
    stream.start()
    return stream


def start_camera(index_or_url, width=None, height=None, fps=None):
    return start_stream(str(index_or_url), width=width, height=height, fps=fps)


def set_fps(stream, fps):
    stream.requested_fps = int(fps)
    if stream.cap is not None:
        stream.cap.set(cv2.CAP_PROP_FPS, int(fps))


def read_frame_bgr(stream):
    frame = stream.read_latest()
    if frame is not None:
        return cv2.flip(frame, -1)
    return frame


# cropping helpers removed: cameras now share FOV





def autofocus_and_lock_shared(cam_a, cam_b, use_macro=False, timeout_s=1.2):
    _ = (cam_a, cam_b, use_macro, timeout_s)

def main():
    URL1 = "rtsp://10.37.111.247:8554/cam1"
    URL2 = "rtsp://10.37.111.247:8554/cam2"
    cam1 = start_camera(URL1)
    cam2 = start_camera(URL2)
    cam1.start()
    cam2.start()
    cv2.namedWindow("Frame1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Frame2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame1", 1280//2, 720//2)
    cv2.resizeWindow("Frame2", 1280//2, 720//2)
    while True:
        frame1 = read_frame_bgr(cam1)
        frame2 = read_frame_bgr(cam2)
        if frame1 is None or frame2 is None:
            continue
        cv2.imshow("Frame1", frame1)
        cv2.imshow("Frame2", frame2)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cam1.stop()
    cam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()