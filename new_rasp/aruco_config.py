import numpy as np
import cv2


# Projector/layout constants
PROJ_W = 2880
PROJ_H = 1800
PADDING = 30
MARKER_SIZE = 300
BORDER_BITS = 1
ARUCO_DICT_NAME = "DICT_4X4_1000"
MARGIN = 40

# Grid layout
GRID_COLS = 8
GRID_ROWS = 5
ID_START = 0


def get_aruco_dictionary():
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))


def generate_marker_image(dictionary, marker_id, marker_size_px, border_bits=BORDER_BITS):
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size_px, borderBits=border_bits)
    marker = np.zeros((marker_size_px, marker_size_px), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, marker_id, marker_size_px, marker, borderBits=border_bits)
    return marker


def projector_marker_top_left(marker_id):
    total = GRID_COLS * GRID_ROWS
    idx = int(marker_id) - ID_START
    if idx < 0 or idx >= total:
        return None

    xs = np.linspace(PADDING, PROJ_W - PADDING - MARKER_SIZE, GRID_COLS)
    ys = np.linspace(PADDING, PROJ_H - PADDING - MARKER_SIZE, GRID_ROWS)

    row = idx // GRID_COLS
    col = idx % GRID_COLS
    return float(xs[col]), float(ys[row])


def projector_marker_corners(marker_id):
    tl = projector_marker_top_left(marker_id)
    if tl is None:
        return None
    x, y = tl
    return np.array([
        [x, y],
        [x + MARKER_SIZE, y],
        [x + MARKER_SIZE, y + MARKER_SIZE],
        [x, y + MARKER_SIZE],
    ], dtype=np.float32)


def get_aruco_detector(dictionary):
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)

        def detect_fn(gray):
            return detector.detectMarkers(gray)

        return detect_fn

    params = cv2.aruco.DetectorParameters_create() if hasattr(cv2.aruco, "DetectorParameters_create") else cv2.aruco.DetectorParameters()

    def detect_fn(gray):
        return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

    return detect_fn


