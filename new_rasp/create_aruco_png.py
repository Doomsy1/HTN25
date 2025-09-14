import os
import numpy as np
from aruco_config import PROJ_W, PROJ_H, PADDING, MARKER_SIZE, BORDER_BITS, get_aruco_dictionary, generate_marker_image, GRID_COLS, GRID_ROWS, ID_START
import cv2


def create_marker_image(dictionary, marker_id, marker_size_px, border_bits=BORDER_BITS):
    return generate_marker_image(dictionary, marker_id, marker_size_px, border_bits)


def paste_marker(canvas, marker, top_left_xy):
    x, y = top_left_xy
    h, w = marker.shape[:2]
    region = canvas[y : y + h, x : x + w]
    if region.shape[:2] != marker.shape[:2]:
        raise ValueError("Marker does not fit at the requested position.")
    region[:, :, 0] = marker
    region[:, :, 1] = marker
    region[:, :, 2] = marker


def generate_aruco_corners_image():
    width_px = PROJ_W
    height_px = PROJ_H
    padding_px = PADDING
    marker_size_px = MARKER_SIZE
    border_bits = BORDER_BITS

    if marker_size_px + padding_px > width_px:
        raise ValueError("marker_size + padding must be <= image width")
    if marker_size_px + padding_px > height_px:
        raise ValueError("marker_size + padding must be <= image height")

    canvas = np.full((height_px, width_px, 3), 255, dtype=np.uint8)

    dictionary = get_aruco_dictionary()

    ids = list(range(ID_START, ID_START + GRID_COLS * GRID_ROWS))

    idx = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            marker_id = ids[idx]
            x = int(np.linspace(padding_px, width_px - padding_px - marker_size_px, GRID_COLS)[c])
            y = int(np.linspace(padding_px, height_px - padding_px - marker_size_px, GRID_ROWS)[r])
            marker = create_marker_image(dictionary, marker_id, marker_size_px, border_bits)
            paste_marker(canvas, marker, (x, y))
            idx += 1

    out_name = "aruco_grid.png"
    out_path = os.path.join(os.path.dirname(__file__), out_name)
    success = cv2.imwrite(out_path, canvas)
    if not success:
        raise RuntimeError(f"Failed to write output image to: {out_path}")
    return out_path


if __name__ == "__main__":
    path = generate_aruco_corners_image()
    print(f"Saved: {path}")




