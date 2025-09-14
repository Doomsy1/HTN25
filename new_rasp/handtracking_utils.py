import cv2

# Centralized hand landmark connections for drawing across new_rasp/
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def render_hands(frame_bgr, result, draw_points=True, connections_color=(0, 255, 0), point_color=(0, 255, 255)):
    if result is None or not getattr(result, "hand_landmarks", None):
        return None
    h, w = frame_bgr.shape[:2]
    index_tip = None
    for hand_landmarks in result.hand_landmarks:
        for a, b in HAND_CONNECTIONS:
            ax, ay = int(hand_landmarks[a].x * w), int(hand_landmarks[a].y * h)
            bx, by = int(hand_landmarks[b].x * w), int(hand_landmarks[b].y * h)
            cv2.line(frame_bgr, (ax, ay), (bx, by), connections_color, 1, cv2.LINE_AA)
        if draw_points:
            for lm in hand_landmarks:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame_bgr, (px, py), 2, point_color, -1, cv2.LINE_AA)
        ix, iy = int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h)
        index_tip = (ix, iy)
        break
    return index_tip


def draw_hands_with_index(frame_bgr, result):
    index_px = render_hands(frame_bgr, result, draw_points=True)
    if index_px is not None:
        ix, iy = index_px
        cv2.circle(frame_bgr, (ix, iy), 18, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(frame_bgr, (ix, iy), 16, (0, 0, 255), -1, cv2.LINE_AA)
    return index_px


