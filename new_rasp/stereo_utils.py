import math
import json
from typing import Tuple

import numpy as np


# Try to reuse triangulation utilities from rasp/, fallback to local minimal versions
try:  # pragma: no cover - optional dependency
    from rasp.triangulate import intrinsics_from_hfov as _intrinsics_from_hfov  # type: ignore
    from rasp.triangulate import triangulate_pair as _triangulate_pair  # type: ignore
    from rasp.triangulate import triangulate_pairs as _triangulate_pairs  # type: ignore
    from rasp.triangulate import recommended_hfov_deg_for_imx708 as _recommended_hfov  # type: ignore
except Exception:  # pragma: no cover
    _intrinsics_from_hfov = None
    _triangulate_pair = None
    _triangulate_pairs = None
    _recommended_hfov = None


def _local_intrinsics_from_hfov(width: int, height: int, hfov_deg: float) -> Tuple[float, float, float, float]:
    cx = float(width) * 0.5
    cy = float(height) * 0.5
    hfov_rad = math.radians(float(hfov_deg))
    fx = (float(width) * 0.5) / math.tan(hfov_rad * 0.5)
    vfov_rad = 2.0 * math.atan((float(height) * 0.5) / fx)
    fy = (float(height) * 0.5) / math.tan(vfov_rad * 0.5)
    return float(fx), float(fy), float(cx), float(cy)


def _local_triangulate_pair(x_left: float, y_left: float, x_right: float, y_right: float,
                            fx: float, fy: float, cx: float, cy: float,
                            baseline_m: float) -> Tuple[float, float, float]:
    d = (float(x_left) - float(x_right))
    if abs(d) < 1e-6:
        return float("nan"), float("nan"), float("nan")
    Z = float(fx) * float(baseline_m) / d
    X_left = (float(x_left) - float(cx)) * (Z / float(fx))
    Y_left = (float(y_left) - float(cy)) * (Z / float(fy))
    X_mid = X_left - (float(baseline_m) * 0.5)
    return float(X_mid), float(Y_left), float(Z)


def _local_triangulate_pairs(points_left: np.ndarray, points_right: np.ndarray,
                             fx: float, fy: float, cx: float, cy: float,
                             baseline_m: float) -> np.ndarray:
    xl = points_left[:, 0].astype(np.float64)
    yl = points_left[:, 1].astype(np.float64)
    xr = points_right[:, 0].astype(np.float64)
    yr = points_right[:, 1].astype(np.float64)
    d = xl - xr
    Z = (float(fx) * float(baseline_m)) / np.where(np.abs(d) < 1e-12, np.nan, d)
    X_left = (xl - float(cx)) * (Z / float(fx))
    Y_left = (yl - float(cy)) * (Z / float(fy))
    X_mid = X_left - (float(baseline_m) * 0.5)
    pts3d = np.stack([X_mid, Y_left, Z], axis=1)
    return pts3d.astype(np.float64)


def get_intrinsics(width: int, height: int, hfov_deg: float) -> Tuple[float, float, float, float]:
    if _intrinsics_from_hfov is not None:
        fx, fy, cx, cy, _ = _intrinsics_from_hfov(width, height, hfov_deg)
        return float(fx), float(fy), float(cx), float(cy)
    return _local_intrinsics_from_hfov(width, height, hfov_deg)


def triangulate_pair(u_left: float, v_left: float, u_right: float, v_right: float,
                     fx: float, fy: float, cx: float, cy: float,
                     baseline_m: float) -> Tuple[float, float, float]:
    if _triangulate_pair is not None:
        x, y, z = _triangulate_pair(u_left, v_left, u_right, v_right, fx, fy, cx, cy, baseline_m)
        return float(x), float(y), float(z)
    return _local_triangulate_pair(u_left, v_left, u_right, v_right, fx, fy, cx, cy, baseline_m)


def triangulate_index(u_left: float, v_left: float, u_right: float, v_right: float,
                      fx: float, fy: float, cx: float, cy: float,
                      baseline_m: float) -> Tuple[float, float, float]:
    # Backwards-compatible alias used by existing modules
    return triangulate_pair(u_left, v_left, u_right, v_right, fx, fy, cx, cy, baseline_m)


def triangulate_pairs(points_left: np.ndarray, points_right: np.ndarray,
                      fx: float, fy: float, cx: float, cy: float,
                      baseline_m: float) -> np.ndarray:
    if _triangulate_pairs is not None:
        return _triangulate_pairs(points_left, points_right, fx, fy, cx, cy, baseline_m)
    return _local_triangulate_pairs(points_left, points_right, fx, fy, cx, cy, baseline_m)


def recommended_hfov_deg() -> float:
    if _recommended_hfov is not None:
        return float(_recommended_hfov(normal=True))
    return 66.0



# Shared stereo/projection helpers
def load_calibration(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    intr = data.get("intrinsics", {})
    src_urls = data.get("source_urls", [])
    proj_corners_px = data.get("projector_corners_px", None)
    screen_corners_3d = data.get("screen_corners_3d_m", None)
    image_size = data.get("image_size", None)
    if not src_urls or proj_corners_px is None or screen_corners_3d is None:
        raise ValueError("Calibration JSON missing required fields: source_urls, projector_corners_px, screen_corners_3d_m")
    fx = float(intr.get("fx"))
    fy = float(intr.get("fy"))
    cx = float(intr.get("cx"))
    cy = float(intr.get("cy"))
    baseline_m = float(intr.get("baseline_m"))
    return {
        "sources": src_urls,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "baseline_m": baseline_m,
        "projector_corners_px": np.array(proj_corners_px, dtype=np.float32),
        "screen_corners_3d": np.array(screen_corners_3d, dtype=np.float32),
        "image_size": tuple(image_size) if image_size is not None else None,
    }


def project_point_to_plane(point: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    v = point - plane_point
    dist = float(np.dot(v, plane_normal))
    return point - dist * plane_normal


def signed_distance_to_plane(point: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> float:
    v = point - plane_point
    return float(np.dot(v, plane_normal))


def calc_plane_basis(corners3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Expect corners: TL (P0), TR (P1), BR (P2), BL (P3)
    P0 = corners3d[0]
    P1 = corners3d[1]
    P3 = corners3d[3]
    U = P1 - P0
    V = P3 - P0
    N = np.cross(U, V)
    n_norm = np.linalg.norm(N)
    if n_norm < 1e-9:
        raise ValueError("Screen plane corners are degenerate")
    N = N / n_norm
    return U, V, N


def solve_plane_uv(P: np.ndarray, P0: np.ndarray, U: np.ndarray, V: np.ndarray) -> Tuple[float, float]:
    A = np.stack([U, V], axis=1)  # shape (3,2)
    b = (P - P0)
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a = float(sol[0])
    c = float(sol[1])
    return a, c


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

