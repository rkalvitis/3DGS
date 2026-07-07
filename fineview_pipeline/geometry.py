"""
Pure-math geometry helpers for the FineView → 3DGS pipeline.
No I/O, no side-effects.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple

# 90° CCW rotation about the optical (Z) axis in camera coords.
# Applied to R and t when the image is rotated CCW by 90°.
_RZ_CCW = np.array([[0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0,  0.0, 1.0]])


# ── Rotation ──────────────────────────────────────────────────────────────────

def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """
    Rotation matrix → quaternion in COLMAP order [qw, qx, qy, qz].
    Uses the Shepperd method via eigendecomposition (numerically stable).
    """
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    # Last row must be (Ryz-Rzy, Rzx-Rxz, Rxy-Ryx) — matches COLMAP's
    # reference implementation. A transposed variant here silently produces
    # wrong rotations (qx/qz cross-terms swapped) for general poses.
    K = np.array([
        [Rxx - Ryy - Rzz, 0.0,             0.0,             0.0],
        [Ryx + Rxy,        Ryy - Rxx - Rzz, 0.0,             0.0],
        [Rzx + Rxz,        Rzy + Ryz,        Rzz - Rxx - Ryy, 0.0],
        [Ryz - Rzy,        Rzx - Rxz,        Rxy - Ryx,        Rxx + Ryy + Rzz],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def rotate_camera(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    w: int,
    h: int,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Apply k × 90° CCW rotation (matching np.rot90(img, k)) consistently to
    K, R, t.  Returns (K, R, t, new_w, new_h).

    One CCW step:
        K' = [[fy, 0, cy], [0, fx, w-1-cx], [0,0,1]]
        R' = RZ_CCW @ R
        t' = RZ_CCW @ t
        (w, h) -> (h, w)

    Call verify.check_reproject after export and flip k sign if reprojection
    is wrong — np.rot90 CCW direction vs. this derivation may need reconciling.
    """
    k = k % 4
    K = K.copy()
    R = R.copy()
    t = t.reshape(3, 1).copy()
    for _ in range(k):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        K = np.array([[fy, 0.0, cy],
                      [0.0, fx, w - 1 - cx],
                      [0.0, 0.0, 1.0]])
        R = _RZ_CCW @ R
        t = _RZ_CCW @ t
        w, h = h, w
    return K, R, t, w, h


# ── Re-centering (crop-based, no black borders) ────────────────────────────────

def recenter(
    img: np.ndarray,
    cx: float,
    cy: float,
) -> Tuple[np.ndarray, float, float]:
    """
    Crop image so the principal point lands exactly at the image centre.
    Removes the off-centre edge rather than padding with black.
    Returns (cropped_img, new_cx, new_cy).
    """
    h, w = img.shape[:2]
    half_w = int(min(cx, w - cx))
    half_h = int(min(cy, h - cy))
    left = int(round(cx)) - half_w
    top  = int(round(cy)) - half_h
    cropped = img[top:top + 2 * half_h, left:left + 2 * half_w]
    return cropped, float(half_w), float(half_h)


def recenter_K(K: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, int, int]:
    """Compute cropped K and dimensions so pp lands at image centre."""
    cx, cy = K[0, 2], K[1, 2]
    half_w = int(min(cx, w - cx))
    half_h = int(min(cy, h - cy))
    K = K.copy()
    K[0, 2] = float(half_w)
    K[1, 2] = float(half_h)
    return K, 2 * half_w, 2 * half_h


# ── Legacy pad helpers (kept for verify.py) ───────────────────────────────────

def pad_params(cx: float, cy: float, w: int, h: int) -> Tuple[int, int, int, int]:
    """Padding (pl, pr, pt, pb) so that after padding cx == W'/2, cy == H'/2."""
    pl = max(0, round(w   - 2 * cx))
    pr = max(0, round(2 * cx - w))
    pt = max(0, round(h   - 2 * cy))
    pb = max(0, round(2 * cy - h))
    return pl, pr, pt, pb


# ── Camera geometry ────────────────────────────────────────────────────────────

def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World position of camera: C = -R^T @ t"""
    return -(R.T @ t.reshape(3, 1)).ravel()


def reproject_check(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    X_world: np.ndarray,
) -> Tuple[float, float]:
    """Project a 3-D world point to (u, v) pixel coordinates."""
    X = np.asarray(X_world, dtype=float).reshape(3, 1)
    x_cam = R @ X + t.reshape(3, 1)
    x_img = K @ x_cam
    return float(x_img[0, 0] / x_img[2, 0]), float(x_img[1, 0] / x_img[2, 0])
