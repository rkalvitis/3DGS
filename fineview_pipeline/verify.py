"""
Sanity checks for the FineView → 3DGS pipeline.

All checks are silent-pass / loud-fail with clear diagnostic messages.
Run before committing to a full 3DGS training run.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np

from .fineview_io import intrinsics_for_species, extrinsics, list_cameras, n_poses
from .geometry import reproject_check, camera_center, recenter_K


# ── 1. Reprojection check ──────────────────────────────────────────────────────

def check_reproject(
    h5_path: str,
    species_id: int,
    X_world: np.ndarray,
    cam: str = "camera1",
    pose_idx: int = 0,
    img_path: Optional[str] = None,
    save_overlay: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Project a known 3-D world point into one view and optionally overlay it.

    Returns (u, v) pixel coordinates.

    If it lands hundreds of pixels off:
        - crop-offset not applied → check intrinsics_for_species
        - wrong R (Rodrigues applied to a matrix) → rvec is already R
        - rotate_k sign wrong → flip the sign and retry
    """
    species_idx = species_id - 1
    with h5py.File(h5_path, "r") as f:
        K, w, h = intrinsics_for_species(f, cam, species_idx)
        R, t    = extrinsics(f, cam, pose_idx)

    u, v = reproject_check(K, R, t, X_world)
    print(f"Reprojection check  cam={cam}  pose={pose_idx}")
    print(f"  3D point   : {X_world}")
    print(f"  Pixel (u,v): ({u:.1f}, {v:.1f})")
    print(f"  Image size : {w} × {h}")

    in_bounds = (0 <= u < w) and (0 <= v < h)
    print("  " + ("OK: inside image." if in_bounds else "WARNING: OUTSIDE image frame!"))

    if img_path and save_overlay:
        _save_overlay(img_path, u, v, save_overlay)

    return u, v


def _save_overlay(img_path: str, u: float, v: float, save_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
        img = np.array(Image.open(img_path))
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        ax.plot(u, v, "r+", markersize=20, markeredgewidth=3,
                label=f"reprojected ({u:.0f},{v:.0f})")
        ax.legend()
        ax.set_title("Reprojection overlay")
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"  Overlay saved → {save_path}")
    except ImportError:
        print("  (matplotlib not available; install it for visual overlay)")


# ── 2. Re-center invariant ────────────────────────────────────────────────────

def check_recenter(
    K: np.ndarray,
    w: int,
    h: int,
    tol_px: float = 1.0,
) -> bool:
    """
    Assert cx ≈ W/2 and cy ≈ H/2 after padding. Returns True if invariant holds.
    """
    K_c, w_c, h_c = recenter_K(K, w, h)
    cx_err = abs(K_c[0, 2] - w_c / 2)
    cy_err = abs(K_c[1, 2] - h_c / 2)
    ok = (cx_err <= tol_px) and (cy_err <= tol_px)
    print(f"Re-center invariant: {'OK' if ok else 'FAIL'}")
    print(f"  cx={K_c[0,2]:.2f}  W/2={w_c/2:.2f}  err={cx_err:.2f} px")
    print(f"  cy={K_c[1,2]:.2f}  H/2={h_c/2:.2f}  err={cy_err:.2f} px")
    if not ok:
        print("  FAIL: pad_params() arithmetic is wrong or K was already modified.")
    return ok


# ── 3. Camera-centre visualisation ────────────────────────────────────────────

def plot_camera_centres(
    h5_path: str,
    species_id: int,
    save_path: Optional[str] = None,
) -> None:
    """
    3-D scatter of all camera centres. Should look like a sphere for FineView
    (8 cameras × 40 turntable poses).
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("matplotlib not available — skipping camera centre plot.")
        return

    species_idx = species_id - 1
    centres = []
    with h5py.File(h5_path, "r") as f:
        cams = list_cameras(f)
        np_  = n_poses(f)
        for cam in cams:
            for p in range(np_):
                R, t = extrinsics(f, cam, p)
                centres.append(camera_center(R, t))

    centres = np.array(centres)
    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(centres[:, 0], centres[:, 1], centres[:, 2], s=5, alpha=0.5)
    ax.set_title(f"Camera centres  species_id={species_id}  ({len(cams)}×{np_} views)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
        print(f"Camera centre plot → {save_path}")
    else:
        plt.show()
    plt.close()


# ── Combined smoke test ───────────────────────────────────────────────────────

def run_all_checks(
    h5_path: str,
    species_id: int,
    X_world: Optional[np.ndarray] = None,
) -> bool:
    """
    Run re-center invariant + optional reprojection check.
    Returns True if all pass.

    X_world  known 3-D scene point for reprojection; skip if None.
    """
    all_ok = True
    species_idx = species_id - 1

    print("=" * 60)
    print("FineView → 3DGS pipeline verification")
    print("=" * 60)

    with h5py.File(h5_path, "r") as f:
        cam     = list_cameras(f)[0]
        K, w, h = intrinsics_for_species(f, cam, species_idx)

    print("\n[1] Re-center invariant")
    all_ok &= check_recenter(K, w, h)

    if X_world is not None:
        print("\n[2] Reprojection check")
        u, v      = check_reproject(h5_path, species_id, X_world)
        all_ok   &= (0 <= u < w) and (0 <= v < h)

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return all_ok
