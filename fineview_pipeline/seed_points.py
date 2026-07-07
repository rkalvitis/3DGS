"""
Seed points3D.txt from FineView's 2-D correspondences via DLT triangulation.

FineView stores correspondences in:
    correspondence_undistort/<SPECIES_NAME>/correspondence_coordinate.h5

INSPECT FIRST:
    python3 -c "
    import h5py
    with h5py.File('correspondence_coordinate.h5','r') as f:
        def show(name, obj): print(name, getattr(obj,'shape',''), getattr(obj,'dtype',''))
        f.visititems(show)
    "
Expected layout (based on the repo's 3d_reconstruction.py usage):
    /pts2d   float (N_points, N_cameras, 2)   pixel (u,v) per view; NaN = invisible
    /pts3d   float (N_points, 3)              optional DLT output stored alongside
    (may differ — adjust load_correspondences() accordingly)
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np

from .fineview_io import intrinsics_for_species, extrinsics, list_cameras, n_poses


# ── DLT triangulation ─────────────────────────────────────────────────────────

def triangulate_dlt(
    Ps: List[np.ndarray],   # list of (3,4) camera projection matrices
    xys: List[np.ndarray],  # list of (2,) pixel coords aligned with Ps
) -> np.ndarray:
    """
    DLT triangulation from N≥2 views. Returns homogeneous 3-D point (3,).
    Builds: A[2i]   = x[i]*P[i][2] - P[i][0]
            A[2i+1] = y[i]*P[i][2] - P[i][1]
    Solution: right singular vector of A corresponding to smallest singular value.
    """
    rows = []
    for P, xy in zip(Ps, xys):
        u, v = float(xy[0]), float(xy[1])
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.array(rows)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return (X[:3] / X[3]).astype(float)


def build_projection(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 3×4 camera projection matrix P = K @ [R | t]."""
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


# ── H5 reader (adjust to actual layout) ───────────────────────────────────────

def load_correspondences(
    corr_h5_path: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load 2-D correspondences from correspondence_coordinate.h5.

    Returns
    -------
    pts2d : (N, C, 2)  pixel coords per point per camera; NaN = invisible
    pts3d : (N, 3) or None  if pre-triangulated coords are stored

    *** VERIFY the actual key names against your download before running. ***
    """
    with h5py.File(corr_h5_path, "r") as f:
        keys = list(f.keys())

        # Try known key patterns; adjust if your file differs
        if "pts2d" in keys:
            pts2d = np.array(f["pts2d"], dtype=float)
        elif "correspondence_coordinate" in keys:
            pts2d = np.array(f["correspondence_coordinate"], dtype=float)
        else:
            raise KeyError(
                f"Could not find 2D correspondence key. Available keys: {keys}\n"
                "Edit load_correspondences() to match your h5 layout."
            )

        pts3d = np.array(f["pts3d"], dtype=float) if "pts3d" in keys else None

    # Normalise to (N, C, 2) if needed
    if pts2d.ndim == 2:
        # Possibly (N, C*2) — reshape
        n = pts2d.shape[0]
        c = pts2d.shape[1] // 2
        pts2d = pts2d.reshape(n, c, 2)

    return pts2d, pts3d


# ── Build coloured point cloud ────────────────────────────────────────────────

def build_point_cloud(
    h5_params_path: str,
    corr_h5_path: str,
    species_id: int,
    img_dir: Optional[str] = None,
    min_views: int = 2,
    pose_index: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Triangulate 3-D points from correspondences and optionally colour them from images.

    Parameters
    ----------
    h5_params_path  camera_parameters.h5
    corr_h5_path    correspondence_coordinate.h5 for the species
    species_id      1-based
    img_dir         directory containing processed images (for colour sampling)
    min_views       minimum visible views to triangulate a point
    pose_index      turntable pose whose correspondences to use

    Returns list of (X_world (3,), rgb (3,)) tuples.
    """
    pts2d, pts3d_precomp = load_correspondences(corr_h5_path)
    # pts2d: (N, C, 2) where C = number of cameras

    species_idx = species_id - 1

    with h5py.File(h5_params_path, "r") as f:
        cameras = list_cameras(f)

        # Build (K, R, t) for each camera at the given pose
        cam_data = []
        for cam in cameras:
            K, w, h = intrinsics_for_species(f, cam, species_idx)
            R, t    = extrinsics(f, cam, pose_index)
            cam_data.append((K, R, t, w, h, cam))

    # If pre-triangulated 3D points exist, skip DLT
    if pts3d_precomp is not None:
        print(f"  Using {len(pts3d_precomp)} pre-triangulated points from h5.")
        points = []
        for i, X in enumerate(pts3d_precomp):
            rgb = _sample_colour(pts2d[i], cam_data, img_dir)
            points.append((np.array(X, float), rgb))
        return points

    # DLT triangulation
    points = []
    n_pts  = pts2d.shape[0]
    n_cams = min(pts2d.shape[1], len(cam_data))

    for i in range(n_pts):
        Ps, xys = [], []
        for c in range(n_cams):
            xy = pts2d[i, c]
            if np.any(np.isnan(xy)) or np.any(xy < 0):
                continue
            K, R, t = cam_data[c][:3]
            Ps.append(build_projection(K, R, t))
            xys.append(xy)

        if len(Ps) < min_views:
            continue

        try:
            X = triangulate_dlt(Ps, xys)
        except Exception:
            continue

        rgb = _sample_colour(pts2d[i], cam_data, img_dir)
        points.append((X, rgb))

    print(f"  Triangulated {len(points)} / {n_pts} points (min_views={min_views})")
    return points


def _sample_colour(
    pt2d_row: np.ndarray,   # (C, 2) per-camera px coords for this point
    cam_data: list,
    img_dir: Optional[str],
) -> np.ndarray:
    """Sample RGB from the first visible camera; returns [128,128,128] if unavailable."""
    if img_dir is None:
        return np.array([128, 128, 128], dtype=np.uint8)

    from PIL import Image as PILImage
    img_dir = Path(img_dir)

    for c, xy in enumerate(pt2d_row):
        if np.any(np.isnan(xy)) or np.any(xy < 0):
            continue
        cam = cam_data[c][5]  # cam name stored at index 5
        # Try to open any image for this camera (pose 00 as reference)
        for pp in range(40):
            p = img_dir / cam / f"{pp:02d}.png"
            if p.exists():
                img = np.array(PILImage.open(p))
                u, v = int(round(xy[0])), int(round(xy[1]))
                h_im, w_im = img.shape[:2]
                if 0 <= v < h_im and 0 <= u < w_im:
                    return img[v, u, :3]
                break

    return np.array([128, 128, 128], dtype=np.uint8)


# ── PCD → PLY conversion (structured-light seed cloud) ───────────────────────

def pcd_to_ply(pcd_path: Path, ply_path: Path) -> int:
    """
    Convert a binary PCD file (FIELDS x y z rgb, TYPE F F F F) to a PLY file
    readable by 3DGS.  The rgb float encodes 4 bytes as 0xAARRGGBB.

    Returns the number of points written.
    """
    import struct

    pcd_path = Path(pcd_path)
    ply_path = Path(ply_path)

    # ── Parse PCD header ──────────────────────────────────────────────────────
    n_pts = 0
    with pcd_path.open("rb") as f:
        while True:
            raw = f.readline()
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("POINTS"):
                n_pts = int(line.split()[1])
            if line.startswith("DATA"):
                break
        binary_data = f.read()

    # Each point: 4 × float32 → x, y, z, rgb_packed
    pts = np.frombuffer(binary_data, dtype=np.float32).reshape(-1, 4)
    xyz = pts[:, :3]
    rgb_packed = pts[:, 3].view(np.uint32)
    r = ((rgb_packed >> 16) & 0xFF).astype(np.uint8)
    g = ((rgb_packed >>  8) & 0xFF).astype(np.uint8)
    b = ( rgb_packed        & 0xFF).astype(np.uint8)

    # ── Write PLY ─────────────────────────────────────────────────────────────
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)
    with ply_path.open("wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack("<fff", float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])))
            f.write(struct.pack("<fff", 0.0, 0.0, 0.0))
            f.write(struct.pack("<BBB", int(r[i]), int(g[i]), int(b[i])))

    return n


def install_pcd_seed(corr_dir: str, species_name: str, sparse_dir: Path) -> int:
    """
    Find <corr_dir>/<species_name>/<species_name>.pcd and write it as
    sparse/0/points3D.ply so 3DGS uses it instead of the COLMAP sparse cloud.

    Returns number of points installed, or 0 if no pcd found.
    """
    pcd = Path(corr_dir) / species_name / f"{species_name}.pcd"
    if not pcd.exists():
        print(f"  WARNING: no .pcd found at {pcd}")
        return 0
    ply_path = Path(sparse_dir) / "points3D.ply"
    n = pcd_to_ply(pcd, ply_path)
    print(f"  Installed {n} structured-light points → {ply_path}")
    return n


# ── Write points3D.txt ────────────────────────────────────────────────────────

def write_points3d(
    points: List[Tuple[np.ndarray, np.ndarray]],
    out_path: str,
) -> None:
    """
    Write a COLMAP-format points3D.txt from a list of (X_world, rgb) tuples.
    All points have zero reprojection error and an empty track (no image links).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        for pid, (X, rgb) in enumerate(points, 1):
            r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
            f.write(f"{pid} {X[0]:.6f} {X[1]:.6f} {X[2]:.6f} "
                    f"{r} {g} {b} 0.0\n")
    print(f"Wrote {len(points)} points → {out_path}")
