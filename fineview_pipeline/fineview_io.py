"""
H5 readers for FineView camera_parameters.h5.

Schema (verified by inspection):
  /camera1 … /camera8
      mtx              (3,3)   intrinsic K, full-frame
      mtx_undistort    (3,3)   K for undistorted images  ← we use this
      dist             (1,5)   already applied, ignored
      rvec             (40,3,3) rotation matrices (NOT Rodrigues)
      tvec             (40,3,1) translations
  /crop
      offset           (213,2) (x,y) crop top-left per species
      img_size         (213,2) (w,h) crop size per species
"""
from __future__ import annotations
import h5py
import numpy as np
from typing import List, Tuple


def intrinsics_for_species(
    f_param: h5py.File,
    cam: str,
    species_index: int,
) -> Tuple[np.ndarray, int, int]:
    """
    Return (K, w, h) with principal point shifted from full-frame to crop coords.

    species_index is 0-based (species_id - 1).
    This is the single most important correction: without it cx/cy is hundreds
    of pixels wrong (full-frame pp vs. cropped-image pp).
    """
    g = f_param[cam]
    K = np.array(g["mtx_undistort"], dtype=float).copy()  # (3,3)

    offset = np.array(f_param["crop"]["offset"][species_index])   # (x, y)
    size   = np.array(f_param["crop"]["img_size"][species_index])  # (w, h)

    K[0, 2] -= float(offset[0])   # cx -= offset_x
    K[1, 2] -= float(offset[1])   # cy -= offset_y

    return K, int(size[0]), int(size[1])


def extrinsics(
    f_param: h5py.File,
    cam: str,
    pose_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (R, t) for the world→camera transform (OpenCV convention).
    R : (3,3) — rvec is already a rotation matrix, do NOT call cv2.Rodrigues.
    t : (3,1)
    """
    g = f_param[cam]
    R = np.array(g["rvec"][pose_index], dtype=float)          # (3,3)
    t = np.array(g["tvec"][pose_index], dtype=float).reshape(3, 1)
    return R, t


def list_cameras(f_param: h5py.File) -> List[str]:
    return sorted(k for k in f_param.keys() if k.startswith("camera"))


def n_poses(f_param: h5py.File, cam: str = "camera1") -> int:
    return int(f_param[cam]["rvec"].shape[0])
