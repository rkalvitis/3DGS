"""
Export FineView species to COLMAP format for Inria gaussian-splatting.

Output layout:
    <out_dir>/
        sparse/0/
            cameras.txt     8 PINHOLE cameras, one per physical lens
            images.txt      one record per (camera, pose)
            points3D.txt    empty stub (or seeded by seed_points.py)
        images/
            camera1/00.png … camera8/39.png  (processed)
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from PIL import Image

from .fineview_io import intrinsics_for_species, extrinsics, list_cameras, n_poses
from .geometry import rotmat2qvec, rotate_camera, recenter_K, pad_params, recenter


# ── Image discovery ────────────────────────────────────────────────────────────

def find_image(
    img_root: Path,
    species_name: str,
    cam: str,                 # 'camera1' … 'camera8'
    pose_idx: int,
    pattern: Optional[str],   # None → auto-detect
) -> Optional[Path]:
    """
    Locate the source PNG for a (camera, pose) pair under img_root/species_name.

    Auto-detection tries several common layouts; supply --img_pattern to override.
    pattern tokens: {species_short}, {cam}, {pp}
    """
    pp = f"{pose_idx:02d}"
    species_short = species_name[4:]  # repo convention: strip 4-char prefix

    if pattern:
        name = pattern.format(species_short=species_short, cam=cam, pp=pp)
        p = img_root / species_name / name
        return p if p.exists() else None

    candidates = [
        img_root / species_name / cam / f"{pp}.png",
        img_root / species_name / f"{species_short}-{cam}-{pp}.png",
        img_root / species_name / f"{cam}-{pp}.png",
        img_root / species_name / f"{species_short}_{cam}_{pp}.png",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Last resort: glob
    import glob
    hits = glob.glob(str(img_root / species_name / "**" / f"*{cam}*{pp}*.png"),
                     recursive=True)
    return Path(hits[0]) if hits else None


# ── Per-camera preprocessing ──────────────────────────────────────────────────

def _peek_image_size(
    img_root: Path,
    species_name: str,
    cam: str,
    n_p: int,
    pattern: Optional[str],
) -> Optional[tuple]:
    """
    Read actual pixel (w, h) from the first available image for a camera.
    Returns the raw stored dimensions (before any rotate_k transformation).
    Returns None if no image is found (e.g. copy_images=False case).
    """
    for pose_idx in range(n_p):
        src = find_image(img_root, species_name, cam, pose_idx, pattern)
        if src is not None:
            with Image.open(src) as img:
                w, h = img.size  # PIL returns (width, height)
            return w, h
    return None


def _compute_camera_params(K, w, h, rotate_k, do_recenter):
    """
    Apply rotation then recentering to K and image dimensions.
    Returns (K_final, w_final, h_final, pl, pt) where (pl, pt) are left/top
    padding in pixels (needed when processing image pixels).
    """
    if rotate_k:
        K, _, _, w, h = rotate_camera(K, np.eye(3), np.zeros((3, 1)), w, h, rotate_k)

    pl = pt = 0
    if do_recenter:
        cx, cy = K[0, 2], K[1, 2]
        pl, pr, pt, pb = pad_params(cx, cy, w, h)
        K, w, h = recenter_K(K, w, h)

    return K, w, h, pl, pt


def find_mask(mask_root: Path, species_name: str, cam: str, pose_idx: int,
              img_pattern: Optional[str]) -> Optional[Path]:
    """
    Locate the binary mask for a (camera, pose) pair.
    Expects the same directory layout as the images but with '_mask' appended
    to the filename stem (e.g. Neophasia_Menapia-001-camera1-00_mask.png).
    """
    pp = f"{pose_idx:02d}"
    species_short = species_name[4:]

    if img_pattern:
        stem = img_pattern.format(species_short=species_short, cam=cam, pp=pp)
        stem = stem.replace(".png", "")
        p = mask_root / species_name / f"{stem}_mask.png"
        return p if p.exists() else None

    candidates = [
        mask_root / species_name / cam / f"{pp}_mask.png",
        mask_root / species_name / f"{species_short}-{cam}-{pp}_mask.png",
        mask_root / species_name / f"{cam}-{pp}_mask.png",
        mask_root / species_name / f"{species_short}_{cam}_{pp}_mask.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _process_and_save_image(
    src: Path, dst: Path, K_cropped, _w, _h, rotate_k, do_recenter,
    mask_src: Optional[Path] = None,
):
    """Load, apply mask, rotate, recenter and save one image.

    With a mask, background pixels are set to black and the mask is written as
    an alpha channel (RGBA) so 3DGS can ignore the background during training
    (render * alpha matches the black GT background → zero loss there).
    Without a mask, a plain RGB image is saved.
    """
    img = np.array(Image.open(src).convert("RGB"))

    if mask_src is not None and mask_src.exists():
        mask = np.array(Image.open(mask_src).convert("L"))
        bg = mask < 128                              # True where background
        img[bg] = 0                                  # black background
        alpha = np.where(bg, 0, 255).astype(np.uint8)
        img = np.dstack([img, alpha])                # RGBA; rotated/recentred together

    if rotate_k:
        img = np.rot90(img, k=rotate_k)

    if do_recenter:
        cx, cy = K_cropped[0, 2], K_cropped[1, 2]
        img, _, _ = recenter(img, cx, cy)

    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(dst)


# ── Main exporter ─────────────────────────────────────────────────────────────

def export_colmap(
    h5_path: str,
    base_path: str,
    species_id: int,
    species_name: str,
    out_dir: str,
    max_poses: int = 40,
    rotate_k: int = 0,
    do_recenter: bool = True,
    resolution: str = "full",
    img_pattern: Optional[str] = None,
    copy_images: bool = True,
    img_dir: Optional[str] = None,
    mask_dir: Optional[str] = None,
    copy_raw_images: bool = False,
) -> None:
    """
    Export one FineView species to COLMAP sparse/0 format.

    Parameters
    ----------
    h5_path          path to camera_parameters.h5
    base_path        FineView dataset root (parent of crop_undistort/)
    species_id       1-based species index (matches /crop arrays)
    species_name     directory name under crop_undistort/
    out_dir          destination directory (created if missing)
    max_poses        number of turntable poses to export (≤ 40)
    rotate_k         k × 90° CCW pixel rotation (0 = off); must pass reprojection check
    do_recenter      pad images so pp → centre (required for vanilla 3DGS)
    resolution       'full' or '8' (selects crop_undistort vs crop_undistort_8)
    img_pattern      override filename template; tokens: {species_short} {cam} {pp}
    copy_images      if False, skip image I/O (useful for pose-only testing)
    img_dir          override image root directory (default: <base_path>/crop_undistort[_8])
    mask_dir         directory containing binary masks (white=insect, black=background).
                     When provided, background is set to black and the mask is written
                     as an alpha channel (RGBA) so 3DGS can ignore the background.
                     Expected layout mirrors img_dir with '_mask' appended to each stem.
    copy_raw_images  if True, also export unmasked (original background) images to
                     images_raw/ alongside the masked images/. Used by COLMAP feature
                     extraction for better SIFT matches on non-white backgrounds.
    """
    out_dir      = Path(out_dir)
    sparse       = out_dir / "sparse" / "0"
    imgs_out     = out_dir / "images"
    raw_imgs_out = out_dir / "images_raw" if copy_raw_images else None
    sparse.mkdir(parents=True, exist_ok=True)
    imgs_out.mkdir(parents=True, exist_ok=True)
    if raw_imgs_out is not None:
        raw_imgs_out.mkdir(parents=True, exist_ok=True)

    if img_dir is not None:
        img_root = Path(img_dir)
    else:
        suffix   = "_8" if resolution == "8" else ""
        img_root = Path(base_path) / f"crop_undistort{suffix}"

    mask_root = Path(mask_dir) if mask_dir else None
    species_idx = species_id - 1

    with h5py.File(h5_path, "r") as f:
        cameras = list_cameras(f)
        n_p     = min(max_poses, n_poses(f))

        # ── cameras.txt ──────────────────────────────────────────────────────
        cam_params = {}  # cam → (K_final, w_final, h_final, pl, pt, K_scaled)
        for cam in cameras:
            K_crop, w_crop, h_crop = intrinsics_for_species(f, cam, species_idx)

            # Use actual on-disk pixel dimensions so cameras.txt matches what
            # COLMAP feature_extractor reads from the image files.  Falls back
            # to h5 metadata when images are not copied (copy_images=False).
            actual = _peek_image_size(img_root, species_name, cam, n_p, img_pattern)
            if actual is not None:
                w_use, h_use = actual
                # Scale K to match actual image resolution.
                # h5 K is for the full-resolution crop; resolution='8' images
                # are 8× smaller, so fx/fy/cx/cy must be scaled accordingly.
                sx = w_use / w_crop
                sy = h_use / h_crop
                K_scaled = K_crop.copy()
                K_scaled[0, 0] *= sx   # fx
                K_scaled[1, 1] *= sy   # fy
                K_scaled[0, 2] *= sx   # cx
                K_scaled[1, 2] *= sy   # cy
            else:
                w_use, h_use = w_crop, h_crop
                K_scaled = K_crop.copy()

            K_final, w_f, h_f, pl, pt = _compute_camera_params(
                K_scaled, w_use, h_use, rotate_k, do_recenter
            )
            cam_params[cam] = (K_final, w_f, h_f, pl, pt, K_scaled)

        with open(sparse / "cameras.txt", "w") as cf:
            cf.write("# Camera list with one line of data per camera:\n")
            cf.write("#   CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
            for cam_id, cam in enumerate(cameras, 1):
                K, w, h = cam_params[cam][:3]
                fx, fy  = K[0, 0], K[1, 1]
                cx, cy  = K[0, 2], K[1, 2]
                cf.write(f"{cam_id} PINHOLE {w} {h} "
                         f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

        # ── images.txt ───────────────────────────────────────────────────────
        with open(sparse / "images.txt", "w") as imf:
            imf.write("# Image list with two lines of data per image:\n")
            imf.write("#   IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
            imf.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

            from tqdm import tqdm

            total = len(cameras) * n_p
            image_id = 0
            with tqdm(total=total, desc="Exporting images", unit="img", ncols=60) as bar:
                for cam_id, cam in enumerate(cameras, 1):
                    K_final, w_cam, h_cam, _, _, K_scaled = cam_params[cam]

                    for pose_idx in range(n_p):
                        R, t = extrinsics(f, cam, pose_idx)

                        if rotate_k:
                            # Use scaled K and actual dims for correct R/t rotation
                            _, R, t, _, _ = rotate_camera(
                                K_scaled.copy(), R, t, w_cam, h_cam, rotate_k
                            )

                        t_vec = t.ravel()
                        qvec  = rotmat2qvec(R)
                        image_id += 1
                        img_name = f"{cam}/{pose_idx:02d}.png"

                        imf.write(
                            f"{image_id} "
                            f"{qvec[0]:.9f} {qvec[1]:.9f} {qvec[2]:.9f} {qvec[3]:.9f} "
                            f"{t_vec[0]:.9f} {t_vec[1]:.9f} {t_vec[2]:.9f} "
                            f"{cam_id} {img_name}\n"
                        )
                        imf.write("\n")  # empty POINTS2D line

                        # ── image file ───────────────────────────────────────
                        if copy_images:
                            src = find_image(img_root, species_name, cam,
                                             pose_idx, img_pattern)
                            if src is None:
                                tqdm.write(f"WARNING: image not found for {cam}/pose{pose_idx:02d}")
                            else:
                                mask_src = find_mask(
                                    mask_root, species_name, cam, pose_idx, img_pattern
                                ) if mask_root else None
                                dst = imgs_out / cam / f"{pose_idx:02d}.png"
                                _process_and_save_image(
                                    src, dst, K_scaled, w_cam, h_cam,
                                    rotate_k, do_recenter, mask_src=mask_src
                                )
                                if raw_imgs_out is not None:
                                    raw_dst = raw_imgs_out / cam / f"{pose_idx:02d}.png"
                                    _process_and_save_image(
                                        src, raw_dst, K_scaled, w_cam, h_cam,
                                        rotate_k, do_recenter, mask_src=None
                                    )

                        bar.set_postfix_str(cam)
                        bar.update()

        # ── points3D.txt (empty stub) ────────────────────────────────────────
        with open(sparse / "points3D.txt", "w") as pf:
            pf.write("# 3D point list (empty — run seed_points.py to populate)\n")
            pf.write("#   POINT3D_ID X Y Z R G B ERROR TRACK[]\n")

    print(f"COLMAP export done → {out_dir}")
    print(f"  cameras: {len(cameras)}, images: {image_id}, poses/cam: {n_p}")
    if do_recenter:
        print("  principal-point recentering: ON (cropped to centre, no black borders)")
    print("  Smoke-test: train ~7k iters; renders should align with no blurry edges.")
