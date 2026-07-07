"""
COLMAP triangulation for FineView known-pose data.

Mirrors the pattern in convert.py (which handles the unknown-pose SfM case)
but replaces `mapper` with `point_triangulator` because FineView camera poses
are already known from calibration.

Pipeline
--------
1. feature_extractor  – detect SIFT keypoints in every image (masked images/
                        by default; --raw_images switches to images_raw/)
2. exhaustive_matcher – match keypoints across all image pairs
3. point_triangulator – triangulate 3-D points using known poses from sparse/0/
4. bundle_adjuster    – OPT-IN via --bundle_adjustment; off by default because
                        the calibrated h5 poses are more accurate than BA on
                        this weakly connected match graph

Prerequisites
-------------
  - export_colmap() must have run first (creates sparse/0/cameras.txt,
    images.txt, points3D.txt and copies images/ into <out_dir>)
  - COLMAP must be on PATH  (or pass --colmap_executable)

Usage
-----
python3.10 -m fineview_pipeline.run_colmap \\
    --out      /output/species1 \\
    --no_gpu                          # omit for GPU systems

Or call from Python:
    from fineview_pipeline.run_colmap import run_colmap_triangulation
    run_colmap_triangulation("/output/species1")
"""
from __future__ import annotations
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def _run(cmd: list[str]) -> None:
    """Run a COLMAP sub-command; raise on non-zero exit."""
    print("\n$ " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error("Command failed (exit %d): %s", result.returncode, cmd[0])
        sys.exit(result.returncode)


def _sync_db_cameras(db_path: Path, sparse_dir: Path) -> None:
    """
    Align the COLMAP database with cameras.txt so that point_triangulator's
    camera-consistency checks pass in COLMAP ≥ 4.x.

    COLMAP 4.x stores cameras, rigs, frames, and frame_data.  After
    feature_extractor runs (one camera per image), the DB has N cameras
    (IDs 1…N) and N rigs/frames.  cameras.txt has 8 cameras (IDs 1…8), one
    per physical lens.

    This function:
      1. Replaces the cameras table with the 8 cameras from cameras.txt.
      2. Rebuilds rigs (8 single-camera rigs, one per lens).
      3. Re-links each frame_data row and image row to its correct camera_id
         based on subfolder name (camera1/ → camera_id=1, etc.).
    """
    import sqlite3
    import re
    import struct

    # ── Parse cameras.txt ──────────────────────────────────────────────────
    cameras_txt = sparse_dir / "cameras.txt"
    db_cameras: dict = {}       # {cam_id: (model, w, h, params_str)}
    with cameras_txt.open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            cam_id = int(parts[0])
            db_cameras[cam_id] = (parts[1], int(parts[2]), int(parts[3]),
                                  ",".join(parts[4:]))

    # ── Parse images.txt → {image_name: cam_id} ───────────────────────────
    images_txt = sparse_dir / "images.txt"
    img_cam: dict = {}
    with images_txt.open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 10:        # skip POINTS2D lines
                continue
            img_cam[parts[9]] = int(parts[8])

    MODEL_IDS = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2,
                 "RADIAL": 3, "OPENCV": 4, "FULL_OPENCV": 5}

    conn = sqlite3.connect(str(db_path))
    cur  = conn.cursor()

    # ── Build image_id → cam_id mapping ───────────────────────────────────
    cur.execute("SELECT image_id, name FROM images")
    img_rows = cur.fetchall()
    img_id_to_cam: dict = {}
    for image_id, name in img_rows:
        cam_id = img_cam.get(name)
        if cam_id is None:
            folder = name.split("/")[0]
            m = re.search(r"(\d+)$", folder)
            cam_id = int(m.group(1)) if m else 1
        img_id_to_cam[image_id] = cam_id

    # ── Replace cameras ────────────────────────────────────────────────────
    cur.execute("DELETE FROM cameras")
    for cam_id, (model, w, h, params_str) in db_cameras.items():
        params = [float(x) for x in params_str.split(",")]
        blob   = struct.pack(f"{len(params)}d", *params)
        cur.execute(
            "INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (cam_id, MODEL_IDS.get(model, 1), w, h, blob),
        )

    # ── Update images table ────────────────────────────────────────────────
    for image_id, cam_id in img_id_to_cam.items():
        cur.execute("UPDATE images SET camera_id = ? WHERE image_id = ?",
                    (cam_id, image_id))

    # ── Rebuild rigs (8 single-camera rigs, one per physical lens) ─────────
    cur.execute("DELETE FROM rigs")
    cur.execute("DELETE FROM rig_sensors")
    for cam_id in db_cameras:
        cur.execute(
            "INSERT INTO rigs (rig_id, ref_sensor_id, ref_sensor_type) VALUES (?, ?, 0)",
            (cam_id, cam_id),
        )

    # ── Update frames: assign each frame to the rig of its image ──────────
    # frame_data: (frame_id, data_id=image_id, sensor_id=cam_id, sensor_type)
    cur.execute("SELECT frame_id, data_id FROM frame_data")
    fd_rows = cur.fetchall()
    for frame_id, image_id in fd_rows:
        cam_id = img_id_to_cam.get(image_id, 1)
        cur.execute("UPDATE frame_data SET sensor_id = ? WHERE frame_id = ? AND data_id = ?",
                    (cam_id, frame_id, image_id))
        # Each frame → one rig (the rig for that camera)
        cur.execute("UPDATE frames SET rig_id = ? WHERE frame_id = ?",
                    (cam_id, frame_id))

    conn.commit()
    conn.close()
    print(f"  DB synced: {len(db_cameras)} cameras, {len(img_rows)} images reassigned.")


def _rewrite_images_txt(db_path: Path, sparse_dir: Path) -> None:
    """
    Rewrite images.txt so image_ids exactly match the database.

    COLMAP 4.x compares frame DataIds() between the sparse model and the
    database.  With --clear_points 1 (default), COLMAP updates image_ids in
    the model by filename but may not propagate that update into Frame.DataIds,
    causing a mismatch.  By rewriting images.txt with the DB's image_ids and
    using --clear_points 0, we bypass the recomputation entirely.
    """
    import sqlite3

    # Build name → db_image_id map
    conn = sqlite3.connect(str(db_path))
    name_to_id = {name: iid for iid, name in
                  conn.execute("SELECT image_id, name FROM images").fetchall()}
    conn.close()

    images_txt = sparse_dir / "images.txt"
    lines = images_txt.read_text().splitlines(keepends=True)
    out_lines = []
    skip_next = False
    for line in lines:
        if line.startswith("#"):
            out_lines.append(line)
            continue
        if skip_next:
            # POINTS2D line (may be blank or have data) — pass through verbatim
            out_lines.append(line)
            skip_next = False
            continue
        parts = line.split()
        if not parts:
            out_lines.append(line)
            continue
        # Pose line: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        if len(parts) >= 10:
            img_name = parts[9]
            db_id    = name_to_id.get(img_name)
            if db_id is None:
                raise ValueError(f"Image '{img_name}' from images.txt not found in DB")
            parts[0] = str(db_id)
            out_lines.append(" ".join(parts) + "\n")
            skip_next = True          # next line is POINTS2D
        else:
            out_lines.append(line)

    images_txt.write_text("".join(out_lines))
    print(f"  images.txt rewritten with DB image_ids ({len(name_to_id)} images).")


def _filter_points3d(pts_bin: Path, max_dist_factor: float = 4.0) -> int:
    """
    Remove spatial outliers from points3D.bin in-place.

    Points whose distance from the median centroid exceeds
    max_dist_factor × median_distance are dropped.  This prevents extreme
    outlier points (sometimes triangulated at infinity) from inflating the
    3DGS scene bounding box and causing colour divergence (green renders).

    Returns the number of points kept.
    """
    import struct
    import numpy as np

    with pts_bin.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        records = []
        for _ in range(n):
            pid   = struct.unpack("<Q", f.read(8))[0]
            xyz   = struct.unpack("<3d", f.read(24))
            rgb   = struct.unpack("<3B", f.read(3))
            err   = struct.unpack("<d",  f.read(8))[0]
            tlen  = struct.unpack("<Q",  f.read(8))[0]
            track = f.read(8 * tlen)
            records.append((pid, xyz, rgb, err, tlen, track))

    pts = np.array([r[1] for r in records])
    centroid = np.median(pts, axis=0)
    dists    = np.linalg.norm(pts - centroid, axis=1)
    threshold = max_dist_factor * np.median(dists)
    keep = dists <= threshold

    n_removed = int((~keep).sum())
    if n_removed:
        print(f"  Outlier filter: removed {n_removed} points "
              f"(>{threshold:.1f} units from median centre). "
              f"{keep.sum()} points kept.")

    with pts_bin.open("wb") as f:
        f.write(struct.pack("<Q", int(keep.sum())))
        for i, (pid, xyz, rgb, err, tlen, track) in enumerate(records):
            if not keep[i]:
                continue
            f.write(struct.pack("<Q", pid))
            f.write(struct.pack("<3d", *xyz))
            f.write(struct.pack("<3B", *rgb))
            f.write(struct.pack("<d",  err))
            f.write(struct.pack("<Q",  tlen))
            f.write(track)

    return int(keep.sum())


def run_colmap_triangulation(
    out_dir: str,
    colmap_bin: str = "colmap",
    use_gpu: bool = True,
    use_raw_images: bool = False,
    run_ba: bool = False,
) -> None:
    """
    Run COLMAP feature extraction → matching → triangulation on a directory
    that was already prepared by export_colmap().

    Expected layout of out_dir before calling this function:
        sparse/0/cameras.txt        known camera intrinsics
        sparse/0/images.txt         known camera poses (world→cam)
        sparse/0/points3D.txt       empty stub (will be overwritten)
        images/camera1/00.png …    processed images

    After this function completes, sparse/0/points3D.txt is populated with
    COLMAP-triangulated 3-D points coloured from image features — ready for
    3DGS training.

    Parameters
    ----------
    out_dir         directory produced by export_colmap()
    colmap_bin      path to colmap executable (default: 'colmap' on PATH)
    use_gpu         enable GPU for SIFT extraction and matching
    use_raw_images  extract SIFT on images_raw/ (unmasked) instead of images/.
                    Off by default: the background is static while the specimen
                    rotates, so background keypoints contradict the known rig
                    poses and pollute triangulation.
    run_ba          run bundle_adjuster after triangulation. Off by default:
                    the h5 calibration is metrically accurate, and the match
                    graph here is too weakly connected (mostly two-view,
                    within-camera tracks) to constrain BA — on
                    009-Neophasia_Menapia it moved cameras by up to 418 units
                    (orbit radius ~343) and broke training.
    """
    out_dir    = Path(out_dir)
    db_path    = out_dir / "database.db"
    sparse_dir = out_dir / "sparse" / "0"
    images_dir = out_dir / "images"
    raw_dir    = out_dir / "images_raw"
    feat_dir   = raw_dir if (use_raw_images and raw_dir.exists()) else images_dir

    if not sparse_dir.exists():
        raise FileNotFoundError(
            f"sparse/0/ not found in {out_dir}. "
            "Run export_colmap() first."
        )
    if not images_dir.exists():
        raise FileNotFoundError(
            f"images/ not found in {out_dir}. "
            "Run export_colmap() first (with copy_images=True)."
        )

    from tqdm import tqdm
    import sqlite3 as _sqlite3

    # Check whether features are already extracted so we can skip re-extraction
    # on a restart after a matching crash.
    def _features_done(db: Path, expected: int) -> bool:
        if not db.exists():
            return False
        try:
            with _sqlite3.connect(str(db)) as c:
                n = c.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            return n >= expected
        except Exception:
            return False

    # The DB is only reusable if its features came from the same image set
    # (masked vs raw) — the image count is identical either way, so track the
    # source dir in a sidecar file and invalidate on mismatch.
    feat_src_marker = db_path.with_suffix(".feat_src")
    same_source = (feat_src_marker.exists()
                   and feat_src_marker.read_text().strip() == feat_dir.name)

    n_images = len(list(feat_dir.rglob("*.png")))
    if same_source and _features_done(db_path, n_images):
        print(f"  Feature extraction already done ({n_images} images in DB) — skipping.")
        steps = []
    else:
        if db_path.exists():
            db_path.unlink()
        feat_src_marker.write_text(feat_dir.name)
        if feat_dir != images_dir:
            print(f"  Using raw (unmasked) images for feature extraction: {feat_dir}")
        feat_cmd = [
            colmap_bin, "feature_extractor",
            "--database_path",             str(db_path),
            "--image_path",                str(feat_dir),
            "--ImageReader.camera_model",  "PINHOLE",
            "--FeatureExtraction.use_gpu", "1" if use_gpu else "0",
        ]
        steps = [("Feature extraction", feat_cmd)]

    # The COLMAP 4.x CPU SIFT matcher crashes with SIGSEGV on Apple Silicon
    # (deterministic, not thread-related).  Force Metal GPU matching on Mac
    # regardless of use_gpu — fall back to CPU only if explicitly requested
    # via use_gpu=False AND the platform is not Darwin.
    import platform as _platform
    match_use_gpu = use_gpu or (_platform.system() == "Darwin")
    match_cmd = [
        colmap_bin, "exhaustive_matcher",
        "--database_path",           str(db_path),
        "--FeatureMatching.use_gpu", "1" if match_use_gpu else "0",
    ]
    steps.append(("Feature matching", match_cmd))

    with tqdm(steps, desc="COLMAP", unit="step", ncols=60) as bar:
        for label, cmd in bar:
            bar.set_postfix_str(label)
            _run(cmd)

    # Align DB cameras/rigs/frames with cameras.txt (COLMAP 4.x compatibility).
    print("\nSyncing database cameras with cameras.txt …")
    _sync_db_cameras(db_path, sparse_dir)

    # Rewrite images.txt with the DB's actual image_ids so we can use
    # --clear_points 0 and avoid the DataIds frame-consistency check in COLMAP 4.x.
    print("Rewriting images.txt with DB image_ids …")
    _rewrite_images_txt(db_path, sparse_dir)

    tri_cmd = [
        colmap_bin, "point_triangulator",
        "--database_path",                     str(db_path),
        "--image_path",                        str(images_dir),
        "--input_path",                        str(sparse_dir),
        "--output_path",                       str(sparse_dir),
        "--clear_points",                      "0",
        "--Mapper.tri_ignore_two_view_tracks", "0",
        "--Mapper.tri_min_angle",              "1.0",
        "--Mapper.tri_complete_max_reproj_error", "8.0",
    ]

    with tqdm([("Point triangulation", tri_cmd)],
              desc="COLMAP", unit="step", ncols=60) as bar:
        for label, cmd in bar:
            bar.set_postfix_str(label)
            _run(cmd)

    # Count triangulated points (before any filtering so BA sees a consistent model)
    pts_bin = sparse_dir / "points3D.bin"
    pts_txt = sparse_dir / "points3D.txt"
    if pts_bin.exists() and pts_bin.stat().st_size > 8:
        import struct as _struct
        with pts_bin.open("rb") as _f:
            n_pts = _struct.unpack("<Q", _f.read(8))[0]
        print(f"\nTriangulation done. points3D.bin has {n_pts} raw points.")
    elif pts_txt.exists():
        n_pts = sum(1 for l in pts_txt.open() if l.strip() and not l.startswith("#"))
        print(f"\nTriangulation done. points3D.txt has {n_pts} points.")
    else:
        print("\nWARNING: could not find points3D output.")
        n_pts = 0

    # Bundle adjustment — opt-in (run_ba=True). Disabled by default: the h5
    # calibration is the ground truth here, and the weakly connected match
    # graph lets BA shear individual camera poses far from the rig geometry.
    # If enabled, it must run BEFORE outlier filtering so that images.bin and
    # points3D.bin are consistent (filtering removes points but not the
    # corresponding image tracks, causing BA to crash with "Point3D does not exist").
    if run_ba and n_pts > 0:
        print("\nRunning bundle adjustment to refine camera poses …")
        ba_cmd = [
            colmap_bin, "bundle_adjuster",
            "--input_path",                                   str(sparse_dir),
            "--output_path",                                  str(sparse_dir),
            "--BundleAdjustment.refine_focal_length",         "0",
            "--BundleAdjustment.refine_principal_point",      "0",
            "--BundleAdjustment.refine_extra_params",         "0",
        ]
        _run(ba_cmd)
        print("  Bundle adjustment done — images.bin updated with refined poses.")

    # Outlier filter — runs after the (optional) BA so we filter from the final model.
    if pts_bin.exists() and pts_bin.stat().st_size > 8:
        n_pts = _filter_points3d(pts_bin)
        print(f"  After outlier filter: {n_pts} points kept.")

    print(f"\nCOLMAP done. Output: {sparse_dir}")
    print("Next step: train 3DGS")
    print(f"  python train.py -s {out_dir} -m <model_out> --eval")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Run COLMAP triangulation on a FineView export directory"
    )
    p.add_argument("--out",     required=True,
                   help="Directory produced by export_colmap() "
                        "(contains sparse/0/ and images/)")
    p.add_argument("--colmap_executable", default="colmap",
                   help="Path to COLMAP binary (default: 'colmap' on PATH)")
    p.add_argument("--no_gpu",  action="store_true",
                   help="Disable GPU for SIFT extraction and matching")
    p.add_argument("--raw_images", action="store_true",
                   help="Extract SIFT on images_raw/ (unmasked) instead of images/. "
                        "Not recommended: static-background keypoints contradict "
                        "the known rig poses.")
    p.add_argument("--bundle_adjustment", action="store_true",
                   help="Run bundle_adjuster after triangulation. Not recommended: "
                        "the h5 calibration is more accurate than BA on this data.")
    args = p.parse_args(argv)

    run_colmap_triangulation(
        out_dir        = args.out,
        colmap_bin     = args.colmap_executable,
        use_gpu        = not args.no_gpu,
        use_raw_images = args.raw_images,
        run_ba         = args.bundle_adjustment,
    )


if __name__ == "__main__":
    main()
