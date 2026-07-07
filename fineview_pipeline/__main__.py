"""
CLI entry point for FineView → 3DGS (COLMAP) conversion.

Usage
-----
# Export poses + images, then run COLMAP triangulation in one command:
python3.10 -m fineview_pipeline \\
    --base_path /data/fineview \\
    --h5        /data/fineview/camera_parameters.h5 \\
    --species_id   1 \\
    --species_name 0001Abies_balsamea \\
    --out          /output/species1 \\
    --resolution   8 \\
    --recenter \\
    --run_colmap

# Inspect h5 schema before running (no export):
python3.10 -m fineview_pipeline --inspect_h5 /data/fineview/camera_parameters.h5
python3.10 -m fineview_pipeline --inspect_h5 /data/fineview/correspondence_undistort/<SPECIES>/correspondence_coordinate.h5
"""
from __future__ import annotations
import argparse
import sys


def _inspect_h5(path: str) -> None:
    import h5py
    print(f"\nH5 schema: {path}\n{'─'*60}")
    with h5py.File(path, "r") as f:
        def _show(name, obj):
            shape = getattr(obj, "shape", "")
            dtype = getattr(obj, "dtype", "")
            print(f"  {name:<50} {str(shape):<20} {dtype}")
        f.visititems(_show)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Convert FineView species → COLMAP format for 3DGS training",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # H5 / paths
    p.add_argument("--h5",           required=False,
                   help="Path to camera_parameters.h5  "
                        "(default: <base_path>/camera_parameters.h5)")
    p.add_argument("--base_path",    required=False,
                   help="FineView dataset root (parent of crop_undistort/)")
    p.add_argument("--species_id",   type=int, required=False,
                   help="1-based species index into /crop arrays")
    p.add_argument("--species_name", required=False,
                   help="Species directory name under crop_undistort/")
    p.add_argument("--out",          required=False, default="./fineview_out",
                   help="Output directory (default: ./fineview_out)")

    # Export options
    p.add_argument("--resolution",   choices=["full", "8"], default="full",
                   help="'full' = crop_undistort  '8' = crop_undistort_8")
    p.add_argument("--recenter",     action="store_true", default=True,
                   help="Pad images so pp → image centre (required for vanilla 3DGS)")
    p.add_argument("--no_recenter",  action="store_false", dest="recenter",
                   help="Disable recentering (only for a cx/cy-aware 3DGS build)")
    p.add_argument("--rotate_k",     type=int, default=0,
                   help="k × 90° CCW pixel rotation; verify with reprojection check")
    p.add_argument("--max_poses",    type=int, default=40,
                   help="Max turntable poses per camera (default: 40)")
    p.add_argument("--img_pattern",  default=None,
                   help="Override image filename template. "
                        "Tokens: {species_short} {cam} {pp}  "
                        "e.g. '{species_short}-{cam}-{pp}.png'")
    p.add_argument("--no_images",    action="store_true", default=False,
                   help="Skip image I/O (pose-only export for quick debugging)")
    p.add_argument("--img_dir",      default=None,
                   help="Override image root directory "
                        "(default: <base_path>/crop_undistort or crop_undistort_8). "
                        "Use e.g. <base_path>/imgs_filtered/crop_undistort.")
    p.add_argument("--mask_dir",     default=None,
                   help="Directory containing binary masks (white=insect, black=background). "
                        "Mirrors img_dir structure with '_mask' appended to filenames. "
                        "e.g. <base_path>/imgs_filtered/crop_mask_undistort")

    # COLMAP triangulation
    p.add_argument("--run_colmap",        action="store_true", default=False,
                   help="After export, run COLMAP feature extraction + matching + "
                        "point_triangulator to populate points3D.txt")
    p.add_argument("--colmap_executable", default="colmap",
                   help="Path to colmap binary (default: 'colmap' on PATH)")
    p.add_argument("--no_gpu",            action="store_true", default=False,
                   help="Disable GPU for COLMAP SIFT extraction and matching")
    p.add_argument("--copy_raw_images",   action="store_true", default=False,
                   help="Also export unmasked (original background) images to "
                        "images_raw/. Only useful together with --raw_features.")
    p.add_argument("--raw_features",      action="store_true", default=False,
                   help="Extract SIFT on images_raw/ instead of masked images/. "
                        "Not recommended: the background is static while the "
                        "specimen rotates, so background keypoints contradict "
                        "the known rig poses and pollute triangulation.")
    p.add_argument("--bundle_adjustment", action="store_true", default=False,
                   help="Run COLMAP bundle_adjuster after triangulation. "
                        "Not recommended: the calibrated h5 poses are more "
                        "accurate than BA on this weakly connected match graph.")

    # Seed points from structured-light correspondence point clouds
    p.add_argument("--corr_h5",      default=None,
                   help="Path to correspondence_coordinate.h5 for DLT seed points")
    p.add_argument("--corr_dir",     default=None,
                   help="Path to correspondence_undistort/ root directory. "
                        "Each species subfolder must contain <species_name>.pcd "
                        "(pre-triangulated structured-light cloud). "
                        "Writes points3D.ply to sparse/0/ — 3DGS uses this in "
                        "preference to COLMAP points, giving a much denser and "
                        "more accurate initialisation on butterfly wing surfaces.")

    # Verification
    p.add_argument("--verify",       action="store_true", default=False,
                   help="Run sanity checks (re-center invariant + reprojection)")

    # Batch mode — run for every species subdirectory in img_dir
    p.add_argument("--batch",        action="store_true", default=False,
                   help="Process all species subdirectories found in --img_dir "
                        "(or <base_path>/imgs_filtered if --img_dir not given). "
                        "Each subdirectory must start with a 3-digit species ID, "
                        "e.g. '001-Papilio_Rutulus-001'. Output goes to "
                        "<out>/<species_name>/.")

    # Schema inspection
    p.add_argument("--inspect_h5",   default=None,
                   help="Print h5 schema and exit (no export performed)")

    args = p.parse_args(argv)

    # ── h5 inspection only ────────────────────────────────────────────────────
    if args.inspect_h5:
        _inspect_h5(args.inspect_h5)
        return

    # ── validate required args ────────────────────────────────────────────────
    if args.base_path is None:
        p.error("--base_path is required")

    h5_path = args.h5 or f"{args.base_path}/camera_parameters.h5"

    # ── build species list ────────────────────────────────────────────────────
    if args.batch:
        import re
        from pathlib import Path as _Path
        img_root = _Path(args.img_dir) if args.img_dir else \
                   _Path(args.base_path) / "imgs_filtered"
        if not img_root.exists():
            p.error(f"--batch: directory not found: {img_root}")
        species_list = []
        for d in sorted(img_root.iterdir()):
            if not d.is_dir():
                continue
            m = re.match(r'^(\d+)', d.name)
            if not m:
                continue
            species_list.append((int(m.group(1)), d.name, str(img_root)))
        if not species_list:
            p.error(f"--batch: no species subdirectories found in {img_root}")
        print(f"Batch mode: {len(species_list)} species found in {img_root}")
        for sid, sname, _ in species_list:
            print(f"  [{sid:03d}] {sname}")
    else:
        missing = [f for f, v in [
            ("--species_id",   args.species_id),
            ("--species_name", args.species_name),
        ] if v is None]
        if missing:
            p.error(f"Required for single-species export: {', '.join(missing)}")
        species_list = [(args.species_id, args.species_name, args.img_dir)]

    # ── run per species ───────────────────────────────────────────────────────
    from .export_colmap import export_colmap
    from pathlib import Path as _Path

    for species_id, species_name, img_dir_for_species in species_list:
        if args.batch:
            out_dir = str(_Path(args.out) / species_name)
            print(f"\n{'='*60}")
            print(f"Processing [{species_id:03d}] {species_name} → {out_dir}")
            print('='*60)
        else:
            out_dir = args.out

        export_kwargs = dict(
            h5_path          = h5_path,
            base_path        = args.base_path,
            species_id       = species_id,
            species_name     = species_name,
            out_dir          = out_dir,
            max_poses        = args.max_poses,
            rotate_k         = args.rotate_k,
            do_recenter      = args.recenter,
            resolution       = args.resolution,
            img_pattern      = args.img_pattern,
            copy_images      = not args.no_images,
            img_dir          = img_dir_for_species,
            mask_dir         = args.mask_dir,
            copy_raw_images  = args.copy_raw_images,
        )

        export_colmap(**export_kwargs)

        if args.run_colmap:
            from .run_colmap import run_colmap_triangulation
            run_colmap_triangulation(
                out_dir        = out_dir,
                colmap_bin     = args.colmap_executable,
                use_gpu        = not args.no_gpu,
                use_raw_images = args.raw_features,
                run_ba         = args.bundle_adjustment,
            )

        if args.corr_h5:
            from .seed_points import build_point_cloud, write_points3d
            points = build_point_cloud(
                h5_path, args.corr_h5, species_id,
                img_dir=f"{out_dir}/images" if not args.no_images else None,
            )
            write_points3d(points, f"{out_dir}/sparse/0/points3D.txt")

        if args.corr_dir:
            from pathlib import Path as _Path
            from .seed_points import install_pcd_seed
            install_pcd_seed(
                corr_dir    = args.corr_dir,
                species_name = species_name,
                sparse_dir   = _Path(out_dir) / "sparse" / "0",
            )

        if args.verify:
            from .verify import run_all_checks
            run_all_checks(h5_path, species_id)


if __name__ == "__main__":
    main()
