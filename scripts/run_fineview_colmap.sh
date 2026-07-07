#!/bin/bash
# Export FineView species to COLMAP format and install the structured-light
# point cloud as the 3DGS init (points3D.ply).
# Run inside the Singularity container (see RUN_FINEVIEW.md for the full command).
#
# Environment variables (override as needed):
#   RAW_DATA     host raw_data dir, mounted at this path inside container
#   DATA_DIR     output dir for COLMAP scenes (one sub-dir per species)
#   H5_PATH      path to camera_parameters.h5
#   RUN_COLMAP   set to 1 to also run COLMAP SIFT + triangulation (default 0).
#                Off by default: training initializes from the structured-light
#                .pcd (installed as points3D.ply, preferred over COLMAP's .bin),
#                so the triangulated cloud is unused. COLMAP's image reader also
#                fails on the masked black-bg RGBA images (drops cameras 5-8),
#                so leave this off unless you specifically need a COLMAP cloud
#                for a pose/recon comparison.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/workspace

RAW_DATA=${RAW_DATA:-/raw_data}
DATA_DIR=${DATA_DIR:-/data}
H5_PATH=${H5_PATH:-$RAW_DATA/camera_parameters.h5}
RUN_COLMAP=${RUN_COLMAP:-0}

echo "=== FineView → COLMAP export ==="
echo "  raw images : $RAW_DATA"
echo "  h5 file    : $H5_PATH"
echo "  output     : $DATA_DIR"
echo "  run colmap : $RUN_COLMAP"
echo ""

EXTRA_ARGS=()
if [ "$RUN_COLMAP" = "1" ]; then
    EXTRA_ARGS+=(--run_colmap)
fi

python -m fineview_pipeline \
    --base_path       "$RAW_DATA" \
    --h5              "$H5_PATH" \
    --img_dir         "$RAW_DATA/crop_undistort" \
    --mask_dir        "$RAW_DATA/crop_mask_undistort" \
    --corr_dir        "$RAW_DATA/correspondence_undistort" \
    --out             "$DATA_DIR" \
    --batch \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo ""
echo "=== Export done. Species exported to $DATA_DIR ==="
