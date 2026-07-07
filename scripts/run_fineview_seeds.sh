#!/bin/bash
# Install structured-light PCD point clouds as 3DGS seed points.
# Replaces COLMAP sparse points with 27k calibrated points per species.
# Run inside the Singularity container (see RUN_FINEVIEW.md).

set -euo pipefail

export PYTHONPATH=/workspace

RAW_DATA=${RAW_DATA:-/raw_data}
DATA_DIR=${DATA_DIR:-/data}
CORR_DIR="$RAW_DATA/correspondence_undistort"

SPECIES=(
    "009-Neophasia_Menapia-001"
    "072-Colias_Eurytheme-002"
    "110-Nymphalis_l_album-001"
    "184-Speyeria_Hydaspe-001"
    "195-Lycaena_Arota-002"
)

echo "=== Installing structured-light seed points ==="
echo "  corr_dir : $CORR_DIR"
echo "  data_dir : $DATA_DIR"
echo ""

for species in "${SPECIES[@]}"; do
    pcd="$CORR_DIR/$species/$species.pcd"
    out="$DATA_DIR/$species/sparse/0/points3D.ply"

    if [ ! -f "$pcd" ]; then
        echo "SKIP $species — no .pcd found at $pcd"
        continue
    fi

    python -c "
import sys; sys.path.insert(0, '/workspace')
from fineview_pipeline.seed_points import install_pcd_seed
from pathlib import Path
n = install_pcd_seed('$CORR_DIR', '$species', Path('$DATA_DIR/$species/sparse/0'))
print('  $species -> ' + str(n) + ' points -> $out')
"
done

echo ""
echo "=== Seed installation done ==="
