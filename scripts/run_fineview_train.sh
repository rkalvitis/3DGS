#!/bin/bash
# Train 3DGS for all 5 FineView species, then render and compute metrics.
# Run inside the Singularity container (see RUN_FINEVIEW.md for the full command).
#
# Environment variables (override as needed):
#   DEVICE     CUDA device index (default: 0)
#   DATA_DIR   COLMAP scenes dir (output of run_fineview_colmap.sh)
#   OUTPUT_DIR training output root
#   EXP_NAME   experiment subdirectory under OUTPUT_DIR (e.g. white_noise_experiment);
#              all scenes + logs land in $OUTPUT_DIR/$EXP_NAME/. Default: no subdir.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export TORCH_HOME=/torch_cache
DATA_DIR=${DATA_DIR:-/data}
OUTPUT_DIR=${OUTPUT_DIR:-/output}
EXP_NAME=${EXP_NAME:-}
if [ -n "$EXP_NAME" ]; then
    OUTPUT_DIR="$OUTPUT_DIR/$EXP_NAME"
fi
echo "Output root: $OUTPUT_DIR"

SPECIES=(
    "009-Neophasia_Menapia-001"
    "072-Colias_Eurytheme-002"
    "110-Nymphalis_l_album-001"
    "184-Speyeria_Hydaspe-001"
    "195-Lycaena_Arota-002"
)

mkdir -p "$OUTPUT_DIR/logs"

for species in "${SPECIES[@]}"; do
    log="$OUTPUT_DIR/logs/${species}.log"
    scene="$DATA_DIR/$species"
    model="$OUTPUT_DIR/$species"

    echo ""
    echo "=========================================="
    echo "Training: $species"
    echo "=========================================="

    python /workspace/train.py \
        -s "$scene" -m "$model" \
        --eval --disable_viewer --seed 1 --device 0 \
        --resolution 1 \
        --data_device cpu \
        --iterations 30000 \
        --test_iterations 30000 \
        --save_iterations 30000 \
        --sh_degree 3 \
        2>&1 | tee "$log"

    python /workspace/render.py \
        -m "$model" --device 0 \
        2>&1 | tee -a "$log"

    python /workspace/metrics.py \
        -m "$model" --device 0 \
        2>&1 | tee -a "$log"

    echo "Done: $species"
done

echo ""
echo "=== All 5 species trained. Results in $OUTPUT_DIR ==="
