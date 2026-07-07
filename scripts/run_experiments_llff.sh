#!/bin/bash
# =============================================================================
# run_experiments_llff.sh
# Runs 3DGS on the NeRF LLFF dataset (8 scenes × 5 seeds = 40 runs)
# sequentially on a single GPU.
#
# Usage:
#   bash /workspace/scripts/run_experiments_llff.sh
#
# Required environment variables:
#   CODE_DIR    — path to your gaussian-splatting code   (e.g. /workspace)
#   DATA_DIR    — root folder containing nerf_llff_data  (e.g. /data/llff)
#   OUTPUT_DIR  — where results are written               (e.g. /output/llff)
#
# Expected dataset layout inside DATA_DIR:
#   $DATA_DIR/{fern,flower,fortress,horns,leaves,orchids,room,trex}/
#     Each scene must contain sparse/0/{cameras.bin,images.bin,points3D.bin}
#     and images_4/ for 4x-downsampled images (standard LLFF eval resolution).
# =============================================================================

set -euo pipefail

CODE_DIR="${CODE_DIR:-/workspace}"
DATA_DIR="${DATA_DIR:-/data/llff}"
OUTPUT_DIR="${OUTPUT_DIR:-/output/llff}"
DEVICE=0

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_HOME=/torch_cache

SEEDS=(0 1 2 3 4)

SCENES=(fern flower fortress horns leaves orchids room trex)

mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

TOTAL=$(( ${#SCENES[@]} * ${#SEEDS[@]} ))
COUNT=0

for scene in "${SCENES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        COUNT=$(( COUNT + 1 ))
        scene_path="$DATA_DIR/$scene"
        out_path="$OUTPUT_DIR/${scene}_seed${seed}"
        log_file="$LOG_DIR/${scene}_seed${seed}.log"

        echo "[$(date '+%H:%M:%S')] ($COUNT/$TOTAL) START  ${scene}  seed=${seed}"

        python "$CODE_DIR/train.py" \
            -s "$scene_path" \
            -m "$out_path" \
            --resolution 4 \
            --eval \
            --disable_viewer \
            --seed "$seed" \
            --device "$DEVICE" \
            >> "$log_file" 2>&1

        python "$CODE_DIR/render.py" \
            -m "$out_path" \
            --iteration 7000 \
            --skip_train \
            --device "$DEVICE" \
            >> "$log_file" 2>&1

        python "$CODE_DIR/render.py" \
            -m "$out_path" \
            --iteration 30000 \
            --skip_train \
            --device "$DEVICE" \
            >> "$log_file" 2>&1

        python "$CODE_DIR/metrics.py" \
            -m "$out_path" \
            --device "$DEVICE" \
            >> "$log_file" 2>&1

        echo "[$(date '+%H:%M:%S')] ($COUNT/$TOTAL) DONE   ${scene}  seed=${seed}"
    done
done

echo ""
echo "=== All $TOTAL experiments complete. Results are in $OUTPUT_DIR ==="
