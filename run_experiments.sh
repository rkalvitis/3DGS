#!/bin/bash
# =============================================================================
# run_experiments.sh
# Runs all 3DGS Table 1 experiments (13 scenes × 5 seeds = 65 runs)
# sequentially on a single GPU.
#
# Usage:
#   bash /workspace/run_experiments.sh
#
# Required environment variables:
#   CODE_DIR    — path to your gaussian-splatting code   (e.g. /workspace)
#   DATA_DIR    — root folder containing all datasets     (e.g. /data)
#   OUTPUT_DIR  — where results are written               (e.g. /output)
#
# Expected dataset layout inside DATA_DIR:
#   $DATA_DIR/mipnerf360/{bicycle,flowers,garden,stump,treehill,room,counter,kitchen,bonsai}/
#   $DATA_DIR/tandt/{truck,train}/
#   $DATA_DIR/db/{playroom,drjohnson}/
# =============================================================================

set -euo pipefail

CODE_DIR="${CODE_DIR:-/workspace}"
DATA_DIR="${DATA_DIR:-/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"

export CUDA_VISIBLE_DEVICES=0

SEEDS=(0 1 2 3 4)

declare -A SCENES
SCENES[bicycle]="mipnerf360/bicycle"
SCENES[flowers]="mipnerf360/flowers"
SCENES[garden]="mipnerf360/garden"
SCENES[stump]="mipnerf360/stump"
SCENES[treehill]="mipnerf360/treehill"
SCENES[room]="mipnerf360/room"
SCENES[counter]="mipnerf360/counter"
SCENES[kitchen]="mipnerf360/kitchen"
SCENES[bonsai]="mipnerf360/bonsai"
SCENES[truck]="tandt/truck"
SCENES[train]="tandt/train"
SCENES[playroom]="db/playroom"
SCENES[drjohnson]="db/drjohnson"

mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

TOTAL=$(( ${#SCENES[@]} * ${#SEEDS[@]} ))
COUNT=0

for scene in "${!SCENES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        COUNT=$(( COUNT + 1 ))
        scene_path="$DATA_DIR/${SCENES[$scene]}"
        out_path="$OUTPUT_DIR/${scene}_seed${seed}"
        log_file="$LOG_DIR/${scene}_seed${seed}.log"

        echo "[$(date '+%H:%M:%S')] ($COUNT/$TOTAL) START  ${scene}  seed=${seed}"

        python "$CODE_DIR/train.py" \
            -s "$scene_path" \
            -m "$out_path" \
            --eval \
            --disable_viewer \
            --seed "$seed" \
            >> "$log_file" 2>&1

        python "$CODE_DIR/render.py" \
            -m "$out_path" \
            >> "$log_file" 2>&1

        python "$CODE_DIR/metrics.py" \
            -m "$out_path" \
            >> "$log_file" 2>&1

        echo "[$(date '+%H:%M:%S')] ($COUNT/$TOTAL) DONE   ${scene}  seed=${seed}"
    done
done

echo ""
echo "=== All $TOTAL experiments complete. Results are in $OUTPUT_DIR ==="
