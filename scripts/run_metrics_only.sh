#!/bin/bash
# =============================================================================
# run_metrics_only.sh
# For all completed runs: renders test views at 7k (30k already exists), then
# evaluates metrics at both iterations so results.json contains ours_7000 and
# ours_30000, reproducing Table 1 of Kerbl et al. 2023.
#
# Usage (inside the container, same env vars as run_experiments.sh):
#   bash /workspace/scripts/run_metrics_only.sh
#
# Optional: DEVICE (CUDA device index, default 1) and EXP_NAME (process the
# runs under $OUTPUT_DIR/$EXP_NAME/ instead of $OUTPUT_DIR).
# =============================================================================

set -euo pipefail

CODE_DIR="${CODE_DIR:-/workspace}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
DEVICE="${DEVICE:-1}"
EXP_NAME="${EXP_NAME:-}"
if [ -n "$EXP_NAME" ]; then
    OUTPUT_DIR="$OUTPUT_DIR/$EXP_NAME"
fi
echo "Output root: $OUTPUT_DIR"

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

TOTAL=0
COUNT=0

for run_dir in "$OUTPUT_DIR"/*/; do
    [[ -d "$run_dir/point_cloud/iteration_7000" ]] && TOTAL=$(( TOTAL + 1 ))
done

echo "Found $TOTAL completed runs. Rendering 7k + evaluating 7k & 30k..."
echo ""

for run_dir in "$OUTPUT_DIR"/*/; do
    [[ ! -d "$run_dir/point_cloud/iteration_7000" ]] && continue

    run_name=$(basename "$run_dir")
    log_file="$LOG_DIR/${run_name}.log"
    COUNT=$(( COUNT + 1 ))

    echo "[$(date '+%H:%M:%S')] ($COUNT/$TOTAL) render 7k  $run_name"
    python "$CODE_DIR/render.py" -m "$run_dir" --iteration 7000 --skip_train \
        --device "$DEVICE" \
        2>&1 | tee -a "$log_file"

    echo "[$(date '+%H:%M:%S')] ($COUNT/$TOTAL) metrics    $run_name"
    python "$CODE_DIR/metrics.py" -m "$run_dir" \
        --device "$DEVICE" \
        2>&1 | tee -a "$log_file"
done

echo ""
echo "=== Done. results.json now contains ours_7000 and ours_30000 for $COUNT runs. ==="
