#!/bin/bash
# =============================================================================
# run_metrics_only.sh
# Re-runs metrics.py on all completed runs (those with a test/ directory).
# Use this to recover after metrics failed while training/rendering succeeded.
#
# Usage (inside the container, same env vars as run_experiments.sh):
#   bash /workspace/run_metrics_only.sh
# =============================================================================

set -euo pipefail

CODE_DIR="${CODE_DIR:-/workspace}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"

export CUDA_VISIBLE_DEVICES=1

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

TOTAL=0
COUNT=0

# Count eligible runs first
for run_dir in "$OUTPUT_DIR"/*/; do
    [[ -d "$run_dir/test" ]] && TOTAL=$(( TOTAL + 1 ))
done

echo "Found $TOTAL runs with rendered output. Running metrics..."
echo ""

for run_dir in "$OUTPUT_DIR"/*/; do
    [[ ! -d "$run_dir/test" ]] && continue

    run_name=$(basename "$run_dir")
    log_file="$LOG_DIR/${run_name}.log"
    COUNT=$(( COUNT + 1 ))

    echo "[$(date '+%H:%M:%S')] ($COUNT/$TOTAL) metrics  $run_name"

    python "$CODE_DIR/metrics.py" \
        -m "$run_dir" \
        2>&1 | tee -a "$log_file"
done

echo ""
echo "=== Metrics complete for $COUNT runs. ==="
