#!/bin/bash
# Generate random labels for SIFT1M.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
BIN="$PROJ_DIR/build/gen_labels"

BASE="${BASE:-/dataset/SIFT1M/sift_base.fbin}"
OUTPUT="${OUTPUT:-$PROJ_DIR/data/sift1m/labels.ibin}"
NUM_LABELS="${NUM_LABELS:-12}"
SEED="${SEED:-42}"

echo "=== Generate SIFT1M Labels ==="
echo "Base:   $BASE"
echo "Output: $OUTPUT"
echo "Labels: $NUM_LABELS, seed=$SEED"
echo

mkdir -p "$(dirname "$OUTPUT")"
"$BIN" --base "$BASE" --output "$OUTPUT" --num-labels "$NUM_LABELS" --seed "$SEED"
