#!/bin/bash
# Generate random labels for SIFT1M.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
BIN="$PROJ_DIR/build/gen_labels"

SEED="${SEED:-42}"
# SIFT1M
# BASE="${BASE:-/dataset/SIFT1M/sift_base.fbin}"
# OUTPUT="${OUTPUT:-$PROJ_DIR/data/sift1m/base_labels.ibin}"
# DIST="${DIST:-uniform}"
# NUM_LABELS="${NUM_LABELS:-12}"

# LAION10M
# BASE="${BASE:-/dataset/LAION/LAION_test_query_textemb_10k.fbin}"
# OUTPUT="${OUTPUT:-$PROJ_DIR/data/laion10m/query_labels_uniform_n10.ibin}"
# DIST="${DIST:-uniform}"
# NUM_LABELS="${NUM_LABELS:-10}"

# DEEP10M
BASE="${BASE:-/dataset/DEEP10M/query.fbin}"
OUTPUT="${OUTPUT:-$PROJ_DIR/data/deep10m/query_labels_uniform_n10.ibin}"
DIST="${DIST:-uniform}"
NUM_LABELS="${NUM_LABELS:-10}"


echo "=== Generate Labels ==="
echo "Base:   $BASE"
echo "Output: $OUTPUT"
echo "Dist:   $DIST, labels=$NUM_LABELS, seed=$SEED"
echo

mkdir -p "$(dirname "$OUTPUT")"
"$BIN" --base "$BASE" --output "$OUTPUT" --dist "$DIST" --num-labels "$NUM_LABELS" --seed "$SEED"
