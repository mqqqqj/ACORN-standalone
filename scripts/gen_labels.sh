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
# SELECTIVITY="${SELECTIVITY:-0.1}"

# LAION10M
# BASE="${BASE:-/dataset/LAION/base.fbin}"
# OUTPUT="${OUTPUT:-$PROJ_DIR/data/laion10m/base_labels_uniform_s0.1.ibin}"
# DIST="${DIST:-uniform}"
# SELECTIVITY="${SELECTIVITY:-0.1}"

# DEEP10M
BASE="${BASE:-/dataset/DEEP10M/base.fbin}"
OUTPUT="${OUTPUT:-$PROJ_DIR/data/deep10m/base_labels_uniform_s0.1.ibin}"
DIST="${DIST:-uniform}"
SELECTIVITY="${SELECTIVITY:-0.1}"


echo "=== Generate Labels ==="
echo "Base:   $BASE"
echo "Output: $OUTPUT"
echo "Dist:   $DIST, selectivity=$SELECTIVITY, seed=$SEED"
echo

mkdir -p "$(dirname "$OUTPUT")"
"$BIN" --base "$BASE" --output "$OUTPUT" --dist "$DIST" --selectivity "$SELECTIVITY" --seed "$SEED"
