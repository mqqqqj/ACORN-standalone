#!/bin/bash
#
# Build ACORN index for SIFT1M (filtered, 13-class labels).
#
# Prerequisites:
#   - Build the project first:  cd build && make build_index
#   - SIFT1M base vectors at /dataset/SIFT1M/sift_base.fbin
#   - Labels file at ../data/sift1m/labels.ibin

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
BIN="$PROJ_DIR/build/build_index"

BASE="${BASE:-/dataset/SIFT1M/sift_base.fbin}"
LABELS="${LABELS:-$PROJ_DIR/data/sift1m/labels.ibin}"
OUTPUT="${OUTPUT:-$PROJ_DIR/data/sift1m/acorn_sift1m.index}"
M="${M:-32}"
GAMMA="${GAMMA:-12}"
EFC="${EFC:-200}"

echo "============================================"
echo "  Build SIFT1M ACORN Index"
echo "============================================"
echo "Base:   $BASE"
echo "Labels: $LABELS"
echo "Output: $OUTPUT"
echo "Params: M=$M, gamma=$GAMMA, efConstruction=$EFC"
echo

mkdir -p "$(dirname "$OUTPUT")"

"$BIN" \
    --base "$BASE" \
    --labels "$LABELS" \
    --output "$OUTPUT" \
    --M "$M" \
    --gamma "$GAMMA" \
    --efc "$EFC" \
    --metric l2

echo
echo "Done. Index saved to $OUTPUT"
