#!/bin/bash
#
# Build ACORN index (filtered, 13-class labels).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
BIN="$PROJ_DIR/build/build_index"

# SIFT1M
# BASE="${BASE:-/dataset/SIFT1M/sift_base.fbin}"
# LABELS="${LABELS:-$PROJ_DIR/data/sift1m/base_labels.ibin}"
# OUTPUT="${OUTPUT:-$PROJ_DIR/data/sift1m/acorn_sift1m_3.index}"
# M="${M:-32}"
# GAMMA="${GAMMA:-12}"
# EFC="${EFC:-200}"
# METRIC="${METRIC:-l2}"

# LAION10M
# BASE="${BASE:-/dataset/LAION/LAION_base_imgemb_10M.fbin}"
# LABELS="${LABELS:-$PROJ_DIR/data/laion10m/base_labels_uniform_n10.ibin}"
# OUTPUT="${OUTPUT:-$PROJ_DIR/data/laion10m/acorn_laion10m_efc500.faiss_index}"
# M="${M:-32}"
# GAMMA="${GAMMA:-12}"
# EFC="${EFC:-500}"
# METRIC="${METRIC:-ip}"

# DEEP10M
BASE="${BASE:-/dataset/DEEP10M/base.fbin}"
LABELS="${LABELS:-$PROJ_DIR/data/deep10m/base_labels_uniform_n10.ibin}"
OUTPUT="${OUTPUT:-$PROJ_DIR/data/deep10m/acorn_deep10m_efc500.index}"
M="${M:-32}"
GAMMA="${GAMMA:-12}"
EFC="${EFC:-500}"
METRIC="${METRIC:-ip}"

echo "============================================"
echo "  Build ACORN Index"
echo "============================================"
echo "Base:   $BASE"
echo "Labels: $LABELS"
echo "Output: $OUTPUT"
echo "Params: M=$M, gamma=$GAMMA, efConstruction=$EFC, metric=$METRIC"
echo

mkdir -p "$(dirname "$OUTPUT")"

"$BIN" \
    --base "$BASE" \
    --labels "$LABELS" \
    --output "$OUTPUT" \
    --M "$M" \
    --gamma "$GAMMA" \
    --efc "$EFC" \
    --metric "$METRIC"

echo
echo "Done. Index saved to $OUTPUT"
