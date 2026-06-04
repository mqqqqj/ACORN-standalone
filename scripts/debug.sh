#!/bin/bash
# Quick debug: build a small ACORN index on SIFT1M and evaluate recall.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
BIN="$PROJ_DIR/build/debug_search"

BASE="${BASE:-/dataset/SIFT1M/sift_base.fbin}"
LABELS="${LABELS:-$PROJ_DIR/data/sift1m/labels.ibin}"
QUERY="${QUERY:-/dataset/SIFT1M/sift_query.fbin}"
GT="${GT:-$PROJ_DIR/data/sift1m/sift1m_gt_top100.ibin}"
OUTPUT="${OUTPUT:-}"
N="${N:-20000}"
NQ="${NQ:-100}"
K="${K:-100}"
M="${M:-16}"
GAMMA="${GAMMA:-8}"
EFC="${EFC:-64}"
EF="${EF:-64}"
METRIC="${METRIC:-l2}"

echo "=== ACORN Debug Search ==="
echo "Base:   $BASE"
echo "Labels: $LABELS"
echo "Query:  $QUERY"
echo "GT:     $GT"
echo "Params: n=$N, nq=$NQ, k=$K, M=$M, gamma=$GAMMA, efc=$EFC, ef=$EF, metric=$METRIC"
echo

ARGS=(
    --base "$BASE"
    --labels "$LABELS"
    --query "$QUERY"
    --gt "$GT"
    --n "$N" --nq "$NQ" --k "$K"
    --M "$M" --gamma "$GAMMA" --efc "$EFC" --ef "$EF" --metric "$METRIC"
)
if [ -n "$OUTPUT" ]; then
    ARGS+=(--output "$OUTPUT")
fi

"$BIN" "${ARGS[@]}"
