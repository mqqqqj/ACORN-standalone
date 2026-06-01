#!/bin/bash
# Compute brute-force L2 ground truth for SIFT1M (no filter).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
BIN="$PROJ_DIR/build/compute_groundtruth"

BASE="${BASE:-/dataset/SIFT1M/sift_base.fbin}"
QUERY="${QUERY:-/dataset/SIFT1M/sift_query.fbin}"
OUTPUT="${OUTPUT:-$PROJ_DIR/data/sift1m/sift1m_gt_top100.ibin}"
K="${K:-100}"
NQ="${NQ:--1}"

echo "=== Compute SIFT1M Ground Truth ==="
echo "Base:   $BASE"
echo "Query:  $QUERY"
echo "Output: $OUTPUT"
echo "k=$K, nq=${NQ}"
echo

mkdir -p "$(dirname "$OUTPUT")"
"$BIN" --base "$BASE" --query "$QUERY" --output "$OUTPUT" --k "$K" --nq "$NQ"
