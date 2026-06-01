#!/bin/bash
# Compute filtered L2 ground truth for SIFT1M.
# Only considers base vectors with the same label as the query.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
BIN="$PROJ_DIR/build/compute_filtered_gt"

BASE="${BASE:-/dataset/SIFT1M/sift_base.fbin}"
QUERY="${QUERY:-/dataset/SIFT1M/sift_query.fbin}"
BASE_LABELS="${BASE_LABELS:-$PROJ_DIR/data/sift1m/labels.ibin}"
QUERY_LABELS="${QUERY_LABELS:-$PROJ_DIR/data/sift1m/query_labels.ibin}"
OUTPUT="${OUTPUT:-$PROJ_DIR/data/sift1m/sift1m_gt_filtered.ibin}"
K="${K:-100}"
NQ="${NQ:--1}"

echo "=== Compute SIFT1M Filtered Ground Truth ==="
echo "Base:         $BASE"
echo "Query:        $QUERY"
echo "Base labels:  $BASE_LABELS"
echo "Query labels: $QUERY_LABELS"
echo "Output:       $OUTPUT"
echo "k=$K, nq=${NQ}"
echo

mkdir -p "$(dirname "$OUTPUT")"
"$BIN" \
    --base "$BASE" \
    --query "$QUERY" \
    --base-labels "$BASE_LABELS" \
    --query-labels "$QUERY_LABELS" \
    --output "$OUTPUT" \
    --k "$K" \
    --nq "$NQ"
