#!/bin/bash
# Compute filtered ground truth.
# Only considers base vectors with the same label as the query.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
BIN="$PROJ_DIR/build/compute_filtered_gt"


K="${K:-100}"
METRIC="${METRIC:-ip}"
# SIFT1M
# BASE="${BASE:-/dataset/SIFT1M/sift_base.fbin}"
# QUERY="${QUERY:-/dataset/SIFT1M/sift_query.fbin}"
# BASE_LABELS="${BASE_LABELS:-$PROJ_DIR/data/sift1m/base_labels.ibin}"
# QUERY_LABELS="${QUERY_LABELS:-$PROJ_DIR/data/sift1m/query_labels.ibin}"
# OUTPUT="${OUTPUT:-$PROJ_DIR/data/sift1m/sift1m_gt_filtered2.ibin}"

# LAION10M
# BASE="${BASE:-/dataset/LAION/LAION_base_imgemb_10M.fbin}"
# QUERY="${QUERY:-/dataset/LAION/LAION_test_query_textemb_10k.fbin}"
# BASE_LABELS="${BASE_LABELS:-$PROJ_DIR/data/laion10m/base_labels_uniform_n10.ibin}"
# QUERY_LABELS="${QUERY_LABELS:-$PROJ_DIR/data/laion10m/query_labels_uniform_n10.ibin}"
# OUTPUT="${OUTPUT:-$PROJ_DIR/data/laion10m/laion10m_gt_filtered_uniform_n10.ibin}"

# DEEP10M
BASE="${BASE:-/dataset/DEEP10M/base.fbin}"
QUERY="${QUERY:-/dataset/DEEP10M/query.fbin}"
BASE_LABELS="${BASE_LABELS:-$PROJ_DIR/data/deep10m/base_labels_uniform_n10.ibin}"
QUERY_LABELS="${QUERY_LABELS:-$PROJ_DIR/data/deep10m/query_labels_uniform_n10.ibin}"
OUTPUT="${OUTPUT:-$PROJ_DIR/data/deep10m/deep10m_gt_filtered_uniform_n10.ibin}"

echo "=== Compute Filtered Ground Truth ==="
echo "Base:         $BASE"
echo "Query:        $QUERY"
echo "Base labels:  $BASE_LABELS"
echo "Query labels: $QUERY_LABELS"
echo "Output:       $OUTPUT"
echo "k=$K, metric=$METRIC"
echo

mkdir -p "$(dirname "$OUTPUT")"
"$BIN" \
    --base "$BASE" \
    --query "$QUERY" \
    --base-labels "$BASE_LABELS" \
    --query-labels "$QUERY_LABELS" \
    --output "$OUTPUT" \
    --k "$K" \
    --metric "$METRIC"
