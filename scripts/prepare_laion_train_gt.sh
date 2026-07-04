#!/bin/bash
# Prepare LAION10M training queries and filtered ground truth for offline profiling.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

BASE="${BASE:-/dataset/LAION/LAION_base_imgemb_10M.fbin}"
QUERY="${QUERY:-/dataset/LAION/LAION_test_query_textemb_10k.fbin}"
DATA_DIR="${DATA_DIR:-$PROJ_DIR/data/laion10m}"
TRAIN_DIR="${TRAIN_DIR:-$DATA_DIR/train}"
QUERY_OFFSET="${QUERY_OFFSET:-2000}"
NQ="${NQ:-1000}"
K="${K:-100}"
METRIC="${METRIC:-ip}"

EXTRACT_BIN="$PROJ_DIR/build/extract_range"
GT_BIN="$PROJ_DIR/build/compute_filtered_gt"

TRAIN_QUERY="$TRAIN_DIR/laion10m_query_${QUERY_OFFSET}_$((QUERY_OFFSET + NQ)).fbin"
TRAIN_QLABEL1="$TRAIN_DIR/query_labels_all1.ibin"
TRAIN_QLABEL2="$TRAIN_DIR/query_labels_all2.ibin"

mkdir -p "$TRAIN_DIR"

echo "=== Preparing LAION10M training query range ==="
echo "Source query:  $QUERY"
echo "Train dir:     $TRAIN_DIR"
echo "Range:         [$QUERY_OFFSET, $((QUERY_OFFSET + NQ)))"
echo "NQ:            $NQ"
echo

"$EXTRACT_BIN" \
    --input "$QUERY" \
    --output "$TRAIN_QUERY" \
    --format fbin \
    --offset "$QUERY_OFFSET" \
    --count "$NQ"

"$EXTRACT_BIN" \
    --input "$DATA_DIR/query_labels_all1.ibin" \
    --output "$TRAIN_QLABEL1" \
    --format ibin \
    --offset "$QUERY_OFFSET" \
    --count "$NQ"

"$EXTRACT_BIN" \
    --input "$DATA_DIR/query_labels_all2.ibin" \
    --output "$TRAIN_QLABEL2" \
    --format ibin \
    --offset "$QUERY_OFFSET" \
    --count "$NQ"

run_gt() {
    local suffix="$1"
    local query_label="$2"
    local output_suffix="$3"

    local base_labels="$DATA_DIR/base_labels_binary_${suffix}.ibin"
    local query_labels="$TRAIN_DIR/query_labels_all${query_label}.ibin"
    local output="$TRAIN_DIR/laion10m_train_gt_binary_${output_suffix}.ibin"

    echo
    echo "=== Train GT ${output_suffix} ==="
    echo "Base labels:  $base_labels"
    echo "Query labels: $query_labels"
    echo "Output:       $output"
    "$GT_BIN" \
        --base "$BASE" \
        --query "$TRAIN_QUERY" \
        --base-labels "$base_labels" \
        --query-labels "$query_labels" \
        --output "$output" \
        --k "$K" \
        --nq "$NQ" \
        --metric "$METRIC"
}

run_gt s0p1 1 s0p1
run_gt s1 1 s1
run_gt s5 1 s5
run_gt s10 1 s10
run_gt s20 1 s20
run_gt s50 1 s50_label1
run_gt s20 2 s80

echo
echo "Training query and GT files written to $TRAIN_DIR"
