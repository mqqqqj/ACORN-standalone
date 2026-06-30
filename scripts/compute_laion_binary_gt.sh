#!/bin/bash
# Compute filtered ground truth for LAION10M binary label selectivity files.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
BIN="$PROJ_DIR/build/compute_filtered_gt"

BASE="${BASE:-/dataset/LAION/LAION_base_imgemb_10M.fbin}"
QUERY="${QUERY:-/dataset/LAION/LAION_test_query_textemb_10k.fbin}"
DATA_DIR="${DATA_DIR:-$PROJ_DIR/data/laion10m}"
K="${K:-100}"
METRIC="${METRIC:-ip}"
NQ="${NQ:-1000}"

run_gt() {
    local suffix="$1"
    local query_label="$2"
    local output_suffix="$3"

    local base_labels="$DATA_DIR/base_labels_binary_${suffix}.ibin"
    local query_labels="$DATA_DIR/query_labels_all${query_label}.ibin"
    local output="$DATA_DIR/laion10m_gt_binary_${output_suffix}.ibin"

    echo "=== GT ${output_suffix} ==="
    echo "Base labels:  $base_labels"
    echo "Query labels: $query_labels"
    echo "Output:       $output"
    echo "NQ:           $NQ"
    "$BIN" \
        --base "$BASE" \
        --query "$QUERY" \
        --base-labels "$base_labels" \
        --query-labels "$query_labels" \
        --output "$output" \
        --k "$K" \
        --nq "$NQ" \
        --metric "$METRIC"
    echo
}

run_gt s0p1 1 s0p1
run_gt s0p1 2 s99p9
run_gt s1 1 s1
run_gt s1 2 s99
run_gt s5 1 s5
run_gt s5 2 s95
run_gt s10 1 s10
run_gt s10 2 s90
run_gt s20 1 s20
run_gt s20 2 s80
run_gt s50 1 s50_label1
run_gt s50 2 s50_label2
