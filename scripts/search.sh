#!/bin/bash
#
# Search ACORN index (filtered).


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
BIN="$PROJ_DIR/build/search"

K="${K:-100}"
EF="${EF:-800}"
THREADS="${THREADS:-4}"
EFS="${EFS:-200}"
NQ="${NQ:-1000}"
MODE="${MODE:-"serial"}"
HELEC="${HELEC:-50}"
FILTER_COST="${FILTER_COST:-0}"
POST_LAMBDA="${POST_LAMBDA:-10}"

# SIFT1M
# INDEX="${INDEX:-$PROJ_DIR/data/sift1m/acorn_sift1m.faiss_index}"
# QUERY="${QUERY:-/dataset/SIFT1M/sift_query.fbin}"
# LABELS="${LABELS:-$PROJ_DIR/data/sift1m/base_labels.ibin}"
# QLABELS="${QLABELS:-$PROJ_DIR/data/sift1m/query_labels.ibin}"
# GT="${GT:-$PROJ_DIR/data/sift1m/sift1m_gt_filtered.ibin}"

# LAION10M
INDEX="${INDEX:-$PROJ_DIR/data/laion10m/acorn_laion10m_efc500.faiss_index}"
QUERY="${QUERY:-/dataset/LAION/LAION_test_query_textemb_10k.fbin}"
LABELS="${LABELS:-$PROJ_DIR/data/laion10m/base_labels_uniform_n10.ibin}"
QLABELS="${QLABELS:-$PROJ_DIR/data/laion10m/query_labels_uniform_n10.ibin}"
GT="${GT:-$PROJ_DIR/data/laion10m/laion10m_gt_filtered_uniform_n10.ibin}"

# DEEP10M
# INDEX="${INDEX:-$PROJ_DIR/data/deep10m/acorn_deep10m_efc500.faiss_index}"
# QUERY="${QUERY:-/dataset/DEEP10M/query.fbin}"
# LABELS="${LABELS:-$PROJ_DIR/data/deep10m/base_labels_uniform_n10.ibin}"
# QLABELS="${QLABELS:-$PROJ_DIR/data/deep10m/query_labels_uniform_n10.ibin}"
# GT="${GT:-$PROJ_DIR/data/deep10m/deep10m_gt_filtered_uniform_n10.ibin}"


taskset -c 0-$((THREADS-1)) "$BIN" \
    --index "$INDEX" \
    --query "$QUERY" \
    --labels "$LABELS" \
    --qlabels "$QLABELS" \
    --gt "$GT" \
    --k "$K" \
    --ef "$EF" \
    --threads "$THREADS" \
    --efs "$EFS" \
    --filter-cost "$FILTER_COST" \
    --post-lambda "$POST_LAMBDA" \
    --nq "$NQ" \
    --mode "$MODE" \
    --Helec "$HELEC"
