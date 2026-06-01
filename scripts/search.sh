#!/bin/bash
#
# Search ACORN index on SIFT1M (filtered, 13-class labels).
#
# Prerequisites:
#   - Build the project first:  cd build && make search
#   - ACORN index at ../data/sift1m/acorn_sift1m.index
#   - SIFT1M query vectors at /dataset/SIFT1M/sift_query.fbin
#   - Labels and ground truth in ../data/sift1m/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
BIN="$PROJ_DIR/build/search"

INDEX="${INDEX:-$PROJ_DIR/data/sift1m/acorn_sift1m.index}"
QUERY="${QUERY:-/dataset/SIFT1M/sift_query.fbin}"
LABELS="${LABELS:-$PROJ_DIR/data/sift1m/labels.ibin}"
QLABELS="${QLABELS:-$PROJ_DIR/data/sift1m/query_labels.ibin}"
GT="${GT:-$PROJ_DIR/data/sift1m/sift1m_gt_filtered.ibin}"
K="${K:-100}"
EF="${EF:-400}"
THREADS="${THREADS:-4}"
EFS="${EFS:-100}"
NQ="${NQ:-1000}"

echo "============================================"
echo "  Search SIFT1M ACORN Index"
echo "============================================"
echo "Index:   $INDEX"
echo "Query:   $QUERY"
echo "Labels:  $LABELS"
echo "GT:      $GT"
echo "Params:  k=$K, ef=$EF, threads=$THREADS, efs=$EFS, nq=$NQ"
echo

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
    --nq "$NQ"
