#!/bin/bash
# Offline profiling grid for adaptive filtered search lookup, compact output.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

INDEX="${INDEX:-$PROJ_DIR/data/laion10m/acorn_laion10m_efc500.faiss_index}"
QUERY="${QUERY:-$PROJ_DIR/data/laion10m/train/laion10m_query_2000_3000.fbin}"
DATA_DIR="${DATA_DIR:-$PROJ_DIR/data/laion10m}"
TRAIN_DIR="${TRAIN_DIR:-$PROJ_DIR/data/laion10m/train}"
OUT_DIR="${OUT_DIR:-$PROJ_DIR/results/offline_lookup_v2}"

NQ="${NQ:-100}"
K="${K:-100}"
THREADS="${THREADS:-4}"
HELEC="${HELEC:-50}"
FILTERED_EXPAND_TARGET="${FILTERED_EXPAND_TARGET:-0}"
COSTS="${COSTS:-0,8,16,24,32}"
EFS_LIST="${EFS_LIST:-100,200,300,400,500,600,700,800,900,1000,1200,1400}"
STRATEGIES="${STRATEGIES:-parallel_pre,parallel_in_scatter,parallel_in_iqan,parallel_post_scatter,parallel_post_iqan}"
PRE_MAX_SELECTIVITY="${PRE_MAX_SELECTIVITY:-0}"
PRE_COSTS="${PRE_COSTS:-0,8,16,24,32}"
MAX_AVG_MS="${MAX_AVG_MS:-0}"

RAW_OUT="$OUT_DIR/laion10m_train_filter_lookup_v2_4t_nq${NQ}.tsv"

mkdir -p "$OUT_DIR"

"$PROJ_DIR/build/profile_filter_lookup" \
    --index "$INDEX" \
    --query "$QUERY" \
    --data-dir "$DATA_DIR" \
    --train-dir "$TRAIN_DIR" \
    --out "$RAW_OUT" \
    --nq "$NQ" \
    --k "$K" \
    --threads "$THREADS" \
    --Helec "$HELEC" \
    --filtered-expand-target "$FILTERED_EXPAND_TARGET" \
    --costs "$COSTS" \
    --efs-list "$EFS_LIST" \
    --strategies "$STRATEGIES" \
    --pre-max-selectivity "$PRE_MAX_SELECTIVITY" \
    --pre-costs "$PRE_COSTS" \
    --max-avg-ms "$MAX_AVG_MS" \
    --compact-output

echo "Raw lookup table: $RAW_OUT"
