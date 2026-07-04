#!/bin/bash
# Offline profiling for the query-adaptive filtered search lookup table.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

INDEX="${INDEX:-$PROJ_DIR/data/laion10m/acorn_laion10m_efc500.faiss_index}"
QUERY="${QUERY:-$PROJ_DIR/data/laion10m/train/laion10m_query_2000_3000.fbin}"
DATA_DIR="${DATA_DIR:-$PROJ_DIR/data/laion10m}"
TRAIN_DIR="${TRAIN_DIR:-$PROJ_DIR/data/laion10m/train}"
OUT_DIR="${OUT_DIR:-$PROJ_DIR/results/offline_lookup}"

NQ="${NQ:-1000}"
K="${K:-100}"
THREADS="${THREADS:-4}"
HELEC="${HELEC:-50}"
COSTS="${COSTS:-0,32,64,128,256}"
EFS_LIST="${EFS_LIST:-100,200,400,700,1000}"
STRATEGIES="${STRATEGIES:-pre_parallel,scatter,iqan,post_parallel,post_parallel_iqan}"
PRE_MAX_SELECTIVITY="${PRE_MAX_SELECTIVITY:-0.02}"
PRE_COSTS="${PRE_COSTS:-0,32}"
MAX_AVG_MS="${MAX_AVG_MS:-0}"
TARGET_RECALL="${TARGET_RECALL:-0.9}"
LOW_SELECTIVITY_PRE_FALLBACK="${LOW_SELECTIVITY_PRE_FALLBACK:-0.01}"

RAW_OUT="$OUT_DIR/laion10m_train_filter_lookup_raw_4t_k${K}_nq${NQ}.tsv"
SELECTOR_OUT="$OUT_DIR/laion10m_train_filter_lookup_selector_r${TARGET_RECALL}_4t_k${K}_nq${NQ}.tsv"

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
    --costs "$COSTS" \
    --efs-list "$EFS_LIST" \
    --strategies "$STRATEGIES" \
    --pre-max-selectivity "$PRE_MAX_SELECTIVITY" \
    --pre-costs "$PRE_COSTS" \
    --max-avg-ms "$MAX_AVG_MS"

python3 "$PROJ_DIR/scripts/build_filter_lookup_selector.py" \
    --raw "$RAW_OUT" \
    --out "$SELECTOR_OUT" \
    --target-recall "$TARGET_RECALL" \
    --low-selectivity-pre-fallback "$LOW_SELECTIVITY_PRE_FALLBACK"

echo "Raw lookup table:      $RAW_OUT"
echo "Selector lookup table: $SELECTOR_OUT"
