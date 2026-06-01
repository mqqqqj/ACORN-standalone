#!/bin/bash
# End-to-end pipeline:
#   1. Generate labels
#   2. Compute ground truth (filtered + unfiltered)
#   3. Build ACORN index
#   4. Search & evaluate
#
# Defaults target SIFT1M; override env vars for other datasets.
#
# Set SKIP_GT=1 to skip brute-force GT computation (slow).
# Set SKIP_BUILD=1 to skip index building (use existing).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  ACORN Pipeline"
echo "============================================"
echo

# Step 1: Generate base labels
echo ">>> Step 1: Generate base labels"
"$SCRIPT_DIR/gen_labels.sh"
echo

# Step 2: Generate query labels
echo ">>> Step 2: Generate query labels"
BASE=/dataset/SIFT1M/sift_query.fbin \
OUTPUT="$(dirname "$SCRIPT_DIR")/data/sift1m/query_labels.ibin" \
"$SCRIPT_DIR/gen_labels.sh"
echo

# Step 3-4: Ground truth
if [ "${SKIP_GT:-0}" = "1" ]; then
    echo ">>> SKIP_GT=1, skipping ground truth"
else
    echo ">>> Step 3: Compute unfiltered ground truth"
    "$SCRIPT_DIR/compute_gt.sh"
    echo

    echo ">>> Step 4: Compute filtered ground truth"
    "$SCRIPT_DIR/compute_filtered_gt.sh"
    echo
fi

# Step 5: Build index
if [ "${SKIP_BUILD:-0}" = "1" ]; then
    echo ">>> SKIP_BUILD=1, skipping index build"
else
    echo ">>> Step 5: Build ACORN index"
    "$SCRIPT_DIR/build.sh"
    echo
fi

# Step 6: Search
echo ">>> Step 6: Search evaluation"
"$SCRIPT_DIR/search.sh"

echo
echo "============================================"
echo "  Pipeline complete."
echo "============================================"
