# ACORN: Approximate Nearest Neighbor Search with Attribute Filters

ACORN is a graph-based approximate nearest neighbor (ANN) search library supporting
**predicate-filtered hybrid search** — filtering by discrete attribute during the
graph traversal itself, not as a post-processing step.

Originally part of [FAISS](https://github.com/facebookresearch/faiss), this is a
**standalone** extraction with zero external dependencies beyond the C++ standard
library and OpenMP.

## How It Works

ACORN builds a layered navigable graph and applies a **predicate-guided pruning
strategy** during both construction and search. Each vector has an integer label.
The algorithm:

- Prioritizes edges to nodes with the same label, keeping the filtered subgraph connected.
- Expands to neighbors-of-neighbors when immediate neighbors have insufficient label matches.
- Uses a `gamma` parameter to trade off index size vs. filtered recall.

Three search modes are provided:

| Mode | Description |
|------|-------------|
| Serial | Single-threaded NSG-style backtracking with 2-hop expansion |
| iQAN | Intra-query parallel: sync-and-redistribute, per-thread pool of `efs` candidates |
| No-Sync | Intra-query parallel: threads search independently, merge at the end |

## Requirements

- C++11 or later
- CMake ≥ 3.10
- OpenMP
- x86-64 with AVX2 + FMA

## Project Structure

```
├── include/acorn/       # Public headers
│   ├── acorn_graph.h    # ACORN index struct + all search methods
│   ├── types.h          # MetricType, SearchNeighbor, InsertIntoPool
│   ├── distance.h       # L2 / inner product with SIMD
│   ├── file_io.h        # fbin / ibin / groundtruth I/O
│   └── random.h         # RandomGenerator (used during construction)
├── src/                 # Library sources → libacorn.a
│   ├── acorn_graph.cpp  # Build, search (serial / iQAN / no-sync), persistence
│   ├── distances.cpp    # SIMD distance kernels
│   ├── file_io.cpp      # Binary format I/O
│   └── random.cpp       # Mersenne Twister wrapper
├── examples/            # Standalone tools (all CLI-driven, no hardcoded paths)
│   ├── build.cpp        # Build an index from base vectors + labels
│   ├── search.cpp       # Serial + iQAN + No-Sync search with recall evaluation
│   ├── gen_labels.cpp   # Generate random labels for a dataset
│   ├── compute_groundtruth.cpp      # Brute-force L2 ground truth
│   ├── compute_filtered_gt.cpp      # Brute-force filtered L2 ground truth
│   └── debug_search.cpp # Quick small-scale build + search for debugging
├── scripts/             # Convenience shell scripts (defaults target SIFT1M)
│   ├── build.sh         # Build index
│   ├── search.sh        # Search evaluation
│   ├── gen_labels.sh    # Generate labels
│   ├── compute_gt.sh    # Compute ground truth
│   ├── compute_filtered_gt.sh  # Compute filtered ground truth
│   ├── debug.sh         # Quick debug run
│   └── run_all.sh       # End-to-end pipeline
└── data/                # Pre-built SIFT1M index, labels, and ground truth
```

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Binaries are placed in `build/`:
`build_index`, `search`, `gen_labels`, `compute_groundtruth`,
`compute_filtered_gt`, `debug_search`.

## Quick Start

```cpp
#include "acorn/acorn_graph.h"
#include "acorn/file_io.h"

int main()
{
    int n = 10000, d = 64;

    // Generate random labels
    std::vector<int> labels(n);
    for (int i = 0; i < n; i++) labels[i] = i % 10;

    // Build index
    acorn::ACORN index(d, /*M=*/32, /*gamma=*/12, labels, /*M_beta=*/32, acorn::METRIC_L2);
    index.efConstruction = 200;
    // ... fill float vectors into base[n*d] ...
    index.add(n, base);

    // Save
    index.save("acorn.index");

    // Load
    acorn::ACORN idx2;
    idx2.load("acorn.index");

    // Serial search: 1 query, top-10
    std::vector<int> labs(10);
    std::vector<float> dists(10);
    int metric = 1;  // L2
    idx2.search(query, idx2.get_xb(), idx2.d, metric, 10, 64,
                labs.data(), dists.data());

    // Filtered search (only label == 3)
    std::vector<char> filter(n, 0);
    for (int i = 0; i < n; i++)
        if (labels[i] == 3) filter[i] = 1;
    idx2.search(query, idx2.get_xb(), idx2.d, metric, 10, 64,
                labs.data(), dists.data(), filter.data());

    // iQAN parallel search (4 threads, efs=100)
    idx2.iqan_search(query, idx2.get_xb(), idx2.d, metric, 10, 100,
                     labs.data(), dists.data(), 4, 100, filter.data());

    // No-sync parallel search (4 threads)
    idx2.no_sync_search(query, idx2.get_xb(), idx2.d, metric, 10, 100,
                        labs.data(), dists.data(), 4, filter.data());
}
```

## CLI Tools

All tools accept `--help` for usage. No file paths are hardcoded.

### Build Index

```bash
./build_index --base sift_base.fbin --labels labels.ibin --output acorn.index \
              --M 32 --gamma 12 --efc 200 --metric l2
```

### Search & Evaluate

```bash
# Unfiltered, no ground truth
./search --index acorn.index --query queries.fbin --k 100 --ef 200 --nq 1000

# Filtered with recall evaluation
./search --index acorn.index --query queries.fbin \
         --labels labels.ibin --qlabels query_labels.ibin \
         --gt gt_filtered.ibin --k 100 --ef 400 --threads 4 --efs 100 --nq 1000
```

### SIFT1M Pipeline

```bash
# One-shot (skips GT if already computed, skips build if index exists)
SKIP_GT=1 SKIP_BUILD=1 ./scripts/run_all.sh

# Or step by step
./scripts/gen_labels.sh
./scripts/compute_gt.sh
./scripts/compute_filtered_gt.sh
./scripts/build.sh
./scripts/search.sh
```

Default paths target the SIFT1M dataset in `/dataset/SIFT1M/`. Override with
environment variables:

```bash
BASE=/data/my_base.fbin QUERY=/data/my_query.fbin OUTPUT=./my.index ./scripts/build.sh
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `M` | Base graph degree | 32 |
| `gamma` | Pruning multiplier — higher improves filtered recall | 12 |
| `M_beta` | Same-label edges to prioritize before 2-hop expansion | M |
| `efConstruction` | Search depth during construction | M × gamma |
| `ef` | Search depth for serial query | 200 |
| `efs` | Pool size per thread in parallel search | 100 |
| `threads` | Number of parallel search threads | 4 |

## File Formats

- **fbin**: 4-byte `int32_t(n)` + 4-byte `int32_t(d)` + `n×d` × `float32` (row-major, LE)
- **ibin**: 4-byte `int32_t(count)` + `count` × `int32_t` (LE)
- **Ground truth** (ibin variant): 4-byte `int32_t(nq)` + 4-byte `int32_t(k)` + `nq×k` × `int32_t` (LE)
- **ACORN index**: custom binary with magic `ACRN` followed by header, vectors, labels, and graph data

## License

MIT — see [LICENSE](LICENSE).
