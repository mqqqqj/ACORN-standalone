# ACORN-standalone: Approximate Nearest Neighbor Search with Attribute Filters

> **ACORN** was originally implemented as part of the [FAISS](https://github.com/facebookresearch/faiss) vector search library. This repository extracts the ACORN index into a **standalone** C++ library with no dependency on FAISS, making it easier to understand, modify, and benchmark the ACORN algorithm in isolation.

ACORN is a high-performance C++ library for approximate nearest neighbor (ANN) search over large-scale vector datasets. It builds a graph-based proximity index and supports **hybrid search** — filtering results by discrete attribute values during the ANN traversal itself, rather than as a post-processing step.

## How It Works

ACORN constructs a layered navigable graph (similar in spirit to HNSW), then applies a **predicate-guided pruning strategy** during both construction and search. Each vector is associated with an integer metadata label. The pruning strategy:

- Prioritizes edges to nodes sharing the same attribute, ensuring the filtered subgraph remains well-connected.
- Expands candidate neighborhoods by looking at neighbors-of-neighbors when the immediate neighborhood has insufficient matches.
- Uses a `gamma` parameter to control the trade-off between raw search speed and filtered recall.

This design means ACORN achieves high recall under attribute filters without needing to build separate per-attribute indices.

## Features

- **Hybrid vector + predicate search** — filter by integer attribute at query time
- **L2 and inner product** distance metrics
- **SIMD-accelerated** distance computations (AVX2 + FMA)
- **OpenMP-parallel** index construction
- **Save/load** indices to disk
- **Single-header-friendly** — the core library compiles to a static library with a clean public API
- **SIFT1M-scale** evaluation harness included

## Requirements

- C++11 or later
- CMake ≥ 3.10
- OpenMP
- x86-64 with AVX2 + FMA support

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

This produces `libacorn.a` and the example binaries.

## Quick Start

```cpp
#include "acorn/index_acorn.h"

// Generate 10000 random 64-dimensional vectors
int n = 10000, d = 64;
std::vector<float> data(n * d);
// ... fill data ...

// Attach random metadata (10 categories)
std::vector<int> metadata(n);
for (int i = 0; i < n; i++) metadata[i] = i % 10;

// Build index
acorn::IndexACORNFlat index(d, /*M=*/32, /*gamma=*/12, metadata, /*M_beta=*/32);
index.add(n, data.data());

// Search: top-10 nearest to query
std::vector<acorn::idx_t> labels(10);
std::vector<float> distances(10);
acorn::SearchParametersACORN params;
params.efSearch = 64;
index.search(1, query.data(), 10, distances.data(), labels.data(), &params);
```

Run the included example:

```bash
./example --n 100000 --d 128 --k 10 --M 32 --gamma 12 --ef 64
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `M` | Base graph degree | 32 |
| `gamma` | Pruning multiplier — higher values improve filtered recall at the cost of index size | 12 |
| `M_beta` | Number of same-attribute edges to prioritize during pruning | M |
| `efConstruction` | Search depth during construction — higher = more accurate but slower build | M × gamma |
| `efSearch` | Search depth at query time — higher = more accurate but slower | 16 |

## Hybrid Search (with Attribute Filters)

```cpp
// Build a filter map: only consider vectors with metadata == 3
std::vector<char> filter_map(n, 0);
for (int i = 0; i < n; i++) {
    if (metadata[i] == 3) filter_map[i] = 1;
}

// Hybrid search
index.search(nq, queries.data(), k, distances.data(), labels.data(),
             filter_map.data(), &params);
```

## SIFT1M Evaluation

The `eval_sift1m` example provides a full evaluation pipeline on the SIFT1M benchmark:

```bash
./eval_sift1m \
    --base   /dataset/SIFT1M/sift_base.fbin \
    --query  /dataset/SIFT1M/sift_query.fbin \
    --labels labels.ibin \
    --gt     sift1m_gt_top100.ibin \
    --M 32 --gamma 12 --efc 200
```

It reports Recall@100 and QPS across a range of `efSearch` values.

## File Formats

- **fbin**: Little-endian binary with `int32_t(n_vectors)`, `int32_t(dimension)`, then `float32` vector data.
- **ibin**: Little-endian binary with `int32_t(count)`, then `int32_t` label data.

## License

MIT License — see [LICENSE](LICENSE) for details.
