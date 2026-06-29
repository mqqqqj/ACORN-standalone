# ACORN Standalone Search

This repository contains the standalone search/evaluation path for ACORN graphs
exported from FAISS. It intentionally does not build ACORN graphs locally: the
supported workflow is to load a FAISS-built `IndexACORN` file, attach label
metadata, and run filtered graph search.

## What Is Included

- FAISS ACORN graph loader: `ACORN::load_from_faiss(...)`
- Serial in-filter, pre-filter, and post-filter search
- Intra-query parallel search modes: `iqan`, `nosync`, and `scatter`
- Label generation and brute-force filtered ground-truth utilities
- Filter-radius analysis utility

## Requirements

- CMake >= 3.10
- C++14 compiler
- OpenMP
- x86-64 with AVX2 + FMA

## Project Structure

```text
include/acorn/
  acorn_graph.h     ACORN graph data, FAISS loader, search APIs
  distance.h        L2 / inner-product kernels
  file_io.h         fbin / ibin / ground-truth I/O
  types.h           shared types and search-pool helpers

src/
  acorn_graph.cpp   FAISS graph loading and search implementations
  distances.cpp     SIMD distance kernels
  file_io.cpp       binary file readers/writers

examples/
  search.cpp                  search and recall evaluation
  gen_labels.cpp              generate label files
  compute_filtered_gt.cpp     brute-force filtered ground truth
  analyze_filter_radius.cpp   filtered/unfiltered radius analysis

scripts/
  search.sh
  gen_labels.sh
  compute_filtered_gt.sh
```

## Build

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Generated binaries:

- `search`
- `gen_labels`
- `compute_filtered_gt`
- `analyze_filter_radius`

There is no `build_index` target in this repository.

## Search

```bash
./build/search \
  --index acorn.faiss_index \
  --query queries.fbin \
  --labels base_labels.ibin \
  --qlabels query_labels.ibin \
  --gt gt_filtered.ibin \
  --k 100 \
  --ef 400 \
  --threads 8 \
  --efs 200 \
  --filter-cost 0 \
  --post-lambda 5 \
  --mode serial
```

Search modes:

| Mode | Description |
|------|-------------|
| `serial` | Single-threaded in-filter graph search |
| `pre` | Pre-filter brute-force search over accepted ids |
| `post` | Unfiltered serial graph search, then filter candidates to top-k |
| `pre_parallel` | Parallel pre-filter brute-force search over id-range chunks |
| `post_parallel` | `no_filter_scatter_search` for candidates, then filter to top-k |
| `iqan` | Sync-and-redistribute intra-query parallel search |
| `nosync` | Independent per-thread search with final merge |
| `scatter` | Leader-guided parallel search |
| `all` | Run all modes and print a summary |

The `scripts/search.sh` wrapper provides dataset-oriented defaults that can be
overridden with environment variables:

```bash
INDEX=/path/to/acorn.faiss_index \
QUERY=/path/to/query.fbin \
LABELS=/path/to/base_labels.ibin \
QLABELS=/path/to/query_labels.ibin \
GT=/path/to/gt_filtered.ibin \
MODE=serial \
FILTER_COST=0 \
POST_LAMBDA=5 \
./scripts/search.sh
```

`--filter-cost` controls synthetic work inside `acorn::check_filter(...)` for
simulating different predicate-check costs. The default is `0`.
`--post-lambda` controls how many candidates post-filter search retrieves before
filtering: `candidate_k = min(ntotal, post_lambda * k)`. The default is `5`.

## C++ Usage

```cpp
#include "acorn/acorn_graph.h"
#include "acorn/file_io.h"

std::vector<int> base_labels = acorn::read_ibin("base_labels.ibin");

acorn::ACORN index;
index.load_from_faiss("acorn.faiss_index", base_labels);

std::vector<char> filter(index.ntotal, 0);
for (int i = 0; i < index.ntotal; i++) {
    if (base_labels[i] == query_label) {
        filter[i] = 1;
    }
}

std::vector<int> ids(k);
std::vector<float> distances(k);
int metric = (index.metric_type == acorn::METRIC_INNER_PRODUCT) ? 0 : 1;

int found = index.search(
    query,
    index.get_xb(),
    index.d,
    metric,
    k,
    ef,
    ids.data(),
    distances.data(),
    filter.data());
```

## File Formats

- `fbin`: `int32_t n`, `int32_t d`, then `n * d` float32 values in row-major order
- `ibin`: `int32_t count`, then `count` int32 values
- filtered ground truth: `int32_t nq`, `int32_t k`, then `nq * k` int32 ids
- index: FAISS `IndexACORN` binary layout loaded by `ACORN::load_from_faiss`

## Notes

The loader expects the ACORN graph and its flat vector storage to be present in
the FAISS index file. Base labels are stored separately in this repo's workflow
and passed to `load_from_faiss`.

## License

MIT. See [LICENSE](LICENSE).
