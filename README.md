# Adaptive Intra-query Parallel Filtered ANNS

This repository implements an adaptive intra-query parallel framework for
filtered approximate nearest-neighbor search (ANNS). ACORN is used as the graph
index backend: the system loads a FAISS-built `IndexACORN` file, attaches
external label metadata, and runs filtered graph search under different
predicate selectivities and predicate evaluation costs.

The main research focus is not ACORN index construction. Instead, this codebase
studies how to execute filtered graph search efficiently when each query may
have a different filter selectivity and a different predicate cost. The system
supports several fixed filtered-search strategies and an adaptive selector that
chooses the strategy and search parameter for each query.

## What Is Included

- ACORN graph loader for FAISS-built `IndexACORN` files
- Serial filtered search baselines:
  - in-filter graph search
  - pre-filter brute-force search
  - post-filter search
- Intra-query parallel filtered search:
  - ScatterSearch
  - iQAN-style sync-and-redistribute search
  - no-sync parallel search
- Adaptive selector for mixed filtered workloads
  - online selectivity estimation by sampling
  - predicate-cost-aware lookup table
  - per-query strategy and `efs` selection
- Offline profiling utilities for recall-latency lookup construction
- Label generation, query-cost generation, and brute-force filtered ground truth
- Plotting and experiment scripts used by the extension experiments

## System Overview

For each query, the system receives:

```text
query vector q
query label l
predicate cost c
```

The query label defines a filter predicate over base vectors:

```text
accept(x) = (label(x) == l)
```

The adaptive selector estimates the predicate selectivity online, maps the
query to a profiled selectivity/cost bucket, and then looks up the best search
strategy and `efs` value for the target recall.

Candidate strategies include:

| Strategy | Description |
| --- | --- |
| `parallel_pre` | Pre-filter accepted ids, then search over the filtered set |
| `parallel_in_scatter` | In-filter ScatterSearch over the ACORN graph |
| `parallel_in_iqan` | In-filter iQAN-style parallel graph search |
| `parallel_post_scatter` | Unfiltered ScatterSearch, then post-filter candidates |
| `parallel_post_iqan` | Unfiltered iQAN-style search, then post-filter candidates |

The selector is designed for heterogeneous online workloads: very selective
queries, medium-selectivity queries, high-selectivity queries, cheap predicates,
and expensive predicates may all prefer different strategies.

## Predicate Cost Model

Predicate evaluation is routed through `acorn::check_filter(...)`. The actual
predicate result is still a label bitmap lookup, but an integer cost level adds
synthetic per-check work:

```cpp
volatile int sink = 0;
for (int i = 0; i < filter_check_cost; i++)
    sink += (i ^ id) & 1;
return filter_map[id] != 0;
```

This gives a controlled predicate-cost knob without changing selectivity. The
cost can be set from command-line tools using:

```text
--filter-cost <int>
```

For analysis, the synthetic cost can be converted to a predicate-to-distance
cost ratio:

```text
filter_to_distance_ratio = C_filter / C_dist
```

and to a selectivity-aware effective ratio:

```text
rho = C_filter / (selectivity * C_dist)
```

The profiling binary `profile_filter_distance_cost` measures both predicate
checking cost and distance-computation cost.

## Requirements

- CMake >= 3.10
- C++14 compiler
- OpenMP
- x86-64 with AVX2 + FMA
- Python 3 for scripts
- Matplotlib for plotting scripts

## Project Structure

```text
include/acorn/
  acorn_graph.h     ACORN graph data, FAISS loader, search APIs
  distance.h        L2 / inner-product kernels
  file_io.h         fbin / ibin / ground-truth I/O
  types.h           shared types and search-pool helpers

src/
  acorn_graph.cpp   ACORN loading, serial search, parallel search, filtering
  distances.cpp     SIMD distance kernels
  file_io.cpp       binary file readers/writers

examples/
  search.cpp                         fixed search modes and recall evaluation
  run_adaptive_selector.cpp          adaptive selector and fixed mixed workload runner
  profile_filter_lookup.cpp          offline recall-latency profiling
  profile_filter_distance_cost.cpp   predicate/distance cost profiling
  gen_labels.cpp                     label and query-cost generation
  compute_filtered_gt.cpp            brute-force filtered ground truth
  extract_range.cpp                  query-range extraction

scripts/
  build_adaptive_selector_lookup.py      build selector lookup tables
  build_experiment2_recall_latency.py    fixed recall-latency curves
  build_experiment3_adaptive_mixed.py    adaptive vs fixed mixed workload curves
  build_cost_rho_table.py                build cost-rho conversion table
  plot_*.py                              plotting helpers

docs/
  extension_experiments_laion10m.md      final LAION10M experiment setup
  query_adaptive_parallel_filter_selector.md
  work_memory.md
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
- `run_adaptive_selector`
- `profile_filter_lookup`
- `profile_filter_distance_cost`
- `gen_labels`
- `compute_filtered_gt`
- `extract_range`

There is no `build_index` target in this repository. ACORN indexes are built
outside this project and loaded from FAISS `IndexACORN` files.

## Fixed Search Evaluation

Use `build/search` to run fixed strategies on binary-selectivity workloads:

```bash
./build/search \
  --index data/laion10m/acorn_laion10m_efc500.faiss_index \
  --query /dataset/LAION/LAION_test_query_textemb_10k.fbin \
  --labels data/laion10m/base_labels_binary_s20.ibin \
  --qlabels data/laion10m/query_labels_all1.ibin \
  --gt data/laion10m/laion10m_gt_binary_s20.ibin \
  --k 100 \
  --nq 1000 \
  --threads 4 \
  --efs 400 \
  --filter-cost 0 \
  --mode scatter
```

Common modes:

| Mode | Description |
| --- | --- |
| `serial` | Single-threaded in-filter graph search |
| `serial_sweep` | Sweep `efSearch` for serial in-filter search |
| `pre` | Pre-filter brute-force search |
| `post` | Unfiltered serial search, then filter candidates |
| `post_sweep` | Sweep post-filter serial search |
| `scatter` | In-filter ScatterSearch |
| `scatter_sweep` | Sweep in-filter ScatterSearch |
| `iqan` | In-filter iQAN-style parallel search |
| `iqan_sweep` | Sweep in-filter iQAN-style search |
| `post_parallel` | Parallel post-filter ScatterSearch |
| `post_parallel_iqan` | Parallel post-filter iQAN-style search |
| `all` | Run all supported modes and print a summary |

## Adaptive Selector Workflow

The adaptive selector uses an offline-then-online workflow.

1. Generate or prepare label files and filtered ground truth.
2. Profile candidate strategies across selectivity, predicate cost, and `efs`.
3. Measure predicate-to-distance cost ratios.
4. Build a lookup table for the target recall.
5. Run `run_adaptive_selector` on a mixed workload.

The final LAION10M experiment setup is documented in:

```text
docs/extension_experiments_laion10m.md
```

### Offline Profiling

`profile_filter_lookup` evaluates candidate strategies and writes
recall-latency rows:

```bash
./build/profile_filter_lookup \
  --index data/laion10m/acorn_laion10m_efc500.faiss_index \
  --query data/laion10m/train/laion10m_query_2000_3000.fbin \
  --label-dir data/laion10m/train \
  --gt-dir data/laion10m/train \
  --out results/offline_lookup/profile.tsv \
  --threads 4 \
  --nq 100 \
  --k 100 \
  --costs 0,1,2,3,4,5
```

`profile_filter_distance_cost` converts synthetic cost levels to measured cost
ratios:

```bash
./build/profile_filter_distance_cost \
  --base /dataset/LAION/LAION_base_imgemb_10M.fbin \
  --dim 512 \
  --selectivity 0.2 \
  --costs 0,1,2,3,4,5
```

Build a selector lookup table:

```bash
python3 scripts/build_adaptive_selector_lookup.py \
  --profile results/offline_lookup/profile.tsv \
  --cost-rho results/offline_lookup/cost_rho.tsv \
  --target-recall 0.95 \
  --out results/offline_lookup/adaptive_selector_lookup_r0p95.tsv
```

### Online Mixed Workload

Run the adaptive selector:

```bash
./build/run_adaptive_selector \
  --index data/laion10m/acorn_laion10m_efc500.faiss_index \
  --query /dataset/LAION/LAION_test_query_textemb_10k.fbin \
  --base-labels data/laion10m/base_labels_mix.ibin \
  --query-labels data/laion10m/query_labels_mix_lesslow_n1000.ibin \
  --query-costs data/laion10m/query_cost_mix_n1000.ibin \
  --gt data/laion10m/laion10m_gt_mix_lesslow_n1000.ibin \
  --cost-rho results/offline_lookup_v2_target32_unfilteredM/cost_rho_cost0_5_formal.tsv \
  --lookup results/offline_lookup_v2_target32_unfilteredM/adaptive_selector_lookup_r0p95_cost0_5_formal.tsv \
  --k 100 \
  --nq 1000 \
  --threads 4 \
  --filtered-expand-target 32 \
  --mode adaptive \
  --out results/adaptive_selector_r0p95_nq1000.tsv
```

The output is per-query and includes:

```text
qid, query label, query cost, sampled selectivity, selectivity bucket,
rho, selected strategy, selected efs, selector time, search time,
total time, recall, result validity, status
```

This detailed output is useful for analyzing how the selector changes decisions
with predicate selectivity and predicate cost.

## Experiment Scripts

The main extension experiments are scripted around existing result files:

- `scripts/build_experiment2_recall_latency.py`
  - Recall-latency curves for in-filter ScatterSearch vs in-filter iQAN
  - Binary selectivities such as `5%, 10%, 20%, 50%`
- `scripts/build_experiment3_adaptive_mixed.py`
  - Adaptive selector vs fixed strategies on mixed workloads
  - Produces recall-latency curves and summary TSVs
- `scripts/plot_thread_scaling.py`
  - Thread-scaling plots for in-filter ScatterSearch

Additional combined figures and paper-ready plots are stored under:

```text
results/extension_combined_figures/
```

## File Formats

- `fbin`: `int32_t n`, `int32_t d`, then `n * d` float32 values in row-major order
- `ibin`: `int32_t count`, then `count` int32 values
- filtered ground truth: `int32_t nq`, `int32_t k`, then `nq * k` int32 ids
- index: FAISS `IndexACORN` binary layout loaded by `ACORN::load_from_faiss`

## Notes

- Base labels are stored separately from the FAISS index and passed to the
  loader or runner at evaluation time.
- The adaptive selector relies on offline profiling tables. If a new dataset is
  used, regenerate labels, ground truth, candidate profiling, cost-rho tables,
  and selector lookup tables for that dataset.
- The code assumes that the ACORN graph and flat vector storage are present in
  the FAISS index file.

## License

MIT. See [LICENSE](LICENSE).
