// -*- c++ -*-
#pragma once

#include <cstring>
#include <vector>
#include <string>
#include <omp.h>

#include "types.h"
#include "distance.h"

namespace acorn
{

    struct ACORN
    {
        using storage_idx_t = int32_t;

        // --- Graph data ---
        std::vector<double> assign_probas;
        std::vector<int> cum_nneighbor_per_level;
        std::vector<int> levels;
        std::vector<storage_idx_t> nb_per_level;
        std::vector<size_t> offsets;
        std::vector<storage_idx_t> neighbors;
        storage_idx_t entry_point;

        int gamma, M, M_beta, max_level;
        int efConstruction, efSearch;

        // Metadata for hybrid search (points to metadata_storage.data())
        const int *metadata;

        // --- Index data (from old IndexACORN) ---
        int d = 0;
        idx_t ntotal = 0;
        MetricType metric_type = METRIC_L2;
        bool verbose = false;
        size_t code_size = 0;
        std::vector<uint8_t> codes;
        std::vector<int> metadata_storage;

        // --- Constructors ---
        ACORN() : entry_point(-1),
                  gamma(0), M(0), M_beta(0), max_level(-1),
                  efConstruction(0), efSearch(16),
                  metadata(nullptr) { offsets.push_back(0); }

        // --- Graph-level methods ---
        int nb_neighbors(int layer_no) const;
        int cum_nb_neighbors(int layer_no) const;
        void neighbor_range(idx_t no, int layer_no, size_t *begin, size_t *end) const;

        // --- Raw graph search (NSG-style sorted pool) ---
        // xb = base vectors, d = dimension, metric: 0=IP, 1=L2
        int search(const float *query, const float *xb, int d, int metric,
                   int k, int efSearch_val,
                   int *indices, float *distances,
                   const char *filter_map) const;

        // --- iQAN search (sync-and-redistribute) ---
        int iqan_search(const float *query, const float *xb, int d, int metric,
                        int k, int efSearch_val,
                        int *indices, float *distances,
                        int num_threads, const char *filter_map) const;

        // --- No-sync parallel search ---
        int no_sync_search(const float *query, const float *xb, int d, int metric,
                           int k, int efSearch_val,
                           int *indices, float *distances,
                           int num_threads,
                           const char *filter_map) const;

    // --- ScatterSearch (ICDE 2026) ---
        int scatter_search(const float* query, const float* xb, int d, int metric,
                           int k, int efSearch_val,
                           int* indices, float* distances,
                           int num_threads, int Helec,
                           const char* filter_map) const;

        // --- FAISS index loading ---
        void load_from_faiss(const char *filename, const std::vector<int> &labels);

        // --- Access ---
        float *get_xb() { return (float *)codes.data(); }
        const float *get_xb() const { return (const float *)codes.data(); }
    };

    // Per-thread NDC profiling
    void reset_thread_ndis(int num_threads);
    void reset_ser_ndis();
    const std::vector<size_t> &get_thread_ndis();
    size_t get_ser_ndis();

    // Phase timing for no_sync / scatter comparison
    struct PhaseTiming { double phase1 = 0, parallel = 0, merge = 0; };
    void reset_phase_timing();
    const PhaseTiming &get_nosync_timing();
    const PhaseTiming &get_scatter_timing();

} // namespace acorn
