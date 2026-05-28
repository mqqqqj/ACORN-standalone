// -*- c++ -*-
#pragma once

#include <vector>
#include "types.h"
#include "acorn_graph.h"

namespace acorn {

struct IndexACORN {
    int d = 0;
    idx_t ntotal = 0;
    MetricType metric_type = METRIC_L2;
    bool is_trained = false;
    bool verbose = false;

    ACORN acorn;
    size_t code_size = 0;
    std::vector<uint8_t> codes;
    std::vector<int> metadata_storage;

    IndexACORN() {}

    IndexACORN(int d, int M, int gamma, std::vector<int>& metadata,
               int M_beta, MetricType metric = METRIC_L2)
        : d(d), metric_type(metric), is_trained(true),
          acorn(M, gamma, metadata, M_beta),
          code_size(sizeof(float) * d) {}

    // --- Vector storage ---
    void add(idx_t n, const float* x);
    void train(idx_t, const float*) { is_trained = true; }
    void reset();

    // --- Search ---
    void search(idx_t n, const float* x, idx_t k,
                float* distances, int* labels,
                const SearchParameters* params = nullptr) const;

    void search(idx_t n, const float* x, idx_t k,
                float* distances, int* labels,
                char* filter_id_map,
                const SearchParameters* params = nullptr) const;

    void parallelSearch(idx_t n, const float* x, idx_t k,
                        float* distances, int* labels,
                        int num_threads, int efs,
                        const SearchParameters* params = nullptr) const;

    void parallelSearch(idx_t n, const float* x, idx_t k,
                        float* distances, int* labels,
                        char* filter_id_map,
                        int num_threads, int efs,
                        const SearchParameters* params = nullptr) const;

    // --- Persistence ---
    void save(const char* filename) const;
    void load(const char* filename);

    // --- Access ---
    float* get_xb() { return (float*)codes.data(); }
    const float* get_xb() const { return (const float*)codes.data(); }
};

} // namespace acorn
