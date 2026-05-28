// -*- c++ -*-
#pragma once

#include <cstring>
#include <vector>
#include <string>
#include <omp.h>

#include "types.h"
#include "random.h"
#include "distance.h"

namespace acorn {

struct ACORN {
    using storage_idx_t = int32_t;

    // --- Graph data ---
    std::vector<double> assign_probas;
    std::vector<int> cum_nneighbor_per_level;
    std::vector<int> levels;
    std::vector<storage_idx_t> nb_per_level;
    std::vector<size_t> offsets;
    std::vector<storage_idx_t> neighbors;
    storage_idx_t entry_point;
    RandomGenerator rng;

    int gamma, M, M_beta, max_level;
    int efConstruction, efSearch;
    bool check_relative_distance;
    int upper_beam;

    // Metadata for hybrid search
    const int* metadata;

    // --- Construction ---
    ACORN() : entry_point(-1), rng(12345),
              gamma(0), M(0), M_beta(0), max_level(-1),
              efConstruction(0), efSearch(16),
              check_relative_distance(true), upper_beam(1),
              metadata(nullptr) { offsets.push_back(0); }

    explicit ACORN(int M, int gamma, std::vector<int>& metadata, int M_beta);

    void set_default_probas(int M, float levelMult, int M_beta, int gamma = 1);
    int random_level();
    int nb_neighbors(int layer_no) const;
    int cum_nb_neighbors(int layer_no) const;
    void neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end) const;
    int prepare_level_tab(size_t n, bool preset_levels = false);
    void reset();

    // --- Search (NSG-style sorted pool) ---
    // xb = base vectors, d = dimension, metric: 0=IP, 1=L2
    int search(const float* query, const float* xb, int d, int metric,
               int k, int efSearch_val,
               int* indices, float* distances,
               const char* filter_map = nullptr) const;

    // --- Parallel search ---
    int parallel_search(const float* query, const float* xb, int d, int metric,
                        int k, int efSearch_val,
                        int* indices, float* distances,
                        int num_threads, int efs,
                        const char* filter_map = nullptr) const;

    // --- Save / Load ---
    void save(FILE* fp) const;
    void load(FILE* fp);

    // --- Construction helpers ---
    void shrink_neighbor_list(const float* query, int q_id,
                              std::vector<std::pair<float, int>>& candidates,
                              int max_size, int gamma_val);
    int search_neighbors_to_add(const float* query, int entry,
                                float d_entry, int level,
                                std::vector<uint8_t>& visited, int vis_mark,
                                std::vector<std::pair<float, int>>& results,
                                int ef);
    void add_links_starting_from(const float* query, int pt_id,
                                  int nearest, float d_nearest, int level,
                                  omp_lock_t* locks,
                                  std::vector<uint8_t>& visited, int vis_mark);
    void add_with_locks(const float* query, int pt_level, int pt_id,
                        std::vector<omp_lock_t>& locks,
                        std::vector<uint8_t>& visited, int vis_mark);

    // --- Stats ---
    void print_neighbor_stats(int level) const;
    void print_neighbor_stats(bool edge_list, bool filtered = false,
                               int filter = -1, Operation op = EQUAL) const;
};

// Simple global stats
struct ACORNStats {
    size_t n1, n2, n3, ndis, nreorder;
    ACORNStats() : n1(0), n2(0), n3(0), ndis(0), nreorder(0) {}
    void combine(const ACORNStats& o) {
        n1 += o.n1; n2 += o.n2; n3 += o.n3;
        ndis += o.ndis; nreorder += o.nreorder;
    }
};
extern ACORNStats acorn_stats;

} // namespace acorn
