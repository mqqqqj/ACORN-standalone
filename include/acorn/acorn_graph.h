// -*- c++ -*-
#pragma once

#include <queue>
#include <unordered_set>
#include <vector>
#include <string>

#include <omp.h>

#include "types.h"
#include "visited.h"
#include "heap.h"
#include "random.h"
#include "distance.h"

namespace acorn {

struct ACORNStats;

struct SearchParametersACORN : SearchParameters {
    int efSearch = 16;
    bool check_relative_distance = true;
    ~SearchParametersACORN() {}
};

struct ACORN {
    using storage_idx_t = int32_t;
    typedef std::pair<float, storage_idx_t> Node;
    typedef storage_idx_t NeighNode;

    /// Heap structure for fast search
    struct MinimaxHeap {
        int n;
        int k;
        int nvalid;
        std::vector<storage_idx_t> ids;
        std::vector<float> dis;
        typedef CMax<float, storage_idx_t> HC;

        explicit MinimaxHeap(int n) : n(n), k(0), nvalid(0), ids(n), dis(n) {}

        void push(storage_idx_t i, float v);
        float max() const;
        int size() const;
        void clear();
        int pop_min(float* vmin_out = nullptr);
        int count_below(float thresh);

        /// Collect all valid (non -1) entries for redistribution
        void collect_valid(std::vector<std::pair<float, storage_idx_t>>& out) const {
            for (int i = 0; i < k; i++) {
                if (ids[i] != -1) out.emplace_back(dis[i], ids[i]);
            }
        }
    };

    /// for sorting (distance, id) pairs
    struct NodeDistCloser {
        float d;
        int id;
        NodeDistCloser(float d, int id) : d(d), id(id) {}
        bool operator<(const NodeDistCloser& obj1) const { return d < obj1.d; }
    };

    struct NodeDistFarther {
        float d;
        int id;
        NodeDistFarther(float d, int id) : d(d), id(id) {}
        bool operator<(const NodeDistFarther& obj1) const { return d > obj1.d; }
    };

    // configuration
    std::vector<double> assign_probas;
    std::vector<int> cum_nneighbor_per_level;
    std::vector<int> levels;
    std::vector<storage_idx_t> nb_per_level;
    std::vector<size_t> offsets;
    std::vector<NeighNode> neighbors;

    storage_idx_t entry_point;
    RandomGenerator rng;

    int gamma;
    int M;
    int M_beta;
    int max_level;
    int efConstruction;
    int efSearch;
    bool check_relative_distance = true;
    int upper_beam;
    bool search_bounded_queue = true;

    // metadata for hybrid search
    const int* metadata;
    std::vector<std::string> metadata_strings;

    void set_default_probas(int M, float levelMult, int M_beta, int gamma = 1);
    void set_nb_neighbors(int level_no, int n);

    int nb_neighbors(int layer_no) const;
    int cum_nb_neighbors(int layer_no) const;
    void neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end) const;

    ACORN() : entry_point(-1), rng(12345),
              gamma(0), M(0), M_beta(0), max_level(-1),
              efConstruction(0), efSearch(16),
              check_relative_distance(true), upper_beam(1),
              search_bounded_queue(true), metadata(nullptr) {
        offsets.push_back(0);
    }

    explicit ACORN(int M, int gamma, std::vector<int>& metadata, int M_beta);

    int random_level();
    void fill_with_random_links(size_t n);

    void add_links_starting_from(
            DistanceComputer& ptdis,
            storage_idx_t pt_id,
            storage_idx_t nearest,
            float d_nearest,
            int level,
            omp_lock_t* locks,
            VisitedTable& vt,
            std::vector<storage_idx_t> ep_per_level = {});

    void add_with_locks(
            DistanceComputer& ptdis,
            int pt_level,
            int pt_id,
            std::vector<omp_lock_t>& locks,
            VisitedTable& vt);

    /// standard search (no filter)
    ACORNStats search(
            DistanceComputer& qdis,
            int k,
            idx_t* I,
            float* D,
            VisitedTable& vt,
            const SearchParametersACORN* params = nullptr) const;

    /// parallel iQAN-style search with shared visited list (lock-free)
    ACORNStats parallel_search(
            DistanceComputer& qdis,
            int k,
            idx_t* I,
            float* D,
            VisitedTable& vt,
            int num_threads,
            int efs,
            const SearchParametersACORN* params = nullptr) const;

    /// hybrid search with attribute filter
    ACORNStats hybrid_search(
            DistanceComputer& qdis,
            int k,
            idx_t* I,
            float* D,
            VisitedTable& vt,
            char* filter_map,
            const SearchParametersACORN* params = nullptr) const;

    void reset();
    void clear_neighbor_tables(int level);
    void print_neighbor_stats(int level) const;
    void print_edges(int level) const;
    void print_edges_filtered(int level, int filter, Operation op) const;
    void print_neighbor_stats(
            bool edge_list,
            bool filtered_edge_list = false,
            int filter = -1,
            Operation op = EQUAL) const;

    int prepare_level_tab(size_t n, bool preset_levels = false);

    void save(FILE* fp) const;
    void load(FILE* fp);

    void shrink_neighbor_list(
            DistanceComputer& qdis,
            std::priority_queue<NodeDistFarther>& input,
            std::vector<NodeDistFarther>& output,
            int max_size,
            int gamma = 1,
            storage_idx_t q_id = 0,
            int q_attr = 0);
};

/// Statistics collected during search
struct ACORNStats {
    size_t n1, n2, n3;
    size_t ndis;
    size_t nreorder;

    double candidates_loop;
    double neighbors_loop;
    double tuple_unwrap;
    double skips;
    double visits;

    ACORNStats(
            size_t n1 = 0, size_t n2 = 0, size_t n3 = 0,
            size_t ndis = 0, size_t nreorder = 0,
            double candidates_loop = 0.0, double neighbors_loop = 0.0,
            double tuple_unwrap = 0.0, double skips = 0.0, double visits = 0.0)
            : n1(n1), n2(n2), n3(n3), ndis(ndis), nreorder(nreorder),
              candidates_loop(candidates_loop), neighbors_loop(neighbors_loop),
              tuple_unwrap(tuple_unwrap), skips(skips), visits(visits) {}

    void reset() {
        n1 = n2 = n3 = 0;
        ndis = 0;
        nreorder = 0;
        candidates_loop = 0.0;
        neighbors_loop = 0.0;
        tuple_unwrap = 0.0;
        skips = 0.0;
        visits = 0.0;
    }

    void combine(const ACORNStats& other) {
        n1 += other.n1;
        n2 += other.n2;
        n3 += other.n3;
        ndis += other.ndis;
        nreorder += other.nreorder;
        candidates_loop += other.candidates_loop;
        neighbors_loop += other.neighbors_loop;
        tuple_unwrap += other.tuple_unwrap;
        skips = other.skips;
        visits = other.visits;
    }
};

extern ACORNStats acorn_stats;

} // namespace acorn
