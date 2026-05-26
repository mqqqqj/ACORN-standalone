// -*- c++ -*-
#include "acorn/index_acorn.h"

#include <omp.h>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <unordered_set>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <stdio.h>
#include <iostream>

namespace acorn {

using MinimaxHeap = ACORN::MinimaxHeap;
using storage_idx_t = ACORN::storage_idx_t;
using NodeDistFarther = ACORN::NodeDistFarther;

static double getmillisecs() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/**************************************************************
 * add / search helpers
 **************************************************************/

namespace {

/// Wrap a distance computer to negate distances (for inner product search)
struct NegativeDistanceComputer : DistanceComputer {
    DistanceComputer* basedis;

    explicit NegativeDistanceComputer(DistanceComputer* basedis)
            : basedis(basedis) {}

    void set_query(const float* x) override {
        basedis->set_query(x);
    }

    float operator()(idx_t i) override {
        return -(*basedis)(i);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return -basedis->symmetric_dis(i, j);
    }

    virtual ~NegativeDistanceComputer() {
        delete basedis;
    }
};

DistanceComputer* storage_distance_computer(const Index* storage) {
    if (storage->metric_type == METRIC_INNER_PRODUCT) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

/// Add vertices to the ACORN graph (parallel)
void acorn_add_vertices(
        IndexACORN& index_acorn,
        size_t n0,
        size_t n,
        const float* x,
        bool verbose,
        bool preset_levels = false) {

    size_t d = index_acorn.d;
    ACORN& acorn = index_acorn.acorn;
    size_t ntotal = n0 + n;
    double t0 = getmillisecs();

    if (verbose) {
        printf("acorn_add_vertices: adding %zd elements on top of %zd "
               "(preset_levels=%d)\n", n, n0, int(preset_levels));
    }

    if (n == 0) return;

    int max_level = acorn.prepare_level_tab(n, preset_levels);

    if (verbose) printf("  max_level = %d\n", max_level);

    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < (int)ntotal; i++)
        omp_init_lock(&locks[i]);

    // add vectors from highest to lowest level
    std::vector<int> hist;
    std::vector<int> order(n);

    {
        // build histogram
        for (int i = 0; i < (int)n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = acorn.levels[pt_id] - 1;
            while (pt_level >= (int)hist.size()) hist.push_back(0);
            hist[pt_level]++;
        }

        // accumulate
        std::vector<int> offsets(hist.size() + 1, 0);
        for (int i = 0; i < (int)hist.size() - 1; i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }

        // bucket sort
        for (int i = 0; i < (int)n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = acorn.levels[pt_id] - 1;
            order[offsets[pt_level]++] = pt_id;
        }
    }

    // perform add
    {
        RandomGenerator rng2(789);
        int i1 = n;

        for (int pt_level = (int)hist.size() - 1; pt_level >= 0; pt_level--) {
            int i0 = i1 - hist[pt_level];

            if (verbose) {
                printf("Adding %d elements at level %d\n", i1 - i0, pt_level);
            }

            // random permutation to remove dataset order bias
            for (int j = i0; j < i1; j++)
                std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

#pragma omp parallel if (i1 > i0 + 100)
            {
                VisitedTable vt(ntotal);

                DistanceComputer* dis = storage_distance_computer(index_acorn.storage);
                ScopeDeleter1<DistanceComputer> del(dis);
                int prev_display = verbose && omp_get_thread_num() == 0 ? 0 : -1;

#pragma omp for schedule(static)
                for (int i = i0; i < i1; i++) {
                    storage_idx_t pt_id = order[i];
                    dis->set_query(x + (pt_id - n0) * d);
                    acorn.add_with_locks(*dis, pt_level, pt_id, locks, vt);

                    if (prev_display >= 0 && i - i0 > prev_display + 10000) {
                        prev_display = i - i0;
                        printf("  %d / %d\r", i - i0, i1 - i0);
                        fflush(stdout);
                    }
                }
            }
            i1 = i0;
        }
        ACORN_ASSERT(i1 == 0);
    }

    if (verbose) printf("Done in %.3f ms\n", getmillisecs() - t0);

    for (int i = 0; i < (int)ntotal; i++) {
        omp_destroy_lock(&locks[i]);
    }
}

} // namespace

/**************************************************************
 * IndexACORN implementation
 **************************************************************/

IndexACORN::IndexACORN(int d, int M, int gamma, std::vector<int>& metadata,
                       int M_beta, MetricType metric)
        : Index(d, metric),
          acorn(M, gamma, metadata, M_beta),
          own_fields(false),
          storage(nullptr) {}

IndexACORN::IndexACORN(Index* storage, int M, int gamma,
                       std::vector<int>& metadata, int M_beta)
        : Index(storage->d, storage->metric_type),
          acorn(M, gamma, metadata, M_beta),
          own_fields(false),
          storage(storage) {}

IndexACORN::~IndexACORN() {
    if (own_fields) delete storage;
}

void IndexACORN::train(idx_t n, const float* x) {
    ACORN_THROW_IF_NOT_MSG(storage,
            "Please use IndexACORNFlat (or variants) instead of IndexACORN directly");
    storage->train(n, x);
    is_trained = true;
}

// Hybrid search (with attribute filters)
void IndexACORN::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        char* filter_id_map,
        const SearchParameters* params_in) const {

    ACORN_THROW_IF_NOT(k > 0);
    ACORN_THROW_IF_NOT_MSG(storage,
            "Please use IndexACORNFlat (or variants) instead of IndexACORN directly");

    const SearchParametersACORN* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const SearchParametersACORN*>(params_in);
        ACORN_THROW_IF_NOT_MSG(params, "params type invalid");
    }

    size_t n1 = 0, n2 = 0, n3 = 0, ndis = 0, nreorder = 0;
    double candidates_loop = 0, neighbors_loop = 0, tuple_unwrap = 0,
           skips = 0, visits = 0;

#pragma omp parallel
    {
        VisitedTable vt(ntotal);

        DistanceComputer* dis = storage_distance_computer(storage);
        ScopeDeleter1<DistanceComputer> del(dis);

#pragma omp for reduction(+ : n1, n2, n3, ndis, nreorder, candidates_loop)
        for (idx_t i = 0; i < n; i++) {
            idx_t* idxi = labels + i * k;
            float* simi = distances + i * k;
            char* filters = filter_id_map + i * ntotal;
            dis->set_query(x + i * d);

            maxheap_heapify(k, simi, idxi);
            ACORNStats stats = acorn.hybrid_search(*dis, k, idxi, simi, vt, filters, params);
            n1 += stats.n1;
            n2 += stats.n2;
            n3 += stats.n3;
            ndis += stats.ndis;
            nreorder += stats.nreorder;
            candidates_loop += stats.candidates_loop;
            neighbors_loop += stats.neighbors_loop;
            tuple_unwrap += stats.tuple_unwrap;
            skips += stats.skips;
            visits += stats.visits;
            maxheap_reorder(k, simi, idxi);
        }
    }

    if (metric_type == METRIC_INNER_PRODUCT) {
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }

    acorn_stats.combine({n1, n2, n3, ndis, nreorder,
                         candidates_loop, neighbors_loop, tuple_unwrap, skips, visits});
}

// Standard search (no filter)
void IndexACORN::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {

    ACORN_THROW_IF_NOT(k > 0);
    ACORN_THROW_IF_NOT_MSG(storage,
            "Please use IndexACORNFlat (or variants) instead of IndexACORN directly");

    const SearchParametersACORN* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const SearchParametersACORN*>(params_in);
        ACORN_THROW_IF_NOT_MSG(params, "params type invalid");
    }

    size_t n1 = 0, n2 = 0, n3 = 0, ndis = 0, nreorder = 0;

#pragma omp parallel
    {
        VisitedTable vt(ntotal);

        DistanceComputer* dis = storage_distance_computer(storage);
        ScopeDeleter1<DistanceComputer> del(dis);

#pragma omp for reduction(+ : n1, n2, n3, ndis, nreorder)
        for (idx_t i = 0; i < n; i++) {
            idx_t* idxi = labels + i * k;
            float* simi = distances + i * k;
            dis->set_query(x + i * d);

            maxheap_heapify(k, simi, idxi);
            ACORNStats stats = acorn.search(*dis, k, idxi, simi, vt, params);
            n1 += stats.n1;
            n2 += stats.n2;
            n3 += stats.n3;
            ndis += stats.ndis;
            nreorder += stats.nreorder;
            maxheap_reorder(k, simi, idxi);
        }
    }

    if (metric_type == METRIC_INNER_PRODUCT) {
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }

    acorn_stats.combine({n1, n2, n3, ndis, nreorder});
}

void IndexACORN::add(idx_t n, const float* x) {
    ACORN_THROW_IF_NOT_MSG(storage,
            "Please use IndexACORNFlat (or variants) instead of IndexACORN directly");
    ACORN_THROW_IF_NOT(is_trained);
    int n0 = ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;

    acorn_add_vertices(*this, n0, n, x, verbose, acorn.levels.size() == ntotal);
}

void IndexACORN::reset() {
    acorn.reset();
    storage->reset();
    ntotal = 0;
}

void IndexACORN::reconstruct(idx_t key, float* recons) const {
    storage->reconstruct(key, recons);
}

void IndexACORN::printStats(bool print_edge_list, bool print_filtered_edge_lists,
                             int filter, Operation op) {
    acorn.print_neighbor_stats(print_edge_list, print_filtered_edge_lists, filter, op);
    printf("METADATA VEC for number nodes per level\n");
    for (int i = 0; i < (int)acorn.nb_per_level.size(); i++) {
        printf("\tlevel %d: %d nodes\n", i, acorn.nb_per_level[i]);
    }
}

void IndexACORN::save(const char* filename) const {
    FILE* fp = fopen(filename, "wb");
    ACORN_THROW_IF_NOT_MSG(fp, "cannot open file for writing");

    const char magic[4] = {'A', 'C', 'R', 'N'};
    int version = 1;
    fwrite(magic, 4, 1, fp);
    fwrite(&version, sizeof(int), 1, fp);

    fwrite(&d, sizeof(int), 1, fp);
    fwrite(&ntotal, sizeof(idx_t), 1, fp);
    int mt = (int)metric_type;
    fwrite(&mt, sizeof(int), 1, fp);

    // storage (IndexFlat codes)
    IndexFlat* flat = dynamic_cast<IndexFlat*>(storage);
    size_t code_size = flat->code_size;
    fwrite(&code_size, sizeof(size_t), 1, fp);
    fwrite(flat->codes.data(), 1, flat->codes.size(), fp);

    // metadata
    size_t meta_sz = (size_t)ntotal;
    fwrite(&meta_sz, sizeof(size_t), 1, fp);
    fwrite(acorn.metadata, sizeof(int), ntotal, fp);

    // ACORN graph
    acorn.save(fp);

    fclose(fp);
}

void IndexACORN::load(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    ACORN_THROW_IF_NOT_MSG(fp, "cannot open file for reading");

    char magic[4];
    int version;
    fread(magic, 4, 1, fp);
    fread(&version, sizeof(int), 1, fp);
    ACORN_THROW_IF_NOT_MSG(
        magic[0] == 'A' && magic[1] == 'C' && magic[2] == 'R' && magic[3] == 'N',
        "invalid index file (bad magic)");

    fread(&d, sizeof(int), 1, fp);
    fread(&ntotal, sizeof(idx_t), 1, fp);
    int mt;
    fread(&mt, sizeof(int), 1, fp);
    metric_type = (MetricType)mt;

    // storage
    size_t code_size;
    fread(&code_size, sizeof(size_t), 1, fp);
    IndexFlat* flat = new IndexFlat(d, metric_type);
    flat->code_size = code_size;
    flat->ntotal = ntotal;
    flat->d = d;
    flat->is_trained = true;
    flat->codes.resize(ntotal * code_size);
    fread(flat->codes.data(), 1, ntotal * code_size, fp);
    storage = flat;
    own_fields = true;

    // metadata
    size_t meta_sz;
    fread(&meta_sz, sizeof(size_t), 1, fp);
    metadata_storage.resize(meta_sz);
    fread(metadata_storage.data(), sizeof(int), meta_sz, fp);

    // ACORN graph
    acorn.load(fp);
    acorn.metadata = metadata_storage.data();

    fclose(fp);
    is_trained = true;
}

/**************************************************************
 * IndexACORNFlat implementation
 **************************************************************/

IndexACORNFlat::IndexACORNFlat(int d, int M, int gamma,
                                std::vector<int>& metadata, int M_beta,
                                MetricType metric)
        : IndexACORN(new IndexFlat(d, metric), M, gamma, metadata, M_beta) {
    own_fields = true;
    is_trained = true;
}

} // namespace acorn
