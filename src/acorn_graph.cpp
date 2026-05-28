// -*- c++ -*-
// ACORN graph: construction + NSG-style clean search
#include "acorn/acorn_graph.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <sys/time.h>

namespace acorn {

ACORNStats acorn_stats;

// ============================================================
// Construction: init + level management
// ============================================================

ACORN::ACORN(int M, int gamma, std::vector<int>& meta, int M_beta)
    : rng(12345) {
    set_default_probas(M, 1.0 / log(M), M_beta, gamma);
    max_level = -1; entry_point = -1;
    efSearch = 16; efConstruction = M * gamma; upper_beam = 1;
    this->gamma = gamma;
    this->metadata = meta.data();
    this->M = M; this->M_beta = M_beta;
    offsets.push_back(0);
    for (size_t i = 0; i < assign_probas.size(); i++) nb_per_level.push_back(0);
}

int ACORN::random_level() {
    double f = rng.rand_float();
    for (int level = 0; level < (int)assign_probas.size(); level++) {
        if (f < assign_probas[level]) return level;
        f -= assign_probas[level];
    }
    return (int)assign_probas.size() - 1;
}

void ACORN::set_default_probas(int M_val, float levelMult, int M_beta_val, int gamma_val) {
    int nn = 0;
    cum_nneighbor_per_level.push_back(0);
    for (int level = 0;; level++) {
        float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
        if (proba < 1e-9) break;
        assign_probas.push_back(proba);
        nn += level == 0 ? (int)M_beta_val + (int)(1.5 * M_val) : M_val * gamma_val;
        cum_nneighbor_per_level.push_back(nn);
    }
}

int ACORN::nb_neighbors(int layer_no) const {
    return cum_nneighbor_per_level[layer_no + 1] - cum_nneighbor_per_level[layer_no];
}

int ACORN::cum_nb_neighbors(int layer_no) const { return cum_nneighbor_per_level[layer_no]; }

void ACORN::neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end) const {
    size_t o = offsets[no];
    *begin = o + cum_nb_neighbors(layer_no);
    *end = o + cum_nb_neighbors(layer_no + 1);
}

int ACORN::prepare_level_tab(size_t n, bool preset_levels) {
    size_t n0 = offsets.size() - 1;
    if (preset_levels) {
        assert(n0 + n == levels.size());
    } else {
        assert(n0 == levels.size());
        for (size_t i = 0; i < n; i++) { levels.push_back(random_level() + 1); }
    }
    int max_lvl = 0;
    for (size_t i = 0; i < n; i++) {
        int pt_level = levels[i + n0] - 1;
        if (pt_level > max_lvl) max_lvl = pt_level;
        offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
        neighbors.resize(offsets.back(), -1);
    }
    return max_lvl;
}

void ACORN::reset() {
    max_level = -1; entry_point = -1;
    offsets.clear(); offsets.push_back(0);
    levels.clear(); neighbors.clear();
}

// ============================================================
// Distance helpers
// ============================================================

static inline float compute_dist(const float* query, const float* xb, int d,
                                  int metric, int v) {
    if (metric == 0)  // inner product: negate
        return -fvec_inner_product(query, xb + (size_t)v * d, d);
    else
        return fvec_L2sqr(query, xb + (size_t)v * d, d);
}

static inline void greedy_update(
    const ACORN& hnsw, const float* query, const float* xb, int d, int metric,
    const char* filter_map, int level, int& nearest, float& d_nearest)
{
    for (;;) {
        int prev = nearest;
        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);
        int num = 0;
        for (size_t i = begin; i < end; i++) {
            int v = hnsw.neighbors[i];
            if (v < 0) break;
            if (filter_map && !filter_map[v]) continue;
            num++;
            float dist_v = compute_dist(query, xb, d, metric, v);
            if (dist_v < d_nearest) { d_nearest = dist_v; nearest = v; }
            if (num >= hnsw.M) break;
        }
        if (nearest == prev) return;
    }
}

// ============================================================
// NSG-style search
// ============================================================

int ACORN::search(const float* query, const float* xb, int d, int metric,
                   int k, int efSearch_val,
                   int* indices, float* distances,
                   const char* filter_map) const
{
    if (entry_point == -1) return 0;
    if (entry_point >= (int)(offsets.size() - 1)) return 0;

    int L = std::max(efSearch_val, k);
    int ntotal = (int)(offsets.size() - 1);
    if (ntotal <= 0) return 0;

    std::vector<SearchNeighbor> pool(L + 1);
    int pool_size = 0;
    std::vector<uint8_t> visited(ntotal, 0);
    int vis_mark = 1;

    // Phase 1: greedy descent
    int nearest = entry_point;
    float d_nearest = compute_dist(query, xb, d, metric, nearest);
    for (int lvl = max_level; lvl >= 1; lvl--)
        greedy_update(*this, query, xb, d, metric, filter_map, lvl, nearest, d_nearest);

    // Phase 2: init pool with nearest and its level-0 neighbors
    visited[nearest] = vis_mark;
    if (!filter_map || filter_map[nearest])
        InsertIntoPool(pool.data(), pool_size, L, SearchNeighbor(nearest, d_nearest, true));

    size_t begin, end;
    neighbor_range(nearest, 0, &begin, &end);
    for (size_t j = begin; j < end; j++) {
        int v = neighbors[j];
        if (v < 0) break;
        if (filter_map && !filter_map[v]) continue;
        if (visited[v] == vis_mark) continue;
        visited[v] = vis_mark;
        float dv = compute_dist(query, xb, d, metric, v);
        InsertIntoPool(pool.data(), pool_size, L, SearchNeighbor(v, dv, true));
    }

    // Backtrack search loop
    int cur = 0;
    while (cur < pool_size) {
        auto& cur_nb = pool[cur];
        if (cur_nb.expanded) {
            cur_nb.expanded = false;
            neighbor_range(cur_nb.id, 0, &begin, &end);
            for (size_t j = begin; j < end; j++) {
                int v = neighbors[j];
                if (v < 0) break;
                if (filter_map && !filter_map[v]) continue;
                if (visited[v] == vis_mark) continue;
                visited[v] = vis_mark;
                float dv = compute_dist(query, xb, d, metric, v);
                if (pool_size == L && dv >= pool[L - 1].distance) continue;
                int r = InsertIntoPool(pool.data(), pool_size, L,
                                       SearchNeighbor(v, dv, true));
                if (r < cur) cur = r;
            }
        }
        cur++;
    }

    int out_n = std::min(k, pool_size);
    for (int i = 0; i < out_n; i++) {
        indices[i] = pool[i].id;
        distances[i] = pool[i].distance;
    }
    return out_n;
}

// ============================================================
// Parallel search (iQAN-style, NSG clean)
// ============================================================

int ACORN::parallel_search(const float* query, const float* xb, int d, int metric,
                            int k, int efSearch_val,
                            int* indices, float* distances,
                            int num_threads, int efs,
                            const char* filter_map) const
{
    if (entry_point == -1) return 0;
    if (num_threads <= 1)
        return search(query, xb, d, metric, k, efSearch_val, indices, distances, filter_map);

    int L = std::max(efSearch_val, k);
    int ntotal = (int)(offsets.size() - 1);
    std::vector<SearchNeighbor> shared_pool(L + 1);
    int shared_size = 0;
    std::vector<uint8_t> visited(ntotal, 0);
    int vis_mark = 1;

    auto comp_dist = [&](int v) { return compute_dist(query, xb, d, metric, v); };

    // Phase 1: greedy descent
    int nearest = entry_point;
    float d_nearest = comp_dist(nearest);
    for (int lvl = max_level; lvl >= 1; lvl--)
        greedy_update(*this, query, xb, d, metric, filter_map, lvl, nearest, d_nearest);

    // Phase 2: init
    visited[nearest] = vis_mark;
    if (!filter_map || filter_map[nearest])
        InsertIntoPool(shared_pool.data(), shared_size, L,
                       SearchNeighbor(nearest, d_nearest, false));

    std::vector<std::pair<float, int>> batch;
    size_t begin, end;
    neighbor_range(nearest, 0, &begin, &end);
    for (size_t j = begin; j < end; j++) {
        int v = neighbors[j];
        if (v < 0) break;
        if (filter_map && !filter_map[v]) continue;
        if (visited[v] == vis_mark) continue;
        visited[v] = vis_mark;
        float dv = comp_dist(v);
        batch.emplace_back(dv, v);
        InsertIntoPool(shared_pool.data(), shared_size, L,
                       SearchNeighbor(v, dv, false));
    }

    // Phase 3: parallel rounds
    while (!batch.empty()) {
        int to_process = std::min(num_threads * efs, (int)batch.size());
        int per_thread = (to_process + num_threads - 1) / num_threads;
        std::vector<std::vector<std::pair<float, int>>> thread_unexpanded(num_threads);

#pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            std::vector<SearchNeighbor> local_pool(efs + 1);
            int local_size = 0;
            int start = tid * per_thread;
            int end = std::min(start + per_thread, to_process);
            for (int i = start; i < end; i++) {
                auto& p = batch[i];
                if (filter_map && !filter_map[p.second]) continue;
                InsertIntoPool(local_pool.data(), local_size, efs,
                               SearchNeighbor(p.second, p.first, true));
            }

            int cur = 0, step = 0;
            while (cur < local_size && step < efs) {
                auto& cn = local_pool[cur];
                if (cn.expanded) {
                    cn.expanded = false; step++;
                    size_t b, e;
                    neighbor_range(cn.id, 0, &b, &e);
                    for (size_t j = b; j < e; j++) {
                        int v = neighbors[j];
                        if (v < 0) break;
                        if (filter_map && !filter_map[v]) continue;
                        if (visited[v] == vis_mark) continue;
                        visited[v] = vis_mark;
                        float dv = comp_dist(v);
                        if (local_size == efs && dv >= local_pool[efs - 1].distance) continue;
                        int r = InsertIntoPool(local_pool.data(), local_size, efs,
                                               SearchNeighbor(v, dv, true));
                        if (r < cur) cur = r;
                    }
                }
                cur++;
            }

            thread_unexpanded[tid].clear();
            for (int i = 0; i < local_size; i++)
                if (local_pool[i].expanded)
                    thread_unexpanded[tid].emplace_back(local_pool[i].distance, local_pool[i].id);

#pragma omp critical
            for (int i = 0; i < local_size; i++)
                InsertIntoPool(shared_pool.data(), shared_size, L,
                               SearchNeighbor(local_pool[i].id, local_pool[i].distance, false));
        }

        batch.erase(batch.begin(), batch.begin() + to_process);
        std::vector<std::pair<float, int>> all_unexpanded;
        for (int t = 0; t < num_threads; t++)
            for (auto& p : thread_unexpanded[t])
                if (!filter_map || filter_map[p.second])
                    all_unexpanded.push_back(p);

        if (!all_unexpanded.empty()) {
            if ((int)all_unexpanded.size() > efs) {
                std::nth_element(all_unexpanded.begin(),
                                 all_unexpanded.begin() + efs, all_unexpanded.end());
                all_unexpanded.resize(efs);
            }
            for (auto& p : all_unexpanded) batch.push_back(p);
        }

        if (shared_size >= k && !batch.empty()) {
            float min_d = batch[0].first;
            for (auto& p : batch) { if (p.first < min_d) min_d = p.first; }
            if (min_d > shared_pool[shared_size - 1].distance) batch.clear();
        }
    }

    int out_n = std::min(k, shared_size);
    for (int i = 0; i < out_n; i++) {
        indices[i] = shared_pool[i].id;
        distances[i] = shared_pool[i].distance;
    }
    return out_n;
}

// ============================================================
// Save / Load
// ============================================================

void ACORN::save(FILE* fp) const {
    fwrite(&M, sizeof(int), 1, fp);
    fwrite(&gamma, sizeof(int), 1, fp);
    fwrite(&M_beta, sizeof(int), 1, fp);
    fwrite(&max_level, sizeof(int), 1, fp);
    fwrite(&entry_point, sizeof(storage_idx_t), 1, fp);
    fwrite(&efConstruction, sizeof(int), 1, fp);
    size_t sz;
    sz = assign_probas.size(); fwrite(&sz, sizeof(size_t), 1, fp);
    fwrite(assign_probas.data(), sizeof(double), sz, fp);
    sz = cum_nneighbor_per_level.size(); fwrite(&sz, sizeof(size_t), 1, fp);
    fwrite(cum_nneighbor_per_level.data(), sizeof(int), sz, fp);
    sz = levels.size(); fwrite(&sz, sizeof(size_t), 1, fp);
    fwrite(levels.data(), sizeof(int), sz, fp);
    sz = nb_per_level.size(); fwrite(&sz, sizeof(size_t), 1, fp);
    fwrite(nb_per_level.data(), sizeof(storage_idx_t), sz, fp);
    sz = offsets.size(); fwrite(&sz, sizeof(size_t), 1, fp);
    fwrite(offsets.data(), sizeof(size_t), sz, fp);
    sz = neighbors.size(); fwrite(&sz, sizeof(size_t), 1, fp);
    fwrite(neighbors.data(), sizeof(storage_idx_t), sz, fp);
}

void ACORN::load(FILE* fp) {
    fread(&M, sizeof(int), 1, fp);
    fread(&gamma, sizeof(int), 1, fp);
    fread(&M_beta, sizeof(int), 1, fp);
    fread(&max_level, sizeof(int), 1, fp);
    fread(&entry_point, sizeof(storage_idx_t), 1, fp);
    fread(&efConstruction, sizeof(int), 1, fp);
    size_t sz;
    fread(&sz, sizeof(size_t), 1, fp); assign_probas.resize(sz);
    fread(assign_probas.data(), sizeof(double), sz, fp);
    fread(&sz, sizeof(size_t), 1, fp); cum_nneighbor_per_level.resize(sz);
    fread(cum_nneighbor_per_level.data(), sizeof(int), sz, fp);
    fread(&sz, sizeof(size_t), 1, fp); levels.resize(sz);
    fread(levels.data(), sizeof(int), sz, fp);
    fread(&sz, sizeof(size_t), 1, fp); nb_per_level.resize(sz);
    fread(nb_per_level.data(), sizeof(storage_idx_t), sz, fp);
    fread(&sz, sizeof(size_t), 1, fp); offsets.resize(sz);
    fread(offsets.data(), sizeof(size_t), sz, fp);
    fread(&sz, sizeof(size_t), 1, fp); neighbors.resize(sz);
    fread(neighbors.data(), sizeof(storage_idx_t), sz, fp);
}

void ACORN::print_neighbor_stats(int level) const {
    printf("* stats on level %d, max neighbors: %d\n", level, nb_neighbors(level));
}

void ACORN::print_neighbor_stats(bool, bool, int, Operation) const {
    printf("========= ACORN Stats =======\n");
    printf("  entry_point=%d, max_level=%d, gamma=%d, M=%d, M_beta=%d\n",
           entry_point, max_level, gamma, M, M_beta);
}

} // namespace acorn
