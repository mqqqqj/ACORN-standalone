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
    if (!filter_map) {
        // non-filter: simple greedy descent, check up to M neighbors
        for (;;) {
            int prev = nearest;
            size_t begin, end;
            hnsw.neighbor_range(nearest, level, &begin, &end);
            int numIters = 0;
            for (size_t i = begin; i < end; i++) {
                int v = hnsw.neighbors[i];
                if (v < 0) break;
                numIters++;
                if (numIters > hnsw.M) break;
                float dist_v = compute_dist(query, xb, d, metric, v);
                if (dist_v < d_nearest) { d_nearest = dist_v; nearest = v; }
            }
            if (nearest == prev) return;
        }
    }

    // filter search: gamma-aware greedy descent with 2-hop expansion for gamma==1
    for (;;) {
        int num_found = 0;
        int prev = nearest;
        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);

        for (size_t i = begin; i < end; i++) {
            int v = hnsw.neighbors[i];
            if (v < 0) break;

            if (filter_map[v]) {
                num_found++;
            } else {
                if (hnsw.gamma > 1) continue;
            }

            if (filter_map[v]) {
                float dist_v = compute_dist(query, xb, d, metric, v);
                if (dist_v < d_nearest || !filter_map[nearest]) {
                    nearest = v;
                    d_nearest = dist_v;
                }
                if (num_found >= hnsw.M) break;
            }

            // gamma==1: expand 2-hop to compensate for sparse graph
            if (hnsw.gamma == 1) {
                size_t b2, e2;
                hnsw.neighbor_range(v, level, &b2, &e2);
                for (size_t j = b2; j < e2; j++) {
                    int v2 = hnsw.neighbors[j];
                    if (v2 < 0) break;

                    if (filter_map[v2]) {
                        num_found++;
                        float dist_v2 = compute_dist(query, xb, d, metric, v2);
                        if (dist_v2 < d_nearest || !filter_map[nearest]) {
                            nearest = v2;
                            d_nearest = dist_v2;
                        }
                        if (num_found >= hnsw.M) break;
                    }
                }
            }
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
    std::vector<bool> visited(ntotal, false);

    // Phase 1: greedy descent
    int nearest = entry_point;
    float d_nearest = compute_dist(query, xb, d, metric, nearest);
    for (int lvl = max_level; lvl >= 1; lvl--)
        greedy_update(*this, query, xb, d, metric, filter_map, lvl, nearest, d_nearest);

    // Phase 2: init pool with nearest and its level-0 neighbors
    visited[nearest] = true;
    if (!filter_map || filter_map[nearest])
        InsertIntoPool(pool.data(), pool_size, L, SearchNeighbor(nearest, d_nearest, true));

    size_t begin, end;
    neighbor_range(nearest, 0, &begin, &end);
    for (size_t j = begin; j < end; j++) {
        int v = neighbors[j];
        if (v < 0) break;
        if (filter_map && !filter_map[v]) continue;
        if (visited[v]) continue;
        visited[v] = true;
        float dv = compute_dist(query, xb, d, metric, v);
        InsertIntoPool(pool.data(), pool_size, L, SearchNeighbor(v, dv, true));
    }

    // Backtrack search loop (NSG style: track earliest insertion for backtrack)
    int cur = 0;
    while (cur < pool_size) {
        int next_cur = pool_size;  // default: no backtrack
        auto& cur_nb = pool[cur];
        if (cur_nb.expanded) {
            cur_nb.expanded = false;
            neighbor_range(cur_nb.id, 0, &begin, &end);

            if (!filter_map) {
                for (size_t j = begin; j < end; j++) {
                    int v = neighbors[j];
                    if (v < 0) break;
                    if (visited[v]) continue;
                    visited[v] = true;
                    float dv = compute_dist(query, xb, d, metric, v);
                    if (pool_size == L && dv >= pool[L - 1].distance) continue;
                    int r = InsertIntoPool(pool.data(), pool_size, L,
                                           SearchNeighbor(v, dv, true));
                    if (r < next_cur) next_cur = r;
                }
            } else {
                int num_found = 0;
                bool keep_expanding = true;
                int neighbor_idx = 0;

                for (size_t j = begin; j < end; j++, neighbor_idx++) {
                    int v1 = neighbors[j];
                    if (v1 < 0) break;

                    if (filter_map[v1]) num_found++;

                    if (!visited[v1] && filter_map[v1]) {
                        visited[v1] = true;
                        float dv = compute_dist(query, xb, d, metric, v1);
                        if (pool_size < L || dv < pool[L - 1].distance) {
                            int r = InsertIntoPool(pool.data(), pool_size, L,
                                                   SearchNeighbor(v1, dv, true));
                            if (r < next_cur) next_cur = r;
                        }
                        if (num_found >= M * 2) { keep_expanding = false; break; }
                    }

                    if ((neighbor_idx >= M_beta && keep_expanding) || gamma == 1) {
                        size_t b2, e2;
                        neighbor_range(v1, 0, &b2, &e2);
                        for (size_t j2 = b2; j2 < e2; j2++) {
                            int v2 = neighbors[j2];
                            if (v2 < 0) break;

                            if (filter_map[v2]) num_found++;
                            else continue;

                            if (visited[v2]) continue;
                            visited[v2] = true;
                            float d2 = compute_dist(query, xb, d, metric, v2);
                            if (pool_size < L || d2 < pool[L - 1].distance) {
                                int r = InsertIntoPool(pool.data(), pool_size, L,
                                                       SearchNeighbor(v2, d2, true));
                                if (r < next_cur) next_cur = r;
                            }
                            if (num_found >= M * 2) { keep_expanding = false; break; }
                        }
                    }
                }
            }
        }
        if (next_cur <= cur) cur = next_cur;
        else cur++;
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
    std::vector<bool> visited(ntotal, false);

    auto comp_dist = [&](int v) { return compute_dist(query, xb, d, metric, v); };

    // Phase 1: greedy descent
    int nearest = entry_point;
    float d_nearest = comp_dist(nearest);
    for (int lvl = max_level; lvl >= 1; lvl--)
        greedy_update(*this, query, xb, d, metric, filter_map, lvl, nearest, d_nearest);

    // Phase 2: init
    visited[nearest] = true;
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
        if (visited[v]) continue;
        visited[v] = true;
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
                int next_cur = local_size;
                auto& cn = local_pool[cur];
                if (cn.expanded) {
                    cn.expanded = false; step++;
                    size_t b, e;
                    neighbor_range(cn.id, 0, &b, &e);

                    if (!filter_map) {
                        for (size_t j = b; j < e; j++) {
                            int v = neighbors[j];
                            if (v < 0) break;
                            if (visited[v] ) continue;
                            visited[v] = true;
                            float dv = comp_dist(v);
                            if (local_size == efs && dv >= local_pool[efs - 1].distance) continue;
                            int r = InsertIntoPool(local_pool.data(), local_size, efs,
                                                   SearchNeighbor(v, dv, true));
                            if (r < next_cur) next_cur = r;
                        }
                    } else {
                        int num_found = 0;
                        bool keep_expanding = true;
                        int neighbor_idx = 0;
                        for (size_t j = b; j < e; j++, neighbor_idx++) {
                            int v1 = neighbors[j];
                            if (v1 < 0) break;
                            if (filter_map[v1]) num_found++;
                            if (!visited[v1] && filter_map[v1]) {
                                visited[v1] = true;
                                float dv = comp_dist(v1);
                                if (local_size < efs || dv < local_pool[efs - 1].distance) {
                                    int r = InsertIntoPool(local_pool.data(), local_size, efs,
                                                           SearchNeighbor(v1, dv, true));
                                    if (r < next_cur) next_cur = r;
                                }
                                if (num_found >= M * 2) { keep_expanding = false; break; }
                            }
                            if ((neighbor_idx >= M_beta && keep_expanding) || gamma == 1) {
                                size_t b2, e2;
                                neighbor_range(v1, 0, &b2, &e2);
                                for (size_t j2 = b2; j2 < e2; j2++) {
                                    int v2 = neighbors[j2];
                                    if (v2 < 0) break;
                                    if (filter_map[v2]) num_found++;
                                    else continue;
                                    if (!visited[v2]) {
                                        visited[v2] = true;
                                        float d2 = comp_dist(v2);
                                        if (local_size < efs || d2 < local_pool[efs - 1].distance) {
                                            int r = InsertIntoPool(local_pool.data(), local_size, efs,
                                                                   SearchNeighbor(v2, d2, true));
                                            if (r < next_cur) next_cur = r;
                                        }
                                    }
                                    if (num_found >= M * 2) { keep_expanding = false; break; }
                                }
                            }
                        }
                    }
                }
                if (next_cur <= cur) cur = next_cur;
                else cur++;
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

// ============================================================
// Construction
// ============================================================

// Build ACORN graph for n new vertices appended to existing n0
void acorn_build(ACORN& acorn, int n0, int n, const float* xb,
                 int d, int metric, bool verbose) {
    int ntotal = n0 + n;
    if (n == 0) return;

    acorn.prepare_level_tab(n, false);
    if (verbose) printf("  max_level = %d\n", acorn.max_level);

    // Init locks
    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++) omp_init_lock(&locks[i]);

    // Bucket-sort new vertices by level
    std::vector<int> hist, order(n);
    for (int i = 0; i < n; i++) {
        int lvl = acorn.levels[i + n0] - 1;
        while (lvl >= (int)hist.size()) hist.push_back(0);
        hist[lvl]++;
    }
    std::vector<int> off(hist.size() + 1, 0);
    for (int i = 0; i < (int)hist.size() - 1; i++)
        off[i + 1] = off[i] + hist[i];
    for (int i = 0; i < n; i++) {
        int lvl = acorn.levels[i + n0] - 1;
        order[off[lvl]++] = i + n0;
    }

    // Precompute efConstruction (same as search ef)
    int ef = acorn.efConstruction;

    // Add from highest to lowest level
    RandomGenerator rng2(789);
    int i1 = n;
    for (int pt_level = (int)hist.size() - 1; pt_level >= 0; pt_level--) {
        int i0 = i1 - hist[pt_level];
        if (verbose) printf("  Adding %d elements at level %d\n", i1 - i0, pt_level);

        // Random permutation
        for (int j = i0; j < i1; j++)
            std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

#pragma omp parallel if (i1 > i0 + 100)
        {
            std::vector<bool> visited(ntotal, false);

#pragma omp for schedule(static)
            for (int i = i0; i < i1; i++) {
                int pt_id = order[i];
                const float* q = xb + (size_t)pt_id * d;
                visited.assign(ntotal, false);  // fresh visited per vertex

                // Get entry point (critical section for first vertex)
                int nearest;
#pragma omp critical
                {
                    nearest = acorn.entry_point;
                    if (nearest == -1) {
                        acorn.max_level = pt_level;
                        acorn.entry_point = pt_id;
                        for (int l = 0; l <= acorn.max_level; l++)
                            acorn.nb_per_level[l]++;
                    }
                }
                if (nearest < 0) continue;

                omp_set_lock(&locks[pt_id]);

                // Greedy descent on upper levels
                int level = acorn.max_level;
                float d_nearest = compute_dist(q, xb, d, metric, nearest);
                std::vector<int> ep_per_level(acorn.max_level + 1);
                ep_per_level[level] = nearest;

                for (; level > pt_level; level--) {
                    // Greedy update at upper level
                    for (;;) {
                        int prev = nearest;
                        size_t b, e;
                        acorn.neighbor_range(nearest, level, &b, &e);
                        int num = 0;
                        for (size_t j = b; j < e; j++) {
                            int v = acorn.neighbors[j];
                            if (v < 0) break;
                            num++;
                            float dv = compute_dist(q, xb, d, metric, v);
                            if (dv < d_nearest) { d_nearest = dv; nearest = v; }
                            if (num >= acorn.M) break;
                        }
                        if (nearest == prev) break;
                    }
                    ep_per_level[level] = nearest;
                }

                // Add links at each level from pt_level down to 0
                for (; level >= 0; level--) {
                    // Search for neighbor candidates at this level
                    // level 0: wider search (2*M*gamma) for richer bridge edge candidates
                    int pool_cap = (level == 0) ? (2 * acorn.M * acorn.gamma) : ef;
                    std::vector<SearchNeighbor> pool(pool_cap + 1);
                    int pool_sz = 0;

                    // Initialize pool with nearest (ep_per_level for level > pt_level)
                    int ep = (level > pt_level) ? ep_per_level[level] : nearest;
                    float ep_d = compute_dist(q, xb, d, metric, ep);
                    visited[ep] = true;
                    InsertIntoPool(pool.data(), pool_sz, pool_cap,
                                   SearchNeighbor(ep, ep_d, true));

                    // BFiS at this level
                    int cur = 0;
                    while (cur < pool_sz) {
                        int next_cur = pool_sz;
                        auto& cn = pool[cur];
                        if (cn.expanded) {
                            cn.expanded = false;
                            size_t b, e;
                            acorn.neighbor_range(cn.id, level, &b, &e);
                            int num_iters = 0;
                            for (size_t j = b; j < e; j++) {
                                int v = acorn.neighbors[j];
                                if (v < 0) break;
                                if (visited[v] ) continue;
                                visited[v] = true;
                                float dv = compute_dist(q, xb, d, metric, v);
                                num_iters++;
                                if (pool_sz == pool_cap && dv >= pool[pool_cap - 1].distance)
                                    continue;
                                int r = InsertIntoPool(pool.data(), pool_sz, pool_cap,
                                                       SearchNeighbor(v, dv, true));
                                if (r < next_cur) next_cur = r;
                                if (num_iters > acorn.M) break;
                            }
                        }
                        cur++;
                    }

                    // visited is cleared at top of loop via visited.assign()

                    // Convert pool to sorted (dist, id) list
                    std::vector<std::pair<float, int>> candidates;
                    for (int kk = 0; kk < pool_sz; kk++)
                        candidates.emplace_back(pool[kk].distance, pool[kk].id);
                    std::sort(candidates.begin(), candidates.end());

                    // At level 0, apply ACORN pruning
                    int max_neighbors = acorn.nb_neighbors(level);
                    if (level == 0 && (int)candidates.size() > max_neighbors) {
                        std::unordered_set<int> neigh_of_neigh;
                        std::vector<std::pair<float, int>> pruned;
                        int node_num = 0;
                        for (auto& c : candidates) {
                            node_num++;
                            bool good = true;
                            if (node_num > acorn.M_beta && neigh_of_neigh.count(c.second))
                                good = false;
                            if (good) {
                                pruned.push_back(c);
                                if ((int)pruned.size() >= max_neighbors) break;
                                neigh_of_neigh.insert(c.second);
                                if (node_num > acorn.M_beta) {
                                    size_t b, e;
                                    acorn.neighbor_range(c.second, 0, &b, &e);
                                    for (size_t j = b; j < e; j++) {
                                        if (acorn.neighbors[j] < 0) break;
                                        neigh_of_neigh.insert(acorn.neighbors[j]);
                                    }
                                }
                                if ((int)neigh_of_neigh.size() >= max_neighbors) break;
                            }
                        }
                        candidates = std::move(pruned);
                    } else if ((int)candidates.size() > max_neighbors) {
                        candidates.resize(max_neighbors);
                    }

                    // Add bidirectional links
                    std::vector<int> added;
                    for (auto& c : candidates) {
                        int other = c.second;

                        // Add pt_id -> other link
                        {
                            size_t b, e;
                            acorn.neighbor_range(pt_id, level, &b, &e);
                            if (acorn.neighbors[e - 1] == -1) {
                                // Find empty slot
                                size_t pos = e;
                                while (pos > b && acorn.neighbors[pos - 1] == -1) pos--;
                                acorn.neighbors[pos] = other;
                            } else {
                                // Collect existing + new, sort by sym dist
                                std::vector<std::pair<float, int>> nl;
                                float d_sym = compute_dist(xb + (size_t)other * d, xb, d, metric, pt_id);
                                nl.emplace_back(d_sym, other);
                                for (size_t j = b; j < e; j++) {
                                    int vv = acorn.neighbors[j];
                                    float ds = compute_dist(xb + (size_t)vv * d, xb, d, metric, pt_id);
                                    nl.emplace_back(ds, vv);
                                }
                                std::sort(nl.begin(), nl.end());
                                if (level == 0 && (int)nl.size() > max_neighbors) {
                                    std::unordered_set<int> non;
                                    std::vector<std::pair<float, int>> pr;
                                    int nn = 0;
                                    for (auto& p : nl) {
                                        nn++;
                                        bool ok = true;
                                        if (nn > acorn.M_beta && non.count(p.second)) ok = false;
                                        if (ok) {
                                            pr.push_back(p);
                                            if ((int)pr.size() >= max_neighbors) break;
                                            non.insert(p.second);
                                            if (nn > acorn.M_beta) {
                                                size_t b2, e2;
                                                acorn.neighbor_range(p.second, 0, &b2, &e2);
                                                for (size_t j2 = b2; j2 < e2; j2++) {
                                                    if (acorn.neighbors[j2] < 0) break;
                                                    non.insert(acorn.neighbors[j2]);
                                                }
                                            }
                                        }
                                    }
                                    nl = std::move(pr);
                                } else if ((int)nl.size() > max_neighbors) {
                                    nl.resize(max_neighbors);
                                }
                                size_t pos = b;
                                for (auto& p : nl) acorn.neighbors[pos++] = p.second;
                                while (pos < e) acorn.neighbors[pos++] = -1;
                            }
                        }
                        added.push_back(other);
                    }

                    // Add reverse links (other -> pt_id)
                    omp_unset_lock(&locks[pt_id]);
                    for (int other : added) {
                        omp_set_lock(&locks[other]);
                        size_t b, e;
                        acorn.neighbor_range(other, level, &b, &e);
                        if (acorn.neighbors[e - 1] == -1) {
                            size_t pos = e;
                            while (pos > b && acorn.neighbors[pos - 1] == -1) pos--;
                            acorn.neighbors[pos] = pt_id;
                        } else {
                            std::vector<std::pair<float, int>> nl;
                            float d_sym = compute_dist(xb + (size_t)pt_id * d, xb, d, metric, other);
                            nl.emplace_back(d_sym, pt_id);
                            for (size_t j = b; j < e; j++) {
                                int vv = acorn.neighbors[j];
                                float ds = compute_dist(xb + (size_t)vv * d, xb, d, metric, other);
                                nl.emplace_back(ds, vv);
                            }
                            std::sort(nl.begin(), nl.end());
                            if (level == 0 && (int)nl.size() > max_neighbors) {
                                std::unordered_set<int> non;
                                std::vector<std::pair<float, int>> pr;
                                int nn = 0;
                                for (auto& p : nl) {
                                    nn++;
                                    bool ok = true;
                                    if (nn > acorn.M_beta && non.count(p.second)) ok = false;
                                    if (ok) {
                                        pr.push_back(p);
                                        if ((int)pr.size() >= max_neighbors) break;
                                        non.insert(p.second);
                                        if (nn > acorn.M_beta) {
                                            size_t b2, e2;
                                            acorn.neighbor_range(p.second, 0, &b2, &e2);
                                            for (size_t j2 = b2; j2 < e2; j2++) {
                                                if (acorn.neighbors[j2] < 0) break;
                                                non.insert(acorn.neighbors[j2]);
                                            }
                                        }
                                    }
                                }
                                nl = std::move(pr);
                            } else if ((int)nl.size() > max_neighbors) {
                                nl.resize(max_neighbors);
                            }
                            size_t pos = b;
                            for (auto& p : nl) acorn.neighbors[pos++] = p.second;
                            while (pos < e) acorn.neighbors[pos++] = -1;
                        }
                        omp_unset_lock(&locks[other]);
                    }
                    omp_set_lock(&locks[pt_id]);

                    acorn.nb_per_level[level]++;
                }

                omp_unset_lock(&locks[pt_id]);

                // Update max_level / entry_point if needed
                if (pt_level > acorn.max_level) {
                    acorn.max_level = pt_level;
                    acorn.entry_point = pt_id;
                }
            }
        }
        i1 = i0;
    }

    for (int i = 0; i < ntotal; i++) omp_destroy_lock(&locks[i]);
}

} // namespace acorn

