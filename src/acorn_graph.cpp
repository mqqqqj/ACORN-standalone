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

namespace acorn
{

    static inline void checked_fread(void *ptr, size_t size, size_t count, FILE *fp)
    {
        size_t n = fread(ptr, size, count, fp);
        assert(n == count);
    }

    // ============================================================
    // Construction: init + level management
    // ============================================================

    ACORN::ACORN(int M, int gamma, std::vector<int> &meta, int M_beta)
        : rng(12345)
    {
        set_default_probas(M, 1.0 / log(M), M_beta, gamma);
        max_level = -1;
        entry_point = -1;
        efSearch = 16;
        efConstruction = M * gamma;
        this->gamma = gamma;
        this->metadata = meta.data();
        this->M = M;
        this->M_beta = M_beta;
        offsets.push_back(0);
        for (size_t i = 0; i < assign_probas.size(); i++)
            nb_per_level.push_back(0);
    }

    int ACORN::random_level()
    {
        double f = rng.rand_float();
        for (int level = 0; level < (int)assign_probas.size(); level++)
        {
            if (f < assign_probas[level])
                return level;
            f -= assign_probas[level];
        }
        return (int)assign_probas.size() - 1;
    }

    void ACORN::set_default_probas(int M_val, float levelMult, int M_beta_val, int gamma_val)
    {
        int nn = 0;
        cum_nneighbor_per_level.push_back(0);
        for (int level = 0;; level++)
        {
            float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
            if (proba < 1e-9)
                break;
            assign_probas.push_back(proba);
            nn += level == 0 ? (int)M_beta_val + (int)(1.5 * M_val) : M_val * gamma_val;
            cum_nneighbor_per_level.push_back(nn);
        }
    }

    int ACORN::nb_neighbors(int layer_no) const
    {
        return cum_nneighbor_per_level[layer_no + 1] - cum_nneighbor_per_level[layer_no];
    }

    int ACORN::cum_nb_neighbors(int layer_no) const { return cum_nneighbor_per_level[layer_no]; }

    void ACORN::neighbor_range(idx_t no, int layer_no, size_t *begin, size_t *end) const
    {
        size_t o = offsets[no];
        *begin = o + cum_nb_neighbors(layer_no);
        *end = o + cum_nb_neighbors(layer_no + 1);
    }

    int ACORN::prepare_level_tab(size_t n, bool preset_levels)
    {
        size_t n0 = offsets.size() - 1;
        if (preset_levels)
        {
            assert(n0 + n == levels.size());
        }
        else
        {
            assert(n0 == levels.size());
            for (size_t i = 0; i < n; i++)
            {
                levels.push_back(random_level() + 1);
            }
        }
        int max_lvl = 0;
        for (size_t i = 0; i < n; i++)
        {
            int pt_level = levels[i + n0] - 1;
            if (pt_level > max_lvl)
                max_lvl = pt_level;
            offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
            neighbors.resize(offsets.back(), -1);
        }
        return max_lvl;
    }

    ACORN::ACORN(int d, int M, int gamma, std::vector<int> &meta,
                 int M_beta, MetricType metric)
        : rng(12345)
    {
        set_default_probas(M, 1.0 / log(M), M_beta, gamma);
        max_level = -1;
        entry_point = -1;
        efSearch = 16;
        efConstruction = M * gamma;
        this->gamma = gamma;
        this->d = d;
        this->metric_type = metric;
        this->code_size = sizeof(float) * d;
        metadata_storage.assign(meta.begin(), meta.end());
        this->metadata = metadata_storage.data();
        this->M = M;
        this->M_beta = M_beta;
        offsets.push_back(0);
        for (size_t i = 0; i < assign_probas.size(); i++)
            nb_per_level.push_back(0);
    }

    void ACORN::reset()
    {
        max_level = -1;
        entry_point = -1;
        offsets.clear();
        offsets.push_back(0);
        levels.clear();
        neighbors.clear();
        codes.clear();
        ntotal = 0;
    }

    // ============================================================
    // Distance helpers
    // ============================================================

    // Per-thread NDC profiling (accumulated across queries)
    static std::vector<size_t> g_thread_ndis_total;
    static size_t g_ser_ndis = 0;

    void reset_thread_ndis(int nt) { g_thread_ndis_total.assign(nt, 0); }
    const std::vector<size_t> &get_thread_ndis() { return g_thread_ndis_total; }
    void reset_ser_ndis() { g_ser_ndis = 0; }
    size_t get_ser_ndis() { return g_ser_ndis; }

    static inline float compute_dist(const float *query, const float *xb, int d,
                                     int metric, int v)
    {
        g_ser_ndis++;
        if (metric == 0) // inner product: negate
            return -fvec_inner_product(query, xb + (size_t)v * d, d);
        else
            return fvec_L2sqr(query, xb + (size_t)v * d, d);
    }

    static inline void greedy_update(
        const ACORN &hnsw, const float *query, const float *xb, int d, int metric,
        const char *filter_map, int level, int &nearest, float &d_nearest)
    {
        if (!filter_map)
        {
            // non-filter: simple greedy descent, check up to M neighbors
            for (;;)
            {
                int prev = nearest;
                size_t begin, end;
                hnsw.neighbor_range(nearest, level, &begin, &end);
                int numIters = 0;
                for (size_t i = begin; i < end; i++)
                {
                    int v = hnsw.neighbors[i];
                    if (v < 0)
                        break;
                    numIters++;
                    if (numIters > hnsw.M)
                        break;
                    float dist_v = compute_dist(query, xb, d, metric, v);
                    if (dist_v < d_nearest)
                    {
                        d_nearest = dist_v;
                        nearest = v;
                    }
                }
                if (nearest == prev)
                    return;
            }
        }

        // filter search: gamma-aware greedy descent with 2-hop expansion for gamma==1
        for (;;)
        {
            int num_found = 0;
            int prev = nearest;
            size_t begin, end;
            hnsw.neighbor_range(nearest, level, &begin, &end);

            for (size_t i = begin; i < end; i++)
            {
                int v = hnsw.neighbors[i];
                if (v < 0)
                    break;

                if (filter_map[v])
                {
                    num_found++;
                }
                else
                {
                    if (hnsw.gamma > 1)
                        continue;
                }

                if (filter_map[v])
                {
                    float dist_v = compute_dist(query, xb, d, metric, v);
                    if (dist_v < d_nearest || !filter_map[nearest])
                    {
                        nearest = v;
                        d_nearest = dist_v;
                    }
                    if (num_found >= hnsw.M)
                        break;
                }

                // gamma==1: expand 2-hop to compensate for sparse graph
                if (hnsw.gamma == 1)
                {
                    size_t b2, e2;
                    hnsw.neighbor_range(v, level, &b2, &e2);
                    for (size_t j = b2; j < e2; j++)
                    {
                        int v2 = hnsw.neighbors[j];
                        if (v2 < 0)
                            break;

                        if (filter_map[v2])
                        {
                            num_found++;
                            float dist_v2 = compute_dist(query, xb, d, metric, v2);
                            if (dist_v2 < d_nearest || !filter_map[nearest])
                            {
                                nearest = v2;
                                d_nearest = dist_v2;
                            }
                            if (num_found >= hnsw.M)
                                break;
                        }
                    }
                }
            }
            if (nearest == prev)
                return;
        }
    }

    // ============================================================
    // NSG-style search
    // ============================================================

    int ACORN::search(const float *query, const float *xb, int d, int metric,
                      int k, int efSearch_val,
                      int *indices, float *distances,
                      const char *filter_map) const
    {
        if (entry_point == -1)
            return 0;
        if (entry_point >= (int)(offsets.size() - 1))
            return 0;

        int L = std::max(efSearch_val, k);
        int ntotal = (int)(offsets.size() - 1);
        if (ntotal <= 0)
            return 0;

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
        for (size_t j = begin; j < end; j++)
        {
            int v = neighbors[j];
            if (v < 0)
                break;
            if (filter_map && !filter_map[v])
                continue;
            if (visited[v])
                continue;
            visited[v] = true;
            float dv = compute_dist(query, xb, d, metric, v);
            InsertIntoPool(pool.data(), pool_size, L, SearchNeighbor(v, dv, true));
        }

        // Backtrack search loop (NSG style: track earliest insertion for backtrack)
        int cur = 0;
        while (cur < pool_size)
        {
            int next_cur = pool_size; // default: no backtrack
            auto &cur_nb = pool[cur];
            if (cur_nb.expanded)
            {
                cur_nb.expanded = false;
                neighbor_range(cur_nb.id, 0, &begin, &end);

                if (!filter_map)
                {
                    for (size_t j = begin; j < end; j++)
                    {
                        int v = neighbors[j];
                        if (v < 0)
                            break;
                        if (visited[v])
                            continue;
                        visited[v] = true;
                        float dv = compute_dist(query, xb, d, metric, v);
                        if (pool_size == L && dv >= pool[L - 1].distance)
                            continue;
                        int r = InsertIntoPool(pool.data(), pool_size, L,
                                               SearchNeighbor(v, dv, true));
                        if (r < next_cur)
                            next_cur = r;
                    }
                }
                else
                {
                    int num_found = 0;
                    bool keep_expanding = true;
                    int neighbor_idx = 0;

                    for (size_t j = begin; j < end; j++, neighbor_idx++)
                    {
                        int v1 = neighbors[j];
                        if (v1 < 0)
                            break;

                        if (filter_map[v1])
                            num_found++;

                        if (!visited[v1] && filter_map[v1])
                        {
                            visited[v1] = true;
                            float dv = compute_dist(query, xb, d, metric, v1);
                            if (pool_size < L || dv < pool[L - 1].distance)
                            {
                                int r = InsertIntoPool(pool.data(), pool_size, L,
                                                       SearchNeighbor(v1, dv, true));
                                if (r < next_cur)
                                    next_cur = r;
                            }
                            if (num_found >= M * 2)
                            {
                                keep_expanding = false;
                                break;
                            }
                        }

                        if ((neighbor_idx >= M_beta && keep_expanding) || gamma == 1)
                        {
                            size_t b2, e2;
                            neighbor_range(v1, 0, &b2, &e2);
                            for (size_t j2 = b2; j2 < e2; j2++)
                            {
                                int v2 = neighbors[j2];
                                if (v2 < 0)
                                    break;

                                if (filter_map[v2])
                                    num_found++;
                                else
                                    continue;

                                if (visited[v2])
                                    continue;
                                visited[v2] = true;
                                float d2 = compute_dist(query, xb, d, metric, v2);
                                if (pool_size < L || d2 < pool[L - 1].distance)
                                {
                                    int r = InsertIntoPool(pool.data(), pool_size, L,
                                                           SearchNeighbor(v2, d2, true));
                                    if (r < next_cur)
                                        next_cur = r;
                                }
                                if (num_found >= M * 2)
                                {
                                    keep_expanding = false;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            if (next_cur <= cur)
                cur = next_cur;
            else
                cur++;
        }

        int out_n = std::min(k, pool_size);
        for (int i = 0; i < out_n; i++)
        {
            indices[i] = pool[i].id;
            distances[i] = pool[i].distance;
        }
        return out_n;
    }

    // ============================================================
    // iQAN search (sync-and-redistribute, NSG clean)
    // ============================================================

    int ACORN::iqan_search(const float *query, const float *xb, int d, int metric,
                           int k, int efSearch_val,
                           int *indices, float *distances,
                           int num_threads, int efs,
                           const char *filter_map) const
    {
        if (entry_point == -1)
            return 0;
        if (num_threads <= 1)
            return search(query, xb, d, metric, k, efSearch_val, indices, distances, filter_map);

        int L = std::max(efSearch_val, k);
        int ntotal = (int)(offsets.size() - 1);
        std::vector<SearchNeighbor> shared_pool(L + 1);
        int shared_size = 0;
        std::vector<bool> visited(ntotal, false);

        auto comp_dist = [&](int v)
        { return compute_dist(query, xb, d, metric, v); };
        auto now_ms = []()
        { timeval tv; gettimeofday(&tv,0); return tv.tv_sec*1000.0+tv.tv_usec/1000.0; };
        double t_start = now_ms();

        // Phase 1: greedy descent
        int nearest = entry_point;
        float d_nearest = comp_dist(nearest);
        for (int lvl = max_level; lvl >= 1; lvl--)
            greedy_update(*this, query, xb, d, metric, filter_map, lvl, nearest, d_nearest);
        double t_phase1 = now_ms() - t_start;

        // Phase 2: init
        visited[nearest] = true;
        if (!filter_map || filter_map[nearest])
            InsertIntoPool(shared_pool.data(), shared_size, L,
                           SearchNeighbor(nearest, d_nearest, false));

        std::vector<std::pair<float, int>> batch;
        // Add nearest to batch so it gets expanded in round 1
        // (needed when nearest has no filtered direct neighbors at level 0,
        //  but 2-hop expansion via ACORN hybrid logic can reach them)
        if (!filter_map || filter_map[nearest])
            batch.emplace_back(d_nearest, nearest);

        size_t begin, end;
        neighbor_range(nearest, 0, &begin, &end);
        for (size_t j = begin; j < end; j++)
        {
            int v = neighbors[j];
            if (v < 0)
                break;
            if (filter_map && !filter_map[v])
                continue;
            if (visited[v])
                continue;
            visited[v] = true;
            float dv = comp_dist(v);
            batch.emplace_back(dv, v);
            InsertIntoPool(shared_pool.data(), shared_size, L,
                           SearchNeighbor(v, dv, false));
        }
        double t_phase2 = now_ms() - t_start;

        // Phase 3: parallel rounds
        double t_parallel = 0, t_sync = 0;
        std::vector<size_t> thread_ndis(num_threads, 0);
        while (!batch.empty())
        {
            int to_process = std::min(num_threads * efs, (int)batch.size());
            int per_thread = (to_process + num_threads - 1) / num_threads;
            std::vector<std::vector<std::pair<float, int>>> thread_unexpanded(num_threads);

            double t_round = now_ms();
#pragma omp parallel num_threads(num_threads)
            {
                int tid = omp_get_thread_num();
                size_t ndis_local = 0;
                std::vector<SearchNeighbor> local_pool(efs + 1);
                int local_size = 0;
                int start = tid * per_thread;
                int end = std::min(start + per_thread, to_process);
                for (int i = start; i < end; i++)
                {
                    auto &p = batch[i];
                    if (filter_map && !filter_map[p.second])
                        continue;
                    InsertIntoPool(local_pool.data(), local_size, efs,
                                   SearchNeighbor(p.second, p.first, true));
                }

                int cur = 0, step = 0;
                while (cur < local_size && step < efs)
                {
                    int next_cur = local_size;
                    auto &cn = local_pool[cur];
                    if (cn.expanded)
                    {
                        cn.expanded = false;
                        step++;
                        size_t b, e;
                        neighbor_range(cn.id, 0, &b, &e);

                        if (!filter_map)
                        {
                            for (size_t j = b; j < e; j++)
                            {
                                int v = neighbors[j];
                                if (v < 0)
                                    break;
                                if (visited[v])
                                    continue;
                                visited[v] = true;
                                float dv = comp_dist(v);
                                ndis_local++;
                                if (local_size == efs && dv >= local_pool[efs - 1].distance)
                                    continue;
                                int r = InsertIntoPool(local_pool.data(), local_size, efs,
                                                       SearchNeighbor(v, dv, true));
                                if (r < next_cur)
                                    next_cur = r;
                            }
                        }
                        else
                        {
                            int num_found = 0;
                            bool keep_expanding = true;
                            int neighbor_idx = 0;
                            for (size_t j = b; j < e; j++, neighbor_idx++)
                            {
                                int v1 = neighbors[j];
                                if (v1 < 0)
                                    break;
                                if (filter_map[v1])
                                    num_found++;
                                if (!visited[v1] && filter_map[v1])
                                {
                                    visited[v1] = true;
                                    float dv = comp_dist(v1);
                                    ndis_local++;
                                    if (local_size < efs || dv < local_pool[efs - 1].distance)
                                    {
                                        int r = InsertIntoPool(local_pool.data(), local_size, efs,
                                                               SearchNeighbor(v1, dv, true));
                                        if (r < next_cur)
                                            next_cur = r;
                                    }
                                    if (num_found >= M * 2)
                                    {
                                        keep_expanding = false;
                                        break;
                                    }
                                }
                                if ((neighbor_idx >= M_beta && keep_expanding) || gamma == 1)
                                {
                                    size_t b2, e2;
                                    neighbor_range(v1, 0, &b2, &e2);
                                    for (size_t j2 = b2; j2 < e2; j2++)
                                    {
                                        int v2 = neighbors[j2];
                                        if (v2 < 0)
                                            break;
                                        if (filter_map[v2])
                                            num_found++;
                                        else
                                            continue;
                                        if (!visited[v2])
                                        {
                                            visited[v2] = true;
                                            float d2 = comp_dist(v2);
                                            ndis_local++;
                                            if (local_size < efs || d2 < local_pool[efs - 1].distance)
                                            {
                                                int r = InsertIntoPool(local_pool.data(), local_size, efs,
                                                                       SearchNeighbor(v2, d2, true));
                                                if (r < next_cur)
                                                    next_cur = r;
                                            }
                                        }
                                        if (num_found >= M * 2)
                                        {
                                            keep_expanding = false;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (next_cur <= cur)
                        cur = next_cur;
                    else
                        cur++;
                }

                thread_unexpanded[tid].clear();
                for (int i = 0; i < local_size; i++)
                    if (local_pool[i].expanded)
                        thread_unexpanded[tid].emplace_back(local_pool[i].distance, local_pool[i].id);
                thread_ndis[tid] = ndis_local;

#pragma omp critical
                for (int i = 0; i < local_size; i++)
                    InsertIntoPool(shared_pool.data(), shared_size, L,
                                   SearchNeighbor(local_pool[i].id, local_pool[i].distance, false));
            }
            t_parallel += now_ms() - t_round;

            double t_sync_start = now_ms();
            batch.erase(batch.begin(), batch.begin() + to_process);
            std::vector<std::pair<float, int>> all_unexpanded;
            for (int t = 0; t < num_threads; t++)
                for (auto &p : thread_unexpanded[t])
                    if (!filter_map || filter_map[p.second])
                        all_unexpanded.push_back(p);

            if (!all_unexpanded.empty())
            {
                if ((int)all_unexpanded.size() > efs)
                {
                    std::nth_element(all_unexpanded.begin(),
                                     all_unexpanded.begin() + efs, all_unexpanded.end());
                    all_unexpanded.resize(efs);
                }
                for (auto &p : all_unexpanded)
                    batch.push_back(p);
            }

            if (shared_size >= k && !batch.empty())
            {
                float min_d = batch[0].first;
                for (auto &p : batch)
                {
                    if (p.first < min_d)
                        min_d = p.first;
                }
                if (min_d > shared_pool[shared_size - 1].distance)
                    batch.clear();
            }
            t_sync += now_ms() - t_sync_start;
        }

        // Accumulate per-thread NDC for profiling
        if ((int)g_thread_ndis_total.size() < num_threads)
            g_thread_ndis_total.resize(num_threads, 0);
        for (int t = 0; t < num_threads; t++)
            g_thread_ndis_total[t] += thread_ndis[t];

        int out_n = std::min(k, shared_size);
        for (int i = 0; i < out_n; i++)
        {
            indices[i] = shared_pool[i].id;
            distances[i] = shared_pool[i].distance;
        }
        return out_n;
    }

    // ============================================================
    // No-sync parallel search
    // ============================================================

    int ACORN::no_sync_search(const float *query, const float *xb, int d, int metric,
                              int k, int efSearch_val,
                              int *indices, float *distances,
                              int num_threads,
                              const char *filter_map) const
    {
        if (entry_point == -1)
            return 0;
        if (num_threads <= 1)
            return search(query, xb, d, metric, k, efSearch_val, indices, distances, filter_map);

        int L = std::max(efSearch_val, k);
        int ntotal = (int)(offsets.size() - 1);
        if (ntotal <= 0)
            return 0;

        auto comp_dist = [&](int v)
        { return compute_dist(query, xb, d, metric, v); };

        // Phase 1: greedy descent
        int nearest = entry_point;
        float d_nearest = comp_dist(nearest);
        for (int lvl = max_level; lvl >= 1; lvl--)
            greedy_update(*this, query, xb, d, metric, filter_map, lvl, nearest, d_nearest);

        // Phase 2: collect batch from nearest + its level-0 neighbors
        std::vector<bool> visited(ntotal, false);
        visited[nearest] = true;

        std::vector<std::pair<float, int>> batch;
        if (!filter_map || filter_map[nearest])
            batch.emplace_back(d_nearest, nearest);

        size_t begin, end;
        neighbor_range(nearest, 0, &begin, &end);
        for (size_t j = begin; j < end; j++)
        {
            int v = neighbors[j];
            if (v < 0)
                break;
            if (filter_map && !filter_map[v])
                continue;
            if (visited[v])
                continue;
            visited[v] = true;
            float dv = comp_dist(v);
            batch.emplace_back(dv, v);
        }

        if (batch.empty())
            return 0;

        // Phase 3: partition batch, each thread searches to convergence
        int batch_size = (int)batch.size();
        int per_thread = (batch_size + num_threads - 1) / num_threads;

        std::vector<SearchNeighbor> shared_pool(L + 1);
        int shared_size = 0;
        std::vector<size_t> thread_ndis(num_threads, 0);

#pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            size_t ndis_local = 0;

            std::vector<SearchNeighbor> local_pool(L + 1);
            int local_size = 0;

            int start = tid * per_thread;
            int end = std::min(start + per_thread, batch_size);
            for (int i = start; i < end; i++)
            {
                auto &p = batch[i];
                if (filter_map && !filter_map[p.second])
                    continue;
                InsertIntoPool(local_pool.data(), local_size, L,
                               SearchNeighbor(p.second, p.first, true));
            }

            // Search until queue is fully exhausted
            int cur = 0;
            while (cur < local_size)
            {
                int next_cur = local_size;
                auto &cn = local_pool[cur];
                if (cn.expanded)
                {
                    cn.expanded = false;
                    size_t b, e;
                    neighbor_range(cn.id, 0, &b, &e);

                    if (!filter_map)
                    {
                        for (size_t j = b; j < e; j++)
                        {
                            int v = neighbors[j];
                            if (v < 0)
                                break;
                            if (visited[v])
                                continue;
                            visited[v] = true;
                            float dv = comp_dist(v);
                            ndis_local++;
                            if (local_size == L && dv >= local_pool[L - 1].distance)
                                continue;
                            int r = InsertIntoPool(local_pool.data(), local_size, L,
                                                   SearchNeighbor(v, dv, true));
                            if (r < next_cur)
                                next_cur = r;
                        }
                    }
                    else
                    {
                        int num_found = 0;
                        bool keep_expanding = true;
                        int neighbor_idx = 0;
                        for (size_t j = b; j < e; j++, neighbor_idx++)
                        {
                            int v1 = neighbors[j];
                            if (v1 < 0)
                                break;
                            if (filter_map[v1])
                                num_found++;
                            if (!visited[v1] && filter_map[v1])
                            {
                                visited[v1] = true;
                                float dv = comp_dist(v1);
                                ndis_local++;
                                if (local_size < L || dv < local_pool[L - 1].distance)
                                {
                                    int r = InsertIntoPool(local_pool.data(), local_size, L,
                                                           SearchNeighbor(v1, dv, true));
                                    if (r < next_cur)
                                        next_cur = r;
                                }
                                if (num_found >= M * 2)
                                {
                                    keep_expanding = false;
                                    break;
                                }
                            }
                            if ((neighbor_idx >= M_beta && keep_expanding) || gamma == 1)
                            {
                                size_t b2, e2;
                                neighbor_range(v1, 0, &b2, &e2);
                                for (size_t j2 = b2; j2 < e2; j2++)
                                {
                                    int v2 = neighbors[j2];
                                    if (v2 < 0)
                                        break;
                                    if (filter_map[v2])
                                        num_found++;
                                    else
                                        continue;
                                    if (!visited[v2])
                                    {
                                        visited[v2] = true;
                                        float d2 = comp_dist(v2);
                                        ndis_local++;
                                        if (local_size < L || d2 < local_pool[L - 1].distance)
                                        {
                                            int r = InsertIntoPool(local_pool.data(), local_size, L,
                                                                   SearchNeighbor(v2, d2, true));
                                            if (r < next_cur)
                                                next_cur = r;
                                        }
                                    }
                                    if (num_found >= M * 2)
                                    {
                                        keep_expanding = false;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                if (next_cur <= cur)
                    cur = next_cur;
                else
                    cur++;
            }

            thread_ndis[tid] = ndis_local;

#pragma omp critical
            for (int i = 0; i < local_size; i++)
                InsertIntoPool(shared_pool.data(), shared_size, L,
                               SearchNeighbor(local_pool[i].id, local_pool[i].distance, false));
        }

        if ((int)g_thread_ndis_total.size() < num_threads)
            g_thread_ndis_total.resize(num_threads, 0);
        for (int t = 0; t < num_threads; t++)
            g_thread_ndis_total[t] += thread_ndis[t];

        int out_n = std::min(k, shared_size);
        for (int i = 0; i < out_n; i++)
        {
            indices[i] = shared_pool[i].id;
            distances[i] = shared_pool[i].distance;
        }
        return out_n;
    }

    // ============================================================
    // Save / Load
    // ============================================================

    void ACORN::save(const char *filename) const
    {
        FILE *fp = fopen(filename, "wb");
        assert(fp);
        const char magic[4] = {'A', 'C', 'R', 'N'};
        int version = 1;
        fwrite(magic, 4, 1, fp);
        fwrite(&version, sizeof(int), 1, fp);
        fwrite(&d, sizeof(int), 1, fp);
        fwrite(&ntotal, sizeof(idx_t), 1, fp);
        int mt = (int)metric_type;
        fwrite(&mt, sizeof(int), 1, fp);
        size_t cs = code_size;
        fwrite(&cs, sizeof(size_t), 1, fp);
        fwrite(codes.data(), 1, codes.size(), fp);
        size_t meta_sz = (size_t)ntotal;
        fwrite(&meta_sz, sizeof(size_t), 1, fp);
        fwrite(metadata, sizeof(int), ntotal, fp);

        // Graph data
        fwrite(&M, sizeof(int), 1, fp);
        fwrite(&gamma, sizeof(int), 1, fp);
        fwrite(&M_beta, sizeof(int), 1, fp);
        fwrite(&max_level, sizeof(int), 1, fp);
        fwrite(&entry_point, sizeof(storage_idx_t), 1, fp);
        fwrite(&efConstruction, sizeof(int), 1, fp);
        size_t sz;
        sz = assign_probas.size();
        fwrite(&sz, sizeof(size_t), 1, fp);
        fwrite(assign_probas.data(), sizeof(double), sz, fp);
        sz = cum_nneighbor_per_level.size();
        fwrite(&sz, sizeof(size_t), 1, fp);
        fwrite(cum_nneighbor_per_level.data(), sizeof(int), sz, fp);
        sz = levels.size();
        fwrite(&sz, sizeof(size_t), 1, fp);
        fwrite(levels.data(), sizeof(int), sz, fp);
        sz = nb_per_level.size();
        fwrite(&sz, sizeof(size_t), 1, fp);
        fwrite(nb_per_level.data(), sizeof(storage_idx_t), sz, fp);
        sz = offsets.size();
        fwrite(&sz, sizeof(size_t), 1, fp);
        fwrite(offsets.data(), sizeof(size_t), sz, fp);
        sz = neighbors.size();
        fwrite(&sz, sizeof(size_t), 1, fp);
        fwrite(neighbors.data(), sizeof(storage_idx_t), sz, fp);
        fclose(fp);
    }

    // ============================================================
    // Construction
    // ============================================================

    // Build ACORN graph for n new vertices appended to existing n0
    // ============================================================
    // Construction helpers (anonymous namespace)
    // ============================================================
    namespace
    {

        // ACORN pruning: reduce candidates to max_size using neighbor-of-neighbor criterion.
        // candidates must be sorted by distance ascending (nearest first).
        // First M_beta candidates are always kept; beyond that, a candidate is pruned
        // if it already appears in the neighbor-of-neighbor set of kept candidates.
        void shrink_neighbor_list(
            const ACORN &acorn,
            std::vector<std::pair<float, int>> &candidates,
            int max_size)
        {
            if ((int)candidates.size() <= max_size)
                return;

            std::unordered_set<int> neigh_of_neigh;
            std::vector<std::pair<float, int>> pruned;
            int node_num = 0;

            for (auto &c : candidates)
            {
                node_num++;
                bool good = true;
                if (node_num > acorn.M_beta && neigh_of_neigh.count(c.second))
                    good = false;

                if (good)
                {
                    pruned.push_back(c);
                    if ((int)pruned.size() >= max_size)
                        break;
                    neigh_of_neigh.insert(c.second);
                    if (node_num > acorn.M_beta)
                    {
                        size_t b, e;
                        acorn.neighbor_range(c.second, 0, &b, &e);
                        for (size_t j = b; j < e; j++)
                        {
                            if (acorn.neighbors[j] < 0)
                                break;
                            neigh_of_neigh.insert(acorn.neighbors[j]);
                        }
                    }
                    if ((int)neigh_of_neigh.size() >= max_size)
                        break;
                }
            }
            candidates = std::move(pruned);
        }

        // Insert a single directed edge src->dest at the given level.
        // If the neighbor slot is full, collect all candidates, sort by symmetric
        // distance, prune at level 0, and write back the survivors.
        void add_link(
            ACORN &acorn,
            const float *xb, int d, int metric,
            int src, int dest, int level)
        {
            size_t b, e;
            acorn.neighbor_range(src, level, &b, &e);
            int max_nb = acorn.nb_neighbors(level);

            if (acorn.neighbors[e - 1] == -1)
            {
                // Slot available — find the rightmost empty position
                size_t pos = e;
                while (pos > b && acorn.neighbors[pos - 1] == -1)
                    pos--;
                acorn.neighbors[pos] = dest;
                return;
            }

            // Slot full — collect existing + new, sort by symmetric distance
            std::vector<std::pair<float, int>> nl;
            float d_sym = compute_dist(xb + (size_t)dest * d, xb, d, metric, src);
            nl.emplace_back(d_sym, dest);
            for (size_t j = b; j < e; j++)
            {
                int vv = acorn.neighbors[j];
                float ds = compute_dist(xb + (size_t)vv * d, xb, d, metric, src);
                nl.emplace_back(ds, vv);
            }
            std::sort(nl.begin(), nl.end());

            if (level == 0 && (int)nl.size() > max_nb)
                shrink_neighbor_list(acorn, nl, max_nb);
            else if ((int)nl.size() > max_nb)
                nl.resize(max_nb);

            size_t pos = b;
            for (auto &p : nl)
                acorn.neighbors[pos++] = p.second;
            while (pos < e)
                acorn.neighbors[pos++] = -1;
        }

        // BFS at a single level to find candidate neighbors for a new point.
        // Returns candidates in a sorted pool (ascending distance).
        void search_neighbors_to_add(
            const ACORN &acorn,
            const float *query, const float *xb, int d, int metric,
            int entry_point, float d_entry, int level,
            std::vector<bool> &visited,
            std::vector<SearchNeighbor> &pool, int &pool_sz, int pool_cap)
        {
            visited[entry_point] = true;
            InsertIntoPool(pool.data(), pool_sz, pool_cap,
                           SearchNeighbor(entry_point, d_entry, true));

            int cur = 0;
            while (cur < pool_sz)
            {
                auto &cn = pool[cur];
                if (cn.expanded)
                {
                    cn.expanded = false;
                    size_t b, e;
                    acorn.neighbor_range(cn.id, level, &b, &e);
                    int num_iters = 0;
                    for (size_t j = b; j < e; j++)
                    {
                        int v = acorn.neighbors[j];
                        if (v < 0)
                            break;
                        if (visited[v])
                            continue;
                        visited[v] = true;
                        float dv = compute_dist(query, xb, d, metric, v);
                        num_iters++;
                        if (pool_sz == pool_cap && dv >= pool[pool_cap - 1].distance)
                            continue;
                        int r = InsertIntoPool(pool.data(), pool_sz, pool_cap,
                                               SearchNeighbor(v, dv, true));
                        (void)r;
                        if (num_iters > acorn.M)
                            break;
                    }
                }
                cur++;
            }
        }

        // Add bidirectional links for one point at one level.
        void add_links_starting_from(
            ACORN &acorn,
            const float *query, const float *xb, int d, int metric,
            int pt_id, int nearest, float d_nearest, int level, int ef,
            omp_lock_t *locks, std::vector<bool> &visited)
        {
            int pool_cap = (level == 0) ? (2 * acorn.M * acorn.gamma) : ef;
            std::vector<SearchNeighbor> pool(pool_cap + 1);
            int pool_sz = 0;

            search_neighbors_to_add(acorn, query, xb, d, metric,
                                    nearest, d_nearest, level, visited,
                                    pool, pool_sz, pool_cap);

            // Convert pool to sorted candidate list
            std::vector<std::pair<float, int>> candidates;
            for (int kk = 0; kk < pool_sz; kk++)
                candidates.emplace_back(pool[kk].distance, pool[kk].id);
            std::sort(candidates.begin(), candidates.end());

            // Prune at level 0
            int max_neighbors = acorn.nb_neighbors(level);
            if (level == 0 && (int)candidates.size() > max_neighbors)
                shrink_neighbor_list(acorn, candidates, max_neighbors);
            else if ((int)candidates.size() > max_neighbors)
                candidates.resize(max_neighbors);

            // Add forward links (pt_id -> other)
            std::vector<int> added;
            for (auto &c : candidates)
            {
                add_link(acorn, xb, d, metric, pt_id, c.second, level);
                added.push_back(c.second);
            }

            // Add reverse links (other -> pt_id)
            omp_unset_lock(&locks[pt_id]);
            for (int other : added)
            {
                omp_set_lock(&locks[other]);
                add_link(acorn, xb, d, metric, other, pt_id, level);
                omp_unset_lock(&locks[other]);
            }
            omp_set_lock(&locks[pt_id]);

            acorn.nb_per_level[level]++;
        }

        // Insert one point into the graph at all levels from pt_level down to 0.
        void add_point(
            ACORN &acorn,
            const float *query, const float *xb, int d, int metric,
            int pt_id, int pt_level, int ef,
            std::vector<omp_lock_t> &locks, std::vector<bool> &visited)
        {
            // Get entry point
            int nearest;
#pragma omp critical
            {
                nearest = acorn.entry_point;
                if (nearest == -1)
                {
                    acorn.max_level = pt_level;
                    acorn.entry_point = pt_id;
                    for (int l = 0; l <= acorn.max_level; l++)
                        acorn.nb_per_level[l]++;
                }
            }
            if (nearest < 0)
                return;

            omp_set_lock(&locks[pt_id]);

            // Greedy descent on upper levels
            int level = acorn.max_level;
            float d_nearest = compute_dist(query, xb, d, metric, nearest);
            std::vector<int> ep_per_level(acorn.max_level + 1);
            ep_per_level[level] = nearest;

            for (; level > pt_level; level--)
            {
                for (;;)
                {
                    int prev = nearest;
                    size_t b, e;
                    acorn.neighbor_range(nearest, level, &b, &e);
                    int num = 0;
                    for (size_t j = b; j < e; j++)
                    {
                        int v = acorn.neighbors[j];
                        if (v < 0)
                            break;
                        num++;
                        float dv = compute_dist(query, xb, d, metric, v);
                        if (dv < d_nearest)
                        {
                            d_nearest = dv;
                            nearest = v;
                        }
                        if (num >= acorn.M)
                            break;
                    }
                    if (nearest == prev)
                        break;
                }
                ep_per_level[level] = nearest;
            }

            // Add links at each level from pt_level down to 0
            for (; level >= 0; level--)
            {
                int ep = (level > pt_level) ? ep_per_level[level] : nearest;
                float ep_d = compute_dist(query, xb, d, metric, ep);
                add_links_starting_from(acorn, query, xb, d, metric,
                                        pt_id, ep, ep_d, level, ef,
                                        locks.data(), visited);
            }

            omp_unset_lock(&locks[pt_id]);

            if (pt_level > acorn.max_level)
            {
                acorn.max_level = pt_level;
                acorn.entry_point = pt_id;
            }
        }

    } // anonymous namespace

    // ============================================================
    // Build ACORN graph for new vertices
    // ============================================================

    void ACORN::add(idx_t n, const float *x)
    {
        int n0 = ntotal;
        if (n == 0)
            return;

        codes.resize((n0 + n) * code_size);
        memcpy(codes.data() + n0 * code_size, x, n * code_size);
        ntotal = n0 + n;

        int metric = (metric_type == METRIC_INNER_PRODUCT) ? 0 : 1;
        const float *xb = get_xb();

        prepare_level_tab(n, false);
        if (verbose)
            printf("  max_level = %d\n", max_level);

        // Init locks
        std::vector<omp_lock_t> locks(ntotal);
        for (int i = 0; i < ntotal; i++)
            omp_init_lock(&locks[i]);

        // Bucket-sort new vertices by level (high to low)
        std::vector<int> hist, order(n);
        for (int i = 0; i < n; i++)
        {
            int lvl = levels[i + n0] - 1;
            while (lvl >= (int)hist.size())
                hist.push_back(0);
            hist[lvl]++;
        }
        std::vector<int> off(hist.size() + 1, 0);
        for (int i = 0; i < (int)hist.size() - 1; i++)
            off[i + 1] = off[i] + hist[i];
        for (int i = 0; i < n; i++)
        {
            int lvl = levels[i + n0] - 1;
            order[off[lvl]++] = i + n0;
        }

        int ef = efConstruction;

        // Add from highest to lowest level
        RandomGenerator rng2(789);
        int i1 = n;
        for (int pt_level = (int)hist.size() - 1; pt_level >= 0; pt_level--)
        {
            int i0 = i1 - hist[pt_level];
            if (verbose)
                printf("  Adding %d elements at level %d\n", i1 - i0, pt_level);

            // Random permutation to remove dataset-order bias
            for (int j = i0; j < i1; j++)
                std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

#pragma omp parallel if (i1 > i0 + 100)
            {
                std::vector<bool> visited(ntotal, false);

#pragma omp for schedule(static)
                for (int i = i0; i < i1; i++)
                {
                    int pt_id = order[i];
                    const float *q = xb + (size_t)pt_id * d;
                    visited.assign(ntotal, false);
                    add_point(*this, q, xb, d, metric, pt_id, pt_level, ef,
                              locks, visited);
                }
            }
            i1 = i0;
        }

        for (int i = 0; i < ntotal; i++)
            omp_destroy_lock(&locks[i]);
    }

    void ACORN::load(const char *filename)
    {
        FILE *fp = fopen(filename, "rb");
        assert(fp);
        char magic[4];
        int version;
        checked_fread(magic, 4, 1, fp);
        checked_fread(&version, sizeof(int), 1, fp);
        assert(magic[0] == 'A' && magic[1] == 'C' && magic[2] == 'R' && magic[3] == 'N');
        checked_fread(&d, sizeof(int), 1, fp);
        checked_fread(&ntotal, sizeof(idx_t), 1, fp);
        int mt;
        checked_fread(&mt, sizeof(int), 1, fp);
        metric_type = (MetricType)mt;
        size_t cs;
        checked_fread(&cs, sizeof(size_t), 1, fp);
        code_size = cs;
        codes.resize(ntotal * code_size);
        checked_fread(codes.data(), 1, ntotal * code_size, fp);
        size_t meta_sz;
        checked_fread(&meta_sz, sizeof(size_t), 1, fp);
        metadata_storage.resize(meta_sz);
        checked_fread(metadata_storage.data(), sizeof(int), meta_sz, fp);
        metadata = metadata_storage.data();

        // Graph data
        checked_fread(&M, sizeof(int), 1, fp);
        checked_fread(&gamma, sizeof(int), 1, fp);
        checked_fread(&M_beta, sizeof(int), 1, fp);
        checked_fread(&max_level, sizeof(int), 1, fp);
        checked_fread(&entry_point, sizeof(storage_idx_t), 1, fp);
        checked_fread(&efConstruction, sizeof(int), 1, fp);
        size_t sz;
        checked_fread(&sz, sizeof(size_t), 1, fp);
        assign_probas.resize(sz);
        checked_fread(assign_probas.data(), sizeof(double), sz, fp);
        checked_fread(&sz, sizeof(size_t), 1, fp);
        cum_nneighbor_per_level.resize(sz);
        checked_fread(cum_nneighbor_per_level.data(), sizeof(int), sz, fp);
        checked_fread(&sz, sizeof(size_t), 1, fp);
        levels.resize(sz);
        checked_fread(levels.data(), sizeof(int), sz, fp);
        checked_fread(&sz, sizeof(size_t), 1, fp);
        nb_per_level.resize(sz);
        checked_fread(nb_per_level.data(), sizeof(storage_idx_t), sz, fp);
        checked_fread(&sz, sizeof(size_t), 1, fp);
        offsets.resize(sz);
        checked_fread(offsets.data(), sizeof(size_t), sz, fp);
        checked_fread(&sz, sizeof(size_t), 1, fp);
        neighbors.resize(sz);
        checked_fread(neighbors.data(), sizeof(storage_idx_t), sz, fp);
        fclose(fp);
    }

} // namespace acorn
