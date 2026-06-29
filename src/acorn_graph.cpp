// -*- c++ -*-
// ACORN graph: construction + NSG-style clean search
#include "acorn/acorn_graph.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
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

    // Per-function phase timing (ms, accumulated across queries)
    static PhaseTiming g_nosync_timing, g_scatter_timing;
    void reset_phase_timing()
    {
        g_nosync_timing = PhaseTiming();
        g_scatter_timing = PhaseTiming();
    }
    const PhaseTiming &get_nosync_timing() { return g_nosync_timing; }
    const PhaseTiming &get_scatter_timing() { return g_scatter_timing; }

    static inline float compute_dist(const float *query, const float *xb, int d,
                                     int metric, int v)
    {
        if (metric == 0) // inner product: negate
            return -fvec_inner_product(query, xb + (size_t)v * d, d);
        else
            return fvec_L2sqr(query, xb + (size_t)v * d, d);
    }

    template <typename DistFn>
    static inline void greedy_update(
        const ACORN &hnsw,
        const char *filter_map, int level, int &nearest, float &d_nearest,
        DistFn &&comp_dist)
    {
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
                    float dist_v = comp_dist(v);
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
                            float dist_v2 = comp_dist(v2);
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

    template <typename VisitedT, typename DistFn>
    static inline int expand_level0_filtered(
        const ACORN &hnsw,
        int cur,
        SearchNeighbor *pool,
        int &pool_size,
        int L,
        VisitedT &visited,
        const char *filter_map,
        DistFn &&comp_dist,
        size_t *ndis_local = nullptr)
    {
        if (!pool[cur].expanded)
            return pool_size;

        pool[cur].expanded = false;
        int next_cur = pool_size;
        size_t begin, end;
        hnsw.neighbor_range(pool[cur].id, 0, &begin, &end);

        int num_found = 0;
        bool keep_expanding = true;
        int neighbor_idx = 0;

        for (size_t j = begin; j < end; j++, neighbor_idx++)
        {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0)
                break;

            if (filter_map[v1])
                num_found++;

            if (!visited[v1] && filter_map[v1])
            {
                visited[v1] = true;
                float dv = comp_dist(v1);
                if (ndis_local)
                    (*ndis_local)++;
                if (pool_size < L || dv < pool[L - 1].distance)
                {
                    int r = InsertIntoPool(pool, pool_size, L,
                                           SearchNeighbor(v1, dv, true));
                    if (r < next_cur)
                        next_cur = r;
                }
                if (num_found >= hnsw.M * 2)
                {
                    keep_expanding = false;
                    break;
                }
            }

            if ((neighbor_idx >= hnsw.M_beta && keep_expanding) || hnsw.gamma == 1)
            {
                size_t b2, e2;
                hnsw.neighbor_range(v1, 0, &b2, &e2);
                for (size_t j2 = b2; j2 < e2; j2++)
                {
                    int v2 = hnsw.neighbors[j2];
                    if (v2 < 0)
                        break;

                    if (filter_map[v2])
                        num_found++;
                    else
                        continue;

                    if (visited[v2])
                        continue;
                    visited[v2] = true;
                    float d2 = comp_dist(v2);
                    if (ndis_local)
                        (*ndis_local)++;
                    if (pool_size < L || d2 < pool[L - 1].distance)
                    {
                        int r = InsertIntoPool(pool, pool_size, L,
                                               SearchNeighbor(v2, d2, true));
                        if (r < next_cur)
                            next_cur = r;
                    }
                    if (num_found >= hnsw.M * 2)
                    {
                        keep_expanding = false;
                        break;
                    }
                }
            }
        }

        return next_cur;
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
        if (!filter_map)
            return 0;

        std::vector<SearchNeighbor> pool(L + 1);
        int pool_size = 0;
        std::vector<bool> visited(ntotal, false);
        auto comp_dist = [&](int v)
        {
            g_ser_ndis++;
            return compute_dist(query, xb, d, metric, v);
        };

        // Phase 1: greedy descent
        int nearest = entry_point;
        float d_nearest = comp_dist(nearest);
        for (int lvl = max_level; lvl >= 1; lvl--)
            greedy_update(*this, filter_map, lvl, nearest, d_nearest, comp_dist);

        // Phase 2: init pool with nearest and its level-0 neighbors
        visited[nearest] = true;
        if (filter_map[nearest])
            InsertIntoPool(pool.data(), pool_size, L, SearchNeighbor(nearest, d_nearest, true));

        size_t begin, end;
        neighbor_range(nearest, 0, &begin, &end);
        for (size_t j = begin; j < end; j++)
        {
            int v = neighbors[j];
            if (v < 0)
                break;
            if (!filter_map[v])
                continue;
            if (visited[v])
                continue;
            visited[v] = true;
            float dv = comp_dist(v);
            InsertIntoPool(pool.data(), pool_size, L, SearchNeighbor(v, dv, true));
        }

        int cur = 0;
        while (cur < pool_size)
        {
            int next_cur = expand_level0_filtered(
                *this, cur, pool.data(), pool_size, L, visited, filter_map,
                comp_dist);
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
    // iQAN search (sync-and-redistribute)
    // ============================================================

    int ACORN::iqan_search(const float *query, const float *xb, int d, int metric,
                           int k, int efSearch_val,
                           int *indices, float *distances,
                           int num_threads, const char *filter_map) const
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

        // Phase 1: greedy descent
        int nearest = entry_point;
        float d_nearest = comp_dist(nearest);
        for (int lvl = max_level; lvl >= 1; lvl--)
            greedy_update(*this, filter_map, lvl, nearest, d_nearest, comp_dist);

        // Phase 2: init
        visited[nearest] = true;
        if (filter_map[nearest])
            InsertIntoPool(shared_pool.data(), shared_size, L,
                           SearchNeighbor(nearest, d_nearest, false));

        std::vector<std::pair<float, int>> batch;
        // Add nearest to batch so it gets expanded in round 1
        // (needed when nearest has no filtered direct neighbors at level 0,
        //  but 2-hop expansion via ACORN hybrid logic can reach them)
        if (filter_map[nearest])
            batch.emplace_back(d_nearest, nearest);

        size_t begin, end;
        neighbor_range(nearest, 0, &begin, &end);
        for (size_t j = begin; j < end; j++)
        {
            int v = neighbors[j];
            if (v < 0)
                break;
            if (!filter_map[v])
                continue;
            if (visited[v])
                continue;
            visited[v] = true;
            float dv = comp_dist(v);
            batch.emplace_back(dv, v);
            InsertIntoPool(shared_pool.data(), shared_size, L,
                           SearchNeighbor(v, dv, false));
        }

        // Phase 3: parallel rounds
        std::vector<size_t> thread_ndis(num_threads, 0);
        while (!batch.empty())
        {
            int to_process = std::min(num_threads * L, (int)batch.size());
            std::vector<std::vector<std::pair<float, int>>> thread_unexpanded(num_threads);
            std::vector<std::vector<std::pair<float, int>>> thread_work(num_threads);
            for (int i = 0; i < to_process; i++)
                thread_work[i % num_threads].push_back(batch[i]);

#pragma omp parallel num_threads(num_threads)
            {
                int tid = omp_get_thread_num();
                size_t ndis_local = 0;
                std::vector<SearchNeighbor> local_pool(L + 1);
                int local_size = 0;
                for (auto &p : thread_work[tid])
                {
                    if (!filter_map[p.second])
                        continue;
                    InsertIntoPool(local_pool.data(), local_size, L,
                                   SearchNeighbor(p.second, p.first, true));
                }

                int cur = 0, step = 0;
                while (cur < local_size && step < L)
                {
                    bool was_expanded = local_pool[cur].expanded;
                    int next_cur = expand_level0_filtered(
                        *this, cur, local_pool.data(), local_size, L, visited, filter_map,
                        comp_dist, &ndis_local);
                    if (was_expanded)
                        step++;
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

            batch.erase(batch.begin(), batch.begin() + to_process);
            std::vector<std::pair<float, int>> all_unexpanded;
            for (int t = 0; t < num_threads; t++)
                for (auto &p : thread_unexpanded[t])
                    if (filter_map[p.second])
                        all_unexpanded.push_back(p);

            if (!all_unexpanded.empty())
            {
                if ((int)all_unexpanded.size() > L)
                {
                    std::nth_element(all_unexpanded.begin(),
                                     all_unexpanded.begin() + L, all_unexpanded.end());
                    all_unexpanded.resize(L);
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

        double t0 = omp_get_wtime();

        auto comp_dist = [&](int v)
        { return compute_dist(query, xb, d, metric, v); };

        // Phase 1: greedy descent
        int nearest = entry_point;
        float d_nearest = comp_dist(nearest);
        for (int lvl = max_level; lvl >= 1; lvl--)
            greedy_update(*this, filter_map, lvl, nearest, d_nearest, comp_dist);

        // Phase 2: collect batch from nearest + its level-0 neighbors
        std::vector<bool> visited(ntotal, false);
        visited[nearest] = true;

        std::vector<std::pair<float, int>> batch;
        if (filter_map[nearest])
            batch.emplace_back(d_nearest, nearest);

        size_t begin, end;
        neighbor_range(nearest, 0, &begin, &end);
        for (size_t j = begin; j < end; j++)
        {
            int v = neighbors[j];
            if (v < 0)
                break;
            if (!filter_map[v])
                continue;
            if (visited[v])
                continue;
            visited[v] = true;
            float dv = comp_dist(v);
            batch.emplace_back(dv, v);
        }

        if (batch.empty())
            return 0;

        // Phase 3: round-robin distribute batch, each thread searches to convergence
        std::vector<std::vector<std::pair<float, int>>> thread_batch(num_threads);
        for (size_t i = 0; i < batch.size(); i++)
            thread_batch[i % num_threads].push_back(batch[i]);

        std::vector<SearchNeighbor> shared_pool(L + 1);
        int shared_size = 0;
        std::vector<size_t> thread_ndis(num_threads, 0);

        double t1 = omp_get_wtime();

#pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            size_t ndis_local = 0;

            std::vector<SearchNeighbor> local_pool(L + 1);
            int local_size = 0;

            for (auto &p : thread_batch[tid])
            {
                if (!filter_map[p.second])
                    continue;
                InsertIntoPool(local_pool.data(), local_size, L,
                               SearchNeighbor(p.second, p.first, true));
            }

            // Search until queue is fully exhausted
            int cur = 0;
            while (cur < local_size)
            {
                int next_cur = expand_level0_filtered(
                    *this, cur, local_pool.data(), local_size, L, visited, filter_map,
                    comp_dist, &ndis_local);
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

        double t2 = omp_get_wtime();

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

        double t3 = omp_get_wtime();
        g_nosync_timing.phase1 += (t1 - t0) * 1000.0;
        g_nosync_timing.parallel += (t2 - t1) * 1000.0;
        g_nosync_timing.merge += (t3 - t2) * 1000.0;

        return out_n;
    }

    // ============================================================
    // ScatterSearch (ICDE 2026) — decoupled parallel, leader-guided pruning
    // ============================================================

    int ACORN::scatter_search(const float *query, const float *xb, int d, int metric,
                              int k, int efSearch_val,
                              int *indices, float *distances,
                              int num_threads, int Helec,
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

        double t0 = omp_get_wtime();

        auto comp_dist = [&](int v)
        { return compute_dist(query, xb, d, metric, v); };

        // Phase 1: greedy descent to level 0, then collect filtered entry points
        int nearest = entry_point;
        float d_nearest = comp_dist(nearest);
        for (int lvl = max_level; lvl >= 1; lvl--)
            greedy_update(*this, filter_map, lvl, nearest, d_nearest, comp_dist);

        std::vector<bool> visited(ntotal, false);
        visited[nearest] = true;

        std::vector<std::pair<float, int>> entry_points;
        if (filter_map[nearest])
            entry_points.emplace_back(d_nearest, nearest);

        size_t begin, end;
        neighbor_range(nearest, 0, &begin, &end);
        for (size_t j = begin; j < end; j++)
        {
            int v = neighbors[j];
            if (v < 0)
                break;
            if (!filter_map[v])
                continue;
            if (visited[v])
                continue;
            visited[v] = true;
            entry_points.emplace_back(comp_dist(v), v);
        }
        if (entry_points.empty())
            return 0;

        std::vector<std::vector<std::pair<float, int>>> thread_entry_points(num_threads);

        for (size_t i = 0; i < entry_points.size(); i++)
        {
            int tid = (int)(i % num_threads);
            thread_entry_points[tid].push_back(entry_points[i]);
        }

        std::vector<SearchNeighbor> shared_pool(L + 1);
        std::vector<size_t> thread_ndis(num_threads, 0);
        std::mutex elec_mutex;
        int best_thread_id = -1;
        std::atomic<int> decide_num{0};
        float epsilon = std::numeric_limits<float>::max();
        std::atomic<bool> best_thread_finish{false};
        std::vector<int> good_thread(num_threads, 0);
        int election_hops = Helec;
        int shared_size = 0;
        if (L <= 50)
            election_hops = std::max(1, L - 5);

        double t1 = omp_get_wtime();

#pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            size_t ndis_local = 0;
            std::vector<SearchNeighbor> local_pool(L + 1);
            int local_size = 0;
            int hop = 0;
            int cur = 0;
            bool need_identify = true;

            for (auto &ep : thread_entry_points[tid])
            {
                InsertIntoPool(local_pool.data(), local_size, L,
                               SearchNeighbor(ep.second, ep.first, true));
            }

            while (cur < local_size)
            {
                if (best_thread_finish.load())
                    break;
                if (hop == election_hops)
                {
                    float distk = (local_size >= k) ? local_pool[k - 1].distance
                                                    : std::numeric_limits<float>::max();
                    if (distk < epsilon)
                    {
                        std::lock_guard<std::mutex> lock(elec_mutex);
                        if (distk < epsilon)
                        {
                            epsilon = distk;
                            best_thread_id = tid;
                        }
                    }
                    decide_num.fetch_add(1);
                }

                if (need_identify && decide_num.load() == num_threads)
                {
                    need_identify = false;
                    float best_dist = (local_size > 0) ? local_pool[0].distance
                                                       : std::numeric_limits<float>::max();
                    if (epsilon < best_dist)
                    {
                        good_thread[tid] = -1;
                        break;
                    }
                    if (epsilon > best_dist)
                        good_thread[tid] = 1;
                }
                bool was_expanded = local_pool[cur].expanded;
                int next_cur = expand_level0_filtered(
                    *this, cur, local_pool.data(), local_size, L, visited, filter_map,
                    comp_dist, &ndis_local);
                if (was_expanded)
                    hop++;
                if (next_cur <= cur)
                    cur = next_cur;
                else
                    cur++;
            }

            if (best_thread_id == tid)
                best_thread_finish.store(true);

            thread_ndis[tid] = ndis_local;
#pragma omp critical
            for (int i = 0; i < local_size; i++)
                InsertIntoPool(shared_pool.data(), shared_size, L,
                               SearchNeighbor(local_pool[i].id, local_pool[i].distance, false));
        }

        double t2 = omp_get_wtime();

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

        double t3 = omp_get_wtime();
        g_scatter_timing.phase1 += (t1 - t0) * 1000.0;
        g_scatter_timing.parallel += (t2 - t1) * 1000.0;
        g_scatter_timing.merge += (t3 - t2) * 1000.0;

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

            if (level == 0)
                shrink_neighbor_list(acorn, nl, max_nb);

            size_t pos = b;
            for (auto &p : nl)
                acorn.neighbors[pos++] = p.second;
            while (pos < e)
                acorn.neighbors[pos++] = -1;
        }

        // BFS to find candidate neighbors for a new point, matching original ACORN.
        // Uses two heaps: candidates (max-heap, farthest-first) and results (max-heap, top efConstruction).
        void search_neighbors_to_add(
            ACORN &acorn,
            const float *query, const float *xb, int d, int metric,
            int entry_point, float d_entry, int level,
            std::vector<bool> &visited,
            std::vector<std::pair<float, int>> &results)
        {
            // results: max-heap (farthest at top), limited to efConstruction
            std::vector<std::pair<float, int>> res_heap; // max-heap by distance
            auto res_push = [&](float dist, int id)
            {
                res_heap.emplace_back(dist, id);
                std::push_heap(res_heap.begin(), res_heap.end());
            };
            auto res_pop = [&]()
            {
                std::pop_heap(res_heap.begin(), res_heap.end());
                res_heap.pop_back();
            };
            auto res_top = [&]() -> const std::pair<float, int> &
            { return res_heap.front(); };

            // candidates: min-heap (nearest at top) for BFS frontier
            std::vector<std::pair<float, int>> cand_heap;
            auto cand_cmp = [](const std::pair<float, int> &a,
                               const std::pair<float, int> &b)
            {
                return a.first > b.first;
            };
            auto cand_push = [&](float dist, int id)
            {
                cand_heap.emplace_back(dist, id);
                std::push_heap(cand_heap.begin(), cand_heap.end(), cand_cmp);
            };
            auto cand_top = [&]() -> const std::pair<float, int> &
            { return cand_heap.front(); };
            auto cand_pop = [&]()
            {
                std::pop_heap(cand_heap.begin(), cand_heap.end(), cand_cmp);
                cand_heap.pop_back();
            };

            int efConstruction = acorn.efConstruction;
            cand_push(d_entry, entry_point);
            res_push(d_entry, entry_point);
            visited[entry_point] = true;

            // M target: 2*M*gamma at level 0, nb_neighbors(level) at higher levels
            int M_target;
            if (level == 0)
                M_target = 2 * acorn.M * acorn.gamma;
            else
                M_target = acorn.nb_neighbors(level);

            while (!cand_heap.empty())
            {
                const auto &currEv = cand_top();

                // Greedy break
                if ((currEv.first > res_top().first && acorn.gamma == 1) ||
                    (int)res_heap.size() >= M_target)
                    break;

                int currNode = currEv.second;
                cand_pop();

                size_t b, e;
                acorn.neighbor_range(currNode, level, &b, &e);
                int numIters = 0;

                for (size_t i = b; i < e; i++)
                {
                    int nodeId = acorn.neighbors[i];
                    if (nodeId < 0)
                        break;
                    if (visited[nodeId])
                        continue;
                    visited[nodeId] = true;

                    numIters++;
                    if (numIters > acorn.M)
                        break;

                    float dis = compute_dist(query, xb, d, metric, nodeId);

                    if ((int)res_heap.size() < efConstruction || res_top().first > dis)
                    {
                        res_push(dis, nodeId);
                        cand_push(dis, nodeId);
                        if ((int)res_heap.size() > efConstruction)
                            res_pop();
                    }

                    numIters++;
                    if (numIters > acorn.M)
                        break;
                }
            }

            // Copy results to output (sorted ascending)
            results.clear();
            results.reserve(res_heap.size());
            while (!res_heap.empty())
            {
                results.emplace_back(res_top());
                res_pop();
            }
            std::reverse(results.begin(), results.end());
        }

        // Add bidirectional links for one point at one level.
        void add_links_starting_from(
            ACORN &acorn,
            const float *query, const float *xb, int d, int metric,
            int pt_id, int nearest, float d_nearest, int level, int ef,
            omp_lock_t *locks, std::vector<bool> &visited)
        {
            std::vector<std::pair<float, int>> link_targets;

            search_neighbors_to_add(acorn, query, xb, d, metric,
                                    nearest, d_nearest, level, visited,
                                    link_targets);

            // Update nearest
            if (!link_targets.empty())
                nearest = link_targets[0].second;

            int max_neighbors = acorn.nb_neighbors(level);

            // Shrink only at level 0
            if (level == 0 && (int)link_targets.size() > max_neighbors)
                shrink_neighbor_list(acorn, link_targets, max_neighbors);

            // Add forward links
            std::vector<int> added;
            for (auto &c : link_targets)
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

        int batch_max_level = prepare_level_tab(n, false);
        if (verbose)
            printf("  max_level = %d\n", batch_max_level);

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

    void ACORN::load_from_faiss(const char *filename, const std::vector<int> &labels)
    {
        FILE *fp = fopen(filename, "rb");
        assert(fp);

        // --- Outer IndexACORN ---
        char fourcc[5] = {};
        checked_fread(fourcc, 1, 4, fp);
        assert(strncmp(fourcc, "IHNH", 4) == 0);

        int64_t dummy;
        uint8_t is_trained;
        int mt;
        checked_fread(&d, sizeof(int), 1, fp);
        checked_fread(&ntotal, sizeof(int64_t), 1, fp);
        checked_fread(&dummy, sizeof(int64_t), 1, fp);
        checked_fread(&dummy, sizeof(int64_t), 1, fp);
        checked_fread(&is_trained, 1, 1, fp);
        checked_fread(&mt, sizeof(int), 1, fp);
        metric_type = (mt == 0) ? METRIC_INNER_PRODUCT : METRIC_L2;
        if (mt > 1)
        {
            float ma;
            checked_fread(&ma, sizeof(float), 1, fp);
        }

        // --- ACORN graph ---
        auto read_vec = [&](auto &v)
        {
            size_t sz;
            checked_fread(&sz, sizeof(size_t), 1, fp);
            v.resize(sz);
            if (sz > 0)
                checked_fread(v.data(), sizeof(v[0]), sz, fp);
        };
        read_vec(assign_probas);
        read_vec(cum_nneighbor_per_level);
        read_vec(levels);
        read_vec(offsets);
        read_vec(neighbors);
        read_vec(nb_per_level);

        checked_fread(&entry_point, sizeof(storage_idx_t), 1, fp);
        checked_fread(&max_level, sizeof(int), 1, fp);
        checked_fread(&efConstruction, sizeof(int), 1, fp);
        checked_fread(&efSearch, sizeof(int), 1, fp);
        checked_fread(&dummy, sizeof(int), 1, fp); // upper_beam (unused)
        checked_fread(&gamma, sizeof(int), 1, fp);
        checked_fread(&M, sizeof(int), 1, fp);
        checked_fread(&M_beta, sizeof(int), 1, fp);

        // --- Inner IndexFlat storage ---
        char inner_fourcc[5] = {};
        checked_fread(inner_fourcc, 1, 4, fp);

        // write_index_header for inner IndexFlat
        int inner_d;
        int64_t inner_ntotal;
        uint8_t inner_trained;
        int inner_metric;
        checked_fread(&inner_d, sizeof(int), 1, fp);
        checked_fread(&inner_ntotal, sizeof(int64_t), 1, fp);
        checked_fread(&dummy, sizeof(int64_t), 1, fp);
        checked_fread(&dummy, sizeof(int64_t), 1, fp);
        checked_fread(&inner_trained, 1, 1, fp);
        checked_fread(&inner_metric, sizeof(int), 1, fp);
        if (inner_metric > 1)
        {
            float ma;
            checked_fread(&ma, sizeof(float), 1, fp);
        }

        // codes: WRITEXBVECTOR — size_t num_floats (= ntotal * d), then data
        size_t num_floats;
        checked_fread(&num_floats, sizeof(size_t), 1, fp);
        code_size = (size_t)d * sizeof(float);
        codes.resize(ntotal * code_size);
        checked_fread(codes.data(), 1, codes.size(), fp);
        fclose(fp);

        // Store metadata
        metadata_storage.assign(labels.begin(), labels.end());
        metadata = metadata_storage.data();
    }

} // namespace acorn
