// -*- c++ -*-
// ACORN graph: FAISS graph loading + search
#include "acorn/acorn_graph.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <sys/time.h>

namespace acorn
{

    static inline void checked_fread(void *ptr, size_t size, size_t count, FILE *fp)
    {
        size_t n = fread(ptr, size, count, fp);
        assert(n == count);
    }

    // ============================================================
    // Graph layout helpers
    // ============================================================

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

    // ============================================================
    // Distance helpers
    // ============================================================

    // Per-thread NDC profiling (accumulated across queries)
    static std::vector<size_t> g_thread_ndis_total;
    static size_t g_ser_ndis = 0;
    static int g_filter_check_cost = 0;

    void reset_thread_ndis(int nt) { g_thread_ndis_total.assign(nt, 0); }
    const std::vector<size_t> &get_thread_ndis() { return g_thread_ndis_total; }
    void reset_ser_ndis() { g_ser_ndis = 0; }
    size_t get_ser_ndis() { return g_ser_ndis; }
    void set_filter_check_cost(int cost) { g_filter_check_cost = std::max(0, cost); }
    int get_filter_check_cost() { return g_filter_check_cost; }

    static inline bool check_filter(const char *filter_map, int id)
    {
        volatile int sink = 0;
        for (int i = 0; i < g_filter_check_cost; i++)
            sink += (i ^ id) & 1;
        (void)sink;
        return filter_map[id] != 0;
    }

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

                if (check_filter(filter_map, v))
                {
                    num_found++;
                }
                else
                {
                    if (hnsw.gamma > 1)
                        continue;
                }

                if (check_filter(filter_map, v))
                {
                    float dist_v = comp_dist(v);
                    if (dist_v < d_nearest || !check_filter(filter_map, nearest))
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

                        if (check_filter(filter_map, v2))
                        {
                            num_found++;
                            float dist_v2 = comp_dist(v2);
                            if (dist_v2 < d_nearest || !check_filter(filter_map, nearest))
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

    template <typename DistFn>
    static inline void greedy_update_unfiltered(
        const ACORN &hnsw,
        int level, int &nearest, float &d_nearest,
        DistFn &&comp_dist)
    {
        for (;;)
        {
            int prev = nearest;
            size_t begin, end;
            hnsw.neighbor_range(nearest, level, &begin, &end);

            int num_found = 0;
            for (size_t i = begin; i < end; i++)
            {
                int v = hnsw.neighbors[i];
                if (v < 0)
                    break;

                float dist_v = comp_dist(v);
                if (dist_v < d_nearest)
                {
                    nearest = v;
                    d_nearest = dist_v;
                }
                if (++num_found >= hnsw.M)
                    break;
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
            if (visited[v1])
                continue;
            if (check_filter(filter_map, v1))
            {
                num_found++;
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
                    if (visited[v2])
                        continue;
                    if (check_filter(filter_map, v2))
                        num_found++;
                    else
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

    template <typename VisitedT, typename DistFn>
    static inline int expand_level0_filtered_lazy(
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
            if (visited[v1])
                continue;

            float dv = comp_dist(v1);
            if (ndis_local)
                (*ndis_local)++;
            visited[v1] = true;
            bool can_enter_pool = (pool_size < L || dv < pool[L - 1].distance);
            if (can_enter_pool && check_filter(filter_map, v1))
            {
                num_found++;
                int r = InsertIntoPool(pool, pool_size, L,
                                       SearchNeighbor(v1, dv, true));
                if (r < next_cur)
                    next_cur = r;
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
                    if (visited[v2])
                        continue;

                    float d2 = comp_dist(v2);
                    visited[v2] = true;
                    if (ndis_local)
                        (*ndis_local)++;

                    bool can_enter_pool2 = (pool_size < L || d2 < pool[L - 1].distance);
                    if (!can_enter_pool2 || !check_filter(filter_map, v2))
                        continue;
                    num_found++;
                    int r = InsertIntoPool(pool, pool_size, L,
                                           SearchNeighbor(v2, d2, true));
                    if (r < next_cur)
                        next_cur = r;
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

    template <typename VisitedT, typename DistFn>
    static inline int expand_level0_unfiltered(
        const ACORN &hnsw,
        int cur,
        SearchNeighbor *pool,
        int &pool_size,
        int L,
        VisitedT &visited,
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
        for (size_t j = begin; j < end; j++)
        {
            int v = hnsw.neighbors[j];
            if (v < 0)
                break;
            if (visited[v])
                continue;

            visited[v] = true;
            float dv = comp_dist(v);
            if (ndis_local)
                (*ndis_local)++;
            if (pool_size < L || dv < pool[L - 1].distance)
            {
                int r = InsertIntoPool(pool, pool_size, L,
                                       SearchNeighbor(v, dv, true));
                if (r < next_cur)
                    next_cur = r;
            }
            if (++num_found >= hnsw.M * 2)
                break;
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
        if (check_filter(filter_map, nearest))
            InsertIntoPool(pool.data(), pool_size, L, SearchNeighbor(nearest, d_nearest, true));

        size_t begin, end;
        neighbor_range(nearest, 0, &begin, &end);
        for (size_t j = begin; j < end; j++)
        {
            int v = neighbors[j];
            if (v < 0)
                break;
            if (!check_filter(filter_map, v))
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

    int ACORN::no_filter_search(const float *query, const float *xb, int d, int metric,
                                int k, int efSearch_val,
                                int *indices, float *distances) const
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
        auto comp_dist = [&](int v)
        {
            g_ser_ndis++;
            return compute_dist(query, xb, d, metric, v);
        };

        int nearest = entry_point;
        float d_nearest = comp_dist(nearest);
        for (int lvl = max_level; lvl >= 1; lvl--)
            greedy_update_unfiltered(*this, lvl, nearest, d_nearest, comp_dist);

        visited[nearest] = true;
        InsertIntoPool(pool.data(), pool_size, L, SearchNeighbor(nearest, d_nearest, true));

        size_t begin, end;
        neighbor_range(nearest, 0, &begin, &end);
        for (size_t j = begin; j < end; j++)
        {
            int v = neighbors[j];
            if (v < 0)
                break;
            if (visited[v])
                continue;
            visited[v] = true;
            float dv = comp_dist(v);
            InsertIntoPool(pool.data(), pool_size, L, SearchNeighbor(v, dv, true));
        }

        int cur = 0;
        while (cur < pool_size)
        {
            int next_cur = expand_level0_unfiltered(
                *this, cur, pool.data(), pool_size, L, visited,
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

    int ACORN::pre_filter_search(const float *query, const float *xb, int d, int metric,
                                 int k,
                                 int *indices, float *distances,
                                 const char *filter_map) const
    {
        int ntotal = (int)(offsets.size() - 1);
        if (ntotal <= 0 || k <= 0 || !filter_map)
            return 0;

        std::vector<SearchNeighbor> pool(k + 1);
        int pool_size = 0;
        for (int v = 0; v < ntotal; v++)
        {
            if (!check_filter(filter_map, v))
                continue;
            g_ser_ndis++;
            float dist = compute_dist(query, xb, d, metric, v);
            InsertIntoPool(pool.data(), pool_size, k, SearchNeighbor(v, dist, false));
        }

        int out_n = std::min(k, pool_size);
        for (int i = 0; i < out_n; i++)
        {
            indices[i] = pool[i].id;
            distances[i] = pool[i].distance;
        }
        return out_n;
    }

    int ACORN::parallel_pre_filter_search(const float *query, const float *xb, int d, int metric,
                                          int k,
                                          int *indices, float *distances,
                                          int num_threads,
                                          const char *filter_map) const
    {
        int ntotal = (int)(offsets.size() - 1);
        if (ntotal <= 0 || k <= 0 || !filter_map)
            return 0;
        std::vector<std::vector<SearchNeighbor>> thread_pools(num_threads);
        std::vector<int> thread_sizes(num_threads, 0);
        std::vector<size_t> thread_ndis(num_threads, 0);
        for (int t = 0; t < num_threads; t++)
            thread_pools[t].resize(k + 1);

#pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int begin = (int)(((int64_t)ntotal * tid) / num_threads);
            int end = (int)(((int64_t)ntotal * (tid + 1)) / num_threads);
            int local_size = 0;
            size_t ndis_local = 0;
            SearchNeighbor *local_pool = thread_pools[tid].data();

            for (int v = begin; v < end; v++)
            {
                if (!check_filter(filter_map, v))
                    continue;
                float dist = compute_dist(query, xb, d, metric, v);
                ndis_local++;
                InsertIntoPool(local_pool, local_size, k, SearchNeighbor(v, dist, false));
            }

            thread_sizes[tid] = local_size;
            thread_ndis[tid] = ndis_local;
        }

        std::vector<SearchNeighbor> pool(k + 1);
        int pool_size = 0;
        for (int t = 0; t < num_threads; t++)
        {
            for (int i = 0; i < thread_sizes[t]; i++)
            {
                InsertIntoPool(pool.data(), pool_size, k,
                               SearchNeighbor(thread_pools[t][i].id,
                                              thread_pools[t][i].distance,
                                              false));
            }
        }

        if ((int)g_thread_ndis_total.size() < num_threads)
            g_thread_ndis_total.resize(num_threads, 0);
        for (int t = 0; t < num_threads; t++)
            g_thread_ndis_total[t] += thread_ndis[t];

        int out_n = std::min(k, pool_size);
        for (int i = 0; i < out_n; i++)
        {
            indices[i] = pool[i].id;
            distances[i] = pool[i].distance;
        }
        return out_n;
    }

    int ACORN::post_filter_search(const float *query, const float *xb, int d, int metric,
                                  int k, int efSearch_val,
                                  int *indices, float *distances,
                                  const char *filter_map) const
    {
        int ntotal = (int)(offsets.size() - 1);
        if (entry_point == -1 || ntotal <= 0 || k <= 0 || !filter_map)
            return 0;
        int candidate_k = std::min(ntotal, std::max(k, efSearch_val));
        std::vector<int> candidate_ids(candidate_k, -1);
        std::vector<float> candidate_dist(candidate_k, 0.0f);
        int nres = no_filter_search(query, xb, d, metric, candidate_k, candidate_k,
                                    candidate_ids.data(), candidate_dist.data());

        std::vector<SearchNeighbor> pool(k + 1);
        int pool_size = 0;
        for (int i = 0; i < nres; i++)
        {
            int id = candidate_ids[i];
            if (id < 0 || !check_filter(filter_map, id))
                continue;
            InsertIntoPool(pool.data(), pool_size, k,
                           SearchNeighbor(id, candidate_dist[i], false));
        }

        int out_n = std::min(k, pool_size);
        for (int i = 0; i < out_n; i++)
        {
            indices[i] = pool[i].id;
            distances[i] = pool[i].distance;
        }
        return out_n;
    }

    int ACORN::parallel_post_filter_search(const float *query, const float *xb, int d, int metric,
                                           int k, int efSearch_val,
                                           int *indices, float *distances,
                                           int num_threads, int Helec,
                                           const char *filter_map) const
    {
        int ntotal = (int)(offsets.size() - 1);
        if (entry_point == -1 || ntotal <= 0 || k <= 0 || !filter_map)
            return 0;
        int per_thread_k = std::max(k, efSearch_val);
        int candidate_k = std::min(ntotal, std::max(k, per_thread_k * std::max(1, num_threads)));
        std::vector<int> candidate_ids(candidate_k, -1);
        std::vector<float> candidate_dist(candidate_k, 0.0f);

        int nres = no_filter_scatter_search(query, xb, d, metric, candidate_k, per_thread_k,
                                            candidate_ids.data(), candidate_dist.data(),
                                            num_threads, Helec);

        std::vector<SearchNeighbor> pool(k + 1);
        int pool_size = 0;
        for (int i = 0; i < nres; i++)
        {
            int id = candidate_ids[i];
            if (id < 0 || !check_filter(filter_map, id))
                continue;
            InsertIntoPool(pool.data(), pool_size, k,
                           SearchNeighbor(id, candidate_dist[i], false));
        }

        int out_n = std::min(k, pool_size);
        for (int i = 0; i < out_n; i++)
        {
            indices[i] = pool[i].id;
            distances[i] = pool[i].distance;
        }
        return out_n;
    }

    int ACORN::parallel_post_filter_iqan_search(const float *query, const float *xb, int d, int metric,
                                                int k, int efSearch_val,
                                                int *indices, float *distances,
                                                int num_threads,
                                                const char *filter_map) const
    {
        int ntotal = (int)(offsets.size() - 1);
        if (entry_point == -1 || ntotal <= 0 || k <= 0 || !filter_map)
            return 0;
        int per_thread_k = std::max(k, efSearch_val);
        int candidate_k = std::min(ntotal, std::max(k, per_thread_k * std::max(1, num_threads)));
        std::vector<int> candidate_ids(candidate_k, -1);
        std::vector<float> candidate_dist(candidate_k, 0.0f);

        int nres = no_filter_iqan_search(query, xb, d, metric, candidate_k, per_thread_k,
                                         candidate_ids.data(), candidate_dist.data(),
                                         num_threads);

        std::vector<SearchNeighbor> pool(k + 1);
        int pool_size = 0;
        for (int i = 0; i < nres; i++)
        {
            int id = candidate_ids[i];
            if (id < 0 || !check_filter(filter_map, id))
                continue;
            InsertIntoPool(pool.data(), pool_size, k,
                           SearchNeighbor(id, candidate_dist[i], false));
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
        if (check_filter(filter_map, nearest))
            InsertIntoPool(shared_pool.data(), shared_size, L,
                           SearchNeighbor(nearest, d_nearest, false));

        std::vector<std::pair<float, int>> batch;
        // Add nearest to batch so it gets expanded in round 1
        // (needed when nearest has no filtered direct neighbors at level 0,
        //  but 2-hop expansion via ACORN hybrid logic can reach them)
        if (check_filter(filter_map, nearest))
            batch.emplace_back(d_nearest, nearest);

        size_t begin, end;
        neighbor_range(nearest, 0, &begin, &end);
        for (size_t j = begin; j < end; j++)
        {
            int v = neighbors[j];
            if (v < 0)
                break;
            if (!check_filter(filter_map, v))
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
                    if (!check_filter(filter_map, p.second))
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
                    if (check_filter(filter_map, p.second))
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

    int ACORN::no_filter_iqan_search(const float *query, const float *xb, int d, int metric,
                                     int k, int efSearch_val,
                                     int *indices, float *distances,
                                     int num_threads) const
    {
        if (entry_point == -1)
            return 0;
        int L = std::max(1, efSearch_val);
        int shared_cap = std::max(k, L);
        int ntotal = (int)(offsets.size() - 1);
        if (ntotal <= 0)
            return 0;

        double t0 = omp_get_wtime();

        auto comp_dist = [&](int v)
        { return compute_dist(query, xb, d, metric, v); };

        int nearest = entry_point;
        float d_nearest = comp_dist(nearest);
        for (int lvl = max_level; lvl >= 1; lvl--)
            greedy_update_unfiltered(*this, lvl, nearest, d_nearest, comp_dist);

        std::vector<SearchNeighbor> shared_pool(shared_cap + 1);
        int shared_size = 0;
        std::vector<bool> visited(ntotal, false);
        visited[nearest] = true;

        std::vector<std::pair<float, int>> batch;
        batch.emplace_back(d_nearest, nearest);
        InsertIntoPool(shared_pool.data(), shared_size, shared_cap,
                       SearchNeighbor(nearest, d_nearest, false));

        size_t begin, end;
        neighbor_range(nearest, 0, &begin, &end);
        for (size_t j = begin; j < end; j++)
        {
            int v = neighbors[j];
            if (v < 0)
                break;
            if (visited[v])
                continue;
            visited[v] = true;
            float dv = comp_dist(v);
            batch.emplace_back(dv, v);
            InsertIntoPool(shared_pool.data(), shared_size, shared_cap,
                           SearchNeighbor(v, dv, false));
        }

        std::vector<size_t> thread_ndis(num_threads, 0);
        double t1 = omp_get_wtime();

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
                    InsertIntoPool(local_pool.data(), local_size, L,
                                   SearchNeighbor(p.second, p.first, true));
                }

                int cur = 0, step = 0;
                while (cur < local_size && step < L)
                {
                    bool was_expanded = local_pool[cur].expanded;
                    int next_cur = expand_level0_unfiltered(
                        *this, cur, local_pool.data(), local_size, L, visited,
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
                thread_ndis[tid] += ndis_local;

#pragma omp critical
                for (int i = 0; i < local_size; i++)
                    InsertIntoPool(shared_pool.data(), shared_size, shared_cap,
                                   SearchNeighbor(local_pool[i].id, local_pool[i].distance, false));
            }

            batch.erase(batch.begin(), batch.begin() + to_process);
            std::vector<std::pair<float, int>> all_unexpanded;
            for (int t = 0; t < num_threads; t++)
                for (auto &p : thread_unexpanded[t])
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

            if (shared_size >= shared_cap && !batch.empty())
            {
                float min_d = batch[0].first;
                for (auto &p : batch)
                    if (p.first < min_d)
                        min_d = p.first;
                if (min_d > shared_pool[shared_size - 1].distance)
                    batch.clear();
            }
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
        if (check_filter(filter_map, nearest))
            batch.emplace_back(d_nearest, nearest);

        size_t begin, end;
        neighbor_range(nearest, 0, &begin, &end);
        for (size_t j = begin; j < end; j++)
        {
            int v = neighbors[j];
            if (v < 0)
                break;
            if (!check_filter(filter_map, v))
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
                if (!check_filter(filter_map, p.second))
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
        if (check_filter(filter_map, nearest))
            entry_points.emplace_back(d_nearest, nearest);

        size_t begin, end;
        neighbor_range(nearest, 0, &begin, &end);
        for (size_t j = begin; j < end; j++)
        {
            int v = neighbors[j];
            if (v < 0)
                break;
            if (visited[v])
                continue;
            if (!check_filter(filter_map, v))
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

    int ACORN::no_filter_scatter_search(const float *query, const float *xb, int d, int metric,
                                        int k, int efSearch_val,
                                        int *indices, float *distances,
                                        int num_threads, int Helec) const
    {
        if (entry_point == -1)
            return 0;
        int L = std::max(1, efSearch_val);
        int shared_cap = std::max(k, L);
        int ntotal = (int)(offsets.size() - 1);
        if (ntotal <= 0)
            return 0;

        double t0 = omp_get_wtime();

        auto comp_dist = [&](int v)
        { return compute_dist(query, xb, d, metric, v); };

        // Phase 1: unfiltered greedy descent to level 0.
        int nearest = entry_point;
        float d_nearest = comp_dist(nearest);
        for (int lvl = max_level; lvl >= 1; lvl--)
            greedy_update_unfiltered(*this, lvl, nearest, d_nearest, comp_dist);

        std::vector<bool> visited(ntotal, false);
        visited[nearest] = true;

        std::vector<std::pair<float, int>> entry_points;
        entry_points.emplace_back(d_nearest, nearest);

        size_t begin, end;
        neighbor_range(nearest, 0, &begin, &end);
        for (size_t j = begin; j < end; j++)
        {
            int v = neighbors[j];
            if (v < 0)
                break;
            if (visited[v])
                continue;
            visited[v] = true;
            entry_points.emplace_back(comp_dist(v), v);
        }
        if (entry_points.empty())
            return 0;

        std::vector<std::vector<std::pair<float, int>>> thread_entry_points(num_threads);
        for (size_t i = 0; i < entry_points.size(); i++)
            thread_entry_points[i % num_threads].push_back(entry_points[i]);

        std::vector<SearchNeighbor> shared_pool(shared_cap + 1);
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
                    float distk = (local_size >= L) ? local_pool[L - 1].distance
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
                int next_cur = expand_level0_unfiltered(
                    *this, cur, local_pool.data(), local_size, L, visited,
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
                InsertIntoPool(shared_pool.data(), shared_size, shared_cap,
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
    // FAISS Index Loading
    // ============================================================

    void ACORN::load_from_faiss(const char *filename)
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
    }

} // namespace acorn
