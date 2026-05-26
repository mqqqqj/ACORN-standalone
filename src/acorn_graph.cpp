// -*- c++ -*-
#include "acorn/acorn_graph.h"

#include <string>
#include <sys/time.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <fstream>

namespace acorn {

/**************************************************************
 * Debugging helpers
 **************************************************************/

static const int localDebugFlag = 0;
static const char* debugSearchFlagEnv = std::getenv("ACORN_DEBUG_SEARCH");
static int debugSearchFlag = debugSearchFlagEnv ? std::atoi(debugSearchFlagEnv) : 0;

static void debugTime() {
    if (localDebugFlag || debugSearchFlag) {
        struct timeval tval;
        gettimeofday(&tval, NULL);
        struct tm* tm_info = localtime(&tval.tv_sec);
        char timeBuff[25] = "";
        strftime(timeBuff, 25, "%H:%M:%S", tm_info);
        char timeBuffWithMilli[50] = "";
        sprintf(timeBuffWithMilli, "%s.%06ld ", timeBuff, tval.tv_usec);
        std::string timestamp(timeBuffWithMilli);
        std::cout << timestamp << std::flush;
    }
}

#define debug(fmt, ...) \
    do { \
        if (localDebugFlag == 1) { \
            fprintf(stdout, "" fmt, __VA_ARGS__); \
        } \
        if (localDebugFlag == 2) { \
            debugTime(); \
            fprintf(stdout, "%s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, __VA_ARGS__); \
        } \
    } while (0)

#define debug_search(fmt, ...) \
    do { \
        if (debugSearchFlag == 1) { \
            fprintf(stdout, "" fmt, __VA_ARGS__); \
        } \
        if (debugSearchFlag == 2) { \
            fprintf(stdout, "%d:%s(): " fmt, __LINE__, __func__, __VA_ARGS__); \
        } \
        if (debugSearchFlag == 3) { \
            debugTime(); \
            fprintf(stdout, "%s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, __VA_ARGS__); \
        } \
    } while (0)

static double getElapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

ACORNStats acorn_stats;

/**************************************************************
 * ACORN structure implementation
 **************************************************************/

int ACORN::nb_neighbors(int layer_no) const {
    return cum_nneighbor_per_level[layer_no + 1] -
            cum_nneighbor_per_level[layer_no];
}

void ACORN::set_nb_neighbors(int level_no, int n) {
    ACORN_THROW_IF_NOT(levels.size() == 0);
    int cur_n = nb_neighbors(level_no);
    for (int i = level_no + 1; i < (int)cum_nneighbor_per_level.size(); i++) {
        cum_nneighbor_per_level[i] += n - cur_n;
    }
}

int ACORN::cum_nb_neighbors(int layer_no) const {
    return cum_nneighbor_per_level[layer_no];
}

void ACORN::neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end) const {
    size_t o = offsets[no];
    *begin = o + cum_nb_neighbors(layer_no);
    *end = o + cum_nb_neighbors(layer_no + 1);
}

ACORN::ACORN(int M, int gamma, std::vector<int>& metadata, int M_beta) : rng(12345) {
    set_default_probas(M, 1.0 / log(M), M_beta, gamma);
    max_level = -1;
    entry_point = -1;
    efSearch = 16;
    efConstruction = M * gamma;
    upper_beam = 1;
    this->gamma = gamma;
    this->metadata = metadata.data();
    this->M = M;
    this->M_beta = M_beta;
    offsets.push_back(0);
    for (int i = 0; i < (int)assign_probas.size(); i++) nb_per_level.push_back(0);
}

int ACORN::random_level() {
    double f = rng.rand_float();
    for (int level = 0; level < (int)assign_probas.size(); level++) {
        if (f < assign_probas[level]) {
            return level;
        }
        f -= assign_probas[level];
    }
    return (int)assign_probas.size() - 1;
}

void ACORN::set_default_probas(int M, float levelMult, int M_beta, int gamma) {
    int nn = 0;
    cum_nneighbor_per_level.push_back(0);
    if (M_beta > 2 * M * gamma) {
        printf("M_beta: %d, M: %d\n", M_beta, M);
        ACORN_THROW_MSG("M_beta must be less than 2*M*gamma");
    }
    for (int level = 0;; level++) {
        float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
        if (proba < 1e-9) break;
        assign_probas.push_back(proba);
        nn += level == 0 ? (int)M_beta + 1.5 * M : M * gamma;
        cum_nneighbor_per_level.push_back(nn);
    }
}

void ACORN::clear_neighbor_tables(int level) {
    for (int i = 0; i < (int)levels.size(); i++) {
        size_t begin, end;
        neighbor_range(i, level, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            neighbors[j] = ACORN::NeighNode(-1);
        }
    }
}

void ACORN::reset() {
    max_level = -1;
    entry_point = -1;
    offsets.clear();
    offsets.push_back(0);
    levels.clear();
    neighbors.clear();
}

void ACORN::print_neighbor_stats(int level) const {
    ACORN_THROW_IF_NOT(level < (int)cum_nneighbor_per_level.size());
    printf("* stats on level %d, max %d neighbors per vertex:\n",
           level, nb_neighbors(level));

    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
#pragma omp parallel for reduction(+: tot_neigh) reduction(+: tot_common) \
  reduction(+: tot_reciprocal) reduction(+: n_node)
    for (int i = 0; i < (int)levels.size(); i++) {
        if (levels[i] > level) {
            n_node++;
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            std::unordered_set<int> neighset;
            for (size_t j = begin; j < end; j++) {
                if (neighbors[j] < 0) break;
                neighset.insert(neighbors[j]);
            }
            int n_neigh = neighset.size();
            int n_common = 0;
            int n_reciprocal = 0;
            for (size_t j = begin; j < end; j++) {
                storage_idx_t i2 = neighbors[j];
                if (i2 < 0) break;
                size_t begin2, end2;
                neighbor_range(i2, level, &begin2, &end2);
                for (size_t j2 = begin2; j2 < end2; j2++) {
                    storage_idx_t i3 = neighbors[j2];
                    if (i3 < 0) break;
                    if (i3 == i) {
                        n_reciprocal++;
                        continue;
                    }
                    if (neighset.count(i3)) {
                        neighset.erase(i3);
                        n_common++;
                    }
                }
            }
            tot_neigh += n_neigh;
            tot_common += n_common;
            tot_reciprocal += n_reciprocal;
        }
    }
    float normalizer = n_node;
    printf("   1. nb of nodes: %zd\n", n_node);
    printf("   2. neighbors per node: %.2f (%zd)\n", tot_neigh / normalizer, tot_neigh);
    printf("   3. nb of reciprocal neighbors: %.2f\n", tot_reciprocal / normalizer);
    printf("   4. nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
           tot_common / normalizer, tot_common);
}

void ACORN::print_edges(int level) const {
    ACORN_THROW_IF_NOT(level < (int)cum_nneighbor_per_level.size());
    printf("* stats on level %d, max %d neighbors per vertex:\n",
           level, nb_neighbors(level));

    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
    printf("\t edges lists:\n");
    for (int i = 0; i < (int)levels.size(); i++) {
        if (levels[i] > level) {
            n_node++;
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            std::unordered_set<int> neighset;
            printf("\t\t %d: [", i);
            for (size_t j = begin; j < end; j++) {
                if (neighbors[j] < 0) break;
                printf("%d(%d), ", neighbors[j], metadata[neighbors[j]]);
                neighset.insert(neighbors[j]);
            }
            printf("]\n");
            int n_neigh = neighset.size();
            int n_common = 0;
            int n_reciprocal = 0;
            for (size_t j = begin; j < end; j++) {
                storage_idx_t i2 = neighbors[j];
                if (i2 < 0) break;
                ACORN_ASSERT(i2 != i);
                size_t begin2, end2;
                neighbor_range(i2, level, &begin2, &end2);
                for (size_t j2 = begin2; j2 < end2; j2++) {
                    storage_idx_t i3 = neighbors[j2];
                    if (i3 < 0) break;
                    if (i3 == i) {
                        n_reciprocal++;
                        continue;
                    }
                    if (neighset.count(i3)) {
                        neighset.erase(i3);
                        n_common++;
                    }
                }
            }
            tot_neigh += n_neigh;
            tot_common += n_common;
            tot_reciprocal += n_reciprocal;
        }
    }
    float normalizer = n_node;
    printf("   0. level: %d\n", level);
    printf("   1. max neighbors per node: %d\n", nb_neighbors(level));
    printf("   2. nb of nodes: %zd\n", n_node);
    printf("   3. neighbors per node: %.2f (%zd)\n", tot_neigh / normalizer, tot_neigh);
    printf("   4. nb of reciprocal neighbors: %.2f\n", tot_reciprocal / normalizer);
    printf("   5. nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
           tot_common / normalizer, tot_common);
}

void ACORN::print_edges_filtered(int level, int filter, Operation op) const {
    ACORN_THROW_IF_NOT(level < (int)cum_nneighbor_per_level.size());
    printf("* stats on level %d, max %d neighbors per vertex:\n",
           level, nb_neighbors(level));

    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
    printf("\t edges lists:\n");
    for (int i = 0; i < (int)levels.size(); i++) {
        if (levels[i] > level) {
            if (((op == EQUAL) && (metadata[i] == filter)) ||
                (op == OR && ((metadata[i] & filter) != 0))) {
                n_node++;
                size_t begin, end;
                neighbor_range(i, level, &begin, &end);
                std::unordered_set<int> neighset;
                printf("\t\t %d (%d): [", i, metadata[i]);
                for (size_t j = begin; j < end; j++) {
                    if (neighbors[j] < 0) break;
                    if (((op == EQUAL) && (metadata[neighbors[j]] == filter)) ||
                        (op == OR && ((metadata[neighbors[j]] & filter) != 0))) {
                        printf("%d(%d), ", neighbors[j], metadata[neighbors[j]]);
                        neighset.insert(neighbors[j]);
                    }
                }
                printf("]\n");

                int n_neigh = neighset.size();
                int n_common = 0;
                int n_reciprocal = 0;
                for (size_t j = begin; j < end; j++) {
                    storage_idx_t i2 = neighbors[j];
                    if (i2 < 0) break;
                    ACORN_ASSERT(i2 != i);
                    size_t begin2, end2;
                    neighbor_range(i2, level, &begin2, &end2);
                    for (size_t j2 = begin2; j2 < end2; j2++) {
                        storage_idx_t i3 = neighbors[j2];
                        if (i3 < 0) break;
                        if (i3 == i) {
                            n_reciprocal++;
                            continue;
                        }
                        if (neighset.count(i3)) {
                            neighset.erase(i3);
                            n_common++;
                        }
                    }
                }
                tot_neigh += n_neigh;
                tot_common += n_common;
                tot_reciprocal += n_reciprocal;
            }
        }
    }
    float normalizer = n_node;
    printf("   0. level: %d\n", level);
    printf("   1. max neighbors per node: %d\n", nb_neighbors(level));
    printf("   2. nb of nodes: %zd\n", n_node);
    printf("   3. neighbors per node: %.2f (%zd)\n", tot_neigh / normalizer, tot_neigh);
    printf("   4. nb of reciprocal neighbors: %.2f\n", tot_reciprocal / normalizer);
    printf("   5. nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
           tot_common / normalizer, tot_common);
}

void ACORN::print_neighbor_stats(bool edge_list, bool filtered_edge_list,
                                  int filter, Operation op) const {
    printf("========= METADATA =======\n");
    printf("\t* cumulative max num neighbors per level\n");
    for (int i = 0; i < (int)cum_nneighbor_per_level.size() - 1; i++) {
        printf("\t\tindx %d: %d\n", i, cum_nneighbor_per_level[i]);
    }
    printf("\t* level probabilities\n");
    for (int level = 0; level < (int)assign_probas.size(); level++) {
        printf("\t\tlevel %d: %f\n", level, assign_probas[level]);
    }
    printf("\t* efConstruction: %d\n", efConstruction);
    printf("\t* efSearch: %d\n", efSearch);
    printf("\t* max_level: %d\n", max_level);
    printf("\t* entry_point: %d\n", entry_point);
    printf("\t* gamma: %d\n", gamma);

    printf("========= LEVEL STATS OF ACORN =======\n");
    if (edge_list) {
        for (int level = 0; level <= max_level; level++) {
            printf("========= LEVEL %d =======\n", level);
            print_edges(level);
        }
    } else {
        for (int level = 0; level <= max_level; level++) {
            printf("========= LEVEL %d =======\n", level);
            print_neighbor_stats(level);
        }
    }

    if (filtered_edge_list) {
        printf("========= LEVEL STATS OF SUBGRAPH WITH FILTER %d, OP %d =======\n", filter, op);
        for (int level = 0; level <= max_level; level++) {
            printf("========= LEVEL %d =======\n", level);
            print_edges_filtered(level, filter, op);
        }
    }
}

void ACORN::fill_with_random_links(size_t) {
    ACORN_THROW_MSG("UNIMPLEMENTED");
}

int ACORN::prepare_level_tab(size_t n, bool preset_levels) {
    size_t n0 = offsets.size() - 1;

    if (preset_levels) {
        ACORN_ASSERT(n0 + n == levels.size());
    } else {
        ACORN_ASSERT(n0 == levels.size());
        for (int i = 0; i < (int)n; i++) {
            int pt_level = random_level();
            levels.push_back(pt_level + 1);
        }
    }

    int max_lvl = 0;
    for (int i = 0; i < (int)n; i++) {
        int pt_level = levels[i + n0] - 1;
        if (pt_level > max_lvl) max_lvl = pt_level;
        offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
        neighbors.resize(offsets.back(), NeighNode(-1));
    }

    return max_lvl;
}

/**************************************************************
 * ACORN shrink neighbor list (pruning strategy)
 **************************************************************/

void ACORN::shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistFarther>& input,
        std::vector<NodeDistFarther>& output,
        int max_size, int gamma, storage_idx_t q_id, int q_attr) {

    std::unordered_set<storage_idx_t> neigh_of_neigh;
    int node_num = 0;

    while (input.size() > 0) {
        node_num = node_num + 1;
        NodeDistFarther v1 = input.top();
        input.pop();
        float dist_v1_q = v1.d;

        bool good = true;

        if (node_num > this->M_beta && neigh_of_neigh.count(v1.id) > 0) {
            good = false;
        }

        if (good) {
            output.push_back(v1);
            if ((int)output.size() >= max_size) return;

            neigh_of_neigh.insert(v1.id);
            if (node_num > this->M_beta) {
                size_t begin, end;
                neighbor_range(v1.id, 0, &begin, &end);
                for (size_t j = begin; j < end; j++) {
                    if (neighbors[j] < 0) break;
                    neigh_of_neigh.insert(neighbors[j]);
                }
            }
            if (neigh_of_neigh.size() >= (size_t)max_size) break;
        }
    }
}

/**************************************************************
 * Construction subroutines (anonymous namespace)
 **************************************************************/

namespace {

using storage_idx_t = ACORN::storage_idx_t;
using NodeDistCloser = ACORN::NodeDistCloser;
using NodeDistFarther = ACORN::NodeDistFarther;

/// remove neighbors from the list to make it smaller than max_size
void shrink_neighbor_list_helper(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& resultSet1,
        int max_size, int gamma, storage_idx_t q_id, int q_attr, ACORN& hnsw) {

    std::priority_queue<NodeDistFarther> resultSet;
    std::vector<NodeDistFarther> returnlist;

    while (resultSet1.size() > 0) {
        resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
        resultSet1.pop();
    }

    hnsw.shrink_neighbor_list(qdis, resultSet, returnlist, max_size, gamma, q_id, q_attr);

    for (NodeDistFarther curen2 : returnlist) {
        resultSet1.emplace(curen2.d, curen2.id);
    }
}

/// add a link between two elements
void add_link(
        ACORN& hnsw,
        DistanceComputer& qdis,
        storage_idx_t src,
        storage_idx_t dest,
        int level) {
    size_t begin, end;
    hnsw.neighbor_range(src, level, &begin, &end);
    if (hnsw.neighbors[end - 1] == -1) {
        size_t i = end;
        while (i > begin) {
            if (hnsw.neighbors[i - 1] != -1) break;
            i--;
        }
        hnsw.neighbors[i] = dest;
        return;
    }

    std::priority_queue<NodeDistCloser> resultSet;
    resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
    for (size_t i = begin; i < end; i++) {
        auto neigh = hnsw.neighbors[i];
        resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
    }

    if (level == 0) {
        shrink_neighbor_list_helper(qdis, resultSet, end - begin, hnsw.gamma,
                                     src, hnsw.metadata[src], hnsw);
    }

    size_t i = begin;
    while (resultSet.size()) {
        hnsw.neighbors[i++] = ACORN::NeighNode(resultSet.top().id);
        resultSet.pop();
    }
    while (i < end) {
        hnsw.neighbors[i++] = ACORN::NeighNode(-1);
    }
}

/// search neighbors for construction on a single level
void search_neighbors_to_add(
        ACORN& hnsw,
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& results,
        int entry_point,
        float d_entry_point,
        int level,
        VisitedTable& vt,
        std::vector<storage_idx_t> ep_per_level = {}) {

    std::priority_queue<NodeDistFarther> candidates;
    NodeDistFarther ev(d_entry_point, entry_point);
    candidates.push(ev);
    results.emplace(d_entry_point, entry_point);
    vt.set(entry_point);

    int M_target;
    if (level == 0) {
        M_target = 2 * hnsw.M * hnsw.gamma;
    } else {
        M_target = hnsw.nb_neighbors(level);
    }

    while (!candidates.empty()) {
        const NodeDistFarther& currEv = candidates.top();

        if ((currEv.d > results.top().d && hnsw.gamma == 1) ||
            (int)results.size() >= M_target) {
            break;
        }
        int currNode = currEv.id;
        candidates.pop();

        size_t begin, end;
        hnsw.neighbor_range(currNode, level, &begin, &end);

        int numIters = 0;
        for (size_t i = begin; i < end; i++) {
            auto nodeId = hnsw.neighbors[i];
            if (nodeId < 0) break;
            if (vt.get(nodeId)) continue;
            vt.set(nodeId);

            numIters = numIters + 1;
            if (numIters > hnsw.M) break;

            float dis = qdis(nodeId);
            NodeDistFarther evE1(dis, nodeId);

            if ((int)results.size() < hnsw.efConstruction || results.top().d > dis) {
                results.emplace(dis, nodeId);
                candidates.emplace(dis, nodeId);
                if ((int)results.size() > hnsw.efConstruction) {
                    results.pop();
                }
            }

            numIters = numIters + 1;
            if (numIters > hnsw.M) break;
        }
    }

    vt.advance();
}

/**************************************************************
 * Searching subroutines
 **************************************************************/

/// greedily update nearest for construction
void greedy_update_nearest(
        const ACORN& hnsw,
        DistanceComputer& qdis,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {
    for (;;) {
        storage_idx_t prev_nearest = nearest;
        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);

        int numIters = 0;
        for (size_t i = begin; i < end; i++) {
            auto v = hnsw.neighbors[i];
            if (v < 0) break;

            numIters = numIters + 1;
            if (numIters > hnsw.M) break;

            float dis = qdis(v);
            if (dis < d_nearest) {
                nearest = v;
                d_nearest = dis;
            }
        }
        if (nearest == prev_nearest) return;
    }
}

/// greedy update nearest for hybrid search
int hybrid_greedy_update_nearest(
        const ACORN& hnsw,
        DistanceComputer& qdis,
        char* filter_map,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {
    int ndis = 0;
    for (;;) {
        int num_found = 0;
        storage_idx_t prev_nearest = nearest;

        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);

        for (size_t i = begin; i < end; i++) {
            auto v = hnsw.neighbors[i];
            if (v < 0) break;

            if (filter_map[v]) {
                num_found = num_found + 1;
            } else {
                if (hnsw.gamma > 1) continue;
            }

            if (filter_map[v]) {
                float dis = qdis(v);
                ndis += 1;
                if (dis < d_nearest || !filter_map[nearest]) {
                    nearest = v;
                    d_nearest = dis;
                }
                if (num_found >= hnsw.M) break;
            }

            if (hnsw.gamma == 1) {
                size_t begin2, end2;
                hnsw.neighbor_range(v, level, &begin2, &end2);
                for (size_t j = begin2; j < end2; j++) {
                    auto v2 = hnsw.neighbors[j];
                    if (v2 < 0) break;

                    if (filter_map[v2]) {
                        num_found = num_found + 1;
                        float dis2 = qdis(v2);
                        ndis += 1;

                        if (dis2 < d_nearest || !filter_map[nearest]) {
                            nearest = v2;
                            d_nearest = dis2;
                        }
                        if (num_found >= hnsw.M) break;
                    }
                }
            }
        }

        if (nearest == prev_nearest) return ndis;
    }
    return ndis;
}

} // anonymous namespace

/**************************************************************
 * Building, parallel
 **************************************************************/

void ACORN::add_links_starting_from(
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        float d_nearest,
        int level,
        omp_lock_t* locks,
        VisitedTable& vt,
        std::vector<storage_idx_t> ep_per_level) {

    std::priority_queue<NodeDistCloser> link_targets;

    search_neighbors_to_add(*this, ptdis, link_targets, nearest, d_nearest,
                             level, vt, ep_per_level);

    nearest = link_targets.top().id;
    int M_target = nb_neighbors(level);

    if (level == 0) {
        shrink_neighbor_list_helper(ptdis, link_targets, M_target, gamma,
                                     pt_id, this->metadata[pt_id], *this);
    }

    std::vector<storage_idx_t> neighs;
    neighs.reserve(link_targets.size());
    while (!link_targets.empty()) {
        storage_idx_t other_id = link_targets.top().id;
        add_link(*this, ptdis, pt_id, other_id, level);
        neighs.push_back(other_id);
        link_targets.pop();
    }

    omp_unset_lock(&locks[pt_id]);
    for (storage_idx_t other_id : neighs) {
        omp_set_lock(&locks[other_id]);
        add_link(*this, ptdis, other_id, pt_id, level);
        omp_unset_lock(&locks[other_id]);
    }
    omp_set_lock(&locks[pt_id]);
}

void ACORN::add_with_locks(
        DistanceComputer& ptdis,
        int pt_level,
        int pt_id,
        std::vector<omp_lock_t>& locks,
        VisitedTable& vt) {

    storage_idx_t nearest;
#pragma omp critical
    {
        nearest = entry_point;
        if (nearest == -1) {
            max_level = pt_level;
            entry_point = pt_id;
            for (int i = 0; i <= max_level; i++) {
                nb_per_level[i] = nb_per_level[i] + 1;
            }
        }
    }

    if (nearest < 0) return;

    omp_set_lock(&locks[pt_id]);

    int level = max_level;
    float d_nearest = ptdis(nearest);

    std::vector<storage_idx_t> ep_per_level(max_level);
    ep_per_level[level] = nearest;

    for (; level > pt_level; level--) {
        greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
        ep_per_level[level] = nearest;
    }

    for (; level >= 0; level--) {
        add_links_starting_from(ptdis, pt_id, nearest, d_nearest, level,
                                 locks.data(), vt, ep_per_level);
        nb_per_level[level] = nb_per_level[level] + 1;
    }

    omp_unset_lock(&locks[pt_id]);

    if (pt_level > max_level) {
        max_level = pt_level;
        entry_point = pt_id;
    }
}

/**************************************************************
 * Searching
 **************************************************************/

namespace {

using MinimaxHeap = ACORN::MinimaxHeap;

int search_from_candidates(
        const ACORN& hnsw,
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        ACORNStats& stats,
        int level,
        int nres_in = 0,
        const SearchParametersACORN* params = nullptr) {

    int nres = nres_in;
    int ndis = 0;
    bool do_dis_check = params ? params->check_relative_distance : hnsw.check_relative_distance;
    int efSearch = params ? params->efSearch : hnsw.efSearch;
    const IDSelector* sel = params ? params->sel : nullptr;

    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        ACORN_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (nres < k) {
                maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
                maxheap_replace_top(nres, D, I, d, v1);
            }
        }
        vt.set(v1);
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        if (do_dis_check) {
            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) break;
        }

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) break;
            if (vt.get(v1)) continue;
            vt.set(v1);
            ndis++;
            float d = qdis(v1);
            if (!sel || sel->is_member(v1)) {
                if (nres < k) {
                    maxheap_push(++nres, D, I, d, v1);
                } else if (d < D[0]) {
                    maxheap_replace_top(nres, D, I, d, v1);
                }
            }
            candidates.push(v1, d);
        }

        nstep++;
        if (!do_dis_check && nstep > efSearch) break;
    }

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) stats.n2++;
        stats.n3 += ndis;
    }

    return nres;
}

int hybrid_search_from_candidates(
        const ACORN& hnsw,
        DistanceComputer& qdis,
        char* filter_map,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        ACORNStats& stats,
        int level,
        int nres_in = 0,
        const SearchParametersACORN* params = nullptr) {

    int nres = nres_in;
    int ndis = 0;
    bool do_dis_check = params ? params->check_relative_distance : hnsw.check_relative_distance;
    int efSearch = params ? params->efSearch : hnsw.efSearch;
    const IDSelector* sel = params ? params->sel : nullptr;

    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        ACORN_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (nres < k) {
                maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
                maxheap_replace_top(nres, D, I, d, v1);
            }
        }
        vt.set(v1);
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        if (do_dis_check) {
            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) break;
        }

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        int num_found = 0;
        bool keep_expanding = true;

        for (size_t j = begin; j < end; j++) {
            auto v1 = hnsw.neighbors[j];
            if (v1 < 0) break;
            if (filter_map[v1]) {
                num_found = num_found + 1;
            }
            if (vt.get(v1)) continue;

            if (filter_map[v1]) {
                vt.set(v1);
                ndis++;
                float d = qdis(v1);

                if (!sel || sel->is_member(v1)) {
                    if (nres < k) {
                        maxheap_push(++nres, D, I, d, v1);
                    } else if (d < D[0]) {
                        maxheap_replace_top(nres, D, I, d, v1);
                    }
                }
                candidates.push(v1, d);

                if (num_found >= hnsw.M * 2) {
                    keep_expanding = false;
                    break;
                }
            }

            if (((j - begin >= hnsw.M_beta) && keep_expanding) || hnsw.gamma == 1) {
                size_t begin2, end2;
                hnsw.neighbor_range(v1, level, &begin2, &end2);
                for (size_t j2 = begin2; j2 < end2; j2 += 1) {
                    auto v2 = hnsw.neighbors[j2];
                    if (v2 < 0) break;

                    if (filter_map[v2]) {
                        num_found = num_found + 1;
                    } else {
                        continue;
                    }

                    if (vt.get(v2)) continue;

                    vt.set(v2);
                    ndis++;

                    float d2 = qdis(v2);
                    if (!sel || sel->is_member(v2)) {
                        if (nres < k) {
                            maxheap_push(++nres, D, I, d2, v2);
                        } else if (d2 < D[0]) {
                            maxheap_replace_top(nres, D, I, d2, v2);
                        }
                    }
                    candidates.push(v2, d2);
                    if (num_found >= hnsw.M * 2) {
                        keep_expanding = false;
                        break;
                    }
                }
            }
        }

        nstep++;
        if (!do_dis_check && nstep > efSearch) break;
    }

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) stats.n2++;
        stats.n3 += ndis;
    }

    return nres;
}

} // anonymous namespace

ACORNStats ACORN::search(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        VisitedTable& vt,
        const SearchParametersACORN* params) const {

    ACORNStats stats;
    if (entry_point == -1) return stats;

    if (upper_beam == 1) {
        storage_idx_t nearest = entry_point;
        float d_nearest = qdis(nearest);

        for (int level = max_level; level >= 1; level--) {
            greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        }

        int efSearch_val = params ? params->efSearch : efSearch;
        int ef = std::max(efSearch_val, k);
        if (search_bounded_queue) {
            MinimaxHeap candidates(ef);
            candidates.push(nearest, d_nearest);
            search_from_candidates(*this, qdis, k, I, D, candidates, vt, stats, 0, 0, params);
        } else {
            ACORN_THROW_MSG("UNIMPLEMENTED search unbounded queue");
        }
        vt.advance();
    } else {
        int candidates_size = upper_beam;
        MinimaxHeap candidates(candidates_size);

        std::vector<idx_t> I_to_next(candidates_size);
        std::vector<float> D_to_next(candidates_size);

        int nres = 1;
        I_to_next[0] = entry_point;
        D_to_next[0] = qdis(entry_point);

        for (int level = max_level; level >= 0; level--) {
            candidates.clear();

            for (int i = 0; i < nres; i++) {
                candidates.push(I_to_next[i], D_to_next[i]);
            }

            if (level == 0) {
                nres = search_from_candidates(
                        *this, qdis, k, I, D, candidates, vt, stats, 0);
            } else {
                nres = search_from_candidates(
                        *this,
                        qdis,
                        candidates_size,
                        I_to_next.data(),
                        D_to_next.data(),
                        candidates,
                        vt,
                        stats,
                        level);
            }
            vt.advance();
        }
    }

    return stats;
}

ACORNStats ACORN::hybrid_search(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        VisitedTable& vt,
        char* filter_map,
        const SearchParametersACORN* params) const {

    ACORNStats stats;
    if (entry_point == -1) return stats;

    if (upper_beam == 1) {
        storage_idx_t nearest = entry_point;
        float d_nearest = qdis(nearest);

        int ndis_upper = 0;
        for (int level = max_level; level >= 1; level--) {
            ndis_upper += hybrid_greedy_update_nearest(*this, qdis, filter_map,
                                                         level, nearest, d_nearest);
        }
        stats.n3 += ndis_upper;

        int efSearch_val = params ? params->efSearch : efSearch;
        int ef = std::max(efSearch_val, k);
        if (search_bounded_queue) {
            MinimaxHeap candidates(ef);
            candidates.push(nearest, d_nearest);
            hybrid_search_from_candidates(*this, qdis, filter_map, k, I, D,
                                           candidates, vt, stats, 0, 0, params);
        } else {
            ACORN_THROW_MSG("UNIMPLEMENTED search unbounded queue");
        }
        vt.advance();
    } else {
        int candidates_size = upper_beam;
        MinimaxHeap candidates(candidates_size);

        std::vector<idx_t> I_to_next(candidates_size);
        std::vector<float> D_to_next(candidates_size);

        int nres = 1;
        I_to_next[0] = entry_point;
        D_to_next[0] = qdis(entry_point);

        for (int level = max_level; level >= 0; level--) {
            candidates.clear();

            for (int i = 0; i < nres; i++) {
                candidates.push(I_to_next[i], D_to_next[i]);
            }

            if (level == 0) {
                nres = hybrid_search_from_candidates(
                        *this, qdis, filter_map, k, I, D, candidates, vt, stats, 0);
            } else {
                nres = hybrid_search_from_candidates(
                        *this,
                        qdis,
                        filter_map,
                        candidates_size,
                        I_to_next.data(),
                        D_to_next.data(),
                        candidates,
                        vt,
                        stats,
                        level);
            }
            vt.advance();
        }
    }

    return stats;
}

/**************************************************************
 * MinimaxHeap
 **************************************************************/

void ACORN::MinimaxHeap::push(storage_idx_t i, float v) {
    if (k == n) {
        if (v >= dis[0]) return;
        heap_pop<HC>(k--, dis.data(), ids.data());
        --nvalid;
    }
    heap_push<HC>(++k, dis.data(), ids.data(), v, i);
    ++nvalid;
}

float ACORN::MinimaxHeap::max() const {
    return dis[0];
}

int ACORN::MinimaxHeap::size() const {
    return nvalid;
}

void ACORN::MinimaxHeap::clear() {
    nvalid = k = 0;
}

int ACORN::MinimaxHeap::pop_min(float* vmin_out) {
    assert(k > 0);
    int i = k - 1;
    while (i >= 0) {
        if (ids[i] != -1) break;
        i--;
    }
    if (i == -1) return -1;
    int imin = i;
    float vmin = dis[i];
    i--;
    while (i >= 0) {
        if (ids[i] != -1 && dis[i] < vmin) {
            vmin = dis[i];
            imin = i;
        }
        i--;
    }
    if (vmin_out) *vmin_out = vmin;
    int ret = ids[imin];
    ids[imin] = -1;
    --nvalid;
    return ret;
}

int ACORN::MinimaxHeap::count_below(float thresh) {
    int n_below = 0;
    for (int i = 0; i < k; i++) {
        if (ids[i] != -1 && dis[i] < thresh) n_below++;
    }
    return n_below;
}

void ACORN::save(FILE* fp) const {
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
    fwrite(neighbors.data(), sizeof(NeighNode), sz, fp);
}

void ACORN::load(FILE* fp) {
    fread(&M, sizeof(int), 1, fp);
    fread(&gamma, sizeof(int), 1, fp);
    fread(&M_beta, sizeof(int), 1, fp);
    fread(&max_level, sizeof(int), 1, fp);
    fread(&entry_point, sizeof(storage_idx_t), 1, fp);
    fread(&efConstruction, sizeof(int), 1, fp);

    size_t sz;

    fread(&sz, sizeof(size_t), 1, fp);
    assign_probas.resize(sz);
    fread(assign_probas.data(), sizeof(double), sz, fp);

    fread(&sz, sizeof(size_t), 1, fp);
    cum_nneighbor_per_level.resize(sz);
    fread(cum_nneighbor_per_level.data(), sizeof(int), sz, fp);

    fread(&sz, sizeof(size_t), 1, fp);
    levels.resize(sz);
    fread(levels.data(), sizeof(int), sz, fp);

    fread(&sz, sizeof(size_t), 1, fp);
    nb_per_level.resize(sz);
    fread(nb_per_level.data(), sizeof(storage_idx_t), sz, fp);

    fread(&sz, sizeof(size_t), 1, fp);
    offsets.resize(sz);
    fread(offsets.data(), sizeof(size_t), sz, fp);

    fread(&sz, sizeof(size_t), 1, fp);
    neighbors.resize(sz);
    fread(neighbors.data(), sizeof(NeighNode), sz, fp);
}

} // namespace acorn
