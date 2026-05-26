// -*- c++ -*-
#pragma once

#include <vector>
#include <sys/time.h>
#include <stdio.h>
#include <iostream>

#include "types.h"
#include "index.h"
#include "index_flat.h"
#include "acorn_graph.h"

namespace acorn {

/** The ACORN index wraps a flat storage with an ACORN graph for efficient
 *  vector + predicate search.
 */
struct IndexACORN : Index {
    typedef ACORN::storage_idx_t storage_idx_t;

    ACORN acorn;
    bool own_fields;
    Index* storage;
    std::vector<int> metadata_storage;

    IndexACORN() : Index(0, METRIC_L2), own_fields(false), storage(nullptr) {}

    explicit IndexACORN(
            int d, int M, int gamma,
            std::vector<int>& metadata, int M_beta,
            MetricType metric = METRIC_L2);

    explicit IndexACORN(
            Index* storage, int M, int gamma,
            std::vector<int>& metadata, int M_beta);

    ~IndexACORN() override;

    void add(idx_t n, const float* x) override;
    void train(idx_t n, const float* x) override;

    /// standard search
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// hybrid search with attribute filter
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            char* filter_id_map,
            const SearchParameters* params = nullptr) const;

    /// intra-query parallel search (simplified iQAN)
    void parallelSearch(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int num_threads,
            int efs,
            const SearchParameters* params = nullptr) const;

    void reconstruct(idx_t key, float* recons) const override;
    void reset() override;

    void save(const char* filename) const;
    void load(const char* filename);

    void printStats(
            bool print_edge_list = false,
            bool print_filtered_edge_lists = false,
            int filter = -1,
            Operation op = EQUAL);

   private:
    static const int debugFlag = 0;

    void debugTime() {
        if (debugFlag) {
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

    double elapsed() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec + tv.tv_usec * 1e-6;
    }
};

/** Flat index with ACORN graph on top. */
struct IndexACORNFlat : IndexACORN {
    IndexACORNFlat() : IndexACORN() {}
    IndexACORNFlat(
            int d, int M, int gamma,
            std::vector<int>& metadata, int M_beta,
            MetricType metric = METRIC_L2);
};

} // namespace acorn
