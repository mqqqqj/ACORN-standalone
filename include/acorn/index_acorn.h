// -*- c++ -*-
#pragma once

#include <vector>
#include <sys/time.h>
#include <stdio.h>
#include <iostream>

#include "types.h"
#include "index.h"
#include "distance.h"
#include "acorn_graph.h"

namespace acorn {

/// Flat float32 vector storage + ACORN graph for efficient ANN search.
struct IndexACORN : Index {
    typedef ACORN::storage_idx_t storage_idx_t;

    ACORN acorn;
    size_t code_size;
    std::vector<uint8_t> codes;
    std::vector<int> metadata_storage;

    IndexACORN() : Index(0, METRIC_L2), code_size(0) {}

    IndexACORN(int d, int M, int gamma,
               std::vector<int>& metadata, int M_beta,
               MetricType metric = METRIC_L2);

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
    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const;
    void reset() override;

    size_t sa_code_size() const { return code_size; }
    float* get_xb() { return (float*)codes.data(); }
    const float* get_xb() const { return (const float*)codes.data(); }

    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const;
    DistanceComputer* get_distance_computer() const override {
        return get_FlatCodesDistanceComputer();
    }

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

} // namespace acorn
