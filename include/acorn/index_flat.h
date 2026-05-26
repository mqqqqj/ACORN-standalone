// -*- c++ -*-
#pragma once

#include <vector>
#include "index.h"
#include "distance.h"

namespace acorn {

/// Index that encodes all vectors as fixed-size codes.
struct IndexFlatCodes : Index {
    size_t code_size;
    std::vector<uint8_t> codes;

    IndexFlatCodes();
    IndexFlatCodes(size_t code_size, idx_t d, MetricType metric = METRIC_L2);

    void add(idx_t n, const float* x) override;
    void reset() override;
    void reconstruct(idx_t key, float* recons) const override;
    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const;

    size_t sa_code_size() const;

    virtual FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const;
    DistanceComputer* get_distance_computer() const override {
        return get_FlatCodesDistanceComputer();
    }
};

/// Index that stores full vectors and performs exhaustive search.
struct IndexFlat : IndexFlatCodes {
    explicit IndexFlat(idx_t d, MetricType metric = METRIC_L2);

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, float* recons) const override;

    float* get_xb() { return (float*)codes.data(); }
    const float* get_xb() const { return (const float*)codes.data(); }

    IndexFlat() {}

    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;
};

} // namespace acorn
