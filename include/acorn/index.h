// -*- c++ -*-
#pragma once

#include <cstdio>
#include <stdint.h>
#include "types.h"
#include "distance.h"

namespace acorn {

/// Abstract structure for an index.
/// All vectors are 32-bit float arrays in row-major storage.
struct Index {
    using component_t = float;
    using distance_t = float;

    int d;              ///< vector dimension
    idx_t ntotal;       ///< total nb of indexed vectors
    bool verbose;       ///< verbosity level
    bool is_trained;    ///< does the Index require training
    MetricType metric_type;
    float metric_arg;

    explicit Index(idx_t d = 0, MetricType metric = METRIC_L2)
            : d(d), ntotal(0), verbose(false), is_trained(true),
              metric_type(metric), metric_arg(0) {}

    virtual ~Index() {}

    virtual void train(idx_t n, const float* x);

    virtual void add(idx_t n, const float* x) = 0;

    virtual void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const = 0;

    virtual void reset() = 0;

    virtual void reconstruct(idx_t key, float* recons) const;

    virtual DistanceComputer* get_distance_computer() const;
};

} // namespace acorn
