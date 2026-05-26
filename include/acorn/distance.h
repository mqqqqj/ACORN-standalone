// -*- c++ -*-
#pragma once

#include <cstring>
#include <stdint.h>
#include "types.h"

namespace acorn {

/// The distance computer maintains a current query and computes distances
/// to elements in an index that supports random access.
struct DistanceComputer {
    virtual void set_query(const float* x) = 0;
    virtual float operator()(idx_t i) = 0;
    virtual float symmetric_dis(idx_t i, idx_t j) = 0;
    virtual ~DistanceComputer() {}
};

/// DistanceComputer for flat (full-precision) vector storage
struct FlatCodesDistanceComputer : DistanceComputer {
    const uint8_t* codes;
    size_t code_size;

    FlatCodesDistanceComputer(const uint8_t* codes, size_t code_size)
            : codes(codes), code_size(code_size) {}
    FlatCodesDistanceComputer() : codes(nullptr), code_size(0) {}

    float operator()(idx_t i) final {
        return distance_to_code(codes + i * code_size);
    }

    virtual float distance_to_code(const uint8_t* code) = 0;
    virtual ~FlatCodesDistanceComputer() {}
};

// Forward declarations
float fvec_L2sqr(const float* x, const float* y, size_t d);
float fvec_inner_product(const float* x, const float* y, size_t d);

} // namespace acorn
