// -*- c++ -*-
#pragma once

#include <cstring>
#include <cmath>
#include <stdint.h>

namespace acorn {

// SIMD distance functions (implemented in src/distances.cpp)
float fvec_L2sqr(const float* x, const float* y, size_t d);
float fvec_inner_product(const float* x, const float* y, size_t d);

// Inline: compute distance from query to indexed vector
inline float l2_distance(const float* query, const uint8_t* codes, size_t code_size, int idx) {
    return fvec_L2sqr(query, (const float*)(codes + idx * code_size), code_size / sizeof(float));
}
inline float ip_distance(const float* query, const uint8_t* codes, size_t code_size, int idx) {
    return fvec_inner_product(query, (const float*)(codes + idx * code_size), code_size / sizeof(float));
}

// Symmetric distance between two stored vectors
inline float sym_l2_distance(const uint8_t* codes, size_t code_size, int i, int j) {
    return fvec_L2sqr((const float*)(codes + i * code_size),
                       (const float*)(codes + j * code_size), code_size / sizeof(float));
}
inline float sym_ip_distance(const uint8_t* codes, size_t code_size, int i, int j) {
    return fvec_inner_product((const float*)(codes + i * code_size),
                               (const float*)(codes + j * code_size), code_size / sizeof(float));
}

} // namespace acorn
