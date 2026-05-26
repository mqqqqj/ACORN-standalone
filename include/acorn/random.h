// -*- c++ -*-
#pragma once

#include <stdint.h>
#include <random>

namespace acorn {

/// random generator that can be used in multithreaded contexts
struct RandomGenerator {
    std::mt19937 mt;

    RandomGenerator(int64_t seed = 1234);

    int rand_int();
    int64_t rand_int64();
    int rand_int(int max);
    float rand_float();
    double rand_double();
};

} // namespace acorn
