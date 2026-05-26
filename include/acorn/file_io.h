// -*- c++ -*-
#pragma once

#include <cstdint>
#include <vector>

namespace acorn {

/// Read a vector file in fbin format.
///
/// fbin format: 4-byte int num_vectors, 4-byte int dim,
/// then num_vectors * dim float32 values (row-major).
///
/// Returns a pair of (data, (num_vectors, dim)).
std::pair<std::vector<float>, std::pair<int, int>> read_fbin(const char* filename);

/// Write vectors to a file in fbin format.
void write_fbin(const char* filename, const float* data, int n, int d);

} // namespace acorn
