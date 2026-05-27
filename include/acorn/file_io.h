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

/// Read label file in ibin format.
///
/// ibin format: 4-byte int count, then count int32 labels.
/// If expected_n > 0, verifies the count matches.
std::vector<int> read_ibin(const char* filename, int expected_n = -1);

/// Read ground truth file in ibin format.
///
/// Format: 4-byte int nq, 4-byte int k, then nq * k int32 label IDs.
/// Returns a pair of (labels, (nq, k)).
std::pair<std::vector<int>, std::pair<int, int>> read_groundtruth(const char* filename);

} // namespace acorn
