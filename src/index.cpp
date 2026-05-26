#include "acorn/index.h"

namespace acorn {

void Index::train(idx_t, const float*) {
    is_trained = true;
}

void Index::reconstruct(idx_t, float*) const {
    ACORN_THROW_MSG("reconstruct not implemented");
}

DistanceComputer* Index::get_distance_computer() const {
    ACORN_THROW_MSG("get_distance_computer not implemented");
}

} // namespace acorn
