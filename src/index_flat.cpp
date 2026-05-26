#include "acorn/index_flat.h"
#include "acorn/heap.h"
#include <cstring>
#include <cmath>

namespace acorn {

/***********************************************************
 * IndexFlatCodes
 ***********************************************************/

IndexFlatCodes::IndexFlatCodes(size_t code_size, idx_t d, MetricType metric)
        : Index(d, metric), code_size(code_size) {}

IndexFlatCodes::IndexFlatCodes() : code_size(0) {}

void IndexFlatCodes::add(idx_t n, const float* x) {
    ACORN_THROW_IF_NOT(is_trained);
    if (n == 0) return;
    codes.resize((ntotal + n) * code_size);
    memcpy(codes.data() + (ntotal * code_size), x, n * code_size);
    ntotal += n;
}

void IndexFlatCodes::reset() {
    codes.clear();
    ntotal = 0;
}

size_t IndexFlatCodes::sa_code_size() const {
    return code_size;
}

void IndexFlatCodes::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
    ACORN_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));
    memcpy(recons, codes.data() + i0 * code_size, ni * code_size);
}

void IndexFlatCodes::reconstruct(idx_t key, float* recons) const {
    reconstruct_n(key, 1, recons);
}

FlatCodesDistanceComputer* IndexFlatCodes::get_FlatCodesDistanceComputer() const {
    ACORN_THROW_MSG("not implemented");
}

/***********************************************************
 * IndexFlat
 ***********************************************************/

IndexFlat::IndexFlat(idx_t d, MetricType metric)
        : IndexFlatCodes(sizeof(float) * d, d, metric) {}

namespace {

struct FlatL2Dis : FlatCodesDistanceComputer {
    size_t d;
    const float* q;
    const float* b;

    float distance_to_code(const uint8_t* code) final {
        return fvec_L2sqr(q, (const float*)code, d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return fvec_L2sqr(b + j * d, b + i * d, d);
    }

    FlatL2Dis(const IndexFlat& storage, const float* q = nullptr)
            : FlatCodesDistanceComputer(storage.codes.data(), storage.code_size),
              d(storage.d), q(q), b(storage.get_xb()) {}

    void set_query(const float* x) override { q = x; }
};

struct FlatIPDis : FlatCodesDistanceComputer {
    size_t d;
    const float* q;
    const float* b;

    float distance_to_code(const uint8_t* code) final {
        return fvec_inner_product(q, (const float*)code, d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return fvec_inner_product(b + j * d, b + i * d, d);
    }

    FlatIPDis(const IndexFlat& storage, const float* q = nullptr)
            : FlatCodesDistanceComputer(storage.codes.data(), storage.code_size),
              d(storage.d), q(q), b(storage.get_xb()) {}

    void set_query(const float* x) override { q = x; }
};

} // namespace

FlatCodesDistanceComputer* IndexFlat::get_FlatCodesDistanceComputer() const {
    if (metric_type == METRIC_L2) {
        return new FlatL2Dis(*this);
    } else if (metric_type == METRIC_INNER_PRODUCT) {
        return new FlatIPDis(*this);
    } else {
        ACORN_THROW_MSG("metric type not supported");
    }
}

void IndexFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    ACORN_THROW_IF_NOT(k > 0);

    const float* xb = get_xb();

    for (idx_t i = 0; i < n; i++) {
        const float* q = x + i * d;
        float* D = distances + i * k;
        idx_t* I = labels + i * k;

        maxheap_heapify(k, D, I);

        for (idx_t j = 0; j < ntotal; j++) {
            float dist;
            if (metric_type == METRIC_INNER_PRODUCT) {
                dist = fvec_inner_product(q, xb + j * d, d);
            } else {
                dist = fvec_L2sqr(q, xb + j * d, d);
            }
            if (dist < D[0] || I[0] == -1) {
                maxheap_replace_top(k, D, I, dist, j);
            }
        }

        maxheap_reorder(k, D, I);
    }
}

void IndexFlat::reconstruct(idx_t key, float* recons) const {
    memcpy(recons, &(codes[key * code_size]), code_size);
}

} // namespace acorn
