// -*- c++ -*-
#include "acorn/index_acorn.h"
#include "acorn/distance.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/time.h>

namespace acorn {

void IndexACORN::add(idx_t n, const float* x) {
    assert(is_trained);
    int n0 = ntotal;
    if (n > 0) {
        codes.resize((ntotal + n) * code_size);
        memcpy(codes.data() + ntotal * code_size, x, n * code_size);
        ntotal += n;
    }
    // Note: graph construction not implemented in this clean version.
    // For now, we only support loading pre-built indices.
    (void)n0;
}

void IndexACORN::reset() {
    acorn.reset();
    codes.clear();
    ntotal = 0;
}

// ============================================================
// Search wrappers
// ============================================================

void IndexACORN::search(idx_t n, const float* x, idx_t k,
                         float* distances, int* labels,
                         const SearchParameters* params_in) const {
    int ef = params_in ? params_in->efSearch : 200;
    const float* xb = get_xb();
    int metric = (metric_type == METRIC_INNER_PRODUCT) ? 0 : 1;

    for (idx_t i = 0; i < n; i++) {
        const float* q = x + i * d;
        int* idxi = labels + i * k;
        float* simi = distances + i * k;

        int nfound = acorn.search(q, xb, d, metric, k, ef, idxi, simi);

        // Negate distances for inner product
        if (metric == 0) {
            for (int j = 0; j < nfound; j++) simi[j] = -simi[j];
        }
        for (int j = nfound; j < k; j++) {
            simi[j] = std::numeric_limits<float>::max();
            idxi[j] = -1;
        }
    }
}

void IndexACORN::search(idx_t n, const float* x, idx_t k,
                         float* distances, int* labels,
                         char* filter_id_map,
                         const SearchParameters* params_in) const {
    int ef = params_in ? params_in->efSearch : 200;
    const float* xb = get_xb();
    int metric = (metric_type == METRIC_INNER_PRODUCT) ? 0 : 1;

    for (idx_t i = 0; i < n; i++) {
        const float* q = x + i * d;
        int* idxi = labels + i * k;
        float* simi = distances + i * k;
        char* fm = filter_id_map + i * ntotal;

        int nfound = acorn.search(q, xb, d, metric, k, ef, idxi, simi, fm);

        if (metric == 0) {
            for (int j = 0; j < nfound; j++) simi[j] = -simi[j];
        }
        for (int j = nfound; j < k; j++) {
            simi[j] = std::numeric_limits<float>::max();
            idxi[j] = -1;
        }
    }
}

void IndexACORN::parallelSearch(idx_t n, const float* x, idx_t k,
                                 float* distances, int* labels,
                                 int num_threads, int efs,
                                 const SearchParameters* params_in) const {
    int ef = params_in ? params_in->efSearch : 200;
    const float* xb = get_xb();
    int metric = (metric_type == METRIC_INNER_PRODUCT) ? 0 : 1;

    for (idx_t i = 0; i < n; i++) {
        const float* q = x + i * d;
        int* idxi = labels + i * k;
        float* simi = distances + i * k;

        int nfound = acorn.parallel_search(q, xb, d, metric, k, ef,
                                            idxi, simi, num_threads, efs);
        if (metric == 0) {
            for (int j = 0; j < nfound; j++) simi[j] = -simi[j];
        }
        for (int j = nfound; j < k; j++) {
            simi[j] = std::numeric_limits<float>::max();
            idxi[j] = -1;
        }
    }
}

void IndexACORN::parallelSearch(idx_t n, const float* x, idx_t k,
                                 float* distances, int* labels,
                                 char* filter_id_map,
                                 int num_threads, int efs,
                                 const SearchParameters* params_in) const {
    int ef = params_in ? params_in->efSearch : 200;
    const float* xb = get_xb();
    int metric = (metric_type == METRIC_INNER_PRODUCT) ? 0 : 1;

    for (idx_t i = 0; i < n; i++) {
        const float* q = x + i * d;
        int* idxi = labels + i * k;
        float* simi = distances + i * k;
        char* fm = filter_id_map + i * ntotal;

        int nfound = acorn.parallel_search(q, xb, d, metric, k, ef,
                                            idxi, simi, num_threads, efs, fm);
        if (metric == 0) {
            for (int j = 0; j < nfound; j++) simi[j] = -simi[j];
        }
        for (int j = nfound; j < k; j++) {
            simi[j] = std::numeric_limits<float>::max();
            idxi[j] = -1;
        }
    }
}

// ============================================================
// Save / Load
// ============================================================

void IndexACORN::save(const char* filename) const {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    const char magic[4] = {'A', 'C', 'R', 'N'};
    int version = 1;
    fwrite(magic, 4, 1, fp);
    fwrite(&version, sizeof(int), 1, fp);
    fwrite(&d, sizeof(int), 1, fp);
    fwrite(&ntotal, sizeof(idx_t), 1, fp);
    int mt = (int)metric_type;
    fwrite(&mt, sizeof(int), 1, fp);
    size_t cs = code_size;
    fwrite(&cs, sizeof(size_t), 1, fp);
    fwrite(codes.data(), 1, codes.size(), fp);
    size_t meta_sz = (size_t)ntotal;
    fwrite(&meta_sz, sizeof(size_t), 1, fp);
    fwrite(acorn.metadata, sizeof(int), ntotal, fp);
    acorn.save(fp);
    fclose(fp);
}

void IndexACORN::load(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    assert(fp);
    char magic[4]; int version;
    fread(magic, 4, 1, fp);
    fread(&version, sizeof(int), 1, fp);
    assert(magic[0] == 'A' && magic[1] == 'C' && magic[2] == 'R' && magic[3] == 'N');
    fread(&d, sizeof(int), 1, fp);
    fread(&ntotal, sizeof(idx_t), 1, fp);
    int mt; fread(&mt, sizeof(int), 1, fp);
    metric_type = (MetricType)mt;
    size_t cs; fread(&cs, sizeof(size_t), 1, fp);
    code_size = cs;
    codes.resize(ntotal * code_size);
    fread(codes.data(), 1, ntotal * code_size, fp);
    size_t meta_sz; fread(&meta_sz, sizeof(size_t), 1, fp);
    metadata_storage.resize(meta_sz);
    fread(metadata_storage.data(), sizeof(int), meta_sz, fp);
    acorn.load(fp);
    acorn.metadata = metadata_storage.data();
    fclose(fp);
    is_trained = true;
}

} // namespace acorn
