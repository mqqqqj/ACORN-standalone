#include "acorn/distance.h"
#include <cmath>
#include <immintrin.h>

namespace acorn {

static inline float hsum_ps_avx(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

float fvec_L2sqr(const float* x, const float* y, size_t d) {
    size_t i = 0;
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();

    for (; i + 16 <= d; i += 16) {
        __m256 xi0 = _mm256_loadu_ps(x + i);
        __m256 yi0 = _mm256_loadu_ps(y + i);
        __m256 xi1 = _mm256_loadu_ps(x + i + 8);
        __m256 yi1 = _mm256_loadu_ps(y + i + 8);

        __m256 diff0 = _mm256_sub_ps(xi0, yi0);
        __m256 diff1 = _mm256_sub_ps(xi1, yi1);

        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
    }

    for (; i + 8 <= d; i += 8) {
        __m256 xi = _mm256_loadu_ps(x + i);
        __m256 yi = _mm256_loadu_ps(y + i);
        __m256 diff = _mm256_sub_ps(xi, yi);
        sum0 = _mm256_fmadd_ps(diff, diff, sum0);
    }

    float result = hsum_ps_avx(_mm256_add_ps(sum0, sum1));

    for (; i < d; i++) {
        float diff = x[i] - y[i];
        result += diff * diff;
    }
    return result;
}

float fvec_inner_product(const float* x, const float* y, size_t d) {
    size_t i = 0;
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();

    for (; i + 16 <= d; i += 16) {
        __m256 xi0 = _mm256_loadu_ps(x + i);
        __m256 yi0 = _mm256_loadu_ps(y + i);
        __m256 xi1 = _mm256_loadu_ps(x + i + 8);
        __m256 yi1 = _mm256_loadu_ps(y + i + 8);

        sum0 = _mm256_fmadd_ps(xi0, yi0, sum0);
        sum1 = _mm256_fmadd_ps(xi1, yi1, sum1);
    }

    for (; i + 8 <= d; i += 8) {
        __m256 xi = _mm256_loadu_ps(x + i);
        __m256 yi = _mm256_loadu_ps(y + i);
        sum0 = _mm256_fmadd_ps(xi, yi, sum0);
    }

    float result = hsum_ps_avx(_mm256_add_ps(sum0, sum1));

    for (; i < d; i++) {
        result += x[i] * y[i];
    }
    return result;
}

} // namespace acorn
