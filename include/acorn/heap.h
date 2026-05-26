// -*- c++ -*-
#pragma once

#include <climits>
#include <cmath>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <limits>
#include <stdint.h>

namespace acorn {

/***********************************************************
 * C object: uniform handling of min and max heap
 ***********************************************************/

template <typename T_, typename TI_>
struct CMax;

template <typename T>
inline T cmin_nextafter(T x);
template <typename T>
inline T cmax_nextafter(T x);

template <typename T_, typename TI_>
struct CMin {
    typedef T_ T;
    typedef TI_ TI;
    typedef CMax<T_, TI_> Crev;
    inline static bool cmp(T a, T b) { return a < b; }
    inline static bool cmp2(T a1, T b1, TI a2, TI b2) {
        return (a1 < b1) || ((a1 == b1) && (a2 < b2));
    }
    inline static T neutral() { return std::numeric_limits<T>::lowest(); }
    static const bool is_max = false;
    inline static T nextafter(T x) { return cmin_nextafter(x); }
};

template <typename T_, typename TI_>
struct CMax {
    typedef T_ T;
    typedef TI_ TI;
    typedef CMin<T_, TI_> Crev;
    inline static bool cmp(T a, T b) { return a > b; }
    inline static bool cmp2(T a1, T b1, TI a2, TI b2) {
        return (a1 > b1) || ((a1 == b1) && (a2 > b2));
    }
    inline static T neutral() { return std::numeric_limits<T>::max(); }
    static const bool is_max = true;
    inline static T nextafter(T x) { return cmax_nextafter(x); }
};

template <>
inline float cmin_nextafter<float>(float x) { return std::nextafterf(x, -HUGE_VALF); }
template <>
inline float cmax_nextafter<float>(float x) { return std::nextafterf(x, HUGE_VALF); }
template <>
inline uint16_t cmin_nextafter<uint16_t>(uint16_t x) { return x - 1; }
template <>
inline uint16_t cmax_nextafter<uint16_t>(uint16_t x) { return x + 1; }

/***********************************************************
 * Basic heap ops: push and pop
 ***********************************************************/

template <class C>
inline void heap_pop(size_t k, typename C::T* bh_val, typename C::TI* bh_ids) {
    bh_val--; bh_ids--;
    typename C::T val = bh_val[k];
    typename C::TI id = bh_ids[k];
    size_t i = 1, i1, i2;
    while (1) {
        i1 = i << 1; i2 = i1 + 1;
        if (i1 > k) break;
        if ((i2 == k + 1) || C::cmp2(bh_val[i1], bh_val[i2], bh_ids[i1], bh_ids[i2])) {
            if (C::cmp2(val, bh_val[i1], id, bh_ids[i1])) break;
            bh_val[i] = bh_val[i1]; bh_ids[i] = bh_ids[i1]; i = i1;
        } else {
            if (C::cmp2(val, bh_val[i2], id, bh_ids[i2])) break;
            bh_val[i] = bh_val[i2]; bh_ids[i] = bh_ids[i2]; i = i2;
        }
    }
    bh_val[i] = bh_val[k];
    bh_ids[i] = bh_ids[k];
}

template <class C>
inline void heap_push(size_t k, typename C::T* bh_val, typename C::TI* bh_ids,
                       typename C::T val, typename C::TI id) {
    bh_val--; bh_ids--;
    size_t i = k, i_father;
    while (i > 1) {
        i_father = i >> 1;
        if (!C::cmp2(val, bh_val[i_father], id, bh_ids[i_father])) break;
        bh_val[i] = bh_val[i_father]; bh_ids[i] = bh_ids[i_father]; i = i_father;
    }
    bh_val[i] = val; bh_ids[i] = id;
}

template <class C>
inline void heap_replace_top(size_t k, typename C::T* bh_val, typename C::TI* bh_ids,
                              typename C::T val, typename C::TI id) {
    bh_val--; bh_ids--;
    size_t i = 1, i1, i2;
    while (1) {
        i1 = i << 1; i2 = i1 + 1;
        if (i1 > k) break;
        if ((i2 == k + 1) || C::cmp2(bh_val[i1], bh_val[i2], bh_ids[i1], bh_ids[i2])) {
            if (C::cmp2(val, bh_val[i1], id, bh_ids[i1])) break;
            bh_val[i] = bh_val[i1]; bh_ids[i] = bh_ids[i1]; i = i1;
        } else {
            if (C::cmp2(val, bh_val[i2], id, bh_ids[i2])) break;
            bh_val[i] = bh_val[i2]; bh_ids[i] = bh_ids[i2]; i = i2;
        }
    }
    bh_val[i] = val; bh_ids[i] = id;
}

// Convenience wrappers
template <typename T>
inline void maxheap_pop(size_t k, T* bh_val, int64_t* bh_ids) {
    heap_pop<CMax<T, int64_t>>(k, bh_val, bh_ids);
}

template <typename T>
inline void maxheap_push(size_t k, T* bh_val, int64_t* bh_ids, T val, int64_t ids) {
    heap_push<CMax<T, int64_t>>(k, bh_val, bh_ids, val, ids);
}

template <typename T>
inline void maxheap_replace_top(size_t k, T* bh_val, int64_t* bh_ids, T val, int64_t ids) {
    heap_replace_top<CMax<T, int64_t>>(k, bh_val, bh_ids, val, ids);
}

// Heap initialization
template <class C>
inline void heap_heapify(size_t k, typename C::T* bh_val, typename C::TI* bh_ids,
                          const typename C::T* x = nullptr,
                          const typename C::TI* ids = nullptr, size_t k0 = 0) {
    if (k0 > 0) assert(x);
    if (ids) {
        for (size_t i = 0; i < k0; i++)
            heap_push<C>(i + 1, bh_val, bh_ids, x[i], ids[i]);
    } else {
        for (size_t i = 0; i < k0; i++)
            heap_push<C>(i + 1, bh_val, bh_ids, x[i], i);
    }
    for (size_t i = k0; i < k; i++) {
        bh_val[i] = C::neutral();
        bh_ids[i] = -1;
    }
}

template <typename T>
inline void maxheap_heapify(size_t k, T* bh_val, int64_t* bh_ids,
                             const T* x = nullptr, const int64_t* ids = nullptr,
                             size_t k0 = 0) {
    heap_heapify<CMax<T, int64_t>>(k, bh_val, bh_ids, x, ids, k0);
}

// Heap finalization (reorder into sorted order)
template <typename C>
inline size_t heap_reorder(size_t k, typename C::T* bh_val, typename C::TI* bh_ids) {
    size_t i, ii;
    for (i = 0, ii = 0; i < k; i++) {
        typename C::T val = bh_val[0];
        typename C::TI id = bh_ids[0];
        heap_pop<C>(k - i, bh_val, bh_ids);
        bh_val[k - ii - 1] = val;
        bh_ids[k - ii - 1] = id;
        if (id != -1) ii++;
    }
    size_t nel = ii;
    memmove(bh_val, bh_val + k - ii, ii * sizeof(*bh_val));
    memmove(bh_ids, bh_ids + k - ii, ii * sizeof(*bh_ids));
    for (; ii < k; ii++) {
        bh_val[ii] = C::neutral();
        bh_ids[ii] = -1;
    }
    return nel;
}

template <typename T>
inline size_t maxheap_reorder(size_t k, T* bh_val, int64_t* bh_ids) {
    return heap_reorder<CMax<T, int64_t>>(k, bh_val, bh_ids);
}

} // namespace acorn
