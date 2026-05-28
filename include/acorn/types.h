// -*- c++ -*-
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <exception>
#include <string>
#include <limits>

namespace acorn {

using idx_t = int64_t;

enum MetricType {
    METRIC_INNER_PRODUCT = 0,
    METRIC_L2 = 1,
};

enum Operation {
    EQUAL = 0,
    OR = 1,
    REGEX = 2,
};

struct SearchParameters {
    int efSearch = 16;
    bool check_relative_distance = true;
    virtual ~SearchParameters() {}
};

// --- Neighbor for search pool (NSG-style sorted array) ---
struct SearchNeighbor {
    int id;
    float distance;
    bool expanded;

    SearchNeighbor() : id(-1), distance(0), expanded(false) {}
    SearchNeighbor(int id, float dist, bool exp = true)
        : id(id), distance(dist), expanded(exp) {}

    bool operator<(const SearchNeighbor& o) const { return distance < o.distance; }
};

/// Insert nn into a sorted pool. Maintains ascending order by distance.
/// pool: pre-allocated array of size cap+1
/// L:    number of valid entries (0..cap), updated in place
/// cap:  max number of valid entries
/// Returns insertion position, or cap if pool full + too far, or cap+1 if duplicate.
inline int InsertIntoPool(SearchNeighbor* pool, int& L, int cap, SearchNeighbor nn) {
    if (L == 0) { pool[0] = nn; L = 1; return 0; }
    int left = 0, right = L - 1;
    // Insert at front?
    if (pool[left].distance > nn.distance) {
        if (L < cap) L++;
        memmove(pool + 1, pool, (L - 1) * sizeof(SearchNeighbor));
        pool[0] = nn; return 0;
    }
    // Append?
    if (pool[right].distance <= nn.distance) {
        if (L == cap) return cap;
        pool[L++] = nn; return L - 1;
    }
    // Binary search
    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (pool[mid].distance > nn.distance) right = mid;
        else left = mid;
    }
    // Duplicate check
    for (int i = left; i >= 0; i--) {
        if (pool[i].id == nn.id) return cap + 1;
        if (pool[i].distance < nn.distance) break;
    }
    if (pool[right].id == nn.id) return cap + 1;
    // Insert at 'right'
    if (L < cap) L++;
    memmove(pool + right + 1, pool + right, (L - right - 1) * sizeof(SearchNeighbor));
    pool[right] = nn;
    return right;
}

// --- Exception / Assert ---
class Exception : public std::exception {
public:
    explicit Exception(const std::string& msg) : msg(msg) {}
    Exception(const std::string& msg, const char* func, const char* file, int line)
        : msg(std::string(file) + ":" + std::to_string(line) + " in " + func + ": " + msg) {}
    const char* what() const noexcept override { return msg.c_str(); }
    std::string msg;
};

#define ACORN_THROW_MSG(MSG) \
    do { throw acorn::Exception(MSG, __PRETTY_FUNCTION__, __FILE__, __LINE__); } while (false)
#define ACORN_THROW_IF_NOT(X) \
    do { if (!(X)) { ACORN_THROW_MSG("Error: '" #X "' failed"); } } while (false)
#define ACORN_THROW_IF_NOT_MSG(X, MSG) \
    do { if (!(X)) { ACORN_THROW_MSG("Error: '" #X "' failed: " MSG); } } while (false)

template <class T> struct ScopeDeleter {
    const T* ptr;
    explicit ScopeDeleter(const T* ptr = nullptr) : ptr(ptr) {}
    ~ScopeDeleter() { delete[] ptr; }
};
template <class T> struct ScopeDeleter1 {
    const T* ptr;
    explicit ScopeDeleter1(const T* ptr = nullptr) : ptr(ptr) {}
    ~ScopeDeleter1() { delete ptr; }
};

} // namespace acorn
