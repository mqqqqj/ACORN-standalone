// -*- c++ -*-
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <utility>
#include <vector>

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

struct IDSelector {
    virtual bool is_member(idx_t i) const = 0;
    virtual ~IDSelector() {}
};

struct SearchParameters {
    IDSelector* sel = nullptr;
    virtual ~SearchParameters() {}
};

// Exception class
class Exception : public std::exception {
   public:
    explicit Exception(const std::string& msg) : msg(msg) {}
    Exception(const std::string& msg, const char* func, const char* file, int line)
            : msg(std::string(file) + ":" + std::to_string(line) + " in " + func +
                  ": " + msg) {}
    const char* what() const noexcept override {
        return msg.c_str();
    }
    std::string msg;
};

// Assertions and throw macros
#define ACORN_ASSERT(X)                                 \
    do {                                                \
        if (!(X)) {                                     \
            fprintf(stderr,                             \
                    "ACORN assertion '%s' failed in %s "\
                    "at %s:%d\n",                       \
                    #X,                                 \
                    __PRETTY_FUNCTION__,                \
                    __FILE__,                           \
                    __LINE__);                          \
            abort();                                    \
        }                                               \
    } while (false)

#define ACORN_THROW_MSG(MSG)                                          \
    do {                                                              \
        throw acorn::Exception(MSG, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
    } while (false)

#define ACORN_THROW_IF_NOT(X)                                 \
    do {                                                      \
        if (!(X)) {                                           \
            ACORN_THROW_MSG("Error: '" #X "' failed");        \
        }                                                     \
    } while (false)

#define ACORN_THROW_IF_NOT_MSG(X, MSG)                              \
    do {                                                             \
        if (!(X)) {                                                  \
            ACORN_THROW_MSG("Error: '" #X "' failed: " MSG);        \
        }                                                            \
    } while (false)

// bare-bones unique_ptr with delete []
template <class T>
struct ScopeDeleter {
    const T* ptr;
    explicit ScopeDeleter(const T* ptr = nullptr) : ptr(ptr) {}
    ~ScopeDeleter() { delete[] ptr; }
};

// bare-bones unique_ptr with delete (single object)
template <class T>
struct ScopeDeleter1 {
    const T* ptr;
    explicit ScopeDeleter1(const T* ptr = nullptr) : ptr(ptr) {}
    void release() { ptr = nullptr; }
    void set(const T* ptr_in) { ptr = ptr_in; }
    ~ScopeDeleter1() { delete ptr; }
};

} // namespace acorn
