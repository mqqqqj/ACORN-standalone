#include "acorn/file_io.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace acorn {

std::pair<std::vector<float>, std::pair<int, int>> read_fbin(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        throw std::runtime_error(std::string("cannot open file: ") + filename);
    }

    int n = 0, d = 0;
    if (fread(&n, sizeof(int), 1, fp) != 1 ||
        fread(&d, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        throw std::runtime_error(std::string("failed to read header from: ") + filename);
    }

    if (n <= 0 || d <= 0) {
        fclose(fp);
        throw std::runtime_error(
            std::string("invalid dimensions in fbin header: n=") +
            std::to_string(n) + ", d=" + std::to_string(d));
    }

    size_t total = (size_t)n * d;
    std::vector<float> data(total);
    size_t nread = fread(data.data(), sizeof(float), total, fp);
    fclose(fp);

    if (nread != total) {
        throw std::runtime_error(
            std::string("file truncated: expected ") +
            std::to_string(total) + " floats, got " + std::to_string(nread));
    }

    return {std::move(data), {n, d}};
}

void write_fbin(const char* filename, const float* data, int n, int d) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        throw std::runtime_error(std::string("cannot open file for writing: ") + filename);
    }

    fwrite(&n, sizeof(int), 1, fp);
    fwrite(&d, sizeof(int), 1, fp);
    size_t total = (size_t)n * d;
    size_t nwritten = fwrite(data, sizeof(float), total, fp);
    fclose(fp);

    if (nwritten != total) {
        throw std::runtime_error(
            std::string("write truncated: expected ") +
            std::to_string(total) + " floats, wrote " + std::to_string(nwritten));
    }
}

std::vector<int> read_ibin(const char* filename, int expected_n) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        throw std::runtime_error(std::string("cannot open file: ") + filename);
    }
    int n;
    if (fread(&n, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        throw std::runtime_error(std::string("failed to read header from: ") + filename);
    }
    if (expected_n > 0 && n != expected_n) {
        fclose(fp);
        throw std::runtime_error(
            std::string("label file count mismatch: expected ") +
            std::to_string(expected_n) + ", got " + std::to_string(n));
    }
    std::vector<int> labels(n);
    if (fread(labels.data(), sizeof(int), n, fp) != (size_t)n) {
        fclose(fp);
        throw std::runtime_error(std::string("file truncated: ") + filename);
    }
    fclose(fp);
    return labels;
}

std::pair<std::vector<int>, std::pair<int, int>> read_groundtruth(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        throw std::runtime_error(std::string("cannot open file: ") + filename);
    }
    int nq, k;
    if (fread(&nq, sizeof(int), 1, fp) != 1 ||
        fread(&k, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        throw std::runtime_error(std::string("failed to read header from: ") + filename);
    }
    std::vector<int> ids((size_t)nq * k);
    if (fread(ids.data(), sizeof(int), nq * k, fp) != (size_t)(nq * k)) {
        fclose(fp);
        throw std::runtime_error(std::string("file truncated: ") + filename);
    }
    fclose(fp);
    return {std::move(ids), {nq, k}};
}

} // namespace acorn
