#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include "acorn/index_acorn.h"
#include "acorn/file_io.h"

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --base FILE      fbin file for base vectors (build index)\n");
    printf("  --query FILE     fbin file for query vectors\n");
    printf("  --k INT          number of nearest neighbors (default: 10)\n");
    printf("  --M INT          graph degree (default: 32)\n");
    printf("  --gamma INT      pruning gamma (default: 12)\n");
    printf("  --ef INT         search ef (default: 64)\n");
    printf("  --metric INT     0=inner product, 1=L2 (default: 1)\n");
    printf("  --n INT          number of random vectors (default: 10000)\n");
    printf("  --d INT          dimension of random vectors (default: 64)\n");
    printf("  --nq INT         number of random queries (default: 100)\n");
    printf("  --help           show this help\n");
    printf("\nIf --base is not provided, random vectors are generated.\n");
}

int main(int argc, char* argv[]) {
    const char* base_file = nullptr;
    const char* query_file = nullptr;
    int k = 10;
    int M = 32;
    int gamma = 12;
    int ef = 64;
    int metric = 1; // L2
    int n_rand = 10000;
    int d_rand = 64;
    int nq_rand = 100;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--base") == 0 && i + 1 < argc) {
            base_file = argv[++i];
        } else if (strcmp(argv[i], "--query") == 0 && i + 1 < argc) {
            query_file = argv[++i];
        } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--M") == 0 && i + 1 < argc) {
            M = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gamma") == 0 && i + 1 < argc) {
            gamma = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ef") == 0 && i + 1 < argc) {
            ef = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--metric") == 0 && i + 1 < argc) {
            metric = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            n_rand = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--d") == 0 && i + 1 < argc) {
            d_rand = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--nq") == 0 && i + 1 < argc) {
            nq_rand = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    acorn::MetricType metric_type = (metric == 0)
        ? acorn::METRIC_INNER_PRODUCT : acorn::METRIC_L2;

    std::vector<float> xb;
    std::vector<float> queries;
    int n, d, nq;

    if (base_file) {
        printf("Loading base vectors from: %s\n", base_file);
        auto result = acorn::read_fbin(base_file);
        xb = std::move(result.first);
        n = result.second.first;
        d = result.second.second;
        printf("  loaded n=%d, d=%d\n", n, d);
    } else {
        n = n_rand;
        d = d_rand;
        printf("Generating %d random base vectors (d=%d)...\n", n, d);
        xb.resize(n * d);
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < n * d; i++) xb[i] = dist(rng);
    }

    if (query_file) {
        printf("Loading query vectors from: %s\n", query_file);
        auto result = acorn::read_fbin(query_file);
        queries = std::move(result.first);
        nq = result.second.first;
        int qd = result.second.second;
        printf("  loaded nq=%d, d=%d\n", nq, qd);
        if (qd != d) {
            fprintf(stderr, "Error: query dimension %d != base dimension %d\n", qd, d);
            return 1;
        }
    } else {
        nq = nq_rand;
        printf("Generating %d random query vectors...\n", nq);
        queries.resize(nq * d);
        std::mt19937 rng(123);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < nq * d; i++) queries[i] = dist(rng);
    }

    // Normalize if inner product
    if (metric_type == acorn::METRIC_INNER_PRODUCT) {
        printf("Normalizing vectors for inner product...\n");
        for (int i = 0; i < n; i++) {
            float norm = 0.0f;
            for (int j = 0; j < d; j++) norm += xb[i * d + j] * xb[i * d + j];
            norm = sqrtf(norm);
            if (norm > 0) {
                for (int j = 0; j < d; j++) xb[i * d + j] /= norm;
            }
        }
        for (int i = 0; i < nq; i++) {
            float norm = 0.0f;
            for (int j = 0; j < d; j++) norm += queries[i * d + j] * queries[i * d + j];
            norm = sqrtf(norm);
            if (norm > 0) {
                for (int j = 0; j < d; j++) queries[i * d + j] /= norm;
            }
        }
    }

    // Generate random metadata
    int num_attr_values = 10;
    std::vector<int> metadata(n);
    std::mt19937 rng_meta(42);
    for (int i = 0; i < n; i++) metadata[i] = rng_meta() % num_attr_values;

    printf("\n=== ACORN Config ===\n");
    printf("  d=%d, M=%d, gamma=%d, metric=%s\n",
           d, M, gamma, metric_type == acorn::METRIC_L2 ? "L2" : "IP");
    printf("  n=%d, nq=%d, k=%d, ef=%d\n", n, nq, k, ef);

    // Build index
    printf("\nBuilding ACORN index...\n");
    acorn::IndexACORN index(d, M, gamma, metadata, /*M_beta=*/M, metric_type);
    index.verbose = true;
    index.add(n, xb.data());
    printf("Index built. ntotal=%ld\n", index.ntotal);

    // Pick a random query label for filtering
    int filter_val = metadata[rng_meta() % n];

    // Search with filter
    printf("\nRunning filtered search (%d queries, k=%d, filter=attr==%d)...\n",
           nq, k, filter_val);
    std::vector<acorn::idx_t> labels(k * nq);
    std::vector<float> distances(k * nq);

    std::vector<char> filter_map(n, 0);
    for (int i = 0; i < n; i++) {
        if (metadata[i] == filter_val) filter_map[i] = 1;
    }

    {
        acorn::SearchParametersACORN params;
        params.efSearch = ef;
        index.search(nq, queries.data(), k, distances.data(), labels.data(),
                     filter_map.data(), &params);
    }

    printf("\nTop-5 results for query 0:\n");
    for (int j = 0; j < 5 && j < k; j++) {
        printf("  %2d: id=%6ld  dist=%.4f  attr=%d\n",
               j, labels[j], distances[j],
               labels[j] >= 0 ? metadata[labels[j]] : -1);
    }

    printf("\nSearch stats:\n");
    printf("  n1=%zu, n2=%zu, n3=%zu, ndis=%zu, nreorder=%zu\n",
           acorn::acorn_stats.n1, acorn::acorn_stats.n2, acorn::acorn_stats.n3,
           acorn::acorn_stats.ndis, acorn::acorn_stats.nreorder);

    printf("\nDone!\n");
    return 0;
}
