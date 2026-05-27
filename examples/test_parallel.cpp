#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <sys/time.h>

#include "acorn/index_acorn.h"
#include "acorn/file_io.h"

static double get_ms()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static double compute_recall(int nq, int k, const std::vector<int> &gt_ids,
                             const std::vector<acorn::idx_t> &results)
{
    int total_hits = 0;
    for (int q = 0; q < nq; q++)
    {
        std::vector<int> gt_set(gt_ids.begin() + q * k, gt_ids.begin() + (q + 1) * k);
        std::sort(gt_set.begin(), gt_set.end());
        for (int j = 0; j < k; j++)
        {
            int rid = results[q * k + j];
            if (rid == -1)
                continue;
            if (std::binary_search(gt_set.begin(), gt_set.end(), (int)rid))
                total_hits++;
        }
    }
    return (double)total_hits / (nq * k);
}

int main(int argc, char *argv[])
{
    const char *query_file = "/dataset/SIFT1M/sift_query.fbin";
    const char *label_file = "labels.ibin";
    const char *gt_file = "sift1m_gt_top100.ibin";
    const char *index_file = "acorn_sift1m.index";
    int k = 100;
    int ef = 256;
    int num_threads = 2;
    int num_queries = 10000;
    int efs = 128;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--query") == 0 && i + 1 < argc)
            query_file = argv[++i];
        else if (strcmp(argv[i], "--ef") == 0 && i + 1 < argc)
            ef = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            num_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--nq") == 0 && i + 1 < argc)
            num_queries = atoi(argv[++i]);
        else if (strcmp(argv[i], "--efs") == 0 && i + 1 < argc)
            efs = atoi(argv[++i]);
    }

    printf("=== ACORN Serial vs Parallel Search Test ===\n");
    printf("ef=%d, threads=%d, efs=%d, nq=%d\n\n", ef, num_threads, efs, num_queries);

    // Load index
    printf("Loading index from %s ...\n", index_file);
    double t0 = get_ms();
    acorn::IndexACORN index;
    index.load(index_file);
    printf("  ntotal=%ld, d=%d, M=%d (%.0f ms)\n", index.ntotal, index.d, index.acorn.M, get_ms() - t0);

    // Load queries
    printf("Loading queries from %s ...\n", query_file);
    t0 = get_ms();
    auto qr = acorn::read_fbin(query_file);
    std::vector<float> queries = std::move(qr.first);
    int nq_all = qr.second.first, qd = qr.second.second;
    printf("  nq=%d, d=%d (%.0f ms)\n", nq_all, qd, get_ms() - t0);

    if (num_queries > nq_all)
        num_queries = nq_all;

    // Load labels (needed for index metadata matching)
    printf("Loading labels from %s ...\n", label_file);
    std::vector<int> metadata = acorn::read_ibin(label_file, index.ntotal);
    printf("  %zu labels loaded\n", metadata.size());

    // Load groundtruth
    printf("Loading groundtruth from %s ...\n", gt_file);
    auto gt_result = acorn::read_groundtruth(gt_file);
    std::vector<int> gt_ids = std::move(gt_result.first);
    int gt_k = gt_result.second.second;
    printf("  gt nq=%d, k=%d\n", (int)gt_ids.size() / gt_k, gt_k);

    int search_k = (k <= gt_k) ? k : gt_k;

    acorn::SearchParametersACORN params;
    params.efSearch = ef;

    // --- Serial Search ---
    printf("\n--- Serial Search (%d queries) ---\n", num_queries);
    std::vector<acorn::idx_t> serial_labels(search_k * num_queries);
    std::vector<float> serial_dist(search_k * num_queries);

    t0 = get_ms();
    for (int i = 0; i < num_queries; i++)
    {
        index.search(1, queries.data() + i * qd, search_k,
                     serial_dist.data() + i * search_k,
                     serial_labels.data() + i * search_k,
                     &params);
        if ((i + 1) % 100 == 0)
            printf("  serial %d/%d (%.0f ms)\n", i + 1, num_queries, get_ms() - t0);
    }
    double serial_time = get_ms() - t0;
    double serial_recall = compute_recall(num_queries, search_k, gt_ids, serial_labels);
    printf("  Time: %.1f ms, Recall@%d: %.4f\n", serial_time, k, serial_recall);

    // --- Parallel Search ---
    printf("\n--- Parallel Search (threads=%d, efs=%d) ---\n", num_threads, efs);
    std::vector<acorn::idx_t> parallel_labels(search_k * num_queries);
    std::vector<float> parallel_dist(search_k * num_queries);

    t0 = get_ms();
    for (int i = 0; i < num_queries; i++)
    {
        index.parallelSearch(1, queries.data() + i * qd, search_k,
                             parallel_dist.data() + i * search_k,
                             parallel_labels.data() + i * search_k,
                             num_threads, efs, &params);
        if ((i + 1) % 10 == 0)
            printf("  parallel %d/%d (%.0f ms)\n", i + 1, num_queries, get_ms() - t0);
    }
    double parallel_time = get_ms() - t0;
    double parallel_recall = compute_recall(num_queries, search_k, gt_ids, parallel_labels);
    printf("  Time: %.1f ms, Recall@%d: %.4f\n", parallel_time, k, parallel_recall);

    // --- Result Comparison ---
    printf("\n--- Result Comparison (first 5 queries, top-5) ---\n");
    for (int q = 0; q < 5 && q < num_queries; q++)
    {
        printf("Query %d:\n", q);
        printf("  Serial:  ");
        for (int j = 0; j < 5; j++)
            printf("[%ld,%.4f] ", serial_labels[q * search_k + j], serial_dist[q * search_k + j]);
        printf("\n  Parallel: ");
        for (int j = 0; j < 5; j++)
            printf("[%ld,%.4f] ", parallel_labels[q * search_k + j], parallel_dist[q * search_k + j]);

        // Count overlap
        int overlap = 0;
        std::vector<acorn::idx_t> s, p;
        for (int j = 0; j < search_k; j++)
            s.push_back(serial_labels[q * search_k + j]);
        for (int j = 0; j < search_k; j++)
            p.push_back(parallel_labels[q * search_k + j]);
        std::sort(s.begin(), s.end());
        std::sort(p.begin(), p.end());
        for (int j = 0; j < search_k; j++)
        {
            if (std::binary_search(p.begin(), p.end(), s[j]))
                overlap++;
        }
        printf("\n  Overlap: %d/%d\n", overlap, search_k);
    }

    printf("\n=== Summary ===\n");
    printf("Serial:   time=%.1f ms, recall=%.4f\n", serial_time, serial_recall);
    printf("Parallel: time=%.1f ms, recall=%.4f, speedup=%.2fx\n",
           parallel_time, parallel_recall, serial_time / parallel_time);

    return 0;
}
