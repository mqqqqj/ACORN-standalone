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
    const char *label_file = "../data/labels.ibin";
    const char *qlabel_file = "../data/query_labels.ibin";
    const char *gt_file = "../data/sift1m_gt_filtered.ibin";
    const char *index_file = "../data/acorn_sift1m.index";
    int k = 100;
    int ef = 200;
    int num_threads = 2;
    int num_queries = 1000;
    int efs = 100;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--ef") == 0 && i + 1 < argc)
            ef = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            num_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--nq") == 0 && i + 1 < argc)
            num_queries = atoi(argv[++i]);
        else if (strcmp(argv[i], "--efs") == 0 && i + 1 < argc)
            efs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc)
            k = atoi(argv[++i]);
    }

    printf("=== ACORN Filtered Search Test ===\n");
    printf("ef=%d, threads=%d, efs=%d, nq=%d, k=%d\n\n",
           ef, num_threads, efs, num_queries, k);

    // Load index
    printf("Loading index from %s ...\n", index_file);
    double t0 = get_ms();
    acorn::IndexACORN index;
    index.load(index_file);
    printf("  ntotal=%ld, d=%d, M=%d (%.0f ms)\n",
           index.ntotal, index.d, index.acorn.M, get_ms() - t0);

    // Load queries
    printf("Loading queries from %s ...\n", query_file);
    t0 = get_ms();
    auto qr = acorn::read_fbin(query_file);
    std::vector<float> queries = std::move(qr.first);
    int nq_all = qr.second.first, qd = qr.second.second;
    printf("  nq=%d, d=%d (%.0f ms)\n", nq_all, qd, get_ms() - t0);
    if (num_queries <= 0 || num_queries > nq_all)
        num_queries = nq_all;

    // Load base labels
    printf("Loading base labels from %s ...\n", label_file);
    std::vector<int> base_labels = acorn::read_ibin(label_file, index.ntotal);
    printf("  %zu labels loaded\n", base_labels.size());

    // Load query labels
    printf("Loading query labels from %s ...\n", qlabel_file);
    std::vector<int> query_labels = acorn::read_ibin(qlabel_file, nq_all);
    printf("  %zu labels loaded\n", query_labels.size());

    // Load filtered GT
    printf("Loading filtered GT from %s ...\n", gt_file);
    auto gt_result = acorn::read_groundtruth(gt_file);
    std::vector<int> gt_ids = std::move(gt_result.first);
    int gt_k = gt_result.second.second;
    printf("  gt nq=%d, k=%d\n", (int)gt_ids.size() / gt_k, gt_k);

    int search_k = (k <= gt_k) ? k : gt_k;

    // Pre-index base labels for fast filter-map generation
    std::vector<std::vector<int>> label_to_ids(13);
    for (int i = 0; i < (int)index.ntotal; i++)
        label_to_ids[base_labels[i]].push_back(i);

    acorn::SearchParametersACORN params;
    params.efSearch = ef;

    // Warmup
    printf("\n--- Warmup (100 queries, filtered) ---\n");
    {
        std::vector<acorn::idx_t> dummy_labels(search_k * 100);
        std::vector<float> dummy_dist(search_k * 100);
        int ql0 = query_labels[0];
        std::vector<char> wf(index.ntotal, 0);
        for (int id : label_to_ids[ql0])
            wf[id] = 1;
        for (int i = 0; i < 100; i++)
        {
            index.search(1, queries.data() + i * qd, search_k,
                         dummy_dist.data() + i * search_k,
                         dummy_labels.data() + i * search_k,
                         wf.data(), &params);
        }
    }
    printf("  Done.\n");

    // --- Filtered Search ---
    printf("\n--- Filtered Search (%d queries, ef=%d) ---\n", num_queries, ef);
    std::vector<acorn::idx_t> labels(search_k * num_queries);
    std::vector<float> distances(search_k * num_queries);

    t0 = get_ms();
    for (int i = 0; i < num_queries; i++)
    {
        int ql = query_labels[i];
        std::vector<char> filter_map(index.ntotal, 0);
        for (int id : label_to_ids[ql])
            filter_map[id] = 1;

        index.search(1, queries.data() + i * qd, search_k,
                     distances.data() + i * search_k,
                     labels.data() + i * search_k,
                     filter_map.data(), &params);
    }
    double search_time = get_ms() - t0;

    double recall = compute_recall(num_queries, search_k, gt_ids, labels);
    printf("  Time: %.1f ms, Avg: %.3f ms/q\n", search_time, search_time / num_queries);
    printf("  Recall@%d: %.4f\n", k, recall);

    // Filter compliance
    int pass = 0, total = num_queries * search_k;
    for (int i = 0; i < num_queries; i++)
    {
        int ql = query_labels[i];
        for (int j = 0; j < search_k; j++)
        {
            int id = labels[i * search_k + j];
            if (id >= 0 && base_labels[id] == ql)
                pass++;
        }
    }
    printf("  Filter compliance: %d/%d (%.1f%%)\n", pass, total, 100.0 * pass / total);

    // --- Parallel Filtered Search ---
    printf("\n--- Parallel Filtered Search (%d queries, threads=%d, efs=%d) ---\n",
           num_queries, num_threads, efs);
    std::vector<acorn::idx_t> par_labels(search_k * num_queries);
    std::vector<float> par_dist(search_k * num_queries);

    // Build full filter_id_map array: num_queries * ntotal
    std::vector<char> all_filter_maps(num_queries * index.ntotal, 0);
    for (int i = 0; i < num_queries; i++)
    {
        int ql = query_labels[i];
        char *fm = all_filter_maps.data() + i * index.ntotal;
        for (int id : label_to_ids[ql])
            fm[id] = 1;
    }

    t0 = get_ms();
    index.parallelSearch(num_queries, queries.data(), search_k,
                         par_dist.data(), par_labels.data(),
                         all_filter_maps.data(),
                         num_threads, efs, &params);
    double par_time = get_ms() - t0;

    double par_recall = compute_recall(num_queries, search_k, gt_ids, par_labels);
    printf("  Time: %.1f ms, Avg: %.3f ms/q\n", par_time, par_time / num_queries);
    printf("  Recall@%d: %.4f\n", k, par_recall);

    // Filter compliance
    pass = 0;
    for (int i = 0; i < num_queries; i++)
    {
        int ql = query_labels[i];
        for (int j = 0; j < search_k; j++)
        {
            int id = par_labels[i * search_k + j];
            if (id >= 0 && base_labels[id] == ql)
                pass++;
        }
    }
    printf("  Filter compliance: %d/%d (%.1f%%)\n", pass, total, 100.0 * pass / total);

    // Debug: check result distribution
    {
        int n_neg1 = 0, n_wrong = 0, n_ok = 0;
        for (int i = 0; i < num_queries; i++) {
            int ql = query_labels[i];
            for (int j = 0; j < search_k; j++) {
                int id = par_labels[i * search_k + j];
                if (id < 0) n_neg1++;
                else if (base_labels[id] != ql) n_wrong++;
                else n_ok++;
            }
        }
        printf("  Result breakdown: ok=%d -1=%d wrong_label=%d\n", n_ok, n_neg1, n_wrong);
    }

    printf("\n=== Summary ===\n");
    printf("Serial (filtered):   time=%.1f ms, recall=%.4f\n", search_time, recall);
    printf("Parallel (filtered): time=%.1f ms, recall=%.4f, speedup=%.2fx\n",
           par_time, par_recall, search_time / par_time);

    return 0;
}
