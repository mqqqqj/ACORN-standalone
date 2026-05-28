#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <sys/time.h>

#include "acorn/index_acorn.h"
#include "acorn/file_io.h"

static double get_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static double compute_recall(int nq, int k, const std::vector<int>& gt_ids,
                             const std::vector<int>& results) {
    int total_hits = 0;
    for (int q = 0; q < nq; q++) {
        std::vector<int> gt_set(gt_ids.begin() + q * k, gt_ids.begin() + (q + 1) * k);
        std::sort(gt_set.begin(), gt_set.end());
        for (int j = 0; j < k; j++) {
            int rid = results[q * k + j];
            if (rid < 0) continue;
            if (std::binary_search(gt_set.begin(), gt_set.end(), rid))
                total_hits++;
        }
    }
    return (double)total_hits / (nq * k);
}

int main(int argc, char* argv[]) {
    const char* base_file  = "/dataset/SIFT1M/sift_base.fbin";
    const char* query_file = "/dataset/SIFT1M/sift_query.fbin";
    const char* label_file = "../data/labels.ibin";
    const char* qlabel_file = "../data/query_labels.ibin";
    const char* gt_file    = "../data/sift1m_gt_filtered.ibin";
    const char* index_file = "../data/acorn_sift1m.index";
    int k = 100, M = 32, gamma = 12, efConstruction = 200;
    bool skip_build = true;
    int num_queries = -1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--base") == 0 && i + 1 < argc)       base_file = argv[++i];
        else if (strcmp(argv[i], "--query") == 0 && i + 1 < argc) query_file = argv[++i];
        else if (strcmp(argv[i], "--labels") == 0 && i + 1 < argc) label_file = argv[++i];
        else if (strcmp(argv[i], "--qlabels") == 0 && i + 1 < argc) qlabel_file = argv[++i];
        else if (strcmp(argv[i], "--gt") == 0 && i + 1 < argc)    gt_file = argv[++i];
        else if (strcmp(argv[i], "--index") == 0 && i + 1 < argc) index_file = argv[++i];
        else if (strcmp(argv[i], "--M") == 0 && i + 1 < argc)     M = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gamma") == 0 && i + 1 < argc) gamma = atoi(argv[++i]);
        else if (strcmp(argv[i], "--efc") == 0 && i + 1 < argc)   efConstruction = atoi(argv[++i]);
        else if (strcmp(argv[i], "--skip-build") == 0)             skip_build = true;
        else if (strcmp(argv[i], "--build") == 0)                  skip_build = false;
        else if (strcmp(argv[i], "--nq") == 0 && i + 1 < argc)    num_queries = atoi(argv[++i]);
    }

    printf("=== SIFT1M ACORN Evaluation ===\n\n");
    printf("Parameters: M=%d, gamma=%d, efConstruction=%d, skip_build=%d\n",
           M, gamma, efConstruction, skip_build);

    // Load base labels (needed if building index)
    std::vector<int> base_labels;
    {
        auto br = acorn::read_fbin(base_file);
        int n = br.second.first;
        printf("\nLoading base labels from %s ...\n", label_file);
        double t0 = get_ms();
        base_labels = acorn::read_ibin(label_file, n);
        printf("  Loaded %zu labels (%.1f ms)\n", base_labels.size(), get_ms() - t0);
    }

    // Load queries
    printf("Loading query vectors from %s ...\n", query_file);
    double t0 = get_ms();
    auto qr = acorn::read_fbin(query_file);
    std::vector<float> queries = std::move(qr.first);
    int nq_all = qr.second.first, qd = qr.second.second;
    printf("  nq=%d, d=%d (%.1f ms)\n", nq_all, qd, get_ms() - t0);
    if (num_queries > 0 && num_queries < nq_all) nq_all = num_queries;

    // Load query labels (all 10000, we'll use first nq_all)
    printf("Loading query labels from %s ...\n", qlabel_file);
    std::vector<int> query_labels_all = acorn::read_ibin(qlabel_file, -1);
    printf("  %zu labels loaded\n", query_labels_all.size());

    // Load filtered GT
    printf("Loading filtered GT from %s ...\n", gt_file);
    t0 = get_ms();
    auto gt_result = acorn::read_groundtruth(gt_file);
    std::vector<int> gt_ids = std::move(gt_result.first);
    int gt_k = gt_result.second.second;
    printf("  gt nq=%d, k=%d (%.1f ms)\n", (int)gt_ids.size() / gt_k, gt_k, get_ms() - t0);

    int search_k = (k <= gt_k) ? k : gt_k;

    // Pre-index labels
    std::vector<std::vector<int>> label_to_ids(13);
    for (int i = 0; i < (int)base_labels.size(); i++)
        label_to_ids[base_labels[i]].push_back(i);

    // Build or load index
    printf("\n--- Index ---\n");
    acorn::IndexACORN index;
    if (skip_build) {
        printf("Loading index from %s ...\n", index_file);
        t0 = get_ms();
        index.load(index_file);
        printf("  Loaded in %.1f ms\n", get_ms() - t0);
        printf("  ntotal=%ld, d=%d, M=%d\n", index.ntotal, index.d, index.acorn.M);
    } else {
        printf("Building ACORN index (M=%d, gamma=%d, efConstruction=%d)...\n",
               M, gamma, efConstruction);
        auto br = acorn::read_fbin(base_file);
        std::vector<float> xb = std::move(br.first);
        int n = br.second.first, d = br.second.second;
        index = acorn::IndexACORN(d, M, gamma, base_labels, M, acorn::METRIC_L2);
        index.verbose = true;
        index.acorn.efConstruction = efConstruction;
        t0 = get_ms();
        index.add(n, xb.data());
        double t_build = get_ms() - t0;
        printf("  Build time: %.1f ms (%.2f sec)\n", t_build, t_build / 1000.0);
        printf("Saving index to %s ...\n", index_file);
        t0 = get_ms();
        index.save(index_file);
        printf("  Saved in %.1f ms\n", get_ms() - t0);
    }

    // Search evaluation
    omp_set_num_threads(1);
    printf("\n--- Filtered Search Evaluation (1 thread) ---\n");
    printf("%-10s %-12s %-12s %-14s\n", "efSearch", "Recall@100", "Avg ms/q", "QPS");
    printf("------------------------------------------------------\n");

    int ef_vals[] = {16, 32, 64, 128, 256, 512};
    for (int efi = 0; efi < 6; efi++) {
        int ef = ef_vals[efi];
        acorn::SearchParameters params;
        params.efSearch = ef;

        std::vector<int> labels(search_k * nq_all);
        std::vector<float> distances(search_k * nq_all);

        // Warmup: 1 query
        {
            int ql0 = query_labels_all[0];
            std::vector<char> wf(index.ntotal, 0);
            for (int id : label_to_ids[ql0]) wf[id] = 1;
            index.search(1, queries.data(), search_k,
                         distances.data(), labels.data(), wf.data(), &params);
        }

        double t_search = -get_ms();
        for (int i = 0; i < nq_all; i++) {
            int ql = query_labels_all[i];
            std::vector<char> filter_map(index.ntotal, 0);
            for (int id : label_to_ids[ql]) filter_map[id] = 1;
            index.search(1, queries.data() + i * qd, search_k,
                         distances.data() + i * search_k,
                         labels.data() + i * search_k,
                         filter_map.data(), &params);
        }
        t_search += get_ms();

        double recall = compute_recall(nq_all, search_k, gt_ids, labels);
        double avg_ms = t_search / nq_all;
        double qps = nq_all / (t_search / 1000.0);

        printf("%-10d %-12.4f %-12.4f %-14.1f\n",
               ef, recall, avg_ms, qps);
        fflush(stdout);
    }

    printf("\nDone!\n");
    return 0;
}
