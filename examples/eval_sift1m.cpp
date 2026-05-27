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

static double compute_recall(int nq, int k, const std::vector<int> &gt_ids,
                             const std::vector<acorn::idx_t> &results) {
    int total_hits = 0;
    for (int q = 0; q < nq; q++) {
        std::vector<int> gt_set(gt_ids.begin() + q * k, gt_ids.begin() + (q + 1) * k);
        std::sort(gt_set.begin(), gt_set.end());
        for (int j = 0; j < k; j++) {
            int rid = results[q * k + j];
            if (rid == -1) continue;
            if (std::binary_search(gt_set.begin(), gt_set.end(), (int)rid))
                total_hits++;
        }
    }
    return (double)total_hits / (nq * k);
}

int main(int argc, char *argv[]) {
    const char *base_file   = "/dataset/SIFT1M/sift_base.fbin";
    const char *query_file  = "/dataset/SIFT1M/sift_query.fbin";
    const char *label_file  = "../data/labels.ibin";
    const char *qlabel_file = "../data/query_labels.ibin";
    const char *gt_file     = "../data/sift1m_gt_filtered.ibin";
    const char *index_file  = "../data/acorn_sift1m.index";
    int k = 100;
    int M = 32;
    int gamma = 12;
    int efConstruction = 200;
    bool skip_build = false;
    int num_queries = -1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--base") == 0 && i + 1 < argc)   base_file = argv[++i];
        else if (strcmp(argv[i], "--query") == 0 && i + 1 < argc) query_file = argv[++i];
        else if (strcmp(argv[i], "--labels") == 0 && i + 1 < argc) label_file = argv[++i];
        else if (strcmp(argv[i], "--qlabels") == 0 && i + 1 < argc) qlabel_file = argv[++i];
        else if (strcmp(argv[i], "--gt") == 0 && i + 1 < argc) gt_file = argv[++i];
        else if (strcmp(argv[i], "--index") == 0 && i + 1 < argc) index_file = argv[++i];
        else if (strcmp(argv[i], "--M") == 0 && i + 1 < argc) M = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gamma") == 0 && i + 1 < argc) gamma = atoi(argv[++i]);
        else if (strcmp(argv[i], "--efc") == 0 && i + 1 < argc) efConstruction = atoi(argv[++i]);
        else if (strcmp(argv[i], "--skip-build") == 0) skip_build = true;
        else if (strcmp(argv[i], "--nq") == 0 && i + 1 < argc) num_queries = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --base FILE     base fbin file\n");
            printf("  --query FILE    query fbin file\n");
            printf("  --labels FILE   base label file\n");
            printf("  --qlabels FILE  query label file\n");
            printf("  --gt FILE       filtered ground truth file\n");
            printf("  --index FILE    index save/load file\n");
            printf("  --M INT         graph degree (default: 32)\n");
            printf("  --gamma INT     pruning gamma (default: 12)\n");
            printf("  --efc INT       efConstruction (default: 200)\n");
            printf("  --skip-build    load index from disk\n");
            printf("  --nq INT        limit number of queries\n");
            return 0;
        }
    }

    printf("=== SIFT1M ACORN Filtered Search Evaluation ===\n\n");
    printf("Parameters: M=%d, gamma=%d, efConstruction=%d\n", M, gamma, efConstruction);

    // 1. Load data
    printf("\nLoading base vectors from %s ...\n", base_file);
    double t0 = get_ms();
    auto base_result = acorn::read_fbin(base_file);
    std::vector<float> xb = std::move(base_result.first);
    int n = base_result.second.first, d = base_result.second.second;
    printf("  Loaded: n=%d, d=%d (%.1f ms)\n", n, d, get_ms() - t0);

    printf("Loading query vectors from %s ...\n", query_file);
    t0 = get_ms();
    auto query_result = acorn::read_fbin(query_file);
    std::vector<float> queries = std::move(query_result.first);
    int nq_all = query_result.second.first, qd = query_result.second.second;
    printf("  Loaded: nq=%d, d=%d (%.1f ms)\n", nq_all, qd, get_ms() - t0);
    if (qd != d) { fprintf(stderr, "Dim mismatch\n"); return 1; }
    if (num_queries > 0 && num_queries < nq_all) {
        nq_all = num_queries;
        printf("  Using first %d queries\n", nq_all);
    }

    // 2. Load labels
    printf("\nLoading base labels from %s ...\n", label_file);
    t0 = get_ms();
    std::vector<int> base_labels = acorn::read_ibin(label_file, n);
    printf("  Loaded %zu labels (%.1f ms)\n", base_labels.size(), get_ms() - t0);

    printf("Loading query labels from %s ...\n", qlabel_file);
    std::vector<int> query_labels = acorn::read_ibin(qlabel_file, nq_all);
    printf("  Loaded %zu labels\n", query_labels.size());

    // 3. Load filtered ground truth
    printf("\nLoading filtered ground truth from %s ...\n", gt_file);
    t0 = get_ms();
    std::pair<std::vector<int>, std::pair<int, int>> gt_result = acorn::read_groundtruth(gt_file);
    std::vector<int> gt_ids = std::move(gt_result.first);
    int gt_nq = gt_result.second.first, gt_k = gt_result.second.second;
    printf("  Loaded: nq=%d, k=%d (%.1f ms)\n", gt_nq, gt_k, get_ms() - t0);
    if (nq_all > gt_nq) { nq_all = gt_nq; printf("  Limited to gt_nq=%d\n", gt_nq); }

    // Pre-index base labels
    std::vector<std::vector<int>> label_to_ids(13);
    for (int i = 0; i < n; i++) label_to_ids[base_labels[i]].push_back(i);

    // 4. Build or load index
    printf("\n--- Index ---\n");
    acorn::IndexACORN *index_ptr = nullptr;

    if (skip_build) {
        printf("Loading index from %s ...\n", index_file);
        index_ptr = new acorn::IndexACORN();
        t0 = get_ms();
        index_ptr->load(index_file);
        printf("  Loaded in %.1f ms\n", get_ms() - t0);
        printf("  ntotal=%ld, d=%d, M=%d\n",
               index_ptr->ntotal, index_ptr->d, index_ptr->acorn.M);
    } else {
        printf("Building ACORN index (M=%d, gamma=%d, efConstruction=%d)...\n",
               M, gamma, efConstruction);
        index_ptr = new acorn::IndexACORN(d, M, gamma, base_labels, /*M_beta=*/M,
                                          acorn::METRIC_L2);
        index_ptr->verbose = true;
        index_ptr->acorn.efConstruction = efConstruction;

        t0 = get_ms();
        index_ptr->add(n, xb.data());
        double t_build = get_ms() - t0;
        printf("  Build time: %.1f ms (%.2f sec)\n", t_build, t_build / 1000.0);

        printf("Saving index to %s ...\n", index_file);
        t0 = get_ms();
        index_ptr->save(index_file);
        printf("  Saved in %.1f ms\n", get_ms() - t0);
    }

    // 5. Filtered search evaluation
    omp_set_num_threads(1);
    printf("\n--- Filtered Search Evaluation (1 thread) ---\n");
    printf("%-10s %-12s %-12s %-14s\n",
           "efSearch", "Recall@100", "Avg ms/q", "QPS");
    printf("------------------------------------------------------\n");

    int search_k = (k <= gt_k) ? k : gt_k;
    std::vector<int> ef_values = {16, 32, 64, 128, 256, 512};

    for (int ef : ef_values) {
        acorn::SearchParametersACORN params;
        params.efSearch = ef;

        std::vector<acorn::idx_t> labels(search_k * nq_all);
        std::vector<float> distances(search_k * nq_all);

        // Warmup: one query
        {
            int ql0 = query_labels[0];
            std::vector<char> wf(n, 0);
            for (int id : label_to_ids[ql0]) wf[id] = 1;
            index_ptr->search(1, queries.data(), search_k,
                              distances.data(), labels.data(),
                              wf.data(), &params);
        }

        // Timed search
        double t_search = -get_ms();
        for (int i = 0; i < nq_all; i++) {
            int ql = query_labels[i];
            std::vector<char> filter_map(n, 0);
            for (int id : label_to_ids[ql]) filter_map[id] = 1;

            index_ptr->search(1, queries.data() + i * qd, search_k,
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

    delete index_ptr;
    printf("\nDone!\n");
    return 0;
}
