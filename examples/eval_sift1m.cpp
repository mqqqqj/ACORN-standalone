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

static std::vector<int> load_labels(const char *filename, int expected_n)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "Error: cannot open %s\n", filename);
        exit(1);
    }
    int n;
    fread(&n, sizeof(int), 1, fp);
    if (n != expected_n)
    {
        fprintf(stderr, "Error: label file has n=%d, expected %d\n", n, expected_n);
        exit(1);
    }
    std::vector<int> labels(n);
    fread(labels.data(), sizeof(int), n, fp);
    fclose(fp);
    return labels;
}

static std::pair<std::vector<int>, std::pair<int, int>> load_groundtruth(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "Error: cannot open %s\n", filename);
        exit(1);
    }
    int nq, k;
    fread(&nq, sizeof(int), 1, fp);
    fread(&k, sizeof(int), 1, fp);
    std::vector<int> gt_ids(nq * k);
    fread(gt_ids.data(), sizeof(int), nq * k, fp);
    fclose(fp);

    printf("Loaded ground truth: nq=%d, k=%d\n", nq, k);
    printf("  First 5 IDs for query 0: %d %d %d %d %d\n",
           gt_ids[0], gt_ids[1], gt_ids[2], gt_ids[3], gt_ids[4]);
    return {std::move(gt_ids), {nq, k}};
}

static double compute_recall(int k, const std::vector<int> &gt_ids,
                             const std::vector<acorn::idx_t> &results)
{
    int nq = gt_ids.size() / k;
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
            {
                total_hits++;
            }
        }
    }
    return (double)total_hits / (nq * k);
}

int main(int argc, char *argv[])
{
    const char *base_file = "/dataset/SIFT1M/sift_base.fbin";
    const char *query_file = "/dataset/SIFT1M/sift_query.fbin";
    const char *label_file = "labels.ibin";
    const char *gt_file = "sift1m_gt_top100.ibin";
    const char *index_file = "acorn_sift1m.index";
    int k = 100;
    int M = 32;
    int gamma = 12;
    int efConstruction = 200;
    bool skip_build = false;
    int num_queries = -1;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--base") == 0 && i + 1 < argc)
            base_file = argv[++i];
        else if (strcmp(argv[i], "--query") == 0 && i + 1 < argc)
            query_file = argv[++i];
        else if (strcmp(argv[i], "--labels") == 0 && i + 1 < argc)
            label_file = argv[++i];
        else if (strcmp(argv[i], "--gt") == 0 && i + 1 < argc)
            gt_file = argv[++i];
        else if (strcmp(argv[i], "--index") == 0 && i + 1 < argc)
            index_file = argv[++i];
        else if (strcmp(argv[i], "--M") == 0 && i + 1 < argc)
            M = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gamma") == 0 && i + 1 < argc)
            gamma = atoi(argv[++i]);
        else if (strcmp(argv[i], "--efc") == 0 && i + 1 < argc)
            efConstruction = atoi(argv[++i]);
        else if (strcmp(argv[i], "--skip-build") == 0)
            skip_build = true;
        else if (strcmp(argv[i], "--nq") == 0 && i + 1 < argc)
            num_queries = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0)
        {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --base FILE     base fbin file\n");
            printf("  --query FILE    query fbin file\n");
            printf("  --labels FILE   pre-computed label file\n");
            printf("  --gt FILE       pre-computed ground truth file\n");
            printf("  --index FILE    index save/load file\n");
            printf("  --M INT         graph degree (default: 32)\n");
            printf("  --gamma INT     pruning gamma (default: 12)\n");
            printf("  --efc INT       efConstruction (default: 200)\n");
            printf("  --skip-build    load index from disk instead of building\n");
            printf("  --nq INT        limit number of queries (default: all)\n");
            return 0;
        }
    }

    printf("=== SIFT1M ACORN Evaluation (single-threaded search) ===\n\n");
    printf("Parameters: M=%d, gamma=%d, efConstruction=%d\n", M, gamma, efConstruction);

    // 1. Load data
    printf("\nLoading base vectors from %s ...\n", base_file);
    double t0 = get_ms();
    auto base_result = acorn::read_fbin(base_file);
    std::vector<float> xb = std::move(base_result.first);
    int n = base_result.second.first;
    int d = base_result.second.second;
    printf("  Loaded: n=%d, d=%d (%.1f ms)\n", n, d, get_ms() - t0);

    printf("Loading query vectors from %s ...\n", query_file);
    t0 = get_ms();
    auto query_result = acorn::read_fbin(query_file);
    std::vector<float> queries = std::move(query_result.first);
    int nq_all = query_result.second.first;
    int qd = query_result.second.second;
    printf("  Loaded: nq=%d, d=%d (%.1f ms)\n", nq_all, qd, get_ms() - t0);

    if (qd != d)
    {
        fprintf(stderr, "Error: query dim %d != base dim %d\n", qd, d);
        return 1;
    }

    if (num_queries > 0 && num_queries < nq_all)
    {
        nq_all = num_queries;
        printf("  Using first %d queries\n", nq_all);
    }

    // 2. Load pre-computed labels
    printf("\nLoading labels from %s ...\n", label_file);
    t0 = get_ms();
    std::vector<int> metadata = load_labels(label_file, n);
    printf("  Loaded %zu labels (%.1f ms)\n", metadata.size(), get_ms() - t0);

    // 3. Load pre-computed ground truth
    printf("\nLoading ground truth from %s ...\n", gt_file);
    t0 = get_ms();
    std::pair<std::vector<int>, std::pair<int, int>> gt_result = load_groundtruth(gt_file);
    std::vector<int> gt_ids = std::move(gt_result.first);
    int gt_nq = gt_result.second.first;
    int gt_k = gt_result.second.second;
    printf("  Loaded in %.1f ms\n", get_ms() - t0);

    if (gt_k != k)
    {
        printf("  Note: gt k=%d, using k=%d for search\n", gt_k, k);
    }
    if (nq_all > gt_nq)
    {
        nq_all = gt_nq;
        printf("  Limited queries to gt_nq=%d\n", gt_nq);
    }

    // 4. Build or load index
    printf("\n--- Index ---\n");
    acorn::IndexACORNFlat *index_ptr = nullptr;

    if (skip_build)
    {
        printf("Loading index from %s ...\n", index_file);
        index_ptr = new acorn::IndexACORNFlat();
        t0 = get_ms();
        index_ptr->load(index_file);
        printf("  Loaded in %.1f ms\n", get_ms() - t0);
        printf("  ntotal=%ld, d=%d, M=%d\n",
               index_ptr->ntotal, index_ptr->d, index_ptr->acorn.M);
    }
    else
    {
        printf("Building ACORN index (M=%d, gamma=%d, efConstruction=%d)...\n",
               M, gamma, efConstruction);
        index_ptr = new acorn::IndexACORNFlat(d, M, gamma, metadata, /*M_beta=*/M,
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

    // 5. Single-threaded search
    omp_set_num_threads(1);
    printf("\n--- Search Evaluation (1 thread) ---\n");
    printf("%-10s %-12s %-12s %-14s\n",
           "efSearch", "Recall@100", "Avg ms/q", "QPS");
    printf("------------------------------------------------------\n");

    std::vector<int> ef_values = {16, 32, 64, 128, 256, 512};

    for (int ef : ef_values)
    {
        acorn::SearchParametersACORN params;
        params.efSearch = ef;

        int search_k = (k <= gt_k) ? k : gt_k;
        std::vector<acorn::idx_t> labels(search_k * nq_all);
        std::vector<float> distances(search_k * nq_all);

        // Warmup: one query
        index_ptr->search(1, queries.data(), search_k, distances.data(), labels.data(), &params);

        // Timed search
        double t_search = -get_ms();
        index_ptr->search(nq_all, queries.data(), search_k, distances.data(), labels.data(), &params);
        t_search += get_ms();

        double recall = compute_recall(search_k, gt_ids, labels);
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
