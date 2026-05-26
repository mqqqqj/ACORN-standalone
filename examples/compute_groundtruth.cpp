#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <sys/time.h>

#include "acorn/distance.h"
#include "acorn/file_io.h"

static double get_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

void brute_force_l2(
    const float* base, int n, int d,
    const float* queries, int nq, int k,
    std::vector<int>& gt_ids)
{
    gt_ids.resize(nq * k);

    #pragma omp parallel for schedule(dynamic)
    for (int q = 0; q < nq; q++) {
        const float* qvec = queries + q * d;
        std::vector<std::pair<float, int>> dists(n);

        for (int i = 0; i < n; i++) {
            dists[i] = {acorn::fvec_L2sqr(qvec, base + i * d, d), i};
        }

        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());

        for (int j = 0; j < k; j++) {
            gt_ids[q * k + j] = dists[j].second;
        }

        if (q % 500 == 0) {
            printf("\r  %d / %d queries done", q, nq);
            fflush(stdout);
        }
    }
    printf("\r  %d / %d queries done\n", nq, nq);
}

int main(int argc, char* argv[]) {
    const char* base_file = "/dataset/SIFT1M/sift_base.fbin";
    const char* query_file = "/dataset/SIFT1M/sift_query.fbin";
    const char* output_file = "groundtruth.ibin";
    int k = 100;
    int num_queries = -1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--base") == 0 && i + 1 < argc) base_file = argv[++i];
        else if (strcmp(argv[i], "--query") == 0 && i + 1 < argc) query_file = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_file = argv[++i];
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--nq") == 0 && i + 1 < argc) num_queries = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --base FILE    base fbin file\n");
            printf("  --query FILE   query fbin file\n");
            printf("  --output FILE  output ground truth file (default: groundtruth.ibin)\n");
            printf("  --k INT        number of nearest neighbors (default: 100)\n");
            printf("  --nq INT       limit number of queries (default: all)\n");
            return 0;
        }
    }

    printf("=== Ground Truth Computation ===\n\n");

    // Load base
    printf("Loading base: %s\n", base_file);
    double t0 = get_ms();
    auto base_res = acorn::read_fbin(base_file);
    std::vector<float> xb = std::move(base_res.first);
    int n = base_res.second.first;
    int d = base_res.second.second;
    printf("  n=%d, d=%d (%.1f ms)\n", n, d, get_ms() - t0);

    // Load queries
    printf("Loading queries: %s\n", query_file);
    t0 = get_ms();
    auto query_res = acorn::read_fbin(query_file);
    std::vector<float> queries = std::move(query_res.first);
    int nq_all = query_res.second.first;
    int qd = query_res.second.second;
    printf("  nq=%d, d=%d (%.1f ms)\n", nq_all, qd, get_ms() - t0);

    if (qd != d) {
        fprintf(stderr, "Error: dim mismatch base=%d query=%d\n", d, qd);
        return 1;
    }

    if (num_queries > 0 && num_queries < nq_all) {
        nq_all = num_queries;
        printf("  Using first %d queries\n", nq_all);
    }

    // Compute
    printf("\nComputing brute-force L2 top-%d...\n", k);
    std::vector<int> gt_ids;
    double t_gt = -get_ms();
    brute_force_l2(xb.data(), n, d, queries.data(), nq_all, k, gt_ids);
    t_gt += get_ms();
    printf("Time: %.1f ms (%.2f sec)\n", t_gt, t_gt / 1000.0);

    // Save: 4B nq, 4B k, then nq*k * 4B IDs
    FILE* fp = fopen(output_file, "wb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s\n", output_file);
        return 1;
    }
    fwrite(&nq_all, sizeof(int), 1, fp);
    fwrite(&k, sizeof(int), 1, fp);
    fwrite(gt_ids.data(), sizeof(int), gt_ids.size(), fp);
    fclose(fp);

    printf("Ground truth saved to %s\n", output_file);
    printf("First 5 IDs for query 0: %d %d %d %d %d\n",
           gt_ids[0], gt_ids[1], gt_ids[2], gt_ids[3], gt_ids[4]);
    return 0;
}
