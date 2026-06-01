#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <sys/time.h>

#include "acorn/distance.h"
#include "acorn/file_io.h"

static double get_ms()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s --base <base.fbin> --query <query.fbin> --output <gt.ibin>\n"
        "\n"
        "Compute brute-force L2 ground truth (no filter).\n"
        "\n"
        "Required:\n"
        "  --base <path>     Base vectors in fbin format\n"
        "  --query <path>    Query vectors in fbin format\n"
        "  --output <path>   Output ground truth file (ibin: nq, k, nq*k ids)\n"
        "\n"
        "Options:\n"
        "  --k <int>         Number of nearest neighbors (default: 100)\n"
        "  --nq <int>        Max queries to compute (default: all)\n",
        prog);
    exit(1);
}

int main(int argc, char *argv[])
{
    const char *base_file = NULL;
    const char *query_file = NULL;
    const char *output_file = NULL;
    int k = 100;
    int num_queries = -1;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--base") == 0 && i + 1 < argc)
            base_file = argv[++i];
        else if (strcmp(argv[i], "--query") == 0 && i + 1 < argc)
            query_file = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output_file = argv[++i];
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc)
            k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--nq") == 0 && i + 1 < argc)
            num_queries = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
            usage(argv[0]);
        else
            { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); }
    }

    if (!base_file || !query_file || !output_file)
        usage(argv[0]);

    printf("=== Ground Truth Computation ===\n\n");

    printf("Loading base: %s\n", base_file);
    double t0 = get_ms();
    auto br = acorn::read_fbin(base_file);
    std::vector<float> xb = std::move(br.first);
    int n = br.second.first, d = br.second.second;
    printf("  n=%d, d=%d (%.0f ms)\n", n, d, get_ms() - t0);

    printf("Loading queries: %s\n", query_file);
    t0 = get_ms();
    auto qr = acorn::read_fbin(query_file);
    std::vector<float> queries = std::move(qr.first);
    int nq_all = qr.second.first, qd = qr.second.second;
    printf("  nq=%d, d=%d (%.0f ms)\n", nq_all, qd, get_ms() - t0);

    if (qd != d) { fprintf(stderr, "Error: dim mismatch base=%d query=%d\n", d, qd); return 1; }
    if (num_queries > 0 && num_queries < nq_all)
    {
        nq_all = num_queries;
        printf("  Using first %d queries\n", nq_all);
    }

    printf("\nComputing brute-force L2 top-%d ...\n", k);
    std::vector<int> gt_ids(nq_all * k);
    double t_gt = -get_ms();

#pragma omp parallel for schedule(dynamic)
    for (int q = 0; q < nq_all; q++)
    {
        const float *qvec = queries.data() + q * d;
        std::vector<std::pair<float, int>> dists(n);
        for (int i = 0; i < n; i++)
            dists[i] = {acorn::fvec_L2sqr(qvec, xb.data() + i * d, d), i};
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        for (int j = 0; j < k; j++)
            gt_ids[q * k + j] = dists[j].second;
        if (q % 500 == 0) { printf("\r  %d / %d queries done", q, nq_all); fflush(stdout); }
    }
    printf("\r  %d / %d queries done\n", nq_all, nq_all);
    t_gt += get_ms();
    printf("  Time: %.1f ms (%.2f sec)\n", t_gt, t_gt / 1000.0);

    FILE *fp = fopen(output_file, "wb");
    if (!fp) { fprintf(stderr, "Error: cannot open %s\n", output_file); return 1; }
    fwrite(&nq_all, sizeof(int), 1, fp);
    fwrite(&k, sizeof(int), 1, fp);
    fwrite(gt_ids.data(), sizeof(int), gt_ids.size(), fp);
    fclose(fp);

    printf("Ground truth saved to %s\n", output_file);
    return 0;
}
