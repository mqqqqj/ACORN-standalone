#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <sys/time.h>

#include "acorn/acorn_graph.h"
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
        "Usage: %s --base <base.fbin> --labels <labels.ibin> --query <query.fbin>\n"
        "       [--gt <gt.ibin>] [--output <index.file>]\n"
        "\n"
        "Quick debug: build a small index and search, report recall.\n"
        "Builds from first N vectors, searches first NQ queries.\n"
        "\n"
        "Required:\n"
        "  --base <path>      Base vectors in fbin format\n"
        "  --labels <path>    Base labels in ibin format\n"
        "  --query <path>     Query vectors in fbin format\n"
        "\n"
        "Options:\n"
        "  --gt <path>        Ground truth in ibin format (for recall)\n"
        "  --output <path>    Save index to file (optional)\n"
        "  --n <int>          Number of base vectors to index (default: 20000)\n"
        "  --nq <int>         Number of queries (default: 100)\n"
        "  --k <int>          Number of results (default: 100)\n"
        "  --M <int>          Max out-degree (default: 16)\n"
        "  --gamma <int>      Level-0 candidate multiplier (default: 8)\n"
        "  --efc <int>        efConstruction (default: 64)\n"
        "  --ef <int>         efSearch (default: 64)\n"
        "  --metric <l2|ip>   Distance metric (default: l2)\n",
        prog);
    exit(1);
}

int main(int argc, char *argv[])
{
    const char *base_file = NULL;
    const char *label_file = NULL;
    const char *query_file = NULL;
    const char *gt_file = NULL;
    const char *output_file = NULL;
    int n = 20000, nq = 100, k = 100;
    int M = 16, gamma = 8, efConstruction = 64, ef = 64;
    acorn::MetricType metric_type = acorn::METRIC_L2;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--base") == 0 && i + 1 < argc)
            base_file = argv[++i];
        else if (strcmp(argv[i], "--labels") == 0 && i + 1 < argc)
            label_file = argv[++i];
        else if (strcmp(argv[i], "--query") == 0 && i + 1 < argc)
            query_file = argv[++i];
        else if (strcmp(argv[i], "--gt") == 0 && i + 1 < argc)
            gt_file = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output_file = argv[++i];
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc)
            n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--nq") == 0 && i + 1 < argc)
            nq = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc)
            k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--M") == 0 && i + 1 < argc)
            M = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gamma") == 0 && i + 1 < argc)
            gamma = atoi(argv[++i]);
        else if (strcmp(argv[i], "--efc") == 0 && i + 1 < argc)
            efConstruction = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ef") == 0 && i + 1 < argc)
            ef = atoi(argv[++i]);
        else if (strcmp(argv[i], "--metric") == 0 && i + 1 < argc)
        {
            const char *m = argv[++i];
            if (strcmp(m, "ip") == 0) metric_type = acorn::METRIC_INNER_PRODUCT;
            else if (strcmp(m, "l2") == 0) metric_type = acorn::METRIC_L2;
            else { fprintf(stderr, "Unknown metric: %s\n", m); return 1; }
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
            usage(argv[0]);
        else
            { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); }
    }

    if (!base_file || !label_file || !query_file)
        usage(argv[0]);

    printf("=== ACORN Debug Search ===\n");
    printf("Params: n=%d, nq=%d, k=%d, M=%d, gamma=%d, efc=%d, ef=%d, metric=%s\n\n",
           n, nq, k, M, gamma, efConstruction, ef,
           metric_type == acorn::METRIC_L2 ? "L2" : "IP");

    // Load base
    printf("Loading base ...\n");
    auto br = acorn::read_fbin(base_file);
    std::vector<float> xb = std::move(br.first);
    int n_all = br.second.first, d = br.second.second;
    printf("  n=%d, d=%d\n", n_all, d);

    // Load labels
    printf("Loading labels ...\n");
    std::vector<int> labels = acorn::read_ibin(label_file, -1);
    printf("  %zu labels\n", labels.size());

    // --- Distance function smoke test ---
    {
        bool use_ip = (metric_type == acorn::METRIC_INNER_PRODUCT);
        int test_n = std::min(n_all, 10);
        int test_d = d;
        printf("\n--- Distance Smoke Test (%s, d=%d, %d vectors) ---\n",
               use_ip ? "IP" : "L2", test_d, test_n);

        // Compute all pairwise distances on the tiny subset
        std::vector<float> results;
        double t_smoke = -get_ms();
        for (int i = 0; i < test_n; i++)
        {
            for (int j = 0; j < test_n; j++)
            {
                float dist;
                if (use_ip)
                    dist = -acorn::fvec_inner_product(xb.data() + i * d, xb.data() + j * d, d);
                else
                    dist = acorn::fvec_L2sqr(xb.data() + i * d, xb.data() + j * d, d);
                results.push_back(dist);
            }
        }
        t_smoke += get_ms();
        printf("  %d distance computations: %.1f ms\n", test_n * test_n, t_smoke);

        // Check for NaN/Inf
        int bad = 0;
        for (size_t i = 0; i < results.size(); i++)
            if (!std::isfinite(results[i])) bad++;
        printf("  NaN/Inf: %d\n", bad);

        // Sample: self-distances (should be 0 for L2, -||v||^2 for IP)
        printf("  Self-distances:");
        for (int i = 0; i < std::min(test_n, 5); i++)
        {
            float sd = results[i * test_n + i];
            printf(" %.4f", sd);
        }
        printf("\n");

        // Sample: cross distances (first vec vs others)
        printf("  vec0 vs others:");
        for (int j = 0; j < std::min(test_n, 5); j++)
            printf(" %.4f", results[j]);
        printf("\n");

        if (bad > 0)
        {
            fprintf(stderr, "ERROR: distance function produced %d NaN/Inf values\n", bad);
            return 1;
        }
        printf("  Distance function OK.\n");
    }

    // Build index from first n vectors
    printf("\nBuilding ACORN index (n=%d) ...\n", n);
    acorn::ACORN idx(d, M, gamma, labels, M, metric_type);
    idx.efConstruction = efConstruction;
    double t0 = get_ms();
    idx.add(n, xb.data());
    printf("  Build: %.0f ms, ntotal=%ld, max_level=%d\n",
           get_ms() - t0, idx.ntotal, idx.max_level);

    if (output_file)
    {
        printf("Saving index to %s ...\n", output_file);
        idx.save(output_file);
        printf("  Saved.\n");
    }

    // Load queries
    printf("\nLoading queries ...\n");
    auto qr = acorn::read_fbin(query_file);
    std::vector<float> queries = std::move(qr.first);
    int nq_all = qr.second.first, qd = qr.second.second;
    printf("  nq=%d, d=%d\n", nq_all, qd);
    if (nq > nq_all) nq = nq_all;
    if (qd != d) { fprintf(stderr, "Error: dim mismatch\n"); return 1; }

    // --- Mini brute-force filtered test (single-threaded) ---
    {
        bool use_ip = (metric_type == acorn::METRIC_INNER_PRODUCT);
        int test_nq = std::min(nq, 10);
        int test_bf_k = std::min(k, 10);
        printf("\n--- Brute-Force Filtered Test (%s, %d queries, k=%d) ---\n",
               use_ip ? "IP" : "L2", test_nq, test_bf_k);

        int max_lbl = 0;
        for (int i = 0; i < n; i++)
            if (labels[i] > max_lbl) max_lbl = labels[i];
        std::vector<std::vector<int>> bf_label_to_ids(max_lbl + 1);
        for (int i = 0; i < n; i++)
            bf_label_to_ids[labels[i]].push_back(i);

        std::vector<int> bf_ql(test_nq);
        for (int i = 0; i < test_nq; i++)
            bf_ql[i] = labels[i % n];

        std::vector<int> bf_gt(test_nq * test_bf_k, -1);
        double t_bf = -get_ms();
        for (int q = 0; q < test_nq; q++)
        {
            int q_label = bf_ql[q];
            if (q_label < 0 || q_label > max_lbl) continue;
            const auto &candidates = bf_label_to_ids[q_label];
            if (candidates.empty()) continue;

            const float *qvec = queries.data() + q * d;
            std::vector<std::pair<float, int>> bf_dists;
            bf_dists.reserve(candidates.size());
            for (int id : candidates)
            {
                float dist = use_ip ? -acorn::fvec_inner_product(qvec, xb.data() + id * d, d)
                                    : acorn::fvec_L2sqr(qvec, xb.data() + id * d, d);
                bf_dists.emplace_back(dist, id);
            }
            int take = std::min(test_bf_k, (int)bf_dists.size());
            std::partial_sort(bf_dists.begin(), bf_dists.begin() + take, bf_dists.end());
            for (int j = 0; j < take; j++)
                bf_gt[q * test_bf_k + j] = bf_dists[j].second;
        }
        t_bf += get_ms();
        printf("  %d queries: %.1f ms", test_nq, t_bf);
        size_t total_cand = 0;
        for (int q = 0; q < test_nq; q++)
            if (bf_ql[q] >= 0 && bf_ql[q] <= max_lbl)
                total_cand += bf_label_to_ids[bf_ql[q]].size();
        printf(", candidates: %zu\n", total_cand);

        printf("  Query 0 (label=%d) top-5:", bf_ql[0]);
        for (int j = 0; j < std::min(5, test_bf_k); j++)
            printf(" id=%d", bf_gt[j]);
        printf("\n");
        printf("  Brute-force filtered test OK.\n");
    }

    // Search (filter: all-pass, since this is a debug tool)
    int metric = (idx.metric_type == acorn::METRIC_INNER_PRODUCT) ? 0 : 1;
    std::vector<char> all_pass(idx.ntotal, 1);
    std::vector<int> labs(k * nq);
    std::vector<float> dists(k * nq);
    t0 = get_ms();
    for (int i = 0; i < nq; i++)
        idx.search(queries.data() + i * d, idx.get_xb(), d, metric,
                   k, ef, labs.data() + i * k, dists.data() + i * k, all_pass.data());
    printf("Search %d queries: %.0f ms, avg %.3f ms/q\n",
           nq, get_ms() - t0, (get_ms() - t0) / nq);

    // Compute recall if GT provided
    if (gt_file)
    {
        printf("\nLoading ground truth ...\n");
        auto gr = acorn::read_groundtruth(gt_file);
        auto gt_ids = std::move(gr.first);
        int gt_k = gr.second.second;
        printf("  nq=%d, k=%d\n", (int)gt_ids.size() / gt_k, gt_k);

        int hits = 0, effective_k = (k <= gt_k) ? k : gt_k;
        for (int i = 0; i < nq; i++)
        {
            std::vector<int> gtset(gt_ids.begin() + i * gt_k,
                                   gt_ids.begin() + i * gt_k + effective_k);
            std::sort(gtset.begin(), gtset.end());
            for (int j = 0; j < effective_k; j++)
            {
                int id = labs[i * k + j];
                if (id >= 0 && std::binary_search(gtset.begin(), gtset.end(), id))
                    hits++;
            }
        }
        printf("Recall@%d: %.4f\n", effective_k, (double)hits / (nq * effective_k));
    }

    return 0;
}
