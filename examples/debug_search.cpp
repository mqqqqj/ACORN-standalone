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
        "  --ef <int>         efSearch (default: 64)\n",
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
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
            usage(argv[0]);
        else
            { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); }
    }

    if (!base_file || !label_file || !query_file)
        usage(argv[0]);

    printf("=== ACORN Debug Search ===\n");
    printf("Params: n=%d, nq=%d, k=%d, M=%d, gamma=%d, efc=%d, ef=%d\n\n",
           n, nq, k, M, gamma, efConstruction, ef);

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

    // Build index from first n vectors
    printf("\nBuilding ACORN index (n=%d) ...\n", n);
    acorn::ACORN idx(d, M, gamma, labels, M, acorn::METRIC_L2);
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

    // Search
    int metric = (idx.metric_type == acorn::METRIC_INNER_PRODUCT) ? 0 : 1;
    std::vector<int> labs(k * nq);
    std::vector<float> dists(k * nq);
    t0 = get_ms();
    for (int i = 0; i < nq; i++)
        idx.search(queries.data() + i * d, idx.get_xb(), d, metric,
                   k, ef, labs.data() + i * k, dists.data() + i * k);
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
