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

static double compute_recall(int nq, int k, int gt_k,
                             const std::vector<int> &gt_ids,
                             const std::vector<int> &results)
{
    int total_hits = 0;
    for (int q = 0; q < nq; q++)
    {
        std::vector<int> gt_set(gt_ids.begin() + q * gt_k,
                                gt_ids.begin() + (q + 1) * gt_k);
        std::sort(gt_set.begin(), gt_set.end());
        for (int j = 0; j < k; j++)
        {
            int rid = results[q * k + j];
            if (rid < 0)
                continue;
            if (std::binary_search(gt_set.begin(), gt_set.end(), rid))
                total_hits++;
        }
    }
    return (double)total_hits / (nq * k);
}

static void check_filter(int nq, int k,
                         const std::vector<int> &results,
                         const std::vector<int> &query_labels,
                         const std::vector<int> &base_labels,
                         int *out_ok, int *out_empty, int *out_wrong)
{
    int ok = 0, empty = 0, wrong = 0;
    for (int i = 0; i < nq; i++)
    {
        int ql = query_labels[i];
        for (int j = 0; j < k; j++)
        {
            int id = results[i * k + j];
            if (id < 0)
                empty++;
            else if (base_labels[id] != ql)
                wrong++;
            else
                ok++;
        }
    }
    *out_ok = ok;
    *out_empty = empty;
    *out_wrong = wrong;
}

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s --index <index.file> --query <query.fbin> [options]\n"
        "\n"
        "Required:\n"
        "  --index <path>     Pre-built ACORN index\n"
        "  --query <path>     Query vectors in fbin format\n"
        "\n"
        "Filtered search (optional):\n"
        "  --labels <path>    Base labels in ibin format (required for filter)\n"
        "  --qlabels <path>   Query labels in ibin format (required for filter)\n"
        "\n"
        "Evaluation:\n"
        "  --gt <path>        Ground truth in ibin format (nq, k, nq*k ids)\n"
        "\n"
        "Search parameters:\n"
        "  --k <int>          Number of results (default: 100)\n"
        "  --ef <int>         Serial efSearch (default: 200)\n"
        "  --threads <int>    Number of threads for parallel search (default: 4)\n"
        "  --efs <int>        Parallel search pool size (default: 100)\n"
        "  --nq <int>         Max queries to run (default: all)\n"
        "\n"
        "Example:\n"
        "  %s --index acorn.index --query sift_query.fbin --gt gt.ibin\n"
        "  %s --index acorn.index --query sift_query.fbin --labels labels.ibin "
        "--qlabels qlabels.ibin --gt gt_filtered.ibin --k 100 --ef 400 --threads 8 --efs 200\n",
        prog, prog, prog);
    exit(1);
}

int main(int argc, char *argv[])
{
    const char *index_file = NULL;
    const char *query_file = NULL;
    const char *label_file = NULL;
    const char *qlabel_file = NULL;
    const char *gt_file = NULL;
    int k = 100, ef = 200, num_threads = 4, efs = 100;
    int num_queries = -1;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--index") == 0 && i + 1 < argc)
            index_file = argv[++i];
        else if (strcmp(argv[i], "--query") == 0 && i + 1 < argc)
            query_file = argv[++i];
        else if (strcmp(argv[i], "--labels") == 0 && i + 1 < argc)
            label_file = argv[++i];
        else if (strcmp(argv[i], "--qlabels") == 0 && i + 1 < argc)
            qlabel_file = argv[++i];
        else if (strcmp(argv[i], "--gt") == 0 && i + 1 < argc)
            gt_file = argv[++i];
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc)
            k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ef") == 0 && i + 1 < argc)
            ef = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            num_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--efs") == 0 && i + 1 < argc)
            efs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--nq") == 0 && i + 1 < argc)
            num_queries = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
            usage(argv[0]);
        else
            { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); }
    }

    if (!index_file || !query_file)
        usage(argv[0]);

    bool use_filter = (label_file != NULL) && (qlabel_file != NULL);

    printf("=== ACORN Search ===\n");
    printf("Index:  %s\n", index_file);
    printf("Query:  %s\n", query_file);
    printf("Filter: %s\n", use_filter ? "yes" : "no");
    if (gt_file) printf("GT:     %s\n", gt_file);
    printf("Params: k=%d, ef=%d, threads=%d, efs=%d, nq=%d\n\n",
           k, ef, num_threads, efs, num_queries);

    // Load index
    printf("Loading index ...\n");
    double t0 = get_ms();
    acorn::ACORN index;
    index.load(index_file);
    printf("  ntotal=%ld, d=%d, M=%d (%.0f ms)\n",
           index.ntotal, index.d, index.M, get_ms() - t0);

    // Load queries
    printf("Loading queries ...\n");
    t0 = get_ms();
    auto qr = acorn::read_fbin(query_file);
    std::vector<float> queries = std::move(qr.first);
    int nq_all = qr.second.first, qd = qr.second.second;
    printf("  nq=%d, d=%d (%.0f ms)\n", nq_all, qd, get_ms() - t0);
    if (num_queries <= 0 || num_queries > nq_all)
        num_queries = nq_all;
    if (qd != index.d)
        { fprintf(stderr, "Error: query dim %d != index dim %d\n", qd, index.d); return 1; }

    int metric = (index.metric_type == acorn::METRIC_INNER_PRODUCT) ? 0 : 1;

    // Load filter data
    std::vector<int> base_labels, query_labels;
    std::vector<std::vector<int>> label_to_ids;
    if (use_filter)
    {
        printf("Loading base labels ...\n");
        base_labels = acorn::read_ibin(label_file, index.ntotal);
        printf("  %zu labels\n", base_labels.size());

        printf("Loading query labels ...\n");
        query_labels = acorn::read_ibin(qlabel_file, nq_all);
        printf("  %zu labels\n", query_labels.size());

        label_to_ids.resize(13);
        for (int i = 0; i < (int)base_labels.size(); i++)
            label_to_ids[base_labels[i]].push_back(i);
    }

    // Load GT
    std::vector<int> gt_ids;
    int gt_k = 0;
    if (gt_file)
    {
        printf("Loading ground truth ...\n");
        auto gr = acorn::read_groundtruth(gt_file);
        gt_ids = std::move(gr.first);
        gt_k = gr.second.second;
        printf("  nq=%d, k=%d\n", (int)gt_ids.size() / gt_k, gt_k);
    }

    int search_k = (gt_k > 0 && k <= gt_k) ? k : k;

    // Warmup
    printf("\n--- Warmup ---\n");
    {
        std::vector<int> tmp_labels(search_k);
        std::vector<float> tmp_dist(search_k);
        if (use_filter)
        {
            int ql = query_labels[0];
            std::vector<char> wf(index.ntotal, 0);
            for (int id : label_to_ids[ql])
                wf[id] = 1;
            index.search(queries.data(), index.get_xb(), index.d, metric,
                         search_k, ef, tmp_labels.data(), tmp_dist.data(), wf.data());
        }
        else
        {
            index.search(queries.data(), index.get_xb(), index.d, metric,
                         search_k, ef, tmp_labels.data(), tmp_dist.data());
        }
    }
    printf("  Done.\n");

    // --- Serial search ---
    printf("\n--- Serial Search (%d queries, ef=%d) ---\n", num_queries, ef);
    std::vector<int> ser_labels(search_k * num_queries);
    std::vector<float> ser_dist(search_k * num_queries);

    acorn::reset_ser_ndis();
    double ser_time = 0;
    for (int i = 0; i < num_queries; i++)
    {
        const float *q = queries.data() + i * qd;
        int *lbl = ser_labels.data() + i * search_k;
        float *dst = ser_dist.data() + i * search_k;

        double tq = -get_ms();
        if (use_filter)
        {
            int ql = query_labels[i];
            std::vector<char> wf(index.ntotal, 0);
            for (int id : label_to_ids[ql])
                wf[id] = 1;
            index.search(q, index.get_xb(), index.d, metric,
                         search_k, ef, lbl, dst, wf.data());
        }
        else
        {
            index.search(q, index.get_xb(), index.d, metric,
                         search_k, ef, lbl, dst);
        }
        ser_time += tq + get_ms();
    }
    printf("  Time: %.1f ms, Avg: %.3f ms/q\n", ser_time, ser_time / num_queries);
    printf("  NDC:  total=%zu avg/q=%.0f\n",
           acorn::get_ser_ndis(), (double)acorn::get_ser_ndis() / num_queries);

    if (gt_file)
    {
        double r = compute_recall(num_queries, search_k, gt_k, gt_ids, ser_labels);
        printf("  Recall@%d: %.4f\n", k, r);
    }
    if (use_filter)
    {
        int ok, empty, wrong;
        check_filter(num_queries, search_k, ser_labels, query_labels, base_labels,
                     &ok, &empty, &wrong);
        printf("  Filter: ok=%d -1=%d wrong=%d\n", ok, empty, wrong);
    }

    // --- iQAN search ---
    printf("\n--- iQAN Search (%d queries, threads=%d, efs=%d) ---\n",
           num_queries, num_threads, efs);
    std::vector<int> iqan_labels(search_k * num_queries);
    std::vector<float> iqan_dist(search_k * num_queries);

    acorn::reset_thread_ndis(num_threads);
    double iqan_time = 0;
    for (int i = 0; i < num_queries; i++)
    {
        const float *q = queries.data() + i * qd;
        int *lbl = iqan_labels.data() + i * search_k;
        float *dst = iqan_dist.data() + i * search_k;

        double tq = -get_ms();
        if (use_filter)
        {
            int ql = query_labels[i];
            std::vector<char> wf(index.ntotal, 0);
            for (int id : label_to_ids[ql])
                wf[id] = 1;
            index.iqan_search(q, index.get_xb(), index.d, metric,
                              search_k, efs, lbl, dst,
                              num_threads, efs, wf.data());
        }
        else
        {
            index.iqan_search(q, index.get_xb(), index.d, metric,
                              search_k, efs, lbl, dst,
                              num_threads, efs);
        }
        iqan_time += tq + get_ms();
    }
    printf("  Time: %.1f ms, Avg: %.3f ms/q\n", iqan_time, iqan_time / num_queries);
    printf("  Speedup: %.2fx\n", ser_time / iqan_time);
    {
        auto &tnd = acorn::get_thread_ndis();
        printf("  NDC per-thread:");
        size_t sum = 0;
        for (size_t nd : tnd) { printf(" %zu", nd); sum += nd; }
        printf(" (total=%zu)\n", sum);
    }
    if (gt_file)
    {
        double r = compute_recall(num_queries, search_k, gt_k, gt_ids, iqan_labels);
        printf("  Recall@%d: %.4f\n", k, r);
    }
    if (use_filter)
    {
        int ok, empty, wrong;
        check_filter(num_queries, search_k, iqan_labels, query_labels, base_labels,
                     &ok, &empty, &wrong);
        printf("  Filter: ok=%d -1=%d wrong=%d\n", ok, empty, wrong);
    }

    // --- No-sync search ---
    printf("\n--- No-Sync Search (%d queries, threads=%d, efs=%d) ---\n",
           num_queries, num_threads, efs);
    std::vector<int> nosync_labels(search_k * num_queries);
    std::vector<float> nosync_dist(search_k * num_queries);

    acorn::reset_thread_ndis(num_threads);
    double nosync_time = 0;
    for (int i = 0; i < num_queries; i++)
    {
        const float *q = queries.data() + i * qd;
        int *lbl = nosync_labels.data() + i * search_k;
        float *dst = nosync_dist.data() + i * search_k;

        double tq = -get_ms();
        if (use_filter)
        {
            int ql = query_labels[i];
            std::vector<char> wf(index.ntotal, 0);
            for (int id : label_to_ids[ql])
                wf[id] = 1;
            index.no_sync_search(q, index.get_xb(), index.d, metric,
                                 search_k, efs, lbl, dst,
                                 num_threads, wf.data());
        }
        else
        {
            index.no_sync_search(q, index.get_xb(), index.d, metric,
                                 search_k, efs, lbl, dst,
                                 num_threads);
        }
        nosync_time += tq + get_ms();
    }
    printf("  Time: %.1f ms, Avg: %.3f ms/q\n", nosync_time, nosync_time / num_queries);
    printf("  Speedup: %.2fx\n", ser_time / nosync_time);
    {
        auto &tnd = acorn::get_thread_ndis();
        printf("  NDC per-thread:");
        size_t sum = 0;
        for (size_t nd : tnd) { printf(" %zu", nd); sum += nd; }
        printf(" (total=%zu)\n", sum);
    }
    if (gt_file)
    {
        double r = compute_recall(num_queries, search_k, gt_k, gt_ids, nosync_labels);
        printf("  Recall@%d: %.4f\n", k, r);
    }
    if (use_filter)
    {
        int ok, empty, wrong;
        check_filter(num_queries, search_k, nosync_labels, query_labels, base_labels,
                     &ok, &empty, &wrong);
        printf("  Filter: ok=%d -1=%d wrong=%d\n", ok, empty, wrong);
    }

    // --- Summary ---
    printf("\n=== Summary ===\n");
    printf("Serial:   time=%.1f ms", ser_time);
    if (gt_file)
        printf(", recall=%.4f", compute_recall(num_queries, search_k, gt_k, gt_ids, ser_labels));
    printf("\n");
    printf("iQAN:     time=%.1f ms, speedup=%.2fx", iqan_time, ser_time / iqan_time);
    if (gt_file)
        printf(", recall=%.4f",
               compute_recall(num_queries, search_k, gt_k, gt_ids, iqan_labels));
    printf("\n");
    printf("No-Sync:  time=%.1f ms, speedup=%.2fx", nosync_time, ser_time / nosync_time);
    if (gt_file)
        printf(", recall=%.4f",
               compute_recall(num_queries, search_k, gt_k, gt_ids, nosync_labels));
    printf("\n");

    return 0;
}
