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

static void check_result_filter(int nq, int k,
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
            "  --labels <path>    Base labels in ibin format\n"
            "  --qlabels <path>   Query labels in ibin format\n"
            "\n"
            "Evaluation:\n"
            "  --gt <path>        Ground truth in ibin format (nq, k, nq*k ids)\n"
            "\n"
            "Search parameters:\n"
            "  --k <int>          Number of results (default: 100)\n"
            "  --ef <int>         Serial efSearch (default: 200)\n"
            "  --threads <int>    Number of threads for parallel search (default: 4)\n"
            "  --efs <int>        Parallel search pool size (default: 100)\n"
            "  --efs-min <int>    Min efs for *_sweep modes\n"
            "  --efs-max <int>    Max efs for *_sweep modes\n"
            "  --efs-step <int>   efs step for *_sweep modes\n"
            "  --filter-cost <n>  Synthetic work per filter check (default: 0)\n"
            "  --mode <name>      Search mode: all|serial|pre|post|pre_parallel|post_parallel|post_parallel_iqan|iqan|nosync|scatter|scatter_sweep|iqan_sweep|post_parallel_sweep|post_parallel_iqan_sweep (default: all)\n"
            "  --nq <int>         Max queries to run (default: all)\n"
            "\n"
            "Example:\n"
            "  %s --index acorn.index --query sift_query.fbin "
            "--labels labels.ibin --qlabels qlabels.ibin --gt gt_filtered.ibin\n"
            "  %s --index acorn.index --query sift_query.fbin "
            "--labels labels.ibin --qlabels qlabels.ibin "
            "--gt gt_filtered.ibin --k 100 --ef 400 --threads 8 --efs 200\n",
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
    const char *mode = "all";
    int k = 100, ef = 400, num_threads = 4, efs = 100, Helec = 50;
    int efs_min = 0, efs_max = 0, efs_step = 100;
    int filter_check_cost = 0;
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
        else if (strcmp(argv[i], "--efs-min") == 0 && i + 1 < argc)
            efs_min = atoi(argv[++i]);
        else if (strcmp(argv[i], "--efs-max") == 0 && i + 1 < argc)
            efs_max = atoi(argv[++i]);
        else if (strcmp(argv[i], "--efs-step") == 0 && i + 1 < argc)
            efs_step = atoi(argv[++i]);
        else if (strcmp(argv[i], "--filter-cost") == 0 && i + 1 < argc)
            filter_check_cost = atoi(argv[++i]);
        else if (strcmp(argv[i], "--post-lambda") == 0 && i + 1 < argc)
            ++i; // Kept for backward-compatible scripts; post-filter now uses ef/efs directly.
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
            mode = argv[++i];
        else if (strcmp(argv[i], "--nq") == 0 && i + 1 < argc)
            num_queries = atoi(argv[++i]);
        else if (strcmp(argv[i], "--Helec") == 0 && i + 1 < argc)
            Helec = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
            usage(argv[0]);
        else
        {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
        }
    }

    if (!index_file || !query_file || !label_file || !qlabel_file)
        usage(argv[0]);

    acorn::set_filter_check_cost(filter_check_cost);

    bool run_serial = strcmp(mode, "all") == 0 || strcmp(mode, "serial") == 0;
    bool run_pre = strcmp(mode, "all") == 0 || strcmp(mode, "pre") == 0 ||
                   strcmp(mode, "prefilter") == 0;
    bool run_post = strcmp(mode, "all") == 0 || strcmp(mode, "post") == 0 ||
                    strcmp(mode, "postfilter") == 0;
    bool run_pre_parallel = strcmp(mode, "all") == 0 || strcmp(mode, "pre_parallel") == 0 ||
                            strcmp(mode, "parallel_pre") == 0;
    bool run_post_parallel = strcmp(mode, "all") == 0 || strcmp(mode, "post_parallel") == 0 ||
                             strcmp(mode, "parallel_post") == 0;
    bool run_post_parallel_iqan = strcmp(mode, "post_parallel_iqan") == 0 ||
                                  strcmp(mode, "parallel_post_iqan") == 0;
    bool run_iqan = strcmp(mode, "all") == 0 || strcmp(mode, "iqan") == 0;
    bool run_nosync = strcmp(mode, "all") == 0 || strcmp(mode, "nosync") == 0;
    bool run_scatter = strcmp(mode, "all") == 0 || strcmp(mode, "scatter") == 0;
    bool run_scatter_sweep = strcmp(mode, "scatter_sweep") == 0 ||
                             strcmp(mode, "scattersearch_sweep") == 0;
    bool run_iqan_sweep = strcmp(mode, "iqan_sweep") == 0;
    bool run_post_parallel_sweep = strcmp(mode, "post_parallel_sweep") == 0 ||
                                   strcmp(mode, "parallel_post_sweep") == 0;
    bool run_post_parallel_iqan_sweep = strcmp(mode, "post_parallel_iqan_sweep") == 0 ||
                                        strcmp(mode, "parallel_post_iqan_sweep") == 0;
    if (!run_serial && !run_pre && !run_post && !run_pre_parallel && !run_post_parallel &&
        !run_post_parallel_iqan &&
        !run_iqan && !run_nosync && !run_scatter && !run_scatter_sweep && !run_iqan_sweep &&
        !run_post_parallel_sweep && !run_post_parallel_iqan_sweep)
    {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        usage(argv[0]);
    }
    if (run_scatter_sweep || run_iqan_sweep || run_post_parallel_sweep || run_post_parallel_iqan_sweep)
    {
        if (efs_min <= 0)
            efs_min = efs;
        if (efs_max <= 0)
            efs_max = efs;
        if (efs_step <= 0 || efs_min > efs_max)
        {
            fprintf(stderr, "Invalid efs sweep: min=%d max=%d step=%d\n",
                    efs_min, efs_max, efs_step);
            return 1;
        }
    }

    printf("=== ACORN Search ===\n");
    printf("Index:  %s\n", index_file);
    printf("Query:  %s\n", query_file);
    if (gt_file)
        printf("GT:     %s\n", gt_file);
    printf("Params: k=%d, ef=%d, threads=%d, efs=%d, nq=%d, mode=%s, filter_cost=%d\n\n",
           k, ef, num_threads, efs, num_queries, mode, filter_check_cost);

    // Load base labels first (needed for load_from_faiss)
    std::vector<int> base_labels;
    printf("Loading base labels ...\n");
    base_labels = acorn::read_ibin(label_file);
    printf("  %zu labels\n", base_labels.size());

    // Load index from FAISS format
    printf("Loading index ...\n");
    double t0 = get_ms();
    acorn::ACORN index;
    index.load_from_faiss(index_file);
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
    {
        fprintf(stderr, "Error: query dim %d != index dim %d\n", qd, index.d);
        return 1;
    }

    int metric = (index.metric_type == acorn::METRIC_INNER_PRODUCT) ? 0 : 1;
    printf("  Metric: %s\n", (metric == 0) ? "Inner Product" : "L2");

    std::vector<int> query_labels;
    std::vector<std::vector<int>> label_to_ids;

    printf("Loading query labels ...\n");
    query_labels = acorn::read_ibin(qlabel_file, nq_all);
    printf("  %zu labels\n", query_labels.size());

    int max_label = 0;
    for (int lbl : base_labels)
        if (lbl > max_label)
            max_label = lbl;
    label_to_ids.resize(max_label + 1);
    for (int i = 0; i < (int)base_labels.size(); i++)
        label_to_ids[base_labels[i]].push_back(i);

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
        int ql = query_labels[0];
        std::vector<char> wf(index.ntotal, 0);
        for (int id : label_to_ids[ql])
            wf[id] = 1;
        index.search(queries.data(), index.get_xb(), index.d, metric,
                     search_k, ef, tmp_labels.data(), tmp_dist.data(), wf.data());
    }
    printf("  Done.\n");

    if (run_scatter_sweep || run_iqan_sweep || run_post_parallel_sweep || run_post_parallel_iqan_sweep)
    {
        const char *sweep_mode = run_iqan_sweep ? "iqan" :
                                 (run_post_parallel_sweep ? "post_parallel" :
                                  (run_post_parallel_iqan_sweep ? "post_parallel_iqan" : "scatter"));
        printf("\n--- %s Sweep (%d queries, threads=%d, efs=%d..%d step=%d, Helec=%d) ---\n",
               run_iqan_sweep ? "iQAN" :
               (run_post_parallel_sweep ? "Parallel Post-Filter" :
                (run_post_parallel_iqan_sweep ? "Parallel Post-Filter iQAN" : "ScatterSearch")),
               num_queries, num_threads, efs_min, efs_max, efs_step, Helec);
        for (int cur_efs = efs_min; cur_efs <= efs_max; cur_efs += efs_step)
        {
            std::vector<int> sweep_labels(search_k * num_queries, -1);
            std::vector<float> sweep_dist(search_k * num_queries);
            double sweep_time = 0;

            acorn::reset_thread_ndis(num_threads);
            acorn::reset_phase_timing();
            for (int i = 0; i < num_queries; i++)
            {
                const float *q = queries.data() + i * qd;
                int *lbl = sweep_labels.data() + i * search_k;
                float *dst = sweep_dist.data() + i * search_k;

                int ql = query_labels[i];
                std::vector<char> wf(index.ntotal, 0);
                for (int id : label_to_ids[ql])
                    wf[id] = 1;

                double tq = -get_ms();
                if (run_iqan_sweep)
                {
                    index.iqan_search(q, index.get_xb(), index.d, metric,
                                      search_k, cur_efs, lbl, dst,
                                      num_threads, wf.data());
                }
                else if (run_post_parallel_sweep)
                {
                    index.parallel_post_filter_search(q, index.get_xb(), index.d, metric,
                                                      search_k, cur_efs, lbl, dst,
                                                      num_threads, Helec, wf.data());
                }
                else if (run_post_parallel_iqan_sweep)
                {
                    index.parallel_post_filter_iqan_search(q, index.get_xb(), index.d, metric,
                                                           search_k, cur_efs, lbl, dst,
                                                           num_threads, wf.data());
                }
                else
                {
                    index.scatter_search(q, index.get_xb(), index.d, metric,
                                         search_k, cur_efs, lbl, dst,
                                         num_threads, Helec, wf.data());
                }
                sweep_time += tq + get_ms();
            }

            double recall = -1.0;
            if (gt_file)
                recall = compute_recall(num_queries, search_k, gt_k, gt_ids, sweep_labels);

            int ok, empty, wrong;
            check_result_filter(num_queries, search_k, sweep_labels, query_labels, base_labels,
                                &ok, &empty, &wrong);

            const auto &tnd = acorn::get_thread_ndis();
            size_t ndis_total = 0;
            for (size_t nd : tnd)
                ndis_total += nd;

            const auto &pt = acorn::get_scatter_timing();
            printf("SWEEP_RESULT mode=%s efs=%d time_ms=%.1f avg_ms=%.3f recall=%.4f ndc=%zu ok=%d empty=%d wrong=%d phase1_ms=%.1f parallel_ms=%.1f merge_ms=%.1f\n",
                   sweep_mode,
                   cur_efs, sweep_time, sweep_time / num_queries, recall,
                   ndis_total, ok, empty, wrong, pt.phase1, pt.parallel, pt.merge);
            fflush(stdout);
        }
        return 0;
    }

    std::vector<int> ser_labels(search_k * num_queries, -1);
    std::vector<float> ser_dist(search_k * num_queries);
    double ser_time = 0;
    if (run_serial)
    {
        printf("\n--- Serial Search (%d queries, ef=%d) ---\n", num_queries, ef);
        acorn::reset_ser_ndis();
        for (int i = 0; i < num_queries; i++)
        {
            const float *q = queries.data() + i * qd;
            int *lbl = ser_labels.data() + i * search_k;
            float *dst = ser_dist.data() + i * search_k;

            int ql = query_labels[i];
            std::vector<char> wf(index.ntotal, 0);
            for (int id : label_to_ids[ql])
                wf[id] = 1;
            double tq = -get_ms();
            index.search(q, index.get_xb(), index.d, metric,
                         search_k, ef, lbl, dst, wf.data());
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
        {
            int ok, empty, wrong;
            check_result_filter(num_queries, search_k, ser_labels, query_labels, base_labels,
                                &ok, &empty, &wrong);
            printf("  Filter: ok=%d -1=%d wrong=%d\n", ok, empty, wrong);
        }
    }

    // --- Pre-filter brute-force search ---
    std::vector<int> pre_labels(search_k * num_queries, -1);
    std::vector<float> pre_dist(search_k * num_queries);
    double pre_time = 0;
    if (run_pre)
    {
        printf("\n--- Pre-Filter Brute-Force Search (%d queries) ---\n", num_queries);
        acorn::reset_ser_ndis();
        for (int i = 0; i < num_queries; i++)
        {
            const float *q = queries.data() + i * qd;
            int *lbl = pre_labels.data() + i * search_k;
            float *dst = pre_dist.data() + i * search_k;

            int ql = query_labels[i];
            std::vector<char> wf(index.ntotal, 0);
            for (int id : label_to_ids[ql])
                wf[id] = 1;
            double tq = -get_ms();
            index.pre_filter_search(q, index.get_xb(), index.d, metric,
                                    search_k, lbl, dst, wf.data());
            pre_time += tq + get_ms();
        }
        printf("  Time: %.1f ms, Avg: %.3f ms/q\n", pre_time, pre_time / num_queries);
        printf("  NDC:  total=%zu avg/q=%.0f\n",
               acorn::get_ser_ndis(), (double)acorn::get_ser_ndis() / num_queries);

        if (gt_file)
        {
            double r = compute_recall(num_queries, search_k, gt_k, gt_ids, pre_labels);
            printf("  Recall@%d: %.4f\n", k, r);
        }
        {
            int ok, empty, wrong;
            check_result_filter(num_queries, search_k, pre_labels, query_labels, base_labels,
                                &ok, &empty, &wrong);
            printf("  Filter: ok=%d -1=%d wrong=%d\n", ok, empty, wrong);
        }
    }

    // --- Parallel pre-filter brute-force search ---
    std::vector<int> pre_parallel_labels(search_k * num_queries, -1);
    std::vector<float> pre_parallel_dist(search_k * num_queries);
    double pre_parallel_time = 0;
    if (run_pre_parallel)
    {
        printf("\n--- Parallel Pre-Filter Brute-Force Search (%d queries, threads=%d) ---\n",
               num_queries, num_threads);
        acorn::reset_thread_ndis(num_threads);
        for (int i = 0; i < num_queries; i++)
        {
            const float *q = queries.data() + i * qd;
            int *lbl = pre_parallel_labels.data() + i * search_k;
            float *dst = pre_parallel_dist.data() + i * search_k;

            int ql = query_labels[i];
            std::vector<char> wf(index.ntotal, 0);
            for (int id : label_to_ids[ql])
                wf[id] = 1;
            double tq = -get_ms();
            index.parallel_pre_filter_search(q, index.get_xb(), index.d, metric,
                                             search_k, lbl, dst,
                                             num_threads, wf.data());
            pre_parallel_time += tq + get_ms();
        }
        printf("  Time: %.1f ms, Avg: %.3f ms/q\n",
               pre_parallel_time, pre_parallel_time / num_queries);
        if (pre_time > 0)
            printf("  Speedup: %.2fx\n", pre_time / pre_parallel_time);
        {
            auto &tnd = acorn::get_thread_ndis();
            printf("  NDC per-thread:");
            size_t sum = 0;
            for (size_t nd : tnd)
            {
                printf(" %zu", nd);
                sum += nd;
            }
            printf(" (total=%zu)\n", sum);
        }

        if (gt_file)
        {
            double r = compute_recall(num_queries, search_k, gt_k, gt_ids, pre_parallel_labels);
            printf("  Recall@%d: %.4f\n", k, r);
        }
        {
            int ok, empty, wrong;
            check_result_filter(num_queries, search_k, pre_parallel_labels, query_labels, base_labels,
                                &ok, &empty, &wrong);
            printf("  Filter: ok=%d -1=%d wrong=%d\n", ok, empty, wrong);
        }
    }

    // --- Post-filter graph search ---
    std::vector<int> post_labels(search_k * num_queries, -1);
    std::vector<float> post_dist(search_k * num_queries);
    double post_time = 0;
    if (run_post)
    {
        printf("\n--- Post-Filter Search (%d queries, ef=%d) ---\n",
               num_queries, ef);
        acorn::reset_ser_ndis();
        for (int i = 0; i < num_queries; i++)
        {
            const float *q = queries.data() + i * qd;
            int *lbl = post_labels.data() + i * search_k;
            float *dst = post_dist.data() + i * search_k;

            int ql = query_labels[i];
            std::vector<char> wf(index.ntotal, 0);
            for (int id : label_to_ids[ql])
                wf[id] = 1;
            double tq = -get_ms();
            index.post_filter_search(q, index.get_xb(), index.d, metric,
                                     search_k, ef, lbl, dst, wf.data());
            post_time += tq + get_ms();
        }
        printf("  Time: %.1f ms, Avg: %.3f ms/q\n", post_time, post_time / num_queries);
        printf("  NDC:  total=%zu avg/q=%.0f\n",
               acorn::get_ser_ndis(), (double)acorn::get_ser_ndis() / num_queries);
        if (ser_time > 0)
            printf("  Speedup: %.2fx\n", ser_time / post_time);

        if (gt_file)
        {
            double r = compute_recall(num_queries, search_k, gt_k, gt_ids, post_labels);
            printf("  Recall@%d: %.4f\n", k, r);
        }
        {
            int ok, empty, wrong;
            check_result_filter(num_queries, search_k, post_labels, query_labels, base_labels,
                                &ok, &empty, &wrong);
            printf("  Filter: ok=%d -1=%d wrong=%d\n", ok, empty, wrong);
        }
    }

    // --- Parallel post-filter graph search ---
    std::vector<int> post_parallel_labels(search_k * num_queries, -1);
    std::vector<float> post_parallel_dist(search_k * num_queries);
    double post_parallel_time = 0;
    if (run_post_parallel)
    {
        printf("\n--- Parallel Post-Filter Search (%d queries, threads=%d, efs=%d, Helec=%d) ---\n",
               num_queries, num_threads, efs, Helec);
        acorn::reset_thread_ndis(num_threads);
        acorn::reset_phase_timing();
        for (int i = 0; i < num_queries; i++)
        {
            const float *q = queries.data() + i * qd;
            int *lbl = post_parallel_labels.data() + i * search_k;
            float *dst = post_parallel_dist.data() + i * search_k;

            int ql = query_labels[i];
            std::vector<char> wf(index.ntotal, 0);
            for (int id : label_to_ids[ql])
                wf[id] = 1;
            double tq = -get_ms();
            index.parallel_post_filter_search(q, index.get_xb(), index.d, metric,
                                              search_k, efs, lbl, dst,
                                              num_threads, Helec, wf.data());
            post_parallel_time += tq + get_ms();
        }
        printf("  Time: %.1f ms, Avg: %.3f ms/q\n",
               post_parallel_time, post_parallel_time / num_queries);
        if (post_time > 0)
            printf("  Speedup: %.2fx\n", post_time / post_parallel_time);
        {
            auto &tnd = acorn::get_thread_ndis();
            printf("  NDC per-thread:");
            size_t sum = 0;
            for (size_t nd : tnd)
            {
                printf(" %zu", nd);
                sum += nd;
            }
            printf(" (total=%zu)\n", sum);
        }
        {
            const auto &pt = acorn::get_scatter_timing();
            printf("  Phases: phase1=%.1f ms parallel=%.1f ms merge=%.1f ms\n",
                   pt.phase1, pt.parallel, pt.merge);
        }
        if (gt_file)
        {
            double r = compute_recall(num_queries, search_k, gt_k, gt_ids, post_parallel_labels);
            printf("  Recall@%d: %.4f\n", k, r);
        }
        {
            int ok, empty, wrong;
            check_result_filter(num_queries, search_k, post_parallel_labels, query_labels, base_labels,
                                &ok, &empty, &wrong);
            printf("  Filter: ok=%d -1=%d wrong=%d\n", ok, empty, wrong);
        }
    }

    // --- Parallel post-filter iQAN graph search ---
    std::vector<int> post_parallel_iqan_labels(search_k * num_queries, -1);
    std::vector<float> post_parallel_iqan_dist(search_k * num_queries);
    double post_parallel_iqan_time = 0;
    if (run_post_parallel_iqan)
    {
        printf("\n--- Parallel Post-Filter iQAN Search (%d queries, threads=%d, efs=%d) ---\n",
               num_queries, num_threads, efs);
        acorn::reset_thread_ndis(num_threads);
        acorn::reset_phase_timing();
        for (int i = 0; i < num_queries; i++)
        {
            const float *q = queries.data() + i * qd;
            int *lbl = post_parallel_iqan_labels.data() + i * search_k;
            float *dst = post_parallel_iqan_dist.data() + i * search_k;

            int ql = query_labels[i];
            std::vector<char> wf(index.ntotal, 0);
            for (int id : label_to_ids[ql])
                wf[id] = 1;
            double tq = -get_ms();
            index.parallel_post_filter_iqan_search(q, index.get_xb(), index.d, metric,
                                                   search_k, efs, lbl, dst,
                                                   num_threads, wf.data());
            post_parallel_iqan_time += tq + get_ms();
        }
        printf("  Time: %.1f ms, Avg: %.3f ms/q\n",
               post_parallel_iqan_time, post_parallel_iqan_time / num_queries);
        {
            auto &tnd = acorn::get_thread_ndis();
            printf("  NDC per-thread:");
            size_t sum = 0;
            for (size_t nd : tnd)
            {
                printf(" %zu", nd);
                sum += nd;
            }
            printf(" (total=%zu)\n", sum);
        }
        {
            const auto &pt = acorn::get_scatter_timing();
            printf("  Phases: phase1=%.1f ms parallel=%.1f ms merge=%.1f ms\n",
                   pt.phase1, pt.parallel, pt.merge);
        }
        if (gt_file)
        {
            double r = compute_recall(num_queries, search_k, gt_k, gt_ids, post_parallel_iqan_labels);
            printf("  Recall@%d: %.4f\n", k, r);
        }
        {
            int ok, empty, wrong;
            check_result_filter(num_queries, search_k, post_parallel_iqan_labels, query_labels, base_labels,
                                &ok, &empty, &wrong);
            printf("  Filter: ok=%d -1=%d wrong=%d\n", ok, empty, wrong);
        }
    }

    // --- iQAN search ---
    std::vector<int> iqan_labels(search_k * num_queries, -1);
    std::vector<float> iqan_dist(search_k * num_queries);
    double iqan_time = 0;
    if (run_iqan)
    {
        printf("\n--- iQAN Search (%d queries, threads=%d, efs=%d) ---\n",
               num_queries, num_threads, efs);
        acorn::reset_thread_ndis(num_threads);
        for (int i = 0; i < num_queries; i++)
        {
            const float *q = queries.data() + i * qd;
            int *lbl = iqan_labels.data() + i * search_k;
            float *dst = iqan_dist.data() + i * search_k;

            int ql = query_labels[i];
            std::vector<char> wf(index.ntotal, 0);
            for (int id : label_to_ids[ql])
                wf[id] = 1;
            double tq = -get_ms();
            index.iqan_search(q, index.get_xb(), index.d, metric,
                              search_k, efs, lbl, dst,
                              num_threads, wf.data());
            iqan_time += tq + get_ms();
        }
        printf("  Time: %.1f ms, Avg: %.3f ms/q\n", iqan_time, iqan_time / num_queries);
        if (ser_time > 0)
            printf("  Speedup: %.2fx\n", ser_time / iqan_time);
        {
            auto &tnd = acorn::get_thread_ndis();
            printf("  NDC per-thread:");
            size_t sum = 0;
            for (size_t nd : tnd)
            {
                printf(" %zu", nd);
                sum += nd;
            }
            printf(" (total=%zu)\n", sum);
        }
        if (gt_file)
        {
            double r = compute_recall(num_queries, search_k, gt_k, gt_ids, iqan_labels);
            printf("  Recall@%d: %.4f\n", k, r);
        }
        {
            int ok, empty, wrong;
            check_result_filter(num_queries, search_k, iqan_labels, query_labels, base_labels,
                                &ok, &empty, &wrong);
            printf("  Filter: ok=%d -1=%d wrong=%d\n", ok, empty, wrong);
        }
    }

    // --- No-sync search ---
    std::vector<int> nosync_labels(search_k * num_queries, -1);
    std::vector<float> nosync_dist(search_k * num_queries);
    double nosync_time = 0;
    if (run_nosync)
    {
        printf("\n--- No-Sync Search (%d queries, threads=%d, efs=%d) ---\n",
               num_queries, num_threads, efs);
        acorn::reset_thread_ndis(num_threads);
        acorn::reset_phase_timing();
        for (int i = 0; i < num_queries; i++)
        {
            const float *q = queries.data() + i * qd;
            int *lbl = nosync_labels.data() + i * search_k;
            float *dst = nosync_dist.data() + i * search_k;

            int ql = query_labels[i];
            std::vector<char> wf(index.ntotal, 0);
            for (int id : label_to_ids[ql])
                wf[id] = 1;
            double tq = -get_ms();
            index.no_sync_search(q, index.get_xb(), index.d, metric,
                                 search_k, efs, lbl, dst,
                                 num_threads, wf.data());
            nosync_time += tq + get_ms();
        }
        printf("  Time: %.1f ms, Avg: %.3f ms/q\n", nosync_time, nosync_time / num_queries);
        if (ser_time > 0)
            printf("  Speedup: %.2fx\n", ser_time / nosync_time);
        {
            auto &tnd = acorn::get_thread_ndis();
            printf("  NDC per-thread:");
            size_t sum = 0;
            for (size_t nd : tnd)
            {
                printf(" %zu", nd);
                sum += nd;
            }
            printf(" (total=%zu)\n", sum);
        }
        {
            const auto &pt = acorn::get_nosync_timing();
            printf("  Phases: phase1=%.1f ms parallel=%.1f ms merge=%.1f ms\n",
                   pt.phase1, pt.parallel, pt.merge);
        }
        if (gt_file)
        {
            double r = compute_recall(num_queries, search_k, gt_k, gt_ids, nosync_labels);
            printf("  Recall@%d: %.4f\n", k, r);
        }
        {
            int ok, empty, wrong;
            check_result_filter(num_queries, search_k, nosync_labels, query_labels, base_labels,
                                &ok, &empty, &wrong);
            printf("  Filter: ok=%d -1=%d wrong=%d\n", ok, empty, wrong);
        }
    }

    // --- ScatterSearch ---
    std::vector<int> scatter_labels(search_k * num_queries, -1);
    std::vector<float> scatter_dist(search_k * num_queries);
    double scatter_time = 0;
    if (run_scatter)
    {
        printf("\n--- ScatterSearch (%d queries, threads=%d, efs=%d, Helec=%d) ---\n",
               num_queries, num_threads, efs, Helec);
        acorn::reset_thread_ndis(num_threads);
        acorn::reset_phase_timing();
        for (int i = 0; i < num_queries; i++)
        {
            const float *q = queries.data() + i * qd;
            int *lbl = scatter_labels.data() + i * search_k;
            float *dst = scatter_dist.data() + i * search_k;

            int ql = query_labels[i];
            std::vector<char> wf(index.ntotal, 0);
            for (int id : label_to_ids[ql])
                wf[id] = 1;

            double tq = -get_ms();
            index.scatter_search(q, index.get_xb(), index.d, metric,
                                 search_k, efs, lbl, dst,
                                 num_threads, Helec, wf.data());
            scatter_time += tq + get_ms();
        }
        printf("  Time: %.1f ms, Avg: %.3f ms/q\n", scatter_time, scatter_time / num_queries);
        if (ser_time > 0)
            printf("  Speedup: %.2fx\n", ser_time / scatter_time);
        {
            auto &tnd = acorn::get_thread_ndis();
            printf("  NDC per-thread:");
            size_t sum = 0;
            for (size_t nd : tnd)
            {
                printf(" %zu", nd);
                sum += nd;
            }
            printf(" (total=%zu)\n", sum);
        }
        {
            const auto &pt = acorn::get_scatter_timing();
            printf("  Phases: phase1=%.1f ms parallel=%.1f ms merge=%.1f ms\n",
                   pt.phase1, pt.parallel, pt.merge);
        }
        if (gt_file)
        {
            double r = compute_recall(num_queries, search_k, gt_k, gt_ids, scatter_labels);
            printf("  Recall@%d: %.4f\n", k, r);
        }
        {
            int ok, empty, wrong;
            check_result_filter(num_queries, search_k, scatter_labels, query_labels, base_labels,
                                &ok, &empty, &wrong);
            printf("  Filter: ok=%d -1=%d wrong=%d\n", ok, empty, wrong);
        }
    }

    // --- Summary ---
    if (strcmp(mode, "all") == 0)
    {
        printf("\n=== Summary ===\n");
        printf("Serial:       time=%.1f ms", ser_time);
        if (gt_file)
            printf(", recall=%.4f", compute_recall(num_queries, search_k, gt_k, gt_ids, ser_labels));
        printf("\n");
        printf("Pre-Filter:   time=%.1f ms", pre_time);
        if (gt_file)
            printf(", recall=%.4f", compute_recall(num_queries, search_k, gt_k, gt_ids, pre_labels));
        printf("\n");
        printf("Post-Filter:  time=%.1f ms", post_time);
        if (gt_file)
            printf(", recall=%.4f", compute_recall(num_queries, search_k, gt_k, gt_ids, post_labels));
        printf("\n");
        printf("Pre-Filter-P: time=%.1f ms", pre_parallel_time);
        if (gt_file)
            printf(", recall=%.4f",
                   compute_recall(num_queries, search_k, gt_k, gt_ids, pre_parallel_labels));
        printf("\n");
        printf("Post-Filter-P: time=%.1f ms", post_parallel_time);
        if (gt_file)
            printf(", recall=%.4f",
                   compute_recall(num_queries, search_k, gt_k, gt_ids, post_parallel_labels));
        printf("\n");
        printf("Post-Filter-iQAN-P: time=%.1f ms", post_parallel_iqan_time);
        if (gt_file)
            printf(", recall=%.4f",
                   compute_recall(num_queries, search_k, gt_k, gt_ids, post_parallel_iqan_labels));
        printf("\n");
        printf("iQAN:         time=%.1f ms, speedup=%.2fx", iqan_time, ser_time / iqan_time);
        if (gt_file)
            printf(", recall=%.4f",
                   compute_recall(num_queries, search_k, gt_k, gt_ids, iqan_labels));
        printf("\n");
        printf("No-Sync:      time=%.1f ms, speedup=%.2fx", nosync_time, ser_time / nosync_time);
        if (gt_file)
            printf(", recall=%.4f",
                   compute_recall(num_queries, search_k, gt_k, gt_ids, nosync_labels));
        printf("\n");
        printf("ScatterSearch: time=%.1f ms, speedup=%.2fx", scatter_time, ser_time / scatter_time);
        if (gt_file)
            printf(", recall=%.4f",
                   compute_recall(num_queries, search_k, gt_k, gt_ids, scatter_labels));
        printf("\n");
    }

    return 0;
}
