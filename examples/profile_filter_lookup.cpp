#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "acorn/acorn_graph.h"
#include "acorn/file_io.h"

static double now_ms()
{
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

static std::vector<int> parse_int_list(const char *s)
{
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        if (item.empty())
            continue;
        out.push_back(std::atoi(item.c_str()));
    }
    return out;
}

static std::vector<std::string> parse_string_list(const char *s)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty())
            out.push_back(item);
    return out;
}

static std::string normalize_strategy(const std::string &s)
{
    if (s == "parallel_pre")
        return "pre_parallel";
    if (s == "parallel_in_scatter")
        return "scatter";
    if (s == "parallel_in_iqan")
        return "iqan";
    if (s == "parallel_post_scatter")
        return "post_parallel";
    if (s == "parallel_post_iqan")
        return "post_parallel_iqan";
    return s;
}

static std::string display_strategy(const std::string &s, bool compact_output)
{
    if (!compact_output)
        return s;
    if (s == "pre_parallel")
        return "parallel_pre";
    if (s == "scatter")
        return "parallel_in_scatter";
    if (s == "iqan")
        return "parallel_in_iqan";
    if (s == "post_parallel")
        return "parallel_post_scatter";
    if (s == "post_parallel_iqan")
        return "parallel_post_iqan";
    return s;
}

static bool contains_int(const std::vector<int> &values, int x)
{
    return std::find(values.begin(), values.end(), x) != values.end();
}

static double compute_recall(int nq, int k, int gt_k,
                             const std::vector<int> &gt_ids,
                             const std::vector<int> &results)
{
    int hits = 0;
    for (int q = 0; q < nq; q++)
    {
        std::vector<int> gt(gt_ids.begin() + q * gt_k,
                            gt_ids.begin() + (q + 1) * gt_k);
        std::sort(gt.begin(), gt.end());
        for (int j = 0; j < k; j++)
        {
            int id = results[(size_t)q * k + j];
            if (id >= 0 && std::binary_search(gt.begin(), gt.end(), id))
                hits++;
        }
    }
    return (double)hits / (nq * k);
}

static void check_filter_result(int nq, int k,
                                const std::vector<int> &results,
                                int qlabel,
                                const std::vector<int> &base_labels,
                                int *ok, int *empty, int *wrong)
{
    *ok = *empty = *wrong = 0;
    for (int i = 0; i < nq * k; i++)
    {
        int id = results[i];
        if (id < 0)
            (*empty)++;
        else if (base_labels[id] != qlabel)
            (*wrong)++;
        else
            (*ok)++;
    }
}

static void usage(const char *prog)
{
    std::fprintf(stderr,
                 "Usage: %s --index <index> --query <train_query.fbin> --out <raw.tsv> [options]\n"
                 "\n"
                 "Options:\n"
                 "  --data-dir <path>          LAION label dir (default: data/laion10m)\n"
                 "  --train-dir <path>         Train query/GT dir (default: data/laion10m/train)\n"
                 "  --gt-prefix <prefix>       Train GT filename prefix (default: laion10m_train_gt_binary_)\n"
                 "  --nq <int>                 Queries to run (default: all)\n"
                 "  --k <int>                  top-k, fixed to 100 for experiments (default: 100)\n"
                 "  --threads <int>            Parallel threads (default: 4)\n"
                 "  --Helec <int>              ScatterSearch Helec (default: 50)\n"
                 "  --repeat <int>             Repeat the same nq queries for profiling (default: 1)\n"
                 "  --filtered-expand-target <int>\n"
                 "                             Override filtered level-0 expand target; 0 uses M*2 (default: 0)\n"
                 "  --costs <csv>              filter_check_cost list (default: 0,32,64,128,256)\n"
                 "  --efs-list <csv>           efs list for non-pre strategies (default: 100,200,400,700,1000)\n"
                 "  --selectivities <csv>      selectivity labels to run, e.g. 5,10 (default: all)\n"
                 "  --strategies <csv>         default: pre_parallel,scatter,iqan,post_parallel,post_parallel_iqan\n"
                 "  --pre-max-selectivity <x>  skip pre-filter above this selectivity; <=0 disables skip (default: 0.02)\n"
                 "  --pre-costs <csv>          filter costs where pre-filter is measured (default: 0,32)\n"
                 "  --max-avg-ms <float>       skip larger efs after a strategy exceeds this avg latency; <=0 disables (default: 0)\n"
                 "  --compact-output           omit fixed k and total time_ms columns; use paper strategy names\n",
                 prog);
    std::exit(1);
}

struct LabelConfig
{
    const char *sel;
    double true_s;
    const char *base_suffix;
    const char *qlabel_file;
    const char *gt_suffix;
};

struct RunResult
{
    double avg_ms = 0.0;
    double total_ms = 0.0;
    double recall = 0.0;
    size_t ndc = 0;
    size_t ndc_max = 0;
    int ok = 0;
    int empty = 0;
    int wrong = 0;
};

static RunResult run_method(const std::string &method,
                            const acorn::ACORN &index,
                            const std::vector<float> &queries,
                            int qd,
                            int nq,
                            int k,
                            int efs,
                            int threads,
                            int Helec,
                            int repeat,
                            int metric,
                            const std::vector<char> &wf,
                            int qlabel,
                            const std::vector<int> &base_labels,
                            const std::vector<int> &gt_ids,
                            int gt_k)
{
    std::vector<int> labels((size_t)nq * k, -1);
    std::vector<float> dist((size_t)nq * k, 0.0f);
    acorn::reset_thread_ndis(threads);
    acorn::reset_phase_timing();

    double total = 0.0;
    for (int rep = 0; rep < repeat; rep++)
    {
        for (int i = 0; i < nq; i++)
        {
            const float *q = queries.data() + (size_t)i * qd;
            int *out = labels.data() + (size_t)i * k;
            float *dst = dist.data() + (size_t)i * k;
            double t0 = now_ms();

            if (method == "pre_parallel")
            {
                index.parallel_pre_filter_search(q, index.get_xb(), index.d, metric, k,
                                                 out, dst, threads, wf.data());
            }
            else if (method == "scatter")
            {
                index.scatter_search(q, index.get_xb(), index.d, metric, k, efs,
                                     out, dst, threads, Helec, wf.data());
            }
            else if (method == "iqan")
            {
                index.iqan_search(q, index.get_xb(), index.d, metric, k, efs,
                                  out, dst, threads, wf.data());
            }
            else if (method == "post_parallel")
            {
                index.parallel_post_filter_search(q, index.get_xb(), index.d, metric, k, efs,
                                                  out, dst, threads, Helec, wf.data());
            }
            else if (method == "post_parallel_iqan")
            {
                index.parallel_post_filter_iqan_search(q, index.get_xb(), index.d, metric, k, efs,
                                                       out, dst, threads, wf.data());
            }
            else
            {
                std::fprintf(stderr, "Unknown strategy: %s\n", method.c_str());
                std::exit(1);
            }

            total += now_ms() - t0;
        }
    }

    RunResult r;
    r.total_ms = total;
    r.avg_ms = total / (nq * repeat);
    r.recall = compute_recall(nq, k, gt_k, gt_ids, labels);
    const auto &tnd = acorn::get_thread_ndis();
    for (size_t v : tnd)
    {
        r.ndc += v;
        r.ndc_max = std::max(r.ndc_max, v);
    }
    check_filter_result(nq, k, labels, qlabel, base_labels, &r.ok, &r.empty, &r.wrong);
    return r;
}

int main(int argc, char **argv)
{
    const char *index_file = nullptr;
    const char *query_file = nullptr;
    const char *out_file = nullptr;
    std::string data_dir = "data/laion10m";
    std::string train_dir = "data/laion10m/train";
    std::string gt_prefix = "laion10m_train_gt_binary_";
    int nq = -1;
    int k = 100;
    int threads = 4;
    int Helec = 50;
    int repeat = 1;
    int filtered_expand_target = 0;
    double pre_max_selectivity = 0.02;
    double max_avg_ms = 0.0;
    bool compact_output = false;
    std::vector<int> costs = {0, 32, 64, 128, 256};
    std::vector<int> pre_costs = {0, 32};
    std::vector<int> efs_list = {100, 200, 400, 700, 1000};
    std::vector<std::string> selected_selectivities;
    std::vector<std::string> strategies = {
        "pre_parallel", "scatter", "iqan", "post_parallel", "post_parallel_iqan"};

    for (int i = 1; i < argc; i++)
    {
        if (std::strcmp(argv[i], "--index") == 0 && i + 1 < argc)
            index_file = argv[++i];
        else if (std::strcmp(argv[i], "--query") == 0 && i + 1 < argc)
            query_file = argv[++i];
        else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc)
            out_file = argv[++i];
        else if (std::strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc)
            data_dir = argv[++i];
        else if (std::strcmp(argv[i], "--train-dir") == 0 && i + 1 < argc)
            train_dir = argv[++i];
        else if (std::strcmp(argv[i], "--gt-prefix") == 0 && i + 1 < argc)
            gt_prefix = argv[++i];
        else if (std::strcmp(argv[i], "--nq") == 0 && i + 1 < argc)
            nq = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc)
            k = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            threads = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--Helec") == 0 && i + 1 < argc)
            Helec = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--repeat") == 0 && i + 1 < argc)
            repeat = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--filtered-expand-target") == 0 && i + 1 < argc)
            filtered_expand_target = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--costs") == 0 && i + 1 < argc)
            costs = parse_int_list(argv[++i]);
        else if (std::strcmp(argv[i], "--efs-list") == 0 && i + 1 < argc)
            efs_list = parse_int_list(argv[++i]);
        else if (std::strcmp(argv[i], "--selectivities") == 0 && i + 1 < argc)
            selected_selectivities = parse_string_list(argv[++i]);
        else if (std::strcmp(argv[i], "--strategies") == 0 && i + 1 < argc)
            strategies = parse_string_list(argv[++i]);
        else if (std::strcmp(argv[i], "--pre-max-selectivity") == 0 && i + 1 < argc)
            pre_max_selectivity = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--pre-costs") == 0 && i + 1 < argc)
            pre_costs = parse_int_list(argv[++i]);
        else if (std::strcmp(argv[i], "--max-avg-ms") == 0 && i + 1 < argc)
            max_avg_ms = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--compact-output") == 0)
            compact_output = true;
        else
            usage(argv[0]);
    }

    for (std::string &strategy : strategies)
        strategy = normalize_strategy(strategy);

    if (!index_file || !query_file || !out_file || k <= 0 || threads <= 0 || repeat <= 0 ||
        costs.empty() || efs_list.empty() || strategies.empty())
        usage(argv[0]);
    if (filtered_expand_target < 0)
        usage(argv[0]);

    acorn::set_filtered_expand_target(filtered_expand_target);

    const LabelConfig configs[] = {
        {"0.1", 0.001, "s0p1", "query_labels_all1.ibin", "s0p1"},
        {"1", 0.01, "s1", "query_labels_all1.ibin", "s1"},
        {"5", 0.05, "s5", "query_labels_all1.ibin", "s5"},
        {"10", 0.10, "s10", "query_labels_all1.ibin", "s10"},
        {"20", 0.20, "s20", "query_labels_all1.ibin", "s20"},
        {"50", 0.50, "s50", "query_labels_all1.ibin", "s50_label1"},
        {"80", 0.80, "s20", "query_labels_all2.ibin", "s80"},
    };

    std::printf("Loading index: %s\n", index_file);
    acorn::ACORN index;
    index.load_from_faiss(index_file);
    int metric = (index.metric_type == acorn::METRIC_INNER_PRODUCT) ? 0 : 1;
    std::printf("  ntotal=%ld d=%d metric=%s\n",
                index.ntotal, index.d, metric == 0 ? "ip" : "l2");

    std::printf("Loading train queries: %s\n", query_file);
    auto qr = acorn::read_fbin(query_file);
    std::vector<float> queries = std::move(qr.first);
    int nq_all = qr.second.first;
    int qd = qr.second.second;
    if (nq <= 0 || nq > nq_all)
        nq = nq_all;
    if (qd != index.d)
    {
        std::fprintf(stderr, "query dim mismatch: %d vs %d\n", qd, index.d);
        return 1;
    }
    std::printf("  nq=%d/%d d=%d\n", nq, nq_all, qd);

    FILE *out = std::fopen(out_file, "w");
    if (!out)
    {
        std::fprintf(stderr, "cannot open output: %s\n", out_file);
        return 1;
    }
    if (compact_output)
        std::fprintf(out, "selectivity\ttrue_s\tfilter_cost\tstrategy\tefs\tthreads\tnq\tavg_ms\trecall\tndc\tndc_max\tok\tempty\twrong\tstatus\n");
    else
        std::fprintf(out, "selectivity\ttrue_s\tfilter_cost\tstrategy\tefs\tthreads\tk\tnq\tavg_ms\ttime_ms\trecall\tndc\tndc_max\tok\tempty\twrong\tstatus\n");

    for (const auto &cfg : configs)
    {
        if (!selected_selectivities.empty() &&
            std::find(selected_selectivities.begin(), selected_selectivities.end(),
                      std::string(cfg.sel)) == selected_selectivities.end())
            continue;

        std::string base_path = data_dir + "/base_labels_binary_" + cfg.base_suffix + ".ibin";
        std::string qlabel_path = train_dir + "/" + cfg.qlabel_file;
        std::string gt_path = train_dir + "/" + gt_prefix + cfg.gt_suffix + ".ibin";

        std::printf("Loading selectivity %s%% labels/GT...\n", cfg.sel);
        std::vector<int> base_labels = acorn::read_ibin(base_path.c_str(), (int)index.ntotal);
        std::vector<int> qlabels = acorn::read_ibin(qlabel_path.c_str(), nq_all);
        auto gr = acorn::read_groundtruth(gt_path.c_str());
        std::vector<int> gt_ids = std::move(gr.first);
        int gt_nq = gr.second.first;
        int gt_k = gr.second.second;
        if (gt_nq < nq || gt_k < k)
        {
            std::fprintf(stderr, "GT too small for %s: nq=%d k=%d\n", cfg.sel, gt_nq, gt_k);
            return 1;
        }
        int qlabel = qlabels[0];

        std::vector<char> wf(index.ntotal, 0);
        for (int i = 0; i < (int)base_labels.size(); i++)
            if (base_labels[i] == qlabel)
                wf[i] = 1;

        for (int cost : costs)
        {
            acorn::set_filter_check_cost(cost);
            for (const std::string &strategy : strategies)
            {
                bool is_pre = strategy == "pre_parallel";
                std::string strategy_out = display_strategy(strategy, compact_output);
                if (is_pre && pre_max_selectivity > 0.0 && cfg.true_s > pre_max_selectivity)
                {
                    if (compact_output)
                        std::fprintf(out, "%s\t%.6f\t%d\t%s\t0\t%d\t%d\tnan\tnan\t0\t0\t0\t0\t0\tskipped_pre_high_selectivity\n",
                                     cfg.sel, cfg.true_s, cost, strategy_out.c_str(), threads, nq);
                    else
                        std::fprintf(out, "%s\t%.6f\t%d\t%s\t0\t%d\t%d\t%d\tnan\tnan\tnan\t0\t0\t0\t0\t0\tskipped_pre_high_selectivity\n",
                                     cfg.sel, cfg.true_s, cost, strategy_out.c_str(), threads, k, nq);
                    std::fflush(out);
                    continue;
                }
                if (is_pre && !contains_int(pre_costs, cost))
                {
                    if (compact_output)
                        std::fprintf(out, "%s\t%.6f\t%d\t%s\t0\t%d\t%d\tnan\tnan\t0\t0\t0\t0\t0\tskipped_pre_cost\n",
                                     cfg.sel, cfg.true_s, cost, strategy_out.c_str(), threads, nq);
                    else
                        std::fprintf(out, "%s\t%.6f\t%d\t%s\t0\t%d\t%d\t%d\tnan\tnan\tnan\t0\t0\t0\t0\t0\tskipped_pre_cost\n",
                                     cfg.sel, cfg.true_s, cost, strategy_out.c_str(), threads, k, nq);
                    std::fflush(out);
                    continue;
                }

                const std::vector<int> *efs_values = &efs_list;
                std::vector<int> pre_efs = {0};
                if (is_pre)
                    efs_values = &pre_efs;

                for (int efs : *efs_values)
                {
                    std::printf("sel=%s cost=%d strategy=%s efs=%d\n",
                                cfg.sel, cost, strategy.c_str(), efs);
                    RunResult r = run_method(strategy, index, queries, qd, nq, k, efs,
                                             threads, Helec, repeat, metric, wf, qlabel,
                                             base_labels, gt_ids, gt_k);
                    if (compact_output)
                        std::fprintf(out, "%s\t%.6f\t%d\t%s\t%d\t%d\t%d\t%.3f\t%.4f\t%zu\t%zu\t%d\t%d\t%d\tok\n",
                                     cfg.sel, cfg.true_s, cost, strategy_out.c_str(), efs,
                                     threads, nq, r.avg_ms, r.recall, r.ndc, r.ndc_max,
                                     r.ok, r.empty, r.wrong);
                    else
                        std::fprintf(out, "%s\t%.6f\t%d\t%s\t%d\t%d\t%d\t%d\t%.3f\t%.1f\t%.4f\t%zu\t%zu\t%d\t%d\t%d\tok\n",
                                     cfg.sel, cfg.true_s, cost, strategy_out.c_str(), efs,
                                     threads, k, nq, r.avg_ms, r.total_ms, r.recall, r.ndc, r.ndc_max,
                                     r.ok, r.empty, r.wrong);
                    std::fflush(out);
                    if (!is_pre && max_avg_ms > 0.0 && r.avg_ms > max_avg_ms)
                    {
                        for (int skipped_efs : *efs_values)
                        {
                            if (skipped_efs <= efs)
                                continue;
                            if (compact_output)
                                std::fprintf(out, "%s\t%.6f\t%d\t%s\t%d\t%d\t%d\tnan\tnan\t0\t0\t0\t0\t0\tskipped_max_avg_ms\n",
                                             cfg.sel, cfg.true_s, cost, strategy_out.c_str(), skipped_efs,
                                             threads, nq);
                            else
                                std::fprintf(out, "%s\t%.6f\t%d\t%s\t%d\t%d\t%d\t%d\tnan\tnan\tnan\t0\t0\t0\t0\t0\tskipped_max_avg_ms\n",
                                             cfg.sel, cfg.true_s, cost, strategy_out.c_str(), skipped_efs,
                                             threads, k, nq);
                        }
                        std::fflush(out);
                        break;
                    }
                }
            }
        }
    }

    std::fclose(out);
    std::printf("Wrote %s\n", out_file);
    return 0;
}
