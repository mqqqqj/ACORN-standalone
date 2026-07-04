#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
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
            int id = results[q * k + j];
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
                 "Usage: %s --index <index> --query <query.fbin> --out <out.tsv> [options]\n"
                 "Options:\n"
                 "  --nq <int>              Queries per bucket (default: 100)\n"
                 "  --k <int>               top-k (default: 100)\n"
                 "  --threads <int>         threads (default: 4)\n"
                 "  --efs <int>             fixed efs (default: 1000)\n"
                 "  --efs-min <int>         target mode minimum efs (default: 100)\n"
                 "  --efs-max <int>         target mode maximum efs (default: fixed efs)\n"
                 "  --efs-step <int>        target mode efs step (default: 100)\n"
                 "  --target-recall <float> enable target mode, e.g. 0.9\n"
                 "  --best-out <out.tsv>    target mode best-strategy output\n"
                 "  --Helec <int>           ScatterSearch Helec (default: 50)\n"
                 "  --sample-size <int>     selectivity sampling size (default: 1024)\n"
                 "  --dist-cost-ns <float>  profiled distance cost (default: 420)\n",
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

static const char *select_strategy(double s_hat, double rho)
{
    if (s_hat <= 0.015 && rho < 0.25)
        return "pre_parallel";
    if (s_hat <= 0.015)
        return "post_parallel";
    if (rho >= 0.8)
        return "post_parallel";
    if (s_hat >= 0.5)
        return "post_parallel";
    return "scatter";
}

static double estimate_selectivity(const std::vector<int> &labels, int qlabel,
                                   int sample_size, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dis(0, (int)labels.size() - 1);
    int hit = 0;
    for (int i = 0; i < sample_size; i++)
    {
        int id = dis(rng);
        if (labels[id] == qlabel)
            hit++;
    }
    const double alpha = 1.0;
    return (hit + alpha) / (sample_size + 2.0 * alpha);
}

static double measure_filter_cost_ns(int cost)
{
    const int iters = 500000;
    const int nlabels = 1 << 20;
    std::vector<char> filter_map(nlabels);
    for (int i = 0; i < nlabels; i++)
        filter_map[i] = (i & 1) ? 1 : 0;

    volatile int sink = 0;
    acorn::set_filter_check_cost(cost);
    for (int i = 0; i < 10000; i++)
        sink += acorn::check_filter(filter_map.data(), i & (nlabels - 1));

    double t0 = now_ms();
    for (int i = 0; i < iters; i++)
        sink += acorn::check_filter(filter_map.data(), i & (nlabels - 1));
    double ms = now_ms() - t0;
    if (sink == 1234567)
        std::fprintf(stderr, "sink=%d\n", (int)sink);
    return ms * 1e6 / iters;
}

struct RunResult
{
    double avg_ms = 0;
    double total_ms = 0;
    double recall = 0;
    size_t ndc = 0;
    int ok = 0;
    int empty = 0;
    int wrong = 0;
};

static RunResult run_method(const char *method,
                            const acorn::ACORN &index,
                            const std::vector<float> &queries,
                            int qd,
                            int nq,
                            int k,
                            int efs,
                            int threads,
                            int Helec,
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

    double total = 0;
    for (int i = 0; i < nq; i++)
    {
        const float *q = queries.data() + (size_t)i * qd;
        int *out = labels.data() + (size_t)i * k;
        float *dst = dist.data() + (size_t)i * k;
        double t0 = now_ms();
        if (std::strcmp(method, "iqan") == 0)
        {
            index.iqan_search(q, index.get_xb(), index.d, metric, k, efs,
                              out, dst, threads, wf.data());
        }
        else if (std::strcmp(method, "scatter") == 0)
        {
            index.scatter_search(q, index.get_xb(), index.d, metric, k, efs,
                                 out, dst, threads, Helec, wf.data());
        }
        else if (std::strcmp(method, "post_parallel_iqan") == 0)
        {
            index.parallel_post_filter_iqan_search(q, index.get_xb(), index.d, metric, k, efs,
                                                   out, dst, threads, wf.data());
        }
        else if (std::strcmp(method, "post_parallel") == 0)
        {
            index.parallel_post_filter_search(q, index.get_xb(), index.d, metric, k, efs,
                                              out, dst, threads, Helec, wf.data());
        }
        else if (std::strcmp(method, "pre_parallel") == 0)
        {
            index.parallel_pre_filter_search(q, index.get_xb(), index.d, metric, k,
                                             out, dst, threads, wf.data());
        }
        total += now_ms() - t0;
    }

    RunResult r;
    r.total_ms = total;
    r.avg_ms = total / nq;
    r.recall = compute_recall(nq, k, gt_k, gt_ids, labels);
    const auto &tnd = acorn::get_thread_ndis();
    for (size_t v : tnd)
        r.ndc += v;
    check_filter_result(nq, k, labels, qlabel, base_labels, &r.ok, &r.empty, &r.wrong);
    return r;
}

struct TargetHit
{
    std::string strategy;
    int efs = -1;
    RunResult result;
    bool met = false;
};

int main(int argc, char **argv)
{
    const char *index_file = nullptr;
    const char *query_file = nullptr;
    const char *out_file = nullptr;
    const char *best_out_file = nullptr;
    int nq = 100;
    int k = 100;
    int threads = 4;
    int efs = 1000;
    int efs_min = 100;
    int efs_max = -1;
    int efs_step = 100;
    int Helec = 50;
    int sample_size = 1024;
    double dist_cost_ns = 420.0;
    double target_recall = -1.0;

    for (int i = 1; i < argc; i++)
    {
        if (std::strcmp(argv[i], "--index") == 0 && i + 1 < argc)
            index_file = argv[++i];
        else if (std::strcmp(argv[i], "--query") == 0 && i + 1 < argc)
            query_file = argv[++i];
        else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc)
            out_file = argv[++i];
        else if (std::strcmp(argv[i], "--nq") == 0 && i + 1 < argc)
            nq = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc)
            k = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            threads = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--efs") == 0 && i + 1 < argc)
            efs = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--efs-min") == 0 && i + 1 < argc)
            efs_min = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--efs-max") == 0 && i + 1 < argc)
            efs_max = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--efs-step") == 0 && i + 1 < argc)
            efs_step = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--target-recall") == 0 && i + 1 < argc)
            target_recall = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--best-out") == 0 && i + 1 < argc)
            best_out_file = argv[++i];
        else if (std::strcmp(argv[i], "--Helec") == 0 && i + 1 < argc)
            Helec = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sample-size") == 0 && i + 1 < argc)
            sample_size = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--dist-cost-ns") == 0 && i + 1 < argc)
            dist_cost_ns = std::atof(argv[++i]);
        else
            usage(argv[0]);
    }
    if (!index_file || !query_file || !out_file || nq <= 0 || k <= 0)
        usage(argv[0]);
    if (efs_max < 0)
        efs_max = efs;
    if (target_recall > 0 && (!best_out_file || efs_min <= 0 || efs_max < efs_min || efs_step <= 0))
        usage(argv[0]);

    const LabelConfig configs[] = {
        {"0.1", 0.001, "s0p1", "query_labels_all1.ibin", "s0p1"},
        {"1", 0.01, "s1", "query_labels_all1.ibin", "s1"},
        {"5", 0.05, "s5", "query_labels_all1.ibin", "s5"},
        {"10", 0.10, "s10", "query_labels_all1.ibin", "s10"},
        {"20", 0.20, "s20", "query_labels_all1.ibin", "s20"},
        {"50", 0.50, "s50", "query_labels_all1.ibin", "s50_label1"},
        {"80", 0.80, "s20", "query_labels_all2.ibin", "s80"},
    };
    const int costs[] = {0, 64, 128, 192, 256, 384, 512};

    std::printf("Loading index...\n");
    acorn::ACORN index;
    index.load_from_faiss(index_file);
    int metric = (index.metric_type == acorn::METRIC_INNER_PRODUCT) ? 0 : 1;

    std::printf("Loading queries...\n");
    auto qr = acorn::read_fbin(query_file);
    std::vector<float> queries = std::move(qr.first);
    int nq_all = qr.second.first;
    int qd = qr.second.second;
    if (nq > nq_all)
        nq = nq_all;
    if (qd != index.d)
    {
        std::fprintf(stderr, "query dim mismatch\n");
        return 1;
    }

    std::vector<double> filter_ns;
    for (int c : costs)
        filter_ns.push_back(measure_filter_cost_ns(c));

    FILE *out = std::fopen(out_file, "w");
    if (!out)
    {
        std::fprintf(stderr, "cannot open output: %s\n", out_file);
        return 1;
    }
    FILE *best_out = nullptr;
    if (target_recall > 0)
    {
        best_out = std::fopen(best_out_file, "w");
        if (!best_out)
        {
            std::fprintf(stderr, "cannot open best output: %s\n", best_out_file);
            std::fclose(out);
            return 1;
        }
        std::fprintf(out, "selectivity\tfilter_cost\tfilter_ns\trho\tstrategy\ts_hat\ttrue_s\tefs\tavg_ms\ttime_ms\trecall\tndc\tok\tempty\twrong\ttarget_recall\ttarget_met\tfirst_hit\n");
        std::fprintf(best_out, "selectivity\tfilter_cost\tfilter_ns\trho\ts_hat\ttrue_s\ttarget_recall\tchosen_strategy\tchosen_efs\tavg_ms\ttime_ms\trecall\tndc\tok\tempty\twrong\n");
    }
    else
    {
        std::fprintf(out, "selectivity\tfilter_cost\tfilter_ns\trho\tmethod\tchosen_strategy\ts_hat\ttrue_s\tefs\tavg_ms\ttime_ms\trecall\tndc\tok\tempty\twrong\n");
    }

    for (const auto &cfg : configs)
    {
        std::string base_path = std::string("data/laion10m/base_labels_binary_") + cfg.base_suffix + ".ibin";
        std::string qlabel_path = std::string("data/laion10m/") + cfg.qlabel_file;
        std::string gt_path = std::string("data/laion10m/laion10m_gt_binary_") + cfg.gt_suffix + ".ibin";

        std::printf("Loading labels for selectivity %s...\n", cfg.sel);
        std::vector<int> base_labels = acorn::read_ibin(base_path.c_str(), (int)index.ntotal);
        std::vector<int> qlabels = acorn::read_ibin(qlabel_path.c_str(), nq_all);
        int qlabel = qlabels[0];
        auto gr = acorn::read_groundtruth(gt_path.c_str());
        std::vector<int> gt_ids = std::move(gr.first);
        int gt_k = gr.second.second;

        std::vector<char> wf(index.ntotal, 0);
        for (int i = 0; i < (int)base_labels.size(); i++)
            if (base_labels[i] == qlabel)
                wf[i] = 1;

        double s_hat = estimate_selectivity(base_labels, qlabel, sample_size, 12345);

        if (target_recall > 0)
        {
            const char *strategies[] = {
                "pre_parallel",
                "scatter",
                "iqan",
                "post_parallel",
                "post_parallel_iqan",
            };
            std::vector<TargetHit> hits;

            std::printf("sel=%s target-discovery cost=0 s_hat=%.5f\n", cfg.sel, s_hat);
            for (const char *strategy : strategies)
            {
                const bool is_pre = std::strcmp(strategy, "pre_parallel") == 0;
                if (is_pre && cfg.true_s > 0.02)
                    continue;

                TargetHit hit;
                hit.strategy = strategy;
                const int first_efs = is_pre ? 0 : efs_min;
                const int last_efs = is_pre ? 0 : efs_max;
                const int step = is_pre ? 1 : efs_step;

                for (int e = first_efs; e <= last_efs; e += step)
                {
                    acorn::set_filter_check_cost(costs[0]);
                    RunResult r = run_method(strategy, index, queries, qd, nq, k, e, threads, Helec,
                                             metric, wf, qlabel, base_labels, gt_ids, gt_k);
                    bool met = r.recall > target_recall;
                    bool first_hit = met && !hit.met;
                    std::fprintf(out, "%s\t%d\t%.3f\t%.6f\t%s\t%.6f\t%.6f\t%d\t%.3f\t%.1f\t%.4f\t%zu\t%d\t%d\t%d\t%.4f\t%d\t%d\n",
                                 cfg.sel, costs[0], filter_ns[0], filter_ns[0] / dist_cost_ns,
                                 strategy, s_hat, cfg.true_s, e, r.avg_ms, r.total_ms,
                                 r.recall, r.ndc, r.ok, r.empty, r.wrong,
                                 target_recall, met ? 1 : 0, first_hit ? 1 : 0);
                    std::fflush(out);

                    if (first_hit)
                    {
                        hit.efs = e;
                        hit.result = r;
                        hit.met = true;
                        break;
                    }
                }
                hits.push_back(hit);
            }

            for (size_t ci = 0; ci < sizeof(costs) / sizeof(costs[0]); ci++)
            {
                int cost = costs[ci];
                double rho = filter_ns[ci] / dist_cost_ns;
                std::printf("sel=%s cost=%d s_hat=%.5f rho=%.3f target-eval\n",
                            cfg.sel, cost, s_hat, rho);

                std::vector<TargetHit> evaluated;
                for (const auto &hit : hits)
                {
                    if (!hit.met)
                        continue;

                    TargetHit eval;
                    eval.strategy = hit.strategy;
                    eval.efs = hit.efs;
                    eval.met = true;
                    if (ci == 0)
                    {
                        eval.result = hit.result;
                    }
                    else
                    {
                        acorn::set_filter_check_cost(cost);
                        eval.result = run_method(hit.strategy.c_str(), index, queries, qd, nq, k,
                                                 hit.efs, threads, Helec, metric, wf, qlabel,
                                                 base_labels, gt_ids, gt_k);
                        std::fprintf(out, "%s\t%d\t%.3f\t%.6f\t%s\t%.6f\t%.6f\t%d\t%.3f\t%.1f\t%.4f\t%zu\t%d\t%d\t%d\t%.4f\t%d\t%d\n",
                                     cfg.sel, cost, filter_ns[ci], rho, hit.strategy.c_str(),
                                     s_hat, cfg.true_s, hit.efs, eval.result.avg_ms,
                                     eval.result.total_ms, eval.result.recall, eval.result.ndc,
                                     eval.result.ok, eval.result.empty, eval.result.wrong,
                                     target_recall, 1, 1);
                        std::fflush(out);
                    }
                    evaluated.push_back(eval);
                }

                TargetHit *best = nullptr;
                for (auto &hit : evaluated)
                {
                    if (!hit.met)
                        continue;
                    if (!best || hit.result.avg_ms < best->result.avg_ms)
                        best = &hit;
                }

                if (best)
                {
                    std::fprintf(best_out, "%s\t%d\t%.3f\t%.6f\t%.6f\t%.6f\t%.4f\t%s\t%d\t%.3f\t%.1f\t%.4f\t%zu\t%d\t%d\t%d\n",
                                 cfg.sel, cost, filter_ns[ci], rho, s_hat, cfg.true_s, target_recall,
                                 best->strategy.c_str(), best->efs, best->result.avg_ms,
                                 best->result.total_ms, best->result.recall, best->result.ndc,
                                 best->result.ok, best->result.empty, best->result.wrong);
                }
                else
                {
                    std::fprintf(best_out, "%s\t%d\t%.3f\t%.6f\t%.6f\t%.6f\t%.4f\tnone\t-1\tnan\tnan\tnan\t0\t0\t0\t0\n",
                                 cfg.sel, cost, filter_ns[ci], rho, s_hat, cfg.true_s, target_recall);
                }
                std::fflush(best_out);
            }
            continue;
        }

        for (size_t ci = 0; ci < sizeof(costs) / sizeof(costs[0]); ci++)
        {
            int cost = costs[ci];
            double rho = filter_ns[ci] / dist_cost_ns;
            const char *adaptive_strategy = select_strategy(s_hat, rho);
            std::printf("sel=%s cost=%d s_hat=%.5f rho=%.3f\n",
                        cfg.sel, cost, s_hat, rho);

            acorn::set_filter_check_cost(cost);
            RunResult fixed = run_method("iqan", index, queries, qd, nq, k, efs, threads, Helec,
                                         metric, wf, qlabel, base_labels, gt_ids, gt_k);
            std::fprintf(out, "%s\t%d\t%.3f\t%.6f\tfixed_iqan\tiqan\t%.6f\t%.6f\t%d\t%.3f\t%.1f\t%.4f\t%zu\t%d\t%d\t%d\n",
                         cfg.sel, cost, filter_ns[ci], rho, s_hat, cfg.true_s, efs,
                         fixed.avg_ms, fixed.total_ms, fixed.recall, fixed.ndc,
                         fixed.ok, fixed.empty, fixed.wrong);
            std::fflush(out);

            acorn::set_filter_check_cost(cost);
            RunResult adaptive = run_method(adaptive_strategy, index, queries, qd, nq, k, efs,
                                            threads, Helec, metric, wf, qlabel, base_labels,
                                            gt_ids, gt_k);
            std::fprintf(out, "%s\t%d\t%.3f\t%.6f\tadaptive\t%s\t%.6f\t%.6f\t%d\t%.3f\t%.1f\t%.4f\t%zu\t%d\t%d\t%d\n",
                         cfg.sel, cost, filter_ns[ci], rho, adaptive_strategy, s_hat,
                         cfg.true_s, efs, adaptive.avg_ms, adaptive.total_ms,
                         adaptive.recall, adaptive.ndc, adaptive.ok, adaptive.empty,
                         adaptive.wrong);
            std::fflush(out);
        }
    }

    if (best_out)
        std::fclose(best_out);
    std::fclose(out);
    std::printf("Wrote %s\n", out_file);
    if (best_out_file)
        std::printf("Wrote %s\n", best_out_file);
    return 0;
}
