#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "acorn/acorn_graph.h"
#include "acorn/file_io.h"

static double now_ms()
{
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

static std::vector<std::string> split_tab(const std::string &line)
{
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, '\t'))
        out.push_back(item);
    return out;
}

static std::vector<int> parse_int_list(const std::string &csv)
{
    std::vector<int> out;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        if (item.empty())
            continue;
        out.push_back(std::atoi(item.c_str()));
    }
    return out;
}

static std::string key3(const std::string &a, int b, const std::string &c)
{
    return a + "\t" + std::to_string(b) + "\t" + c;
}

static void usage(const char *prog)
{
    std::fprintf(stderr,
                 "Usage: %s --index <index> --query <query.fbin> --base-labels <ibin>\n"
                 "          --query-labels <ibin> --query-costs <ibin> --cost-rho <tsv>\n"
                 "          --lookup <tsv> --out <out.tsv> [options]\n"
                 "\n"
                 "Options:\n"
                 "  --gt <gt.ibin>                 optional groundtruth for recall\n"
                 "  --mode <adaptive|fixed_scatter|fixed_iqan|fixed_post_scatter|fixed_post_iqan> default adaptive\n"
                 "  --nq <int>                     default all\n"
                 "  --k <int>                      default 100\n"
                 "  --threads <int>                default 4\n"
                 "  --Helec <int>                  default 50\n"
                 "  --sample-size <int>            default 1024\n"
                 "  --sample-seed <int>            default 12345\n"
                 "  --fixed-efs <int>              default 800 for fixed baselines/fallback\n"
                 "  --fixed-efs-list <csv>         sweep fixed baselines in one loaded process\n"
                 "  --out-prefix <prefix>          output prefix for --fixed-efs-list\n"
                 "  --filtered-expand-target <int> default 0 (ACORN original M*2)\n"
                 "  --rho-threshold <float>        default 0.3\n",
                 prog);
    std::exit(1);
}

struct CostRhoRow
{
    std::string selectivity;
    double true_s = 0.0;
    int filter_cost = 0;
    double rho = 0.0;
};

struct LookupRow
{
    std::string selectivity;
    double true_s = 0.0;
    int filter_cost = 0;
    double rho = 0.0;
    std::string strategy;
    int best_efs = -1;
    double avg_ms = NAN;
    double recall = NAN;
    std::string status;
};

struct QueryResult
{
    int qid = 0;
    int qlabel = 0;
    int qcost = 0;
    std::string selectivity_bucket;
    double bucket_s = 0.0;
    double s_hat = 0.0;
    double rho = 0.0;
    std::string strategy;
    int efs = -1;
    double selector_ms = 0.0;
    double search_ms = 0.0;
    double total_ms = 0.0;
    double recall = NAN;
    size_t ndc = 0;
    size_t ndc_max = 0;
    int ok = 0;
    int empty = 0;
    int wrong = 0;
    std::string status = "ok";
};

static std::vector<CostRhoRow> read_cost_rho(const char *path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error(std::string("cannot open cost-rho table: ") + path);
    std::string line;
    if (!std::getline(in, line))
        throw std::runtime_error("empty cost-rho table");
    std::vector<CostRhoRow> rows;
    while (std::getline(in, line))
    {
        if (line.empty())
            continue;
        auto p = split_tab(line);
        if (p.size() < 6)
            continue;
        CostRhoRow r;
        r.selectivity = p[0];
        r.true_s = std::atof(p[1].c_str());
        r.filter_cost = std::atoi(p[2].c_str());
        r.rho = std::atof(p[5].c_str());
        rows.push_back(r);
    }
    return rows;
}

static std::unordered_map<std::string, LookupRow> read_lookup(const char *path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error(std::string("cannot open selector lookup: ") + path);
    std::string line;
    if (!std::getline(in, line))
        throw std::runtime_error("empty selector lookup");
    std::unordered_map<std::string, LookupRow> out;
    while (std::getline(in, line))
    {
        if (line.empty())
            continue;
        auto p = split_tab(line);
        if (p.size() < 10)
            continue;
        LookupRow r;
        r.selectivity = p[0];
        r.true_s = std::atof(p[1].c_str());
        r.filter_cost = std::atoi(p[2].c_str());
        r.rho = std::atof(p[3].c_str());
        r.strategy = p[4];
        r.best_efs = std::atoi(p[6].c_str());
        r.avg_ms = p[7] == "nan" ? NAN : std::atof(p[7].c_str());
        r.recall = p[8] == "nan" ? NAN : std::atof(p[8].c_str());
        r.status = p[9];
        out[key3(r.selectivity, r.filter_cost, r.strategy)] = r;
    }
    return out;
}

static const CostRhoRow &nearest_selectivity_bucket(
    const std::vector<CostRhoRow> &rows, double s_hat, int cost)
{
    const CostRhoRow *best = nullptr;
    double best_diff = std::numeric_limits<double>::max();
    for (const auto &r : rows)
    {
        if (r.filter_cost != cost)
            continue;
        double diff = std::fabs(std::log(std::max(s_hat, 1e-9)) -
                                std::log(std::max(r.true_s, 1e-9)));
        if (!best || diff < best_diff)
        {
            best = &r;
            best_diff = diff;
        }
    }
    if (!best)
        throw std::runtime_error("query cost not found in cost-rho table: " + std::to_string(cost));
    return *best;
}

static std::string choose_strategy(double s_hat, double rho, double rho_threshold)
{
    if (s_hat < 0.01)
        return "parallel_pre";
    if (s_hat >= 0.01 && s_hat <= 0.10 && rho < rho_threshold)
        return "parallel_in_scatter";
    return "parallel_post_scatter";
}

static double estimate_selectivity(const std::vector<char> &filter_map,
                                   int sample_size,
                                   std::mt19937 &rng)
{
    std::uniform_int_distribution<int> dis(0, (int)filter_map.size() - 1);
    int hit = 0;
    for (int i = 0; i < sample_size; i++)
    {
        int id = dis(rng);
        if (acorn::check_filter(filter_map.data(), id))
            hit++;
    }
    const double alpha = 1.0;
    return (hit + alpha) / (sample_size + 2.0 * alpha);
}

static std::vector<char> build_filter_map(const std::vector<int> &base_labels, int qlabel)
{
    std::vector<char> filter_map(base_labels.size(), 0);
    for (size_t i = 0; i < base_labels.size(); i++)
        if (base_labels[i] == qlabel)
            filter_map[i] = 1;
    return filter_map;
}

static void check_one_filter_result(int k,
                                    const std::vector<int> &ids,
                                    int qlabel,
                                    const std::vector<int> &base_labels,
                                    int *ok, int *empty, int *wrong)
{
    *ok = *empty = *wrong = 0;
    for (int i = 0; i < k; i++)
    {
        int id = ids[i];
        if (id < 0)
            (*empty)++;
        else if (base_labels[id] != qlabel)
            (*wrong)++;
        else
            (*ok)++;
    }
}

static double compute_one_recall(int qid, int k, int gt_k,
                                 const std::vector<int> &gt,
                                 const std::vector<int> &ids)
{
    if (gt.empty() || gt_k < k)
        return NAN;
    std::vector<int> truth(gt.begin() + (size_t)qid * gt_k,
                           gt.begin() + (size_t)qid * gt_k + gt_k);
    std::sort(truth.begin(), truth.end());
    int hits = 0;
    for (int i = 0; i < k; i++)
        if (ids[i] >= 0 && std::binary_search(truth.begin(), truth.end(), ids[i]))
            hits++;
    return (double)hits / k;
}

static void run_search(const std::string &strategy,
                       const acorn::ACORN &index,
                       const float *q,
                       int metric,
                       int k,
                       int efs,
                       int threads,
                       int Helec,
                       const std::vector<char> &filter_map,
                       std::vector<int> &ids,
                       std::vector<float> &dist)
{
    if (strategy == "parallel_pre")
    {
        index.parallel_pre_filter_search(q, index.get_xb(), index.d, metric, k,
                                         ids.data(), dist.data(), threads,
                                         filter_map.data());
    }
    else if (strategy == "parallel_in_scatter")
    {
        index.scatter_search(q, index.get_xb(), index.d, metric, k, efs,
                             ids.data(), dist.data(), threads, Helec,
                             filter_map.data());
    }
    else if (strategy == "parallel_in_iqan")
    {
        index.iqan_search(q, index.get_xb(), index.d, metric, k, efs,
                          ids.data(), dist.data(), threads, filter_map.data());
    }
    else if (strategy == "parallel_post_scatter")
    {
        index.parallel_post_filter_search(q, index.get_xb(), index.d, metric, k, efs,
                                          ids.data(), dist.data(), threads, Helec,
                                          filter_map.data());
    }
    else if (strategy == "parallel_post_iqan")
    {
        index.parallel_post_filter_iqan_search(q, index.get_xb(), index.d, metric, k, efs,
                                               ids.data(), dist.data(), threads,
                                               filter_map.data());
    }
    else
    {
        throw std::runtime_error("unsupported strategy: " + strategy);
    }
}

int main(int argc, char **argv)
{
    const char *index_file = nullptr;
    const char *query_file = nullptr;
    const char *base_labels_file = nullptr;
    const char *query_labels_file = nullptr;
    const char *query_costs_file = nullptr;
    const char *gt_file = nullptr;
    const char *cost_rho_file = nullptr;
    const char *lookup_file = nullptr;
    const char *out_file = nullptr;
    const char *out_prefix = nullptr;
    std::string mode = "adaptive";
    int nq = -1;
    int k = 100;
    int threads = 4;
    int Helec = 50;
    int sample_size = 1024;
    int sample_seed = 12345;
    int fixed_efs = 800;
    std::vector<int> fixed_efs_list;
    int filtered_expand_target = 0;
    double rho_threshold = 0.3;

    for (int i = 1; i < argc; i++)
    {
        if (std::strcmp(argv[i], "--index") == 0 && i + 1 < argc)
            index_file = argv[++i];
        else if (std::strcmp(argv[i], "--query") == 0 && i + 1 < argc)
            query_file = argv[++i];
        else if (std::strcmp(argv[i], "--base-labels") == 0 && i + 1 < argc)
            base_labels_file = argv[++i];
        else if (std::strcmp(argv[i], "--query-labels") == 0 && i + 1 < argc)
            query_labels_file = argv[++i];
        else if (std::strcmp(argv[i], "--query-costs") == 0 && i + 1 < argc)
            query_costs_file = argv[++i];
        else if (std::strcmp(argv[i], "--gt") == 0 && i + 1 < argc)
            gt_file = argv[++i];
        else if (std::strcmp(argv[i], "--cost-rho") == 0 && i + 1 < argc)
            cost_rho_file = argv[++i];
        else if (std::strcmp(argv[i], "--lookup") == 0 && i + 1 < argc)
            lookup_file = argv[++i];
        else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc)
            out_file = argv[++i];
        else if (std::strcmp(argv[i], "--out-prefix") == 0 && i + 1 < argc)
            out_prefix = argv[++i];
        else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
            mode = argv[++i];
        else if (std::strcmp(argv[i], "--nq") == 0 && i + 1 < argc)
            nq = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc)
            k = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            threads = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--Helec") == 0 && i + 1 < argc)
            Helec = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sample-size") == 0 && i + 1 < argc)
            sample_size = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sample-seed") == 0 && i + 1 < argc)
            sample_seed = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--fixed-efs") == 0 && i + 1 < argc)
            fixed_efs = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--fixed-efs-list") == 0 && i + 1 < argc)
            fixed_efs_list = parse_int_list(argv[++i]);
        else if (std::strcmp(argv[i], "--filtered-expand-target") == 0 && i + 1 < argc)
            filtered_expand_target = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--rho-threshold") == 0 && i + 1 < argc)
            rho_threshold = std::atof(argv[++i]);
        else
            usage(argv[0]);
    }

    if (!index_file || !query_file || !base_labels_file || !query_labels_file ||
        !query_costs_file || !cost_rho_file || !lookup_file ||
        k <= 0 || threads <= 0 || sample_size <= 0 ||
        (mode != "adaptive" && mode != "fixed_scatter" && mode != "fixed_iqan" &&
         mode != "fixed_post_scatter" && mode != "fixed_post_iqan"))
        usage(argv[0]);
    if (fixed_efs_list.empty())
    {
        if (!out_file)
            usage(argv[0]);
    }
    else
    {
        if (mode == "adaptive" || !out_prefix)
            usage(argv[0]);
        for (int efs : fixed_efs_list)
            if (efs <= 0)
                usage(argv[0]);
    }

    acorn::set_filtered_expand_target(filtered_expand_target);

    std::printf("Loading index: %s\n", index_file);
    acorn::ACORN index;
    index.load_from_faiss(index_file);
    int metric = (index.metric_type == acorn::METRIC_INNER_PRODUCT) ? 0 : 1;
    std::printf("  ntotal=%ld d=%d metric=%s\n",
                index.ntotal, index.d, metric == 0 ? "ip" : "l2");

    std::printf("Loading queries: %s\n", query_file);
    auto qr = acorn::read_fbin(query_file);
    std::vector<float> queries = std::move(qr.first);
    int nq_all = qr.second.first;
    int qd = qr.second.second;
    if (qd != index.d)
        throw std::runtime_error("query dimension mismatch");
    if (nq <= 0 || nq > nq_all)
        nq = nq_all;

    std::vector<int> base_labels = acorn::read_ibin(base_labels_file, (int)index.ntotal);
    std::vector<int> query_labels = acorn::read_ibin(query_labels_file, -1);
    std::vector<int> query_costs = acorn::read_ibin(query_costs_file, -1);
    if ((int)query_labels.size() < nq || (int)query_costs.size() < nq)
        throw std::runtime_error("query label/cost file shorter than nq");

    std::vector<int> gt;
    int gt_k = 0;
    if (gt_file)
    {
        auto gr = acorn::read_groundtruth(gt_file);
        gt = std::move(gr.first);
        if (gr.second.first < nq || gr.second.second < k)
            throw std::runtime_error("groundtruth too small");
        gt_k = gr.second.second;
    }

    std::vector<CostRhoRow> cost_rho = read_cost_rho(cost_rho_file);
    auto lookup = read_lookup(lookup_file);
    std::map<int, std::vector<char>> filter_maps;
    std::set<int> needed_labels;
    for (int i = 0; i < nq; i++)
        needed_labels.insert(query_labels[i]);
    std::printf("Prebuilding %zu filter maps...\n", needed_labels.size());
    for (int label : needed_labels)
        filter_maps.emplace(label, build_filter_map(base_labels, label));

    std::vector<int> run_efs_list;
    if (fixed_efs_list.empty())
        run_efs_list.push_back(fixed_efs);
    else
        run_efs_list = fixed_efs_list;

    for (int run_efs : run_efs_list)
    {
        std::string run_out_file;
        if (fixed_efs_list.empty())
        {
            run_out_file = out_file;
        }
        else
        {
            run_out_file = std::string(out_prefix) + "_efs" + std::to_string(run_efs) +
                           "_nq" + std::to_string(nq) + ".tsv";
        }

        FILE *out = std::fopen(run_out_file.c_str(), "w");
        if (!out)
            throw std::runtime_error(std::string("cannot open output: ") + run_out_file);
        std::fprintf(out,
                     "qid\tmode\tqlabel\tquery_cost\ts_hat\tselectivity_bucket\tbucket_s\trho\tstrategy\tefs\tselector_ms\tsearch_ms\ttotal_ms\trecall\tndc\tndc_max\tok\tempty\twrong\tstatus\n");

        std::mt19937 rng(sample_seed);
        double sum_selector = 0.0, sum_search = 0.0, sum_total = 0.0;
        double sum_recall = 0.0;
        int recall_count = 0;
        long long sum_ok = 0, sum_empty = 0, sum_wrong = 0;

        for (int qi = 0; qi < nq; qi++)
        {
            QueryResult r;
            r.qid = qi;
            r.qlabel = query_labels[qi];
            r.qcost = query_costs[qi];

            auto it_map = filter_maps.find(r.qlabel);
            if (it_map == filter_maps.end())
                throw std::runtime_error("missing prebuilt filter map");
            const std::vector<char> &filter_map = it_map->second;

            double selector_t0 = now_ms();
            acorn::set_filter_check_cost(r.qcost);
            r.s_hat = estimate_selectivity(filter_map, sample_size, rng);
            const CostRhoRow &bucket = nearest_selectivity_bucket(cost_rho, r.s_hat, r.qcost);
            r.selectivity_bucket = bucket.selectivity;
            r.bucket_s = bucket.true_s;
            r.rho = bucket.rho;

            if (mode == "adaptive")
            {
                r.strategy = choose_strategy(r.s_hat, r.rho, rho_threshold);
            }
            else
            {
                if (mode == "fixed_scatter")
                    r.strategy = "parallel_in_scatter";
                else if (mode == "fixed_post_scatter")
                    r.strategy = "parallel_post_scatter";
                else if (mode == "fixed_post_iqan")
                    r.strategy = "parallel_post_iqan";
                else
                    r.strategy = "parallel_in_iqan";
            }

            if (mode == "fixed_scatter" || mode == "fixed_iqan" ||
                mode == "fixed_post_scatter" || mode == "fixed_post_iqan")
            {
                r.efs = run_efs;
            }
            else
            {
                auto it_lookup = lookup.find(key3(r.selectivity_bucket, r.qcost, r.strategy));
                if (it_lookup == lookup.end() || it_lookup->second.best_efs < 0)
                {
                    auto it_pre = lookup.find(key3(r.selectivity_bucket, r.qcost, "parallel_pre"));
                    if (it_pre != lookup.end() && it_pre->second.best_efs >= 0)
                    {
                        r.strategy = "parallel_pre";
                        r.efs = it_pre->second.best_efs;
                        r.status = "pre_fallback";
                    }
                    else
                    {
                        r.status = "missing_lookup";
                        r.efs = (r.strategy == "parallel_pre") ? 0 : run_efs;
                    }
                }
                else
                {
                    r.efs = it_lookup->second.best_efs;
                }
            }
            r.selector_ms = now_ms() - selector_t0;

            acorn::set_filter_check_cost(r.qcost);
            std::vector<int> ids(k, -1);
            std::vector<float> dist(k, 0.0f);
            acorn::reset_thread_ndis(threads);
            double search_t0 = now_ms();
            run_search(r.strategy, index,
                       queries.data() + (size_t)qi * qd,
                       metric, k, r.efs, threads, Helec, filter_map,
                       ids, dist);
            r.search_ms = now_ms() - search_t0;
            r.total_ms = r.selector_ms + r.search_ms;

            const auto &tnd = acorn::get_thread_ndis();
            for (size_t v : tnd)
            {
                r.ndc += v;
                r.ndc_max = std::max(r.ndc_max, v);
            }
            check_one_filter_result(k, ids, r.qlabel, base_labels, &r.ok, &r.empty, &r.wrong);
            if (!gt.empty())
                r.recall = compute_one_recall(qi, k, gt_k, gt, ids);

            sum_selector += r.selector_ms;
            sum_search += r.search_ms;
            sum_total += r.total_ms;
            if (!std::isnan(r.recall))
            {
                sum_recall += r.recall;
                recall_count++;
            }
            sum_ok += r.ok;
            sum_empty += r.empty;
            sum_wrong += r.wrong;

            std::fprintf(out,
                         "%d\t%s\t%d\t%d\t%.6f\t%s\t%.6f\t%.6f\t%s\t%d\t%.3f\t%.3f\t%.3f\t",
                         r.qid, mode.c_str(), r.qlabel, r.qcost, r.s_hat,
                         r.selectivity_bucket.c_str(), r.bucket_s, r.rho,
                         r.strategy.c_str(), r.efs, r.selector_ms, r.search_ms, r.total_ms);
            if (std::isnan(r.recall))
                std::fprintf(out, "nan");
            else
                std::fprintf(out, "%.4f", r.recall);
            std::fprintf(out, "\t%zu\t%zu\t%d\t%d\t%d\t%s\n",
                         r.ndc, r.ndc_max, r.ok, r.empty, r.wrong, r.status.c_str());
            std::fflush(out);
        }
        std::fclose(out);

        std::printf("Wrote %s\n", run_out_file.c_str());
        std::printf("summary mode=%s efs=%d nq=%d avg_selector_ms=%.3f avg_search_ms=%.3f avg_total_ms=%.3f",
                    mode.c_str(), run_efs, nq, sum_selector / nq, sum_search / nq, sum_total / nq);
        if (recall_count > 0)
            std::printf(" recall=%.4f", sum_recall / recall_count);
        std::printf(" ok=%lld empty=%lld wrong=%lld\n", sum_ok, sum_empty, sum_wrong);
    }

    return 0;
}
