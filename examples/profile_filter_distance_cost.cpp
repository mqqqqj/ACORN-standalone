#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "acorn/acorn_graph.h"
#include "acorn/distance.h"
#include "acorn/file_io.h"

static double now_ms()
{
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

static void usage(const char *prog)
{
    std::fprintf(stderr,
                 "Usage: %s [--iters N] [--dim D] [--max-cost C] [--step S]\n"
                 "          [--costs c1,c2,...] [--base base.fbin] [--selectivity S]\n"
                 "          [--repeats R] [--warmup-runs W] [--seed SEED]\n"
                 "\n"
                 "Defaults: --iters 10000000 --dim 512 --max-cost 256 --step 32\n"
                 "          --repeats 5 --warmup-runs 1 --seed 12345\n"
                 "If --base is set, distance cost uses random base ids to mimic graph search.\n"
                 "--selectivity is used to estimate total filter cost over total distance cost:\n"
                 "  effective_filter/dist = filter_ns / (distance_ns * selectivity).\n",
                 prog);
    std::exit(1);
}

static std::vector<int> parse_costs(const char *s)
{
    std::vector<int> costs;
    const char *p = s;
    while (*p)
    {
        char *end = nullptr;
        long v = std::strtol(p, &end, 10);
        if (end == p || v < 0)
            usage("profile_filter_distance_cost");
        costs.push_back((int)v);
        p = end;
        if (*p == ',')
            ++p;
        else if (*p != '\0')
            usage("profile_filter_distance_cost");
    }
    return costs;
}

static double median(std::vector<double> values)
{
    if (values.empty())
        return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    size_t mid = values.size() / 2;
    if (values.size() % 2 == 1)
        return values[mid];
    return 0.5 * (values[mid - 1] + values[mid]);
}

static double min_value(const std::vector<double> &values)
{
    return *std::min_element(values.begin(), values.end());
}

static double max_value(const std::vector<double> &values)
{
    return *std::max_element(values.begin(), values.end());
}

static double measure_distance_ns(const std::vector<float> &q,
                                  const std::vector<float> &xb,
                                  const std::vector<int> &random_ids,
                                  int dim,
                                  volatile float &dist_sink)
{
    double t0 = now_ms();
    for (size_t i = 0; i < random_ids.size(); i++)
    {
        int id = random_ids[i];
        dist_sink += acorn::fvec_inner_product(q.data(), xb.data() + (size_t)id * dim, dim);
    }
    double ms = now_ms() - t0;
    return ms * 1e6 / random_ids.size();
}

static double measure_filter_ns(const std::vector<char> &filter_map,
                                int iters,
                                int cost,
                                volatile int &filter_sink)
{
    acorn::set_filter_check_cost(cost);
    double t0 = now_ms();
    for (int i = 0; i < iters; i++)
        filter_sink += acorn::check_filter(filter_map.data(), i & ((int)filter_map.size() - 1));
    double ms = now_ms() - t0;
    return ms * 1e6 / iters;
}

int main(int argc, char **argv)
{
    int iters = 10000000;
    int dim = 512;
    int max_cost = 256;
    int step = 32;
    int repeats = 5;
    int warmup_runs = 1;
    int seed = 12345;
    const char *base_file = nullptr;
    double selectivity = 1.0;
    std::vector<int> costs;

    for (int i = 1; i < argc; i++)
    {
        if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
            iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--dim") == 0 && i + 1 < argc)
            dim = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--max-cost") == 0 && i + 1 < argc)
            max_cost = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--step") == 0 && i + 1 < argc)
            step = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--costs") == 0 && i + 1 < argc)
            costs = parse_costs(argv[++i]);
        else if (std::strcmp(argv[i], "--base") == 0 && i + 1 < argc)
            base_file = argv[++i];
        else if (std::strcmp(argv[i], "--selectivity") == 0 && i + 1 < argc)
            selectivity = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc)
            repeats = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmup-runs") == 0 && i + 1 < argc)
            warmup_runs = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0)
            usage(argv[0]);
        else
            usage(argv[0]);
    }

    if (iters <= 0 || dim <= 0 || max_cost < 0 || step <= 0 ||
        repeats <= 0 || warmup_runs < 0 ||
        !(selectivity > 0.0 && selectivity <= 1.0))
        usage(argv[0]);
    if (costs.empty())
    {
        for (int c = 0; c <= max_cost; c += step)
            costs.push_back(c);
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    std::vector<float> q(dim);
    std::vector<float> xb;
    int nb = 1;
    bool using_base = base_file != nullptr;

    if (using_base)
    {
        std::printf("Loading base: %s\n", base_file);
        auto br = acorn::read_fbin(base_file);
        xb = std::move(br.first);
        nb = br.second.first;
        int bd = br.second.second;
        if (bd != dim)
        {
            std::fprintf(stderr, "Error: base dim %d != --dim %d\n", bd, dim);
            return 1;
        }
    }
    else
    {
        xb.resize(dim);
    }

    for (int i = 0; i < dim; i++)
        q[i] = dis(rng);

    if (!using_base)
    {
        for (int i = 0; i < dim; i++)
            xb[i] = dis(rng);
    }

    std::vector<int> random_ids(iters);
    if (using_base)
    {
        std::uniform_int_distribution<int> id_dis(0, nb - 1);
        for (int i = 0; i < iters; i++)
            random_ids[i] = id_dis(rng);
    }
    else
    {
        std::fill(random_ids.begin(), random_ids.end(), 0);
    }

    const int nlabels = 1 << 20;
    std::vector<char> filter_map(nlabels);
    for (int i = 0; i < nlabels; i++)
        filter_map[i] = (i & 1) ? 1 : 0;

    volatile float dist_sink = 0.0f;
    volatile int filter_sink = 0;

    // Short warmup to trigger code paths before full warmup rounds.
    for (int i = 0; i < std::min(10000, iters); i++)
    {
        int id = random_ids[i % iters];
        dist_sink += acorn::fvec_inner_product(q.data(), xb.data() + (size_t)id * dim, dim);
        filter_sink += acorn::check_filter(filter_map.data(), i & (nlabels - 1));
    }

    for (int r = 0; r < warmup_runs; r++)
    {
        (void)measure_distance_ns(q, xb, random_ids, dim, dist_sink);
        for (int cost : costs)
            (void)measure_filter_ns(filter_map, iters, cost, filter_sink);
    }

    std::vector<double> dist_runs;
    dist_runs.reserve(repeats);
    for (int r = 0; r < repeats; r++)
        dist_runs.push_back(measure_distance_ns(q, xb, random_ids, dim, dist_sink));
    double dist_ns = median(dist_runs);

    std::printf("dim=%d metric=ip iters=%d repeats=%d warmup_runs=%d seed=%d distance_access=%s nb=%d selectivity=%.6f\n",
                dim, iters, repeats, warmup_runs, seed,
                using_base ? "random_base_ids" : "single_vector_hot_cache", nb,
                selectivity);
    std::printf("distance_ns_per_op=%.3f\n", dist_ns);
    std::printf("distance_ns_min=%.3f\n", min_value(dist_runs));
    std::printf("distance_ns_max=%.3f\n", max_value(dist_runs));
    std::printf("cost\tfilter_ns_per_op\tfilter/dist\teffective_filter/dist\tfilter_ns_min\tfilter_ns_max\n");

    int first_ge_cost = -1;
    int first_effective_ge_cost = -1;
    for (int cost : costs)
    {
        std::vector<double> filter_runs;
        filter_runs.reserve(repeats);
        for (int r = 0; r < repeats; r++)
            filter_runs.push_back(measure_filter_ns(filter_map, iters, cost, filter_sink));
        double filter_ns = median(filter_runs);
        double ratio = filter_ns / dist_ns;
        double effective_ratio = ratio / selectivity;
        if (first_ge_cost < 0 && ratio >= 1.0)
            first_ge_cost = cost;
        if (first_effective_ge_cost < 0 && effective_ratio >= 1.0)
            first_effective_ge_cost = cost;
        std::printf("%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n",
                    cost, filter_ns, ratio, effective_ratio,
                    min_value(filter_runs), max_value(filter_runs));
    }

    if (first_ge_cost >= 0)
        std::printf("first_cost_where_filter_ge_distance=%d\n", first_ge_cost);
    else
        std::printf("first_cost_where_filter_ge_distance=not_found\n");
    if (first_effective_ge_cost >= 0)
        std::printf("first_cost_where_effective_filter_ge_distance=%d\n", first_effective_ge_cost);
    else
        std::printf("first_cost_where_effective_filter_ge_distance=not_found\n");

    // Keep sinks observable.
    if (filter_sink == 123456789)
        std::fprintf(stderr, "sink=%f\n", (double)dist_sink);
    return 0;
}
