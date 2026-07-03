#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
                 "          [--costs c1,c2,...] [--base base.fbin]\n"
                 "\n"
                 "Defaults: --iters 10000000 --dim 512 --max-cost 512 --step 16\n"
                 "If --base is set, distance cost uses random base ids to mimic graph search.\n",
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

int main(int argc, char **argv)
{
    int iters = 10000000;
    int dim = 512;
    int max_cost = 512;
    int step = 16;
    const char *base_file = nullptr;
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
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0)
            usage(argv[0]);
        else
            usage(argv[0]);
    }

    if (iters <= 0 || dim <= 0 || max_cost < 0 || step <= 0)
        usage(argv[0]);
    if (costs.empty())
    {
        for (int c = 0; c <= max_cost; c += step)
            costs.push_back(c);
    }

    std::mt19937 rng(12345);
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

    // Warm up both paths.
    for (int i = 0; i < 10000; i++)
    {
        int id = random_ids[i % iters];
        dist_sink += acorn::fvec_inner_product(q.data(), xb.data() + (size_t)id * dim, dim);
        filter_sink += acorn::check_filter(filter_map.data(), i & (nlabels - 1));
    }

    double t0 = now_ms();
    for (int i = 0; i < iters; i++)
    {
        int id = random_ids[i];
        dist_sink += acorn::fvec_inner_product(q.data(), xb.data() + (size_t)id * dim, dim);
    }
    double dist_ms = now_ms() - t0;
    double dist_ns = dist_ms * 1e6 / iters;

    std::printf("dim=%d metric=ip iters=%d distance_access=%s nb=%d\n",
                dim, iters, using_base ? "random_base_ids" : "single_vector_hot_cache", nb);
    std::printf("distance_ns_per_op=%.3f\n", dist_ns);
    std::printf("cost\tfilter_ns_per_op\tfilter/dist\n");

    int first_ge_cost = -1;
    for (int cost : costs)
    {
        acorn::set_filter_check_cost(cost);
        double tf0 = now_ms();
        for (int i = 0; i < iters; i++)
            filter_sink += acorn::check_filter(filter_map.data(), i & (nlabels - 1));
        double filter_ms = now_ms() - tf0;
        double filter_ns = filter_ms * 1e6 / iters;
        double ratio = filter_ns / dist_ns;
        if (first_ge_cost < 0 && ratio >= 1.0)
            first_ge_cost = cost;
        std::printf("%d\t%.3f\t%.3f\n", cost, filter_ns, ratio);
    }

    if (first_ge_cost >= 0)
        std::printf("first_cost_where_filter_ge_distance=%d\n", first_ge_cost);
    else
        std::printf("first_cost_where_filter_ge_distance=not_found\n");

    // Keep sinks observable.
    if (filter_sink == 123456789)
        std::fprintf(stderr, "sink=%f\n", (double)dist_sink);
    return 0;
}
