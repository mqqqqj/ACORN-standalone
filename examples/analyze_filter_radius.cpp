#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <sys/time.h>
#include <vector>

#include "acorn/distance.h"
#include "acorn/file_io.h"
#include "acorn/types.h"

static double get_ms()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s --base <base.fbin> --labels <labels.ibin> "
            "--query <query.fbin> --qlabels <query_labels.ibin> [options]\n"
            "\n"
            "Analyze how filtered KNN radius differs from unfiltered KNN radius.\n"
            "For each query:\n"
            "  r1 = distance to filtered top-k boundary\n"
            "  r2 = distance to unfiltered top-k boundary\n"
            "  circle1_total = #all base points within radius r1\n"
            "\n"
            "Required:\n"
            "  --base <path>      Base vectors in fbin format\n"
            "  --labels <path>    Base labels in ibin format\n"
            "  --query <path>     Query vectors in fbin format\n"
            "  --qlabels <path>   Query labels in ibin format\n"
            "\n"
            "Options:\n"
            "  --nq <int>         Number of queries to analyze (default: all)\n"
            "  --base_n <int>     Number of base vectors to analyze (default: all)\n"
            "  --k <int>          K for radius comparison (default: 100)\n"
            "  --metric <l2|ip>   Metric (default: l2)\n",
            prog);
    exit(1);
}

struct QueryStat
{
    int query_id = -1;
    int label = -1;
    int filtered_candidates = 0;
    float r1_raw = 0.0f;
    float r2_raw = 0.0f;
    size_t circle1_total = 0;
    size_t circle1_filtered = 0;
};

int main(int argc, char *argv[])
{
    const char *base_file = NULL;
    const char *label_file = NULL;
    const char *query_file = NULL;
    const char *qlabel_file = NULL;
    int nq = -1;
    int base_n = -1;
    int k = 100;
    acorn::MetricType metric_type = acorn::METRIC_L2;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--base") == 0 && i + 1 < argc)
            base_file = argv[++i];
        else if (strcmp(argv[i], "--labels") == 0 && i + 1 < argc)
            label_file = argv[++i];
        else if (strcmp(argv[i], "--query") == 0 && i + 1 < argc)
            query_file = argv[++i];
        else if (strcmp(argv[i], "--qlabels") == 0 && i + 1 < argc)
            qlabel_file = argv[++i];
        else if (strcmp(argv[i], "--nq") == 0 && i + 1 < argc)
            nq = atoi(argv[++i]);
        else if (strcmp(argv[i], "--base_n") == 0 && i + 1 < argc)
            base_n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc)
            k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--metric") == 0 && i + 1 < argc)
        {
            const char *m = argv[++i];
            if (strcmp(m, "ip") == 0)
                metric_type = acorn::METRIC_INNER_PRODUCT;
            else if (strcmp(m, "l2") == 0)
                metric_type = acorn::METRIC_L2;
            else
            {
                fprintf(stderr, "Unknown metric: %s\n", m);
                return 1;
            }
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
            usage(argv[0]);
        else
        {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
        }
    }

    if (!base_file || !label_file || !query_file || !qlabel_file)
        usage(argv[0]);

    printf("=== Filter Radius Analysis ===\n");
    printf("Base:    %s\n", base_file);
    printf("Labels:  %s\n", label_file);
    printf("Query:   %s\n", query_file);
    printf("QLabels: %s\n", qlabel_file);
    printf("Params:  k=%d, metric=%s\n\n",
           k, metric_type == acorn::METRIC_L2 ? "L2" : "IP");

    printf("Loading base ...\n");
    double t0 = get_ms();
    auto br = acorn::read_fbin(base_file);
    std::vector<float> xb = std::move(br.first);
    int n_all = br.second.first;
    int d = br.second.second;
    printf("  n=%d, d=%d (%.0f ms)\n", n_all, d, get_ms() - t0);

    printf("Loading base labels ...\n");
    t0 = get_ms();
    std::vector<int> base_labels = acorn::read_ibin(label_file, n_all);
    printf("  %zu labels (%.0f ms)\n", base_labels.size(), get_ms() - t0);

    printf("Loading queries ...\n");
    t0 = get_ms();
    auto qr = acorn::read_fbin(query_file);
    std::vector<float> queries = std::move(qr.first);
    int nq_all = qr.second.first;
    int qd = qr.second.second;
    printf("  nq=%d, d=%d (%.0f ms)\n", nq_all, qd, get_ms() - t0);

    if (qd != d)
    {
        fprintf(stderr, "Error: query dim %d != base dim %d\n", qd, d);
        return 1;
    }

    printf("Loading query labels ...\n");
    t0 = get_ms();
    std::vector<int> query_labels = acorn::read_ibin(qlabel_file, nq_all);
    printf("  %zu labels (%.0f ms)\n", query_labels.size(), get_ms() - t0);

    if (base_n <= 0 || base_n > n_all)
        base_n = n_all;
    if (nq <= 0 || nq > nq_all)
        nq = nq_all;

    int max_label = 0;
    for (int i = 0; i < base_n; i++)
        if (base_labels[i] > max_label)
            max_label = base_labels[i];

    std::vector<int> label_freq(max_label + 1, 0);
    for (int i = 0; i < base_n; i++)
        label_freq[base_labels[i]]++;

    auto compute_dist = [&](const float *q, int id) -> float
    {
        const float *x = xb.data() + (size_t)id * d;
        if (metric_type == acorn::METRIC_INNER_PRODUCT)
            return -acorn::fvec_inner_product(q, x, d);
        return acorn::fvec_L2sqr(q, x, d);
    };

    std::vector<QueryStat> stats;
    stats.reserve(nq);
    int skipped_bad_label = 0;
    int skipped_too_few_filtered = 0;

    double t_analysis = -get_ms();
    for (int qi = 0; qi < nq; qi++)
    {
        int ql = query_labels[qi];
        if (ql < 0 || ql > max_label)
        {
            skipped_bad_label++;
            continue;
        }
        if (label_freq[ql] < k)
        {
            skipped_too_few_filtered++;
            continue;
        }

        const float *q = queries.data() + (size_t)qi * d;
        std::vector<float> all_dists(base_n);
        std::vector<float> filtered_dists;
        filtered_dists.reserve(label_freq[ql]);

        for (int bi = 0; bi < base_n; bi++)
        {
            float dist = compute_dist(q, bi);
            all_dists[bi] = dist;
            if (base_labels[bi] == ql)
                filtered_dists.push_back(dist);
        }

        std::nth_element(all_dists.begin(), all_dists.begin() + (k - 1), all_dists.end());
        float r2_raw = all_dists[k - 1];

        std::nth_element(filtered_dists.begin(), filtered_dists.begin() + (k - 1), filtered_dists.end());
        float r1_raw = filtered_dists[k - 1];

        size_t circle1_total = 0;
        size_t circle1_filtered = 0;
        const float eps = 1e-6f;
        for (int bi = 0; bi < base_n; bi++)
        {
            if (all_dists[bi] <= r1_raw + eps)
            {
                circle1_total++;
                if (base_labels[bi] == ql)
                    circle1_filtered++;
            }
        }

        QueryStat st;
        st.query_id = qi;
        st.label = ql;
        st.filtered_candidates = label_freq[ql];
        st.r1_raw = r1_raw;
        st.r2_raw = r2_raw;
        st.circle1_total = circle1_total;
        st.circle1_filtered = circle1_filtered;
        stats.push_back(st);
    }
    t_analysis += get_ms();

    auto report_dist = [&](float raw) -> double
    {
        if (metric_type == acorn::METRIC_L2)
            return std::sqrt(std::max(0.0f, raw));
        return raw;
    };

    std::vector<double> ratios;
    ratios.reserve(stats.size());
    double sum_r1 = 0.0, sum_r2 = 0.0, sum_ratio = 0.0;
    double sum_circle1_total = 0.0, sum_circle1_filtered = 0.0;
    double sum_nonfilter_in_circle1 = 0.0;
    int ratio_gt_1 = 0;

    for (const auto &st : stats)
    {
        double r1 = report_dist(st.r1_raw);
        double r2 = report_dist(st.r2_raw);
        double ratio = (r2 > 0.0) ? (r1 / r2) : 0.0;
        ratios.push_back(ratio);
        sum_r1 += r1;
        sum_r2 += r2;
        sum_ratio += ratio;
        sum_circle1_total += (double)st.circle1_total;
        sum_circle1_filtered += (double)st.circle1_filtered;
        sum_nonfilter_in_circle1 += (double)(st.circle1_total - st.circle1_filtered);
        if (r1 > r2)
            ratio_gt_1++;
    }

    std::sort(ratios.begin(), ratios.end());
    double median_ratio = ratios.empty() ? 0.0 : ratios[ratios.size() / 2];

    printf("\n--- Analysis Summary ---\n");
    printf("Processed queries: %zu / %d\n", stats.size(), nq);
    printf("Skipped: bad_label=%d too_few_filtered=%d\n",
           skipped_bad_label, skipped_too_few_filtered);
    printf("Analysis time: %.1f ms\n", t_analysis);

    if (!stats.empty())
    {
        double inv = 1.0 / stats.size();
        printf("Avg r1 (%s top-%d radius): %.6f\n",
               metric_type == acorn::METRIC_L2 ? "filtered" : "filtered-dist", k, sum_r1 * inv);
        printf("Avg r2 (%s top-%d radius): %.6f\n",
               metric_type == acorn::METRIC_L2 ? "unfiltered" : "unfiltered-dist", k, sum_r2 * inv);
        printf("Avg r1/r2: %.6f, median r1/r2: %.6f\n", sum_ratio * inv, median_ratio);
        printf("Queries with r1 > r2: %d / %zu\n", ratio_gt_1, stats.size());
        printf("Avg circle1 total points: %.2f\n", sum_circle1_total * inv);
        printf("Avg circle1 filtered points: %.2f\n", sum_circle1_filtered * inv);
        printf("Avg circle1 non-filter points: %.2f\n", sum_nonfilter_in_circle1 * inv);
    }

    printf("\n--- First 10 Queries ---\n");
    int show_n = std::min<int>(10, stats.size());
    for (int i = 0; i < show_n; i++)
    {
        const auto &st = stats[i];
        double r1 = report_dist(st.r1_raw);
        double r2 = report_dist(st.r2_raw);
        double ratio = (r2 > 0.0) ? (r1 / r2) : 0.0;
        printf("q=%d label=%d filtered_candidates=%d r1=%.6f r2=%.6f ratio=%.6f "
               "circle1_total=%zu circle1_filtered=%zu nonfilter=%zu\n",
               st.query_id, st.label, st.filtered_candidates,
               r1, r2, ratio,
               st.circle1_total, st.circle1_filtered,
               st.circle1_total - st.circle1_filtered);
    }

    return 0;
}
