#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <string>

#include "acorn/file_io.h"

static bool read_fbin_header(const char *filename, int *n, int *d)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
        return false;

    bool ok = fread(n, sizeof(int), 1, fp) == 1 &&
              fread(d, sizeof(int), 1, fp) == 1 &&
              *n > 0 && *d > 0;
    fclose(fp);
    return ok;
}

static bool derive_num_labels(double selectivity, int *num_labels)
{
    if (selectivity <= 0.0 || selectivity > 1.0)
        return false;

    double reciprocal = 1.0 / selectivity;
    int labels = (int)std::round(reciprocal);
    if (labels <= 0)
        return false;

    double actual = 1.0 / labels;
    double tolerance = std::max(1e-12, selectivity * 1e-9);
    if (std::fabs(actual - selectivity) > tolerance)
        return false;

    *num_labels = labels;
    return true;
}

static std::vector<int> generate_uniform(int n, int num_labels, std::mt19937 &rng)
{
    std::vector<int> labels(n);
    int base_count = n / num_labels;
    int extra = n % num_labels;

    int pos = 0;
    for (int lbl = 1; lbl <= num_labels; lbl++)
    {
        int count = base_count + (lbl <= extra ? 1 : 0);
        for (int j = 0; j < count; j++)
            labels[pos++] = lbl;
    }

    std::shuffle(labels.begin(), labels.end(), rng);
    return labels;
}

static std::vector<int> generate_binary(int n, double selectivity, std::mt19937 &rng)
{
    std::vector<int> labels(n);
    int selected = (int)std::round(n * selectivity);
    selected = std::max(0, std::min(n, selected));

    for (int i = 0; i < selected; i++)
        labels[i] = 1;
    for (int i = selected; i < n; i++)
        labels[i] = 2;

    std::shuffle(labels.begin(), labels.end(), rng);
    return labels;
}

static std::vector<int> generate_constant(int n, int label)
{
    return std::vector<int>(n, label);
}

static bool parse_selectivity(const char *arg, double *selectivity)
{
    std::string s(arg);
    bool is_percent = false;
    if (!s.empty() && s.back() == '%')
    {
        is_percent = true;
        s.pop_back();
    }

    char *end = NULL;
    double value = strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0')
        return false;
    if (is_percent)
        value /= 100.0;
    if (value <= 0.0 || value > 1.0)
        return false;

    *selectivity = value;
    return true;
}

static std::vector<int> generate_skewed(int n, std::mt19937 &rng)
{
    // Selectivity: 50%, 25%, 12.5%, 6.25%, 3.125%, 3.125% = 100%
    double fracs[] = {0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125};
    int num_labels = 6;

    // Compute exact counts per label (handle rounding so sum == n)
    std::vector<int> counts(num_labels);
    int cum = 0;
    for (int i = 0; i < num_labels - 1; i++)
    {
        counts[i] = (int)(n * fracs[i]);
        cum += counts[i];
    }
    counts[num_labels - 1] = n - cum;

    std::vector<int> labels(n);
    int pos = 0;
    for (int lbl = 0; lbl < num_labels; lbl++)
    {
        for (int j = 0; j < counts[lbl]; j++)
            labels[pos++] = lbl + 1;
    }
    std::shuffle(labels.begin(), labels.end(), rng);
    return labels;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s --base <base.fbin> --output <labels.ibin> [options]\n"
            "\n"
            "Generate label file in ibin format.\n"
            "\n"
            "Required:\n"
            "  --base <path>      Base vectors in fbin format (to get count)\n"
            "  --output <path>    Output label file in ibin format\n"
            "\n"
            "Options:\n"
            "  --dist <type>      Label distribution: uniform (default) | skewed\n"
            "  --selectivity <x>  Per-label fraction for uniform dist\n"
            "                     Accepts decimals or percentages, e.g. 0.001 or 0.1%%\n"
            "  --binary           Force binary labels: label 1 has the requested selectivity,\n"
            "                     label 2 has the rest\n"
            "  --constant-label <int>\n"
            "                     Assign this label to every vector\n"
            "  --seed <int>       Random seed (default: 42)\n"
            "\n"
            "Uniform distribution:\n"
            "  All base vectors are assigned one label in 1..round(1/selectivity).\n"
            "  If selectivity is not representable as 1 / num_labels, falls back to\n"
            "  binary labels: label 1 has the requested selectivity, label 2 has the rest.\n"
            "\n"
            "Skewed distribution (6 labels):\n"
            "  50%%, 25%%, 12.5%%, 6.25%%, 3.125%%, 3.125%%\n",
            prog);
    exit(1);
}

int main(int argc, char *argv[])
{
    const char *base_file = NULL;
    const char *output_file = NULL;
    const char *dist_type = "uniform";
    double selectivity = 0.1;
    bool force_binary = false;
    int constant_label = 0;
    int seed = 42;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--base") == 0 && i + 1 < argc)
            base_file = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output_file = argv[++i];
        else if (strcmp(argv[i], "--dist") == 0 && i + 1 < argc)
            dist_type = argv[++i];
        else if (strcmp(argv[i], "--selectivity") == 0 && i + 1 < argc)
        {
            if (!parse_selectivity(argv[++i], &selectivity))
            {
                fprintf(stderr, "Invalid selectivity: %s\n", argv[i]);
                usage(argv[0]);
            }
        }
        else if (strcmp(argv[i], "--binary") == 0)
            force_binary = true;
        else if (strcmp(argv[i], "--constant-label") == 0 && i + 1 < argc)
        {
            constant_label = atoi(argv[++i]);
            if (constant_label <= 0)
            {
                fprintf(stderr, "Invalid constant label: %d\n", constant_label);
                usage(argv[0]);
            }
        }
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
            usage(argv[0]);
        else
        {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
        }
    }

    if (!base_file || !output_file)
        usage(argv[0]);

    printf("Reading base vector header to get count ...\n");
    int n = 0, d = 0;
    if (!read_fbin_header(base_file, &n, &d))
    {
        fprintf(stderr, "Error: cannot read valid fbin header from %s\n", base_file);
        return 1;
    }
    printf("  n=%d, d=%d\n", n, d);

    std::mt19937 rng(seed);

    std::vector<int> labels;
    int max_lbl = 0;
    if (constant_label > 0)
    {
        max_lbl = constant_label;
        printf("Generating constant labels (label %d for every vector) ...\n", constant_label);
        labels = generate_constant(n, constant_label);
    }
    else if (strcmp(dist_type, "skewed") == 0)
    {
        printf("Generating skewed labels (50%%, 25%%, 12.5%%, 6.25%%, 3.125%%, 3.125%%) ...\n");
        labels = generate_skewed(n, rng);
        max_lbl = 6;
    }
    else
    {
        if (force_binary || !derive_num_labels(selectivity, &max_lbl))
        {
            max_lbl = 2;
            printf("Generating binary labels (label 1 selectivity %.6f / %.4f%%, label 2 rest) ...\n",
                   selectivity, 100.0 * selectivity);
            labels = generate_binary(n, selectivity, rng);
        }
        else
        {
            printf("Generating uniform labels (%d labels, per-label selectivity %.6f / %.4f%%) ...\n",
                   max_lbl, 1.0 / max_lbl, 100.0 / max_lbl);
            labels = generate_uniform(n, max_lbl, rng);
        }
    }

    // Print distribution
    std::vector<int> count_vec(max_lbl + 1, 0);
    for (int i = 0; i < n; i++)
        count_vec[labels[i]]++;
    printf("Label distribution:\n");
    for (int lbl = 1; lbl <= max_lbl; lbl++)
        printf("  %2d: %d (%.2f%%)\n", lbl, count_vec[lbl], 100.0 * count_vec[lbl] / n);

    FILE *fp = fopen(output_file, "wb");
    if (!fp)
    {
        fprintf(stderr, "Error: cannot open %s\n", output_file);
        return 1;
    }
    fwrite(&n, sizeof(int), 1, fp);
    fwrite(labels.data(), sizeof(int), n, fp);
    fclose(fp);

    printf("Labels saved to %s (%d integers)\n", output_file, n);
    return 0;
}
