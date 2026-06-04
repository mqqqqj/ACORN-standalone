#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>

#include "acorn/file_io.h"

static std::vector<int> generate_uniform(int n, int num_labels, std::mt19937 &rng)
{
    std::vector<int> labels(n);
    for (int i = 0; i < n; i++)
        labels[i] = (rng() % num_labels) + 1;
    return labels;
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
        "  --num-labels <int> Number of labels for uniform dist (default: 12)\n"
        "  --seed <int>       Random seed (default: 42)\n"
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
    int num_labels = 12;
    int seed = 42;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--base") == 0 && i + 1 < argc)
            base_file = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output_file = argv[++i];
        else if (strcmp(argv[i], "--dist") == 0 && i + 1 < argc)
            dist_type = argv[++i];
        else if (strcmp(argv[i], "--num-labels") == 0 && i + 1 < argc)
            num_labels = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
            usage(argv[0]);
        else
            { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); }
    }

    if (!base_file || !output_file)
        usage(argv[0]);

    printf("Loading base vectors to get count ...\n");
    auto result = acorn::read_fbin(base_file);
    int n = result.second.first;
    printf("  n=%d\n", n);

    std::mt19937 rng(seed);

    std::vector<int> labels;
    if (strcmp(dist_type, "skewed") == 0)
    {
        printf("Generating skewed labels (50%%, 25%%, 12.5%%, 6.25%%, 3.125%%, 3.125%%) ...\n");
        labels = generate_skewed(n, rng);
    }
    else
    {
        printf("Generating uniform labels (1-%d) ...\n", num_labels);
        labels = generate_uniform(n, num_labels, rng);
    }

    // Print distribution
    int max_lbl = (strcmp(dist_type, "skewed") == 0) ? 6 : num_labels;
    std::vector<int> count_vec(max_lbl + 1, 0);
    for (int i = 0; i < n; i++)
        count_vec[labels[i]]++;
    printf("Label distribution:\n");
    for (int lbl = 1; lbl <= max_lbl; lbl++)
        printf("  %2d: %d (%.2f%%)\n", lbl, count_vec[lbl], 100.0 * count_vec[lbl] / n);

    FILE *fp = fopen(output_file, "wb");
    if (!fp) { fprintf(stderr, "Error: cannot open %s\n", output_file); return 1; }
    fwrite(&n, sizeof(int), 1, fp);
    fwrite(labels.data(), sizeof(int), n, fp);
    fclose(fp);

    printf("Labels saved to %s (%d integers)\n", output_file, n);
    return 0;
}
