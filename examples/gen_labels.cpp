#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

#include "acorn/file_io.h"

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s --base <base.fbin> --output <labels.ibin> [options]\n"
        "\n"
        "Generate random label file in ibin format.\n"
        "Labels are uniformly sampled from [1, num_labels].\n"
        "\n"
        "Required:\n"
        "  --base <path>      Base vectors in fbin format (to get count)\n"
        "  --output <path>    Output label file in ibin format\n"
        "\n"
        "Options:\n"
        "  --num-labels <int> Number of label classes (default: 12)\n"
        "  --seed <int>       Random seed (default: 42)\n",
        prog);
    exit(1);
}

int main(int argc, char *argv[])
{
    const char *base_file = NULL;
    const char *output_file = NULL;
    int num_labels = 12;
    int seed = 42;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--base") == 0 && i + 1 < argc)
            base_file = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output_file = argv[++i];
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

    printf("Generating random labels (1-%d) ...\n", num_labels);
    std::vector<int> labels(n);
    std::mt19937 rng(seed);
    for (int i = 0; i < n; i++)
        labels[i] = (rng() % num_labels) + 1;

    std::vector<int> counts(num_labels + 1, 0);
    for (int i = 0; i < n; i++)
        counts[labels[i]]++;
    printf("Label distribution:\n");
    for (int lbl = 1; lbl <= num_labels; lbl++)
        printf("  %2d: %d\n", lbl, counts[lbl]);

    FILE *fp = fopen(output_file, "wb");
    if (!fp) { fprintf(stderr, "Error: cannot open %s\n", output_file); return 1; }
    fwrite(&n, sizeof(int), 1, fp);
    fwrite(labels.data(), sizeof(int), n, fp);
    fclose(fp);

    printf("Labels saved to %s (%d integers)\n", output_file, n);
    return 0;
}
