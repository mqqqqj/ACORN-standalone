#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include "acorn/acorn_graph.h"
#include "acorn/file_io.h"

static double get_ms()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s --base <base.fbin> --labels <labels.ibin> --output <index.file> [options]\n"
        "\n"
        "Required:\n"
        "  --base <path>      Base vectors in fbin format\n"
        "  --labels <path>    Base labels in ibin format\n"
        "  --output <path>    Output index file\n"
        "\n"
        "Options:\n"
        "  --M <int>          Max out-degree (default: 32)\n"
        "  --gamma <int>      Multiplier for level 0 candidate pool (default: 12)\n"
        "  --efc <int>        efConstruction (default: 200)\n"
        "  --metric <l2|ip>   Distance metric (default: l2)\n"
        "\n"
        "Example:\n"
        "  %s --base sift_base.fbin --labels labels.ibin --output acorn.index\n"
        "  %s --base sift_base.fbin --labels labels.ibin --output acorn.index --M 64 --gamma 24 --efc 500\n",
        prog, prog, prog);
    exit(1);
}

int main(int argc, char *argv[])
{
    const char *base_file = NULL;
    const char *label_file = NULL;
    const char *output_file = NULL;
    int M = 32, gamma = 12, efConstruction = 200;
    acorn::MetricType metric_type = acorn::METRIC_L2;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--base") == 0 && i + 1 < argc)
            base_file = argv[++i];
        else if (strcmp(argv[i], "--labels") == 0 && i + 1 < argc)
            label_file = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output_file = argv[++i];
        else if (strcmp(argv[i], "--M") == 0 && i + 1 < argc)
            M = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gamma") == 0 && i + 1 < argc)
            gamma = atoi(argv[++i]);
        else if (strcmp(argv[i], "--efc") == 0 && i + 1 < argc)
            efConstruction = atoi(argv[++i]);
        else if (strcmp(argv[i], "--metric") == 0 && i + 1 < argc)
        {
            const char *m = argv[++i];
            if (strcmp(m, "ip") == 0)
                metric_type = acorn::METRIC_INNER_PRODUCT;
            else if (strcmp(m, "l2") == 0)
                metric_type = acorn::METRIC_L2;
            else { fprintf(stderr, "Unknown metric: %s\n", m); return 1; }
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
            usage(argv[0]);
        else
            { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); }
    }

    if (!base_file || !label_file || !output_file)
        usage(argv[0]);

    printf("=== ACORN Index Build ===\n");
    printf("Base:   %s\n", base_file);
    printf("Labels: %s\n", label_file);
    printf("Output: %s\n", output_file);
    printf("Params: M=%d, gamma=%d, efConstruction=%d, metric=%s\n\n",
           M, gamma, efConstruction,
           metric_type == acorn::METRIC_L2 ? "L2" : "IP");

    // Load base vectors
    printf("Loading base vectors ...\n");
    double t0 = get_ms();
    auto br = acorn::read_fbin(base_file);
    std::vector<float> xb = std::move(br.first);
    int n = br.second.first, d = br.second.second;
    printf("  n=%d, d=%d (%.0f ms)\n", n, d, get_ms() - t0);

    // Load labels
    printf("Loading labels ...\n");
    t0 = get_ms();
    std::vector<int> base_labels = acorn::read_ibin(label_file, n);
    printf("  %zu labels (%.0f ms)\n", base_labels.size(), get_ms() - t0);

    // Build index
    printf("\nBuilding ACORN index ...\n");
    t0 = get_ms();
    acorn::ACORN index(d, M, gamma, base_labels, M, metric_type);
    index.verbose = true;
    index.efConstruction = efConstruction;
    index.add(n, xb.data());
    double t_build = get_ms() - t0;
    printf("  Build time: %.1f ms (%.2f sec)\n", t_build, t_build / 1000.0);

    // Save
    printf("\nSaving index to %s ...\n", output_file);
    t0 = get_ms();
    index.save(output_file);
    printf("  Saved in %.0f ms\n", get_ms() - t0);

    printf("\nDone.\n");
    return 0;
}
