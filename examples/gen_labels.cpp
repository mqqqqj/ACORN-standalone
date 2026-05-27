#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

#include "acorn/file_io.h"

int main(int argc, char* argv[]) {
    const char* base_file = "/dataset/SIFT1M/sift_base.fbin";
    const char* output_file = "../data/labels.ibin";
    int seed = 42;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--base") == 0 && i + 1 < argc) base_file = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_file = argv[++i];
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --base FILE    base fbin file\n");
            printf("  --output FILE  output label file (default: labels.ibin)\n");
            printf("  --seed INT     random seed (default: 42)\n");
            return 0;
        }
    }

    printf("Loading base vectors to get count...\n");
    auto result = acorn::read_fbin(base_file);
    int n = result.second.first;
    printf("  n=%d\n", n);

    printf("Generating random labels (1-12)...\n");
    std::vector<int> labels(n);
    std::mt19937 rng(seed);
    for (int i = 0; i < n; i++) {
        labels[i] = (rng() % 12) + 1;
    }

    // Print distribution
    std::vector<int> counts(13, 0);
    for (int i = 0; i < n; i++) counts[labels[i]]++;
    printf("Label distribution:\n");
    for (int lbl = 1; lbl <= 12; lbl++) {
        printf("  %2d: %d\n", lbl, counts[lbl]);
    }

    // Save: 4B n, then n * 4B labels
    FILE* fp = fopen(output_file, "wb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s\n", output_file);
        return 1;
    }
    fwrite(&n, sizeof(int), 1, fp);
    fwrite(labels.data(), sizeof(int), n, fp);
    fclose(fp);

    printf("Labels saved to %s (%d integers)\n", output_file, n);
    return 0;
}
