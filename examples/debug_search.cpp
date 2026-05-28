#include <cstdio>
#include <vector>
#include "acorn/index_acorn.h"

int main() {
    acorn::IndexACORN idx;
    idx.load("../data/acorn_sift1m.index");

    // Simple: just test search with entry_point itself
    const float* xb = idx.get_xb();
    float q[128] = {0.1f};
    
    std::vector<int> labels(10);
    std::vector<float> dists(10);
    
    printf("Testing search...\n"); fflush(stdout);
    int n = idx.acorn.search(q, xb, 128, 1, 10, 200, labels.data(), dists.data(), nullptr);
    printf("ok: n=%d\n", n);
    for (int i = 0; i < n; i++)
        printf("  [%d] id=%d dist=%.4f\n", i, labels[i], dists[i]);
    return 0;
}
