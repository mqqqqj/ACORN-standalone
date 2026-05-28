#include <cstdio>
#include <vector>
#include <algorithm>
#include <sys/time.h>
#include "acorn/index_acorn.h"
#include "acorn/file_io.h"

static double get_ms() { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec*1000.0+tv.tv_usec/1000.0; }

int main() {
    int n = 20000, nq = 100;
    // Load base
    auto br = acorn::read_fbin("/dataset/SIFT1M/sift_base.fbin");
    auto xb = std::move(br.first); int d = br.second.second;
    printf("Base: n=%d/%d d=%d\n", n, br.second.first, d);
    // Load labels
    auto labels = acorn::read_ibin("../data/labels.ibin", -1);
    printf("Labels: %zu\n", labels.size());

    // Build
    acorn::IndexACORN idx(d, 16, 8, labels, 16, acorn::METRIC_L2);
    idx.acorn.efConstruction = 64;
    double t0 = get_ms();
    idx.add(n, xb.data());
    printf("Build: %.0f ms, ntotal=%ld max_level=%d\n", get_ms()-t0, idx.ntotal, idx.acorn.max_level);

    // Load queries
    auto qr = acorn::read_fbin("/dataset/SIFT1M/sift_query.fbin");
    auto queries = std::move(qr.first);

    // Search
    std::vector<int> labs(100 * nq); std::vector<float> dists(100 * nq);
    t0 = get_ms();
    for (int i = 0; i < nq; i++)
        idx.acorn.search(queries.data() + i*d, idx.get_xb(), d, 1, 100, 64, labs.data()+i*100, dists.data()+i*100);
    printf("Search %d queries: %.0f ms, avg %.3f ms/q\n", nq, get_ms()-t0, (get_ms()-t0)/nq);

    // Compute recall against full GT
    auto gt = acorn::read_groundtruth("../data/sift1m_gt_top100.ibin");
    auto gt_ids = std::move(gt.first);
    int hits = 0;
    for (int i = 0; i < nq; i++) {
        std::vector<int> gtset(gt_ids.begin()+i*100, gt_ids.begin()+(i+1)*100);
        std::sort(gtset.begin(), gtset.end());
        for (int j = 0; j < 100; j++) {
            int id = labs[i*100+j];
            if (id >= 0 && std::binary_search(gtset.begin(), gtset.end(), id)) hits++;
        }
    }
    printf("Recall@100: %.4f\n", (double)hits/(nq*100));
    return 0;
}
