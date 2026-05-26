// -*- c++ -*-
#pragma once

#include <cstring>
#include <vector>
#include <stdint.h>

namespace acorn {

/// set implementation optimized for fast access
struct VisitedTable {
    std::vector<uint8_t> visited;
    int visno;

    explicit VisitedTable(int size) : visited(size), visno(1) {}

    void set(int no) {
        visited[no] = visno;
    }

    bool get(int no) const {
        return visited[no] == visno;
    }

    void advance() {
        visno++;
        if (visno == 250) {
            memset(visited.data(), 0, sizeof(visited[0]) * visited.size());
            visno = 1;
        }
    }

    int num_visited() {
        int num = 0;
        for (int i = 0; i < (int)visited.size(); i++) {
            if (visited[i] == visno) {
                num = num + 1;
            }
        }
        return num;
    }
};

} // namespace acorn
