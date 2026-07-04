#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["filter_cost"] = int(row["filter_cost"])
            row["avg_ms"] = float(row["avg_ms"])
            row["recall"] = float(row["recall"])
            rows.append(row)
    return rows


def plot_by_cost(rows, out_prefix):
    costs = sorted({r["filter_cost"] for r in rows})
    methods = ["fixed_iqan", "adaptive"]
    colors = {"fixed_iqan": "#d62728", "adaptive": "#1f77b4"}
    labels = {"fixed_iqan": "Fixed iQAN in-filter", "adaptive": "Adaptive"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=170)
    for method in methods:
        avg_ms = []
        recall = []
        for cost in costs:
            bucket = [r for r in rows if r["method"] == method and r["filter_cost"] == cost]
            avg_ms.append(sum(r["avg_ms"] for r in bucket) / len(bucket))
            recall.append(sum(r["recall"] for r in bucket) / len(bucket))
        axes[0].plot(costs, avg_ms, marker="o", label=labels[method], color=colors[method])
        axes[1].plot(costs, recall, marker="o", label=labels[method], color=colors[method])

    axes[0].set_xlabel("filter_check_cost")
    axes[0].set_ylabel("Latency (ms/query)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[1].set_xlabel("filter_check_cost")
    axes[1].set_ylabel("Recall@100")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Adaptive Phase-1 Pilot on Mixed Selectivity Workload")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_by_cost.png")
    fig.savefig(f"{out_prefix}_by_cost.pdf")
    plt.close(fig)


def plot_strategy_heatmap(rows, out_prefix):
    selectivities = ["0.1", "1", "5", "10", "20", "50", "80"]
    costs = sorted({r["filter_cost"] for r in rows})
    code = {"pre_parallel": 0, "scatter": 1, "post_parallel": 2}
    names = ["pre", "scatter", "post"]

    matrix = []
    for sel in selectivities:
        line = []
        for cost in costs:
            bucket = [
                r
                for r in rows
                if r["method"] == "adaptive"
                and r["selectivity"] == sel
                and r["filter_cost"] == cost
            ]
            line.append(code[bucket[0]["chosen_strategy"]])
        matrix.append(line)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=170)
    ax.imshow(matrix, aspect="auto", cmap=plt.get_cmap("Set2", 3), vmin=-0.5, vmax=2.5)
    ax.set_xticks(range(len(costs)), costs)
    ax.set_yticks(range(len(selectivities)), [f"{s}%" for s in selectivities])
    ax.set_xlabel("filter_check_cost")
    ax.set_ylabel("selectivity")
    ax.set_title("Adaptive selector decisions")
    for i in range(len(selectivities)):
        for j in range(len(costs)):
            ax.text(j, i, names[matrix[i][j]], ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_strategy_heatmap.png")
    fig.savefig(f"{out_prefix}_strategy_heatmap.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/adaptive_phase1_pilot_nq20.tsv")
    parser.add_argument("--out-prefix", default="results/adaptive_phase1_pilot_nq20")
    args = parser.parse_args()
    rows = read_rows(args.input)
    plot_by_cost(rows, args.out_prefix)
    plot_strategy_heatmap(rows, args.out_prefix)


if __name__ == "__main__":
    main()
