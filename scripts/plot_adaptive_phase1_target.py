#!/usr/bin/env python3
import argparse
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


STRATEGY_CODE = {
    "none": 0,
    "pre_parallel": 1,
    "scatter": 2,
    "iqan": 3,
    "post_parallel": 4,
    "post_parallel_iqan": 5,
}

STRATEGY_LABEL = {
    "none": "none",
    "pre_parallel": "pre",
    "scatter": "scatter",
    "iqan": "iqan",
    "post_parallel": "post-s",
    "post_parallel_iqan": "post-i",
}

STRATEGY_COLORS = [
    "#f2f2f2",
    "#8dd3c7",
    "#80b1d3",
    "#bebada",
    "#fb8072",
    "#fdb462",
]


def read_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["filter_cost"] = int(row["filter_cost"])
            row["rho"] = float(row["rho"])
            row["avg_ms"] = float(row["avg_ms"]) if row["avg_ms"] != "nan" else math.nan
            row["recall"] = float(row["recall"]) if row["recall"] != "nan" else math.nan
            row["chosen_efs"] = int(row["chosen_efs"])
            rows.append(row)
    return rows


def plot_strategy_heatmap(rows, out_prefix):
    selectivities = ["0.1", "1", "5", "10", "20", "50", "80"]
    costs = sorted({r["filter_cost"] for r in rows})
    by_key = {(r["selectivity"], r["filter_cost"]): r for r in rows}

    matrix = []
    labels = []
    for sel in selectivities:
        line = []
        text_line = []
        for cost in costs:
            row = by_key[(sel, cost)]
            strategy = row["chosen_strategy"]
            line.append(STRATEGY_CODE[strategy])
            efs = row["chosen_efs"]
            efs_text = "" if efs < 0 else f"\n{efs}"
            text_line.append(f"{STRATEGY_LABEL[strategy]}{efs_text}")
        matrix.append(line)
        labels.append(text_line)

    fig, ax = plt.subplots(figsize=(10.2, 5.6), dpi=170)
    ax.imshow(
        matrix,
        aspect="auto",
        cmap=ListedColormap(STRATEGY_COLORS),
        vmin=-0.5,
        vmax=len(STRATEGY_COLORS) - 0.5,
    )
    ax.set_xticks(range(len(costs)), costs)
    ax.set_yticks(range(len(selectivities)), [f"{s}%" for s in selectivities])
    ax.set_xlabel("filter_check_cost")
    ax.set_ylabel("selectivity")
    ax.set_title("Phase-1 target-recall strategy choice")

    for i in range(len(selectivities)):
        for j in range(len(costs)):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=7.5)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=STRATEGY_COLORS[code])
        for _, code in sorted(STRATEGY_CODE.items(), key=lambda kv: kv[1])
    ]
    legend_labels = [
        STRATEGY_LABEL[name]
        for name, _ in sorted(STRATEGY_CODE.items(), key=lambda kv: kv[1])
    ]
    ax.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=6)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_strategy_heatmap.png")
    fig.savefig(f"{out_prefix}_strategy_heatmap.pdf")
    plt.close(fig)


def plot_latency_heatmap(rows, out_prefix):
    selectivities = ["0.1", "1", "5", "10", "20", "50", "80"]
    costs = sorted({r["filter_cost"] for r in rows})
    by_key = {(r["selectivity"], r["filter_cost"]): r for r in rows}

    matrix = []
    text = []
    for sel in selectivities:
        line = []
        text_line = []
        for cost in costs:
            row = by_key[(sel, cost)]
            value = row["avg_ms"]
            line.append(value)
            if math.isnan(value):
                text_line.append("none")
            else:
                text_line.append(f"{value:.2f}")
        matrix.append(line)
        text.append(text_line)

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=170)
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(costs)), costs)
    ax.set_yticks(range(len(selectivities)), [f"{s}%" for s in selectivities])
    ax.set_xlabel("filter_check_cost")
    ax.set_ylabel("selectivity")
    ax.set_title("Latency at first Recall@100 > target (ms/query)")
    fig.colorbar(im, ax=ax, label="ms/query")

    for i in range(len(selectivities)):
        for j in range(len(costs)):
            ax.text(j, i, text[i][j], ha="center", va="center", fontsize=7, color="white")

    fig.tight_layout()
    fig.savefig(f"{out_prefix}_latency_heatmap.png")
    fig.savefig(f"{out_prefix}_latency_heatmap.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", required=True)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    rows = read_rows(args.best)
    plot_strategy_heatmap(rows, args.out_prefix)
    plot_latency_heatmap(rows, args.out_prefix)


if __name__ == "__main__":
    main()
