#!/usr/bin/env python3
import argparse
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


STRATEGY_ORDER = [
    "none",
    "parallel_pre",
    "parallel_in_scatter",
    "parallel_in_iqan",
    "parallel_post_scatter",
    "parallel_post_iqan",
]

STRATEGY_LABEL = {
    "none": "N/A",
    "parallel_pre": "pre",
    "parallel_in_scatter": "in-scatter",
    "parallel_in_iqan": "in-iqan",
    "parallel_post_scatter": "post-scatter",
    "parallel_post_iqan": "post-iqan",
}

STRATEGY_SHORT = {
    "none": "N/A",
    "parallel_pre": "pre",
    "parallel_in_scatter": "in-s",
    "parallel_in_iqan": "in-i",
    "parallel_post_scatter": "post-s",
    "parallel_post_iqan": "post-i",
}

STRATEGY_COLORS = [
    "#f2f2f2",
    "#8dd3c7",
    "#80b1d3",
    "#bebada",
    "#fb8072",
    "#fdb462",
]


def parse_float(value):
    if value == "nan":
        return math.nan
    return float(value)


def parse_int(value):
    if value == "nan":
        return 0
    return int(value)


def read_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["true_s"] = float(row["true_s"])
            row["filter_cost"] = int(row["filter_cost"])
            row["efs"] = int(row["efs"])
            row["threads"] = int(row["threads"])
            row["nq"] = int(row["nq"])
            row["avg_ms"] = parse_float(row["avg_ms"])
            row["recall"] = parse_float(row["recall"])
            row["ndc"] = parse_int(row["ndc"])
            row["ok"] = parse_int(row["ok"])
            row["empty"] = parse_int(row["empty"])
            row["wrong"] = parse_int(row["wrong"])
            rows.append(row)
    return rows


def build_best(rows, target_recall):
    buckets = sorted(
        {(r["selectivity"], r["true_s"], r["filter_cost"]) for r in rows},
        key=lambda b: (b[1], b[2]),
    )
    best_rows = []
    for selectivity, true_s, cost in buckets:
        candidates = [
            r
            for r in rows
            if r["selectivity"] == selectivity
            and r["filter_cost"] == cost
            and r["status"] == "ok"
            and not math.isnan(r["avg_ms"])
            and not math.isnan(r["recall"])
            and r["recall"] >= target_recall
        ]
        candidates.sort(key=lambda r: (r["avg_ms"], r["efs"], r["strategy"]))
        if candidates:
            chosen = candidates[0]
            best_rows.append(
                {
                    "selectivity": selectivity,
                    "true_s": f"{true_s:.6f}",
                    "filter_cost": cost,
                    "target_recall": f"{target_recall:.4f}",
                    "chosen_strategy": chosen["strategy"],
                    "chosen_efs": chosen["efs"],
                    "avg_ms": f"{chosen['avg_ms']:.3f}",
                    "recall": f"{chosen['recall']:.4f}",
                    "ndc": chosen["ndc"],
                    "ok": chosen["ok"],
                    "empty": chosen["empty"],
                    "wrong": chosen["wrong"],
                    "status": "ok",
                }
            )
        else:
            best_rows.append(
                {
                    "selectivity": selectivity,
                    "true_s": f"{true_s:.6f}",
                    "filter_cost": cost,
                    "target_recall": f"{target_recall:.4f}",
                    "chosen_strategy": "none",
                    "chosen_efs": -1,
                    "avg_ms": "nan",
                    "recall": "nan",
                    "ndc": 0,
                    "ok": 0,
                    "empty": 0,
                    "wrong": 0,
                    "status": "no_strategy_meets_target",
                }
            )
    return best_rows


def write_best(path, rows):
    fieldnames = [
        "selectivity",
        "true_s",
        "filter_cost",
        "target_recall",
        "chosen_strategy",
        "chosen_efs",
        "avg_ms",
        "recall",
        "ndc",
        "ok",
        "empty",
        "wrong",
        "status",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_strategy_heatmap(rows, out_prefix, target_recall):
    selectivities = sorted({r["selectivity"] for r in rows}, key=lambda s: float(s))
    costs = sorted({int(r["filter_cost"]) for r in rows})
    by_key = {(r["selectivity"], int(r["filter_cost"])): r for r in rows}
    code = {strategy: i for i, strategy in enumerate(STRATEGY_ORDER)}

    matrix = []
    labels = []
    for sel in selectivities:
        matrix_row = []
        label_row = []
        for cost in costs:
            row = by_key[(sel, cost)]
            strategy = row["chosen_strategy"]
            matrix_row.append(code[strategy])
            if strategy == "none":
                label_row.append("N/A")
            else:
                label_row.append(
                    f"{STRATEGY_SHORT[strategy]}\n"
                    f"efs={row['chosen_efs']}\n"
                    f"{float(row['avg_ms']):.2f}ms"
                )
        matrix.append(matrix_row)
        labels.append(label_row)

    fig, ax = plt.subplots(figsize=(11.2, 5.8), dpi=180)
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
    ax.set_title(f"Best Strategy Under Recall@100 >= {target_recall:.2f}")

    for i in range(len(selectivities)):
        for j in range(len(costs)):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=7.3)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=STRATEGY_COLORS[i])
        for i in range(len(STRATEGY_ORDER))
    ]
    legend_labels = [STRATEGY_LABEL[s] for s in STRATEGY_ORDER]
    ax.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=True,
    )
    fig.tight_layout()
    png = f"{out_prefix}_strategy_heatmap.png"
    pdf = f"{out_prefix}_strategy_heatmap.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/offline_lookup_v2/laion10m_train_filter_lookup_v2_4t_nq100.tsv",
    )
    parser.add_argument(
        "--out-dir",
        default="results/offline_lookup_v2/plots",
    )
    parser.add_argument("--target-recall", type=float, default=0.95)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_rows(args.input)
    best = build_best(rows, args.target_recall)

    recall_tag = str(args.target_recall).replace(".", "p")
    table_path = os.path.join(
        args.out_dir,
        f"offline_lookup_v2_best_strategy_r{recall_tag}.tsv",
    )
    out_prefix = os.path.join(args.out_dir, f"offline_lookup_v2_r{recall_tag}")

    write_best(table_path, best)
    written = [table_path]
    written.extend(plot_strategy_heatmap(best, out_prefix, args.target_recall))

    print("Wrote:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
