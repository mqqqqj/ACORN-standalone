#!/usr/bin/env python3
import argparse
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STRATEGY_ORDER = [
    "parallel_in_scatter",
    "parallel_in_iqan",
    "parallel_post_scatter",
    "parallel_post_iqan",
]

STRATEGY_LABEL = {
    "parallel_pre": "pre",
    "parallel_in_scatter": "in-scatter",
    "parallel_in_iqan": "in-iqan",
    "parallel_post_scatter": "post-scatter",
    "parallel_post_iqan": "post-iqan",
}

STRATEGY_STYLE = {
    "parallel_pre": ("#4c78a8", "D"),
    "parallel_in_scatter": ("#f58518", "o"),
    "parallel_in_iqan": ("#e45756", "s"),
    "parallel_post_scatter": ("#54a24b", "^"),
    "parallel_post_iqan": ("#b279a2", "v"),
}


def read_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["status"] != "ok":
                continue
            row["true_s"] = float(row["true_s"])
            row["filter_cost"] = int(row["filter_cost"])
            row["efs"] = int(row["efs"])
            row["avg_ms"] = float(row["avg_ms"]) if row["avg_ms"] != "nan" else math.nan
            row["recall"] = float(row["recall"]) if row["recall"] != "nan" else math.nan
            if math.isnan(row["avg_ms"]) or math.isnan(row["recall"]):
                continue
            rows.append(row)
    return rows


def selectivity_label(value):
    if value.endswith("%"):
        return value
    return f"{value}%"


def plot_selectivity(rows, selectivity, true_s, costs, strategies, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.2), dpi=170)
    axes = axes.flatten()
    handles_by_label = {}

    for i, cost in enumerate(costs):
        ax = axes[i]
        cost_rows = [
            r for r in rows if r["selectivity"] == selectivity and r["filter_cost"] == cost
        ]

        for strategy in strategies:
            series = [r for r in cost_rows if r["strategy"] == strategy]
            if not series:
                continue
            series.sort(key=lambda r: r["efs"])
            color, marker = STRATEGY_STYLE[strategy]
            label = STRATEGY_LABEL[strategy]
            line = ax.plot(
                [r["recall"] for r in series],
                [r["avg_ms"] for r in series],
                color=color,
                marker=marker,
                linewidth=1.8,
                markersize=4.0,
                label=label,
            )[0]
            handles_by_label[label] = line

        plotted = [r for r in cost_rows if r["strategy"] in strategies]
        if plotted:
            recalls = [r["recall"] for r in plotted]
            high_recalls = [r for r in recalls if r >= 0.9]
            if high_recalls:
                xmin = max(0.0, min(high_recalls) - 0.01)
            else:
                xmin = max(0.0, min(recalls) - 0.02)
            xmax = min(1.0, max(recalls) + 0.01)
            if xmax <= xmin:
                xmax = min(1.0, xmin + 0.02)
            ax.set_xlim(xmin, xmax)

        ax.set_title(f"filter_cost={cost}", fontsize=11)
        ax.set_xlabel("Recall@100")
        ax.set_ylabel("Latency (ms/query)")
        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.45)

    for ax in axes[len(costs) :]:
        ax.axis("off")

    labels = [STRATEGY_LABEL[s] for s in strategies if STRATEGY_LABEL[s] in handles_by_label]
    handles = [handles_by_label[label] for label in labels]
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=True)
    fig.suptitle(
        f"LAION10M Offline Lookup v2 Recall-Latency, selectivity={selectivity_label(selectivity)}",
        fontsize=15,
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    safe_sel = selectivity.replace(".", "p")
    png = os.path.join(out_dir, f"offline_lookup_v2_sel_{safe_sel}.png")
    pdf = os.path.join(out_dir, f"offline_lookup_v2_sel_{safe_sel}.pdf")
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
    parser.add_argument(
        "--include-pre",
        action="store_true",
        help="Include parallel_pre curves; omitted by default to keep y-axis readable.",
    )
    args = parser.parse_args()

    rows = read_rows(args.input)
    os.makedirs(args.out_dir, exist_ok=True)

    selectivities = sorted(
        {(r["selectivity"], r["true_s"]) for r in rows}, key=lambda item: item[1]
    )
    costs = sorted({r["filter_cost"] for r in rows})
    strategies = list(STRATEGY_ORDER)
    if args.include_pre:
        strategies.insert(0, "parallel_pre")

    written = []
    for selectivity, true_s in selectivities:
        written.extend(plot_selectivity(rows, selectivity, true_s, costs, strategies, args.out_dir))

    print("Wrote:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
