#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SELECTIVITIES = ["0.1%", "1%", "5%", "10%", "20%", "50%", "80%"]


def normalize_selectivity(value):
    value = value.strip()
    if value.endswith("%"):
        return value
    return f"{value}%"


def read_tsv(path):
    data = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            data[normalize_selectivity(row["selectivity"])].append(
                {
                    "efs": int(row["efs"]),
                    "recall": float(row["recall"]),
                    "avg_ms": float(row["avg_ms"]),
                }
            )
    for rows in data.values():
        rows.sort(key=lambda r: r["efs"])
    return data


def plot_panel(ax, title, series):
    all_rows = []
    for label, rows, color, marker in series:
        if not rows:
            continue
        all_rows.extend(rows)
        ax.plot(
            [r["recall"] for r in rows],
            [r["avg_ms"] for r in rows],
            marker=marker,
            linewidth=1.8,
            markersize=4.0,
            color=color,
            label=label,
        )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Recall@100")
    ax.set_ylabel("Latency (ms/query)")
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.45)

    if all_rows:
        recalls = [r["recall"] for r in all_rows]
        latency = [r["avg_ms"] for r in all_rows]
        xmin, xmax = min(recalls), max(recalls)
        ymin, ymax = min(latency), max(latency)
        xpad = max(0.001, (xmax - xmin) * 0.08)
        ypad = max(0.03, (ymax - ymin) * 0.12)
        ax.set_xlim(max(0.0, xmin - xpad), min(1.0, xmax + xpad))
        ax.set_ylim(max(0.0, ymin - ypad), ymax + ypad)


def plot_grid(title, scatter_data, iqan_data, out_prefix):
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 11.0), dpi=170)
    axes = axes.flatten()

    for i, selectivity in enumerate(SELECTIVITIES):
        plot_panel(
            axes[i],
            f"Selectivity {selectivity}",
            [
                ("ScatterSearch", scatter_data[selectivity], "#1f77b4", "o"),
                ("iQAN", iqan_data[selectivity], "#d62728", "s"),
            ],
        )

    for ax in axes[len(SELECTIVITIES) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
    fig.suptitle(title, fontsize=15, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))

    png_path = f"{out_prefix}.png"
    pdf_path = f"{out_prefix}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-scatter",
        default="results/laion10m_binary_scatter_sweep_4t_nq1000.tsv",
    )
    parser.add_argument(
        "--in-iqan",
        default="results/laion10m_binary_iqan_sweep_4t_nq1000.tsv",
    )
    parser.add_argument(
        "--post-scatter",
        default="results/laion10m_binary_post_parallel_sweep_4t_nq1000.tsv",
    )
    parser.add_argument(
        "--post-iqan",
        default="results/laion10m_binary_post_iqan_sweep_4t_nq1000.tsv",
    )
    parser.add_argument(
        "--out-dir",
        default="results/recall_latency_curves",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    in_scatter = read_tsv(args.in_scatter)
    in_iqan = read_tsv(args.in_iqan)
    post_scatter = read_tsv(args.post_scatter)
    post_iqan = read_tsv(args.post_iqan)

    written = []
    written.extend(
        plot_grid(
            "LAION10M In-Filter Recall-Latency",
            in_scatter,
            in_iqan,
            os.path.join(args.out_dir, "in_filter_recall_latency"),
        )
    )
    written.extend(
        plot_grid(
            "LAION10M Post-Filter Recall-Latency",
            post_scatter,
            post_iqan,
            os.path.join(args.out_dir, "post_filter_recall_latency"),
        )
    )

    print("Wrote:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
