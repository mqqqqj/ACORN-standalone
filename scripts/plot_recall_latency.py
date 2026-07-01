#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SELECTIVITY_ORDER = [
    "0.1%",
    "1%",
    "5%",
    "10%",
    "20%",
    "50%",
    "80%",
    "90%",
    "95%",
    "99%",
    "99.9%",
]

FILE_SUFFIX = {
    "0.1%": "s0p1",
    "1%": "s1",
    "5%": "s5",
    "10%": "s10",
    "20%": "s20",
    "50%": "s50",
    "80%": "s80",
    "90%": "s90",
    "95%": "s95",
    "99%": "s99",
    "99.9%": "s99p9",
}


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


def plot_one(selectivity, scatter_rows, iqan_rows, post_rows, out_dir):
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=160)

    for label, rows, color, marker in [
        ("ScatterSearch", scatter_rows, "#1f77b4", "o"),
        ("iQAN", iqan_rows, "#d62728", "s"),
        ("Post-Filter", post_rows, "#2ca02c", "^"),
    ]:
        if not rows:
            continue
        recall = [r["recall"] for r in rows]
        latency = [r["avg_ms"] for r in rows]
        ax.plot(
            recall,
            latency,
            marker=marker,
            linewidth=2.0,
            markersize=4.5,
            color=color,
            label=label,
        )
        for r in rows:
            ax.annotate(
                str(r["efs"]),
                (r["recall"], r["avg_ms"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color=color,
            )

    ax.set_title(f"LAION10M Recall-Latency, selectivity {selectivity}")
    ax.set_xlabel("Recall@100")
    ax.set_ylabel("Latency (ms/query)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(frameon=True)

    all_rows = scatter_rows + iqan_rows + post_rows
    all_recalls = [r["recall"] for r in all_rows]
    all_latency = [r["avg_ms"] for r in all_rows]
    if all_recalls:
        xmin, xmax = min(all_recalls), max(all_recalls)
        pad = max(0.001, (xmax - xmin) * 0.08)
        ax.set_xlim(max(0.0, xmin - pad), min(1.0, xmax + pad))
    if all_latency:
        ymin, ymax = min(all_latency), max(all_latency)
        pad = max(0.05, (ymax - ymin) * 0.12)
        ax.set_ylim(max(0.0, ymin - pad), ymax + pad)

    fig.tight_layout()
    suffix = FILE_SUFFIX[selectivity]
    png_path = os.path.join(out_dir, f"recall_latency_{suffix}.png")
    pdf_path = os.path.join(out_dir, f"recall_latency_{suffix}.pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scatter",
        default="results/laion10m_binary_scatter_sweep_4t_nq1000.tsv",
    )
    parser.add_argument(
        "--iqan",
        default="results/laion10m_binary_iqan_sweep_4t_nq1000.tsv",
    )
    parser.add_argument(
        "--post",
        default="results/laion10m_binary_post_parallel_sweep_4t_nq1000.tsv",
    )
    parser.add_argument(
        "--out-dir",
        default="results/recall_latency_curves",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scatter = read_tsv(args.scatter)
    iqan = read_tsv(args.iqan)
    post = read_tsv(args.post)

    written = []
    for selectivity in SELECTIVITY_ORDER:
        written.extend(
            plot_one(
                selectivity,
                scatter[selectivity],
                iqan[selectivity],
                post[selectivity],
                args.out_dir,
            )
        )

    print("Wrote:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
