#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for row in rows:
        row["threads"] = int(row["threads"])
        row["scatter_avg_ms"] = float(row["scatter_avg_ms"])
        row["scatter_recall"] = float(row["scatter_recall"])
        row["iqan_avg_ms"] = float(row["iqan_avg_ms"])
        row["iqan_recall"] = float(row["iqan_recall"])
        row["scatter_speedup_vs_iqan"] = float(row["scatter_speedup_vs_iqan"])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    rows = read_rows(args.input)
    x = np.arange(len(rows))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    scatter = [r["scatter_avg_ms"] for r in rows]
    iqan = [r["iqan_avg_ms"] for r in rows]

    b1 = ax.bar(x - width / 2, scatter, width, label="ScatterSearch", color="#2a6fbb")
    b2 = ax.bar(x + width / 2, iqan, width, label="iQAN", color="#d9822b")

    ax.set_title(args.title, fontsize=12, weight="bold", pad=10)
    ax.set_ylabel("Latency (ms/query)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['threads']}t" for r in rows])
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    ymax = max(scatter + iqan) * 1.25
    ax.set_ylim(0, ymax)

    for bars, key in ((b1, "scatter_recall"), (b2, "iqan_recall")):
        for bar, row in zip(bars, rows):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.025,
                f"{bar.get_height():.1f}\nR={row[key]:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    for i, row in enumerate(rows):
        ax.text(
            x[i],
            ymax * 0.04,
            f"{row['scatter_speedup_vs_iqan']:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#111",
        )

    fig.tight_layout(pad=1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
