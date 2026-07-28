#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for row in rows:
        row["threads"] = int(row["threads"])
        row["avg_ms"] = float(row["avg_ms"])
        row["recall"] = float(row["recall"])
        row["speedup"] = float(next(v for k, v in row.items() if k.startswith("speedup_vs_")))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    rows = read_rows(args.input)
    labels = [f"{r['threads']}t\n{r['search_param']}\nR={r['recall']:.3f}" for r in rows]
    latencies = [r["avg_ms"] for r in rows]
    speedups = [r["speedup"] for r in rows]
    colors = ["#9a9a9a"] + ["#2a6fbb"] * (len(rows) - 1)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.bar(range(len(rows)), latencies, color=colors, width=0.58)

    ax.set_title(args.title, fontsize=12, weight="bold", pad=10)
    ax.set_ylabel("Latency (ms/query)")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymax = max(latencies) * 1.18
    ax.set_ylim(0, ymax)
    for bar, latency, speedup in zip(bars, latencies, speedups):
        label = f"{latency:.1f} ms"
        if speedup > 1.001:
            label += f"\n{speedup:.2f}x"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.025,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout(pad=1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
