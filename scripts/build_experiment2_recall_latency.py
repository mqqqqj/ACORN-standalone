#!/usr/bin/env python3
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SELECTIVITIES = ["5", "10", "20", "50"]
METHODS = [
    ("in_scatter", "In-filter ScatterSearch", "#1f77b4", "o"),
    ("in_iqan", "In-filter iQAN", "#d62728", "s"),
]


def normalize_selectivity(value: str) -> str:
    value = value.strip()
    if value.endswith("%"):
        value = value[:-1]
    return str(float(value)).rstrip("0").rstrip(".")


def read_source(path: Path, method: str, selected: set[str]):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            selectivity = normalize_selectivity(row["selectivity"])
            if selectivity not in selected:
                continue
            rows.append(
                {
                    "method": method,
                    "selectivity": selectivity,
                    "efs": int(row["efs"]),
                    "avg_ms": float(row["avg_ms"]),
                    "recall": float(row["recall"]),
                    "empty": int(row["empty"]),
                    "wrong": int(row["wrong"]),
                }
            )
    return rows


def read_sweep_logs(log_dir: Path, method: str, mode: str, selected: set[str]):
    rows = []
    pattern = re.compile(
        rf"SWEEP_RESULT mode={re.escape(mode)} efs=(\d+) time_ms=([0-9.]+) avg_ms=([0-9.]+) "
        r"recall=([0-9.]+).* ok=(\d+) empty=(\d+) wrong=(\d+)"
    )
    for selectivity in sorted(selected, key=int):
        path = log_dir / f"{method}_s{selectivity}_4t_nq1000.log"
        if not path.exists():
            return []
        for line in path.read_text().splitlines():
            match = pattern.search(line)
            if not match:
                continue
            rows.append(
                {
                    "method": f"in_{method}",
                    "selectivity": selectivity,
                    "efs": int(match.group(1)),
                    "avg_ms": float(match.group(3)),
                    "recall": float(match.group(4)),
                    "empty": int(match.group(6)),
                    "wrong": int(match.group(7)),
                }
            )
    return rows


def dedupe_rows(rows):
    by_key = {}
    for row in rows:
        key = (row["method"], row["selectivity"], row["efs"])
        by_key[key] = row
    return sorted(by_key.values(), key=lambda r: (int(r["selectivity"]), r["method"], r["efs"]))


def write_tsv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            lineterminator="\n",
            fieldnames=["method", "selectivity", "efs", "avg_ms", "recall", "empty", "wrong"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row["method"],
                    "selectivity": row["selectivity"],
                    "efs": row["efs"],
                    "avg_ms": f"{row['avg_ms']:.3f}",
                    "recall": f"{row['recall']:.4f}",
                    "empty": row["empty"],
                    "wrong": row["wrong"],
                }
            )


def plot(rows, path: Path, dataset_name: str, log_y: bool = False):
    by_panel = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_panel[row["selectivity"]][row["method"]].append(row)
    for methods in by_panel.values():
        for values in methods.values():
            values.sort(key=lambda r: r["efs"])

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.1), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, selectivity in zip(axes, SELECTIVITIES):
        all_panel_rows = []
        for method, label, color, marker in METHODS:
            values = by_panel[selectivity][method]
            if not values:
                continue
            all_panel_rows.extend(values)
            ax.plot(
                [r["recall"] for r in values],
                [r["avg_ms"] for r in values],
                label=label,
                color=color,
                marker=marker,
                markersize=3.8,
                linewidth=1.7,
            )

        ax.set_title(f"Selectivity {selectivity}%", fontsize=10, weight="bold")
        ax.set_xlabel("Recall@100")
        ax.set_ylabel("Latency (ms/query)")
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if all_panel_rows:
            recalls = [r["recall"] for r in all_panel_rows]
            latencies = [r["avg_ms"] for r in all_panel_rows]
            xpad = max(0.002, (max(recalls) - min(recalls)) * 0.08)
            ypad = max(0.25, (max(latencies) - min(latencies)) * 0.12)
            ax.set_xlim(max(0.0, min(recalls) - xpad), min(1.0, max(recalls) + xpad))
            ymin = min(latencies) * 0.85 if log_y else max(0.0, min(latencies) - ypad)
            ymax = max(latencies) * 1.18 if log_y else max(latencies) + ypad
            ax.set_ylim(ymin, ymax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"In-filter Recall-Latency Curves ({dataset_name}, cost=0, 4 threads)", fontsize=12, weight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results/extension_main_experiments/experiment2_recall_latency"))
    parser.add_argument("--in-scatter", type=Path, default=Path("results/recall_latency_curves/laion10m_binary_scatter_sweep_4t_nq1000.tsv"))
    parser.add_argument("--in-iqan", type=Path, default=Path("results/recall_latency_curves/laion10m_binary_iqan_sweep_4t_nq1000.tsv"))
    parser.add_argument("--current-sweep-log-dir", type=Path, default=Path("results/extension_main_experiments/experiment2_recall_latency/current_sweeps_after_sharedpool_batch"))
    parser.add_argument("--current-iqan-log-dir", type=Path, default=Path("results/extension_main_experiments/experiment2_recall_latency/current_iqan_sweeps"))
    parser.add_argument("--post-scatter", type=Path, default=Path("results/recall_latency_curves/laion10m_binary_post_parallel_sweep_4t_nq1000.tsv"))
    parser.add_argument("--post-scatter-extra", type=Path, default=Path("results/recall_latency_curves/laion10m_binary_post_parallel_extra_efs1200_2000_4t_nq1000.tsv"))
    parser.add_argument("--post-iqan", type=Path, default=Path("results/recall_latency_curves/laion10m_binary_post_iqan_sweep_4t_nq1000.tsv"))
    parser.add_argument("--dataset-name", default="LAION10M")
    parser.add_argument("--log-y", action="store_true")
    args = parser.parse_args()

    selected = set(SELECTIVITIES)
    rows = []
    current_scatter_rows = read_sweep_logs(args.current_sweep_log_dir, "scatter", "scatter", selected)
    if current_scatter_rows:
        rows.extend(current_scatter_rows)
    else:
        rows.extend(read_source(args.in_scatter, "in_scatter", selected))
    current_iqan_rows = read_sweep_logs(args.current_sweep_log_dir, "iqan", "iqan", selected)
    if not current_iqan_rows:
        current_iqan_rows = read_sweep_logs(args.current_iqan_log_dir, "iqan", "iqan", selected)
    if current_iqan_rows:
        rows.extend(current_iqan_rows)
    else:
        rows.extend(read_source(args.in_iqan, "in_iqan", selected))
    rows = dedupe_rows(rows)

    tsv_path = args.out_dir / "recall_latency_s5_s10_s20_s50_cost0.tsv"
    pdf_path = args.out_dir / "recall_latency_s5_s10_s20_s50_cost0.pdf"
    write_tsv(rows, tsv_path)
    plot(rows, pdf_path, dataset_name=args.dataset_name, log_y=args.log_y)
    print(tsv_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
