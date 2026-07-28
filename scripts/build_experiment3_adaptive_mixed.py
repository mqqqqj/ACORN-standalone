#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIXED_MODES = [
    "fixed_scatter",
    "fixed_iqan",
    "fixed_post_scatter",
    "fixed_post_iqan",
]

IN_FILTER_MODES = {"fixed_scatter", "fixed_iqan"}
POST_FILTER_MODES = {"fixed_post_scatter", "fixed_post_iqan"}

LABELS = {
    "adaptive_selector": "Adaptive",
    "fixed_scatter": "Fixed in-filter ScatterSearch",
    "fixed_iqan": "Fixed in-filter iQAN",
    "fixed_post_scatter": "Fixed post-filter ScatterSearch",
    "fixed_post_iqan": "Fixed post-filter iQAN",
}

STYLES = {
    "adaptive_selector": ("#111111", "*"),
    "fixed_scatter": ("#1f77b4", "o"),
    "fixed_iqan": ("#d62728", "s"),
    "fixed_post_scatter": ("#2ca02c", "^"),
    "fixed_post_iqan": ("#9467bd", "D"),
}


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--binary", default="build/run_adaptive_selector")
    parser.add_argument("--index", default="data/laion10m/acorn_laion10m_efc500.faiss_index")
    parser.add_argument("--query", default="/dataset/LAION/LAION_test_query_textemb_10k.fbin")
    parser.add_argument("--base-labels", default="data/laion10m/base_labels_mix.ibin")
    parser.add_argument("--query-labels", default="data/laion10m/query_labels_mix_n1000.ibin")
    parser.add_argument("--query-costs", default="data/laion10m/query_cost_mix_n1000.ibin")
    parser.add_argument("--gt", default="data/laion10m/laion10m_gt_mix_n1000.ibin")
    parser.add_argument("--cost-rho", default="results/offline_lookup_v2_target32_unfilteredM/cost_rho_cost0_5_formal.tsv")
    parser.add_argument("--lookup", default="results/offline_lookup_v2_target32_unfilteredM/adaptive_selector_lookup_r0p95_cost0_5_formal.tsv")
    parser.add_argument("--out-dir", type=Path, default=Path("results/extension_main_experiments/experiment3_adaptive_mixed"))
    parser.add_argument("--nq", type=int, default=1000)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--filtered-expand-target", type=int, default=32)
    parser.add_argument("--efs", type=int, nargs="+", default=[400, 800, 1200, 1600])
    parser.add_argument("--in-efs", type=int, nargs="+")
    parser.add_argument("--post-efs", type=int, nargs="+")
    parser.add_argument("--scatter-efs", type=int, nargs="+")
    parser.add_argument("--iqan-efs", type=int, nargs="+")
    parser.add_argument("--post-scatter-efs", type=int, nargs="+")
    parser.add_argument("--post-iqan-efs", type=int, nargs="+")
    parser.add_argument("--adaptive-files", type=Path, nargs="+")


def efs_for_mode(args: argparse.Namespace, mode: str) -> list[int]:
    if mode == "fixed_scatter" and args.scatter_efs is not None:
        return args.scatter_efs
    if mode == "fixed_iqan" and args.iqan_efs is not None:
        return args.iqan_efs
    if mode == "fixed_post_scatter" and args.post_scatter_efs is not None:
        return args.post_scatter_efs
    if mode == "fixed_post_iqan" and args.post_iqan_efs is not None:
        return args.post_iqan_efs
    if mode in IN_FILTER_MODES:
        return args.in_efs if args.in_efs is not None else args.efs
    if mode in POST_FILTER_MODES:
        return args.post_efs if args.post_efs is not None else args.efs
    return args.efs


def common_command(args: argparse.Namespace) -> list[str]:
    return [
        args.binary,
        "--index",
        args.index,
        "--query",
        args.query,
        "--base-labels",
        args.base_labels,
        "--query-labels",
        args.query_labels,
        "--query-costs",
        args.query_costs,
        "--cost-rho",
        args.cost_rho,
        "--lookup",
        args.lookup,
        "--gt",
        args.gt,
        "--nq",
        str(args.nq),
        "--k",
        str(args.k),
        "--threads",
        str(args.threads),
        "--filtered-expand-target",
        str(args.filtered_expand_target),
    ]


def run_one(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_experiment(args: argparse.Namespace) -> None:
    fixed_dir = args.out_dir / "fixed_sweep"
    fixed_dir.mkdir(parents=True, exist_ok=True)

    base = common_command(args)
    if not args.skip_adaptive:
        run_one(base + ["--mode", "adaptive", "--out", str(args.out_dir / f"adaptive_selector_nq{args.nq}.tsv")])

    for mode in args.modes:
        for efs in efs_for_mode(args, mode):
            out = fixed_dir / f"{mode}_efs{efs}_nq{args.nq}.tsv"
            run_one(base + ["--mode", mode, "--fixed-efs", str(efs), "--out", str(out)])


def summarize_file(path: Path, method: str, efs: int) -> dict:
    total_ms = 0.0
    total_recall = 0.0
    ok = 0
    empty = 0
    wrong = 0
    count = 0
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            count += 1
            total_ms += float(row["total_ms"])
            total_recall += float(row["recall"])
            ok += int(row["ok"])
            empty += int(row["empty"])
            wrong += int(row["wrong"])
    if count == 0:
        raise ValueError(f"{path} has no rows")
    return {
        "method": method,
        "efs": efs,
        "target_recall": target_from_path(path),
        "avg_ms": total_ms / count,
        "recall": total_recall / count,
        "ok": ok,
        "empty": empty,
        "wrong": wrong,
    }


def target_from_path(path: Path) -> str:
    match = re.search(r"r0p(\d+)", path.name)
    if not match:
        return ""
    digits = match.group(1)
    return f"0.{digits}"


def build_summary(args: argparse.Namespace) -> list[dict]:
    adaptive_files = args.adaptive_files or [args.out_dir / f"adaptive_selector_nq{args.nq}.tsv"]
    rows = [summarize_file(path, "adaptive_selector", -1) for path in adaptive_files]
    fixed_dir = args.out_dir / "fixed_sweep"
    for mode in FIXED_MODES:
        for efs in efs_for_mode(args, mode):
            rows.append(summarize_file(fixed_dir / f"{mode}_efs{efs}_nq{args.nq}.tsv", mode, efs))
    return rows


def write_summary(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            lineterminator="\n",
            fieldnames=["method", "efs", "target_recall", "avg_ms", "recall", "ok", "empty", "wrong"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row["method"],
                    "efs": row["efs"],
                    "target_recall": row.get("target_recall", ""),
                    "avg_ms": f"{row['avg_ms']:.3f}",
                    "recall": f"{row['recall']:.4f}",
                    "ok": row["ok"],
                    "empty": row["empty"],
                    "wrong": row["wrong"],
                }
            )


def plot(rows: list[dict], path: Path) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda r: r["efs"])

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=170)

    for method in FIXED_MODES:
        values = grouped[method]
        color, marker = STYLES[method]
        ax.plot(
            [r["recall"] for r in values],
            [r["avg_ms"] for r in values],
            label=LABELS[method],
            color=color,
            marker=marker,
            markersize=4.0,
            linewidth=1.8,
        )

    adaptive_values = grouped["adaptive_selector"]
    adaptive_values.sort(key=lambda r: (r["recall"], r["avg_ms"]))
    color, marker = STYLES["adaptive_selector"]
    if len(adaptive_values) == 1:
        ax.scatter(
            [adaptive_values[0]["recall"]],
            [adaptive_values[0]["avg_ms"]],
            label=LABELS["adaptive_selector"],
            color=color,
            marker=marker,
            s=115,
            zorder=5,
        )
    else:
        ax.plot(
            [r["recall"] for r in adaptive_values],
            [r["avg_ms"] for r in adaptive_values],
            label=LABELS["adaptive_selector"],
            color=color,
            marker=marker,
            markersize=9,
            linewidth=1.8,
            zorder=5,
        )

    all_rows = rows
    recalls = [r["recall"] for r in all_rows]
    latencies = [r["avg_ms"] for r in all_rows]
    xpad = max(0.002, (max(recalls) - min(recalls)) * 0.08)
    ypad = max(0.5, (max(latencies) - min(latencies)) * 0.10)

    ax.set_xlabel("Recall@100")
    ax.set_ylabel("Latency (ms/query)")
    ax.set_title("Adaptive Selector vs Fixed Strategies (LAION10M mixed workload)", fontsize=11, weight="bold", pad=9)
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(max(0.0, min(recalls) - xpad), min(1.0, max(recalls) + xpad))
    ax.set_ylim(max(0.0, min(latencies) - ypad), max(latencies) + ypad)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--skip-adaptive", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=FIXED_MODES, default=FIXED_MODES)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_run:
        run_experiment(args)

    rows = build_summary(args)
    summary_path = args.out_dir / "mixed_recall_latency_fixed_sweep_with_adaptive.tsv"
    pdf_path = args.out_dir / "mixed_recall_latency_fixed_sweep_with_adaptive.pdf"
    write_summary(rows, summary_path)
    plot(rows, pdf_path)
    print(summary_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
