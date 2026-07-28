#!/usr/bin/env python3
import argparse
import csv
import math


def parse_float(value):
    if value == "nan":
        return math.nan
    return float(value)


def read_cost_rho(path):
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sel = row["selectivity"]
            cost = int(row["filter_cost"])
            out[(sel, cost)] = {
                "true_s": float(row["true_s"]),
                "rho": float(row["rho"]),
            }
    return out


def read_raw(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                row["true_s"] = float(row["true_s"])
                row["filter_cost"] = int(row["filter_cost"])
                row["efs"] = int(row["efs"])
                row["avg_ms"] = parse_float(row["avg_ms"])
                row["recall"] = parse_float(row["recall"])
                rows.append(row)
    return rows


def normalize_strategy(strategy):
    aliases = {
        "pre_parallel": "parallel_pre",
        "scatter": "parallel_in_scatter",
        "iqan": "parallel_in_iqan",
        "post_parallel": "parallel_post_scatter",
        "post_parallel_iqan": "parallel_post_iqan",
    }
    return aliases.get(strategy, strategy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="append", required=True, help="Offline lookup TSV; repeatable")
    parser.add_argument("--cost-rho", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument(
        "--strategies",
        default="parallel_pre,parallel_in_scatter,parallel_post_scatter,parallel_in_iqan",
        help="CSV strategies to include in lookup.",
    )
    args = parser.parse_args()

    wanted = set(x for x in args.strategies.split(",") if x)
    cost_rho = read_cost_rho(args.cost_rho)
    rows = read_raw(args.raw)

    buckets = sorted(
        {
            (r["selectivity"], r["true_s"], r["filter_cost"], normalize_strategy(r["strategy"]))
            for r in rows
            if normalize_strategy(r["strategy"]) in wanted
        },
        key=lambda x: (x[1], x[2], x[3]),
    )

    fields = [
        "selectivity",
        "true_s",
        "filter_cost",
        "rho",
        "strategy",
        "target_recall",
        "best_efs",
        "avg_ms",
        "recall",
        "status",
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for sel, true_s, cost, strategy in buckets:
            rho_row = cost_rho.get((sel, cost))
            rho = rho_row["rho"] if rho_row else math.nan
            candidates = [
                r
                for r in rows
                if r["selectivity"] == sel
                and r["filter_cost"] == cost
                and normalize_strategy(r["strategy"]) == strategy
                and r["status"] == "ok"
                and not math.isnan(r["avg_ms"])
                and not math.isnan(r["recall"])
                and r["recall"] >= args.target_recall
            ]
            candidates.sort(key=lambda r: (r["avg_ms"], r["efs"]))
            if candidates:
                best = candidates[0]
                writer.writerow(
                    {
                        "selectivity": sel,
                        "true_s": f"{true_s:.6f}",
                        "filter_cost": cost,
                        "rho": f"{rho:.6f}" if not math.isnan(rho) else "nan",
                        "strategy": strategy,
                        "target_recall": f"{args.target_recall:.4f}",
                        "best_efs": best["efs"],
                        "avg_ms": f"{best['avg_ms']:.3f}",
                        "recall": f"{best['recall']:.4f}",
                        "status": "ok",
                    }
                )
            else:
                writer.writerow(
                    {
                        "selectivity": sel,
                        "true_s": f"{true_s:.6f}",
                        "filter_cost": cost,
                        "rho": f"{rho:.6f}" if not math.isnan(rho) else "nan",
                        "strategy": strategy,
                        "target_recall": f"{args.target_recall:.4f}",
                        "best_efs": -1,
                        "avg_ms": "nan",
                        "recall": "nan",
                        "status": "no_strategy_meets_target",
                    }
                )


if __name__ == "__main__":
    main()
