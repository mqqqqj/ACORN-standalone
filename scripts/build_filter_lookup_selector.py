#!/usr/bin/env python3
import argparse
import csv
import math


def parse_float(value):
    if value == "nan":
        return math.nan
    return float(value)


def read_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["true_s"] = float(row["true_s"])
            row["filter_cost"] = int(row["filter_cost"])
            row["efs"] = int(row["efs"])
            row["threads"] = int(row["threads"])
            row["k"] = int(row["k"])
            row["nq"] = int(row["nq"])
            row["avg_ms"] = parse_float(row["avg_ms"])
            row["time_ms"] = parse_float(row["time_ms"])
            row["recall"] = parse_float(row["recall"])
            row["ndc"] = int(row["ndc"])
            row["ok"] = int(row["ok"])
            row["empty"] = int(row["empty"])
            row["wrong"] = int(row["wrong"])
            rows.append(row)
    return rows


def build_selector(rows, target_recall, low_selectivity_pre_fallback):
    buckets = sorted(
        {(r["selectivity"], r["true_s"], r["filter_cost"]) for r in rows},
        key=lambda b: (b[1], b[2]),
    )
    out = []
    for selectivity, true_s, cost in buckets:
        candidates = [
            r
            for r in rows
            if r["selectivity"] == selectivity
            and r["filter_cost"] == cost
            and r["status"] == "ok"
            and not math.isnan(r["recall"])
            and r["recall"] > target_recall
        ]
        candidates.sort(key=lambda r: (r["avg_ms"], r["efs"], r["strategy"]))
        if candidates:
            best = candidates[0]
            out.append(
                {
                    "selectivity": selectivity,
                    "true_s": f"{true_s:.6f}",
                    "filter_cost": cost,
                    "target_recall": f"{target_recall:.4f}",
                    "chosen_strategy": best["strategy"],
                    "chosen_efs": best["efs"],
                    "avg_ms": f"{best['avg_ms']:.3f}",
                    "time_ms": f"{best['time_ms']:.1f}",
                    "recall": f"{best['recall']:.4f}",
                    "ndc": best["ndc"],
                    "ok": best["ok"],
                    "empty": best["empty"],
                    "wrong": best["wrong"],
                    "status": "ok",
                }
            )
        elif low_selectivity_pre_fallback > 0.0 and true_s <= low_selectivity_pre_fallback:
            out.append(
                {
                    "selectivity": selectivity,
                    "true_s": f"{true_s:.6f}",
                    "filter_cost": cost,
                    "target_recall": f"{target_recall:.4f}",
                    "chosen_strategy": "pre_parallel",
                    "chosen_efs": 0,
                    "avg_ms": "nan",
                    "time_ms": "nan",
                    "recall": "nan",
                    "ndc": 0,
                    "ok": 0,
                    "empty": 0,
                    "wrong": 0,
                    "status": "low_selectivity_pre_fallback",
                }
            )
        else:
            out.append(
                {
                    "selectivity": selectivity,
                    "true_s": f"{true_s:.6f}",
                    "filter_cost": cost,
                    "target_recall": f"{target_recall:.4f}",
                    "chosen_strategy": "none",
                    "chosen_efs": -1,
                    "avg_ms": "nan",
                    "time_ms": "nan",
                    "recall": "nan",
                    "ndc": 0,
                    "ok": 0,
                    "empty": 0,
                    "wrong": 0,
                    "status": "no_strategy_meets_target",
                }
            )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-recall", type=float, default=0.9)
    parser.add_argument(
        "--low-selectivity-pre-fallback",
        type=float,
        default=0.0,
        help="Choose pre_parallel when no measured strategy meets target and true selectivity is <= this threshold; 0 disables.",
    )
    args = parser.parse_args()

    rows = read_rows(args.raw)
    selector = build_selector(rows, args.target_recall, args.low_selectivity_pre_fallback)

    fieldnames = [
        "selectivity",
        "true_s",
        "filter_cost",
        "target_recall",
        "chosen_strategy",
        "chosen_efs",
        "avg_ms",
        "time_ms",
        "recall",
        "ndc",
        "ok",
        "empty",
        "wrong",
        "status",
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selector)


if __name__ == "__main__":
    main()
