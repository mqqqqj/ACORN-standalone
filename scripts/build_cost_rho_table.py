#!/usr/bin/env python3
import argparse
import csv
import re


def parse_csv_floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def parse_profile(path):
    distance_ns = None
    rows = []
    in_table = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"distance_ns_per_op=([0-9.]+)", line)
            if m:
                distance_ns = float(m.group(1))
                continue
            if line.startswith("cost\t"):
                in_table = True
                continue
            if in_table:
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    cost = int(parts[0])
                    filter_ns = float(parts[1])
                except ValueError:
                    continue
                rows.append((cost, filter_ns))
    if distance_ns is None or not rows:
        raise RuntimeError(f"failed to parse profile output: {path}")
    return distance_ns, rows


def selectivity_label(s):
    pct = s * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return str(int(round(pct)))
    text = f"{pct:.6f}".rstrip("0").rstrip(".")
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, help="Output from profile_filter_distance_cost")
    parser.add_argument("--selectivities", required=True, help="CSV true selectivities, e.g. 0.001,0.01,0.05")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    distance_ns, cost_rows = parse_profile(args.profile)
    selectivities = parse_csv_floats(args.selectivities)

    fields = [
        "selectivity",
        "true_s",
        "filter_cost",
        "c_filter_ns",
        "c_dist_ns",
        "rho",
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for s in selectivities:
            if not (0.0 < s <= 1.0):
                raise RuntimeError(f"invalid selectivity: {s}")
            for cost, filter_ns in cost_rows:
                rho = filter_ns / (s * distance_ns)
                writer.writerow(
                    {
                        "selectivity": selectivity_label(s),
                        "true_s": f"{s:.6f}",
                        "filter_cost": cost,
                        "c_filter_ns": f"{filter_ns:.3f}",
                        "c_dist_ns": f"{distance_ns:.3f}",
                        "rho": f"{rho:.6f}",
                    }
                )


if __name__ == "__main__":
    main()
