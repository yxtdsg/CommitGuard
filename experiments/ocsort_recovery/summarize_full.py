#!/usr/bin/env python3
"""Rebuild the E24 full-sequence OC-SORT macro table from video rows."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

METRICS = ("IDF1", "HOTA", "AssA", "MOTA", "IDSW", "FP", "FN")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with open(args.video_summary, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["condition"])].append(row)
    output = []
    for (method, condition), members in groups.items():
        output.append({"method": method, "condition": condition, **{
            metric: sum(float(row[metric]) for row in members) / len(members)
            for metric in METRICS}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "condition", *METRICS])
        writer.writeheader()
        writer.writerows(output)
    print(f"Wrote {args.output} ({len(output)} macro rows)")


if __name__ == "__main__":
    main()
