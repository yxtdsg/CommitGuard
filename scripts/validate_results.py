#!/usr/bin/env python3
"""Validate result hashes, formal scopes, and macro aggregation without raw data."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
PAPER = RESULTS / "paper"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected):
    return abs(float(actual) - float(expected)) <= 1e-10 * max(1.0, abs(float(expected)))


def check_macro(video_path, macro_path, group_fields, metrics, percent_metrics=()):
    videos = read_csv(video_path)
    macros = read_csv(macro_path)
    grouped = defaultdict(list)
    for row in videos:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    expected = {tuple(row[field] for field in group_fields): row for row in macros}
    assert set(grouped) == set(expected), f"Group mismatch: {macro_path}"
    for key, rows in grouped.items():
        for metric in metrics:
            mean = sum(float(row[metric]) for row in rows) / len(rows)
            if metric in percent_metrics:
                mean *= 100.0
            assert close(mean, expected[key][metric]), (
                f"{macro_path.name} {key} {metric}: {mean} != {expected[key][metric]}")
    return len(macros)


def main():
    with open(PAPER / "manifest.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for record in manifest["files"]:
        path = PAPER / record["path"]
        assert path.is_file(), f"Missing {path}"
        assert path.stat().st_size == record["bytes"], f"Size changed: {path}"
        assert sha256(path) == record["sha256"], f"Hash changed: {path}"

    with open(RESULTS / "supplementary" / "manifest.json", encoding="utf-8") as handle:
        supplementary = json.load(handle)
    assert not supplementary["formal_statistics_included"]
    for record in supplementary["files"]:
        path = RESULTS / "supplementary" / record["path"]
        assert path.is_file() and sha256(path) == record["sha256"], f"Supplement changed: {path}"

    checks = {"supplementary_files": len(supplementary["files"])}
    checks["E19_video_rows"] = len(read_csv(PAPER / "mot17_development/video_summary.csv"))
    assert checks["E19_video_rows"] == 48
    checks["E20_macro_rows"] = check_macro(
        PAPER / "mot17_additional/video_summary.csv",
        PAPER / "mot17_additional/macro_summary.csv", ["method", "condition"],
        ["IDF1", "IDP", "IDR", "HOTA", "AssA", "MOTA", "IDSW", "FP", "FN", "TP",
         "mean_update_ms"])
    assert checks["E20_macro_rows"] == 6
    checks["E24_full_macro_rows"] = check_macro(
        PAPER / "ocsort_recovery/ocsort_full_video_summary.csv",
        PAPER / "ocsort_recovery/ocsort_full_macro_summary.csv", ["method", "condition"],
        ["IDF1", "HOTA", "AssA", "MOTA", "IDSW", "FP", "FN"])
    assert checks["E24_full_macro_rows"] == 1
    dance_videos = read_csv(PAPER / "dancetrack_main/video_summary.csv")
    assert {row["video"] for row in dance_videos} == {"0004", "0005", "0007", "0010"}
    assert "0014" not in {row["video"] for row in dance_videos}
    checks["E26_macro_rows"] = check_macro(
        PAPER / "dancetrack_main/video_summary.csv",
        PAPER / "dancetrack_main/macro_summary.csv", ["label", "method", "condition"],
        ["MOTA", "IDF1", "HOTA", "AssA", "IDSW", "FP", "FN", "TP", "mean_update_ms"])
    assert len(dance_videos) == 20 and checks["E26_macro_rows"] == 5
    with open(PAPER / "dancetrack_main/audit_equivalence.json", encoding="utf-8") as handle:
        audit = json.load(handle)
    assert audit["all_identical"] and audit["files"] == 48
    with open(PAPER / "dancetrack_main/runtime_validation.json", encoding="utf-8") as handle:
        runtime = json.load(handle)
    assert runtime["runs"] == 60

    checks["E29_comparison_macro_rows"] = check_macro(
        PAPER / "method_comparison/video_summary.csv",
        PAPER / "method_comparison/paper_macro.csv", ["method", "label"],
        ["MOTA", "IDF1", "HOTA", "IDSW", "FN", "FP"],
        percent_metrics=("MOTA", "IDF1", "HOTA"))
    assert checks["E29_comparison_macro_rows"] == 8
    with open(PAPER / "method_comparison/coverage.json", encoding="utf-8") as handle:
        coverage = json.load(handle)
    assert coverage["complete"] and coverage["valid_runs"] == 96

    trasw_segments = read_csv(PAPER / "trasw/segment_metrics.csv")
    assert len(trasw_segments) == 120
    assert {row["video"] for row in trasw_segments} == {"02", "04", "05"}
    metrics = ["IDF1", "IDP", "IDR", "HOTA", "AssA", "MOTA", "IDSW", "FP", "FN", "TP",
               "alarm_frames", "intervention_frames", "aux_review_frames",
               "aux_filter_removed", "aux_fn_added", "aux_birth_rejected"]
    checks["E29_trasw_macro_rows"] = check_macro(
        PAPER / "trasw/segment_metrics.csv", PAPER / "trasw/macro_summary.csv",
        ["arm", "condition", "method"], metrics)
    assert checks["E29_trasw_macro_rows"] == 4
    with open(PAPER / "trasw/result_summary.json", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["runs"]["completed_and_validated"] == 120
    assert not summary["conclusion"]["overall_IDF1_or_HOTA_defense_supported"]

    print(json.dumps({"status": "ok", "manifest_files": len(manifest["files"]),
                      "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
