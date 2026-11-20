#!/usr/bin/env python3
"""Prepare or run the 187-run E24 C0/CommitGuard/OC-SORT recovery matrix."""

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from commitguard.runner import run, sha256
from commitguard.tracker import CommitGuardTracker
from ocsort_adapter import OCSortAdapter

HERE = Path(__file__).resolve().parent
with open(HERE / "protocol.json", encoding="utf-8") as handle:
    PROTOCOL = json.load(handle)


def path(input_root, kind, video, condition="A", rep=1):
    values = {"video": video, "condition": condition, "rep": rep,
              "annotation": PROTOCOL["annotations"][video]}
    return input_root / PROTOCOL["paths"][kind].format(**values)


def jobs(scope, method, condition):
    scopes = PROTOCOL["scopes"] if scope == "all" else {scope: PROTOCOL["scopes"][scope]}
    for scope_name, settings in scopes.items():
        for arm in PROTOCOL["conditions"]:
            if condition and arm != condition:
                continue
            methods = ("c0", "ocsort") if arm == "clean" else ("c0", "e19_fi40", "ocsort")
            for method_name in methods:
                if method and method_name != method:
                    continue
                for video in settings["videos"]:
                    for rep in settings["repeats"]:
                        yield scope_name, method_name, arm, video, rep


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run"])
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/ocsort_recovery/results"))
    parser.add_argument("--ocsort-root", type=Path, default=None)
    parser.add_argument("--scope", choices=["all", "development", "additional"], default="all")
    parser.add_argument("--method", choices=["c0", "e19_fi40", "ocsort"], default="")
    parser.add_argument("--condition", choices=["clean", "B", "A", "C"], default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    selected = list(jobs(args.scope, args.method, args.condition))
    if not selected:
        raise ValueError("No E24 jobs match the filters")
    if args.ocsort_root:
        os.environ["OCSORT_ROOT"] = str(args.ocsort_root.resolve())
    for scope, method, condition, video, rep in selected:
        annotation = path(args.input_root, "annotation", video, condition, rep)
        detections = path(args.input_root, "detections", video, condition, rep)
        auxiliary = path(args.input_root, "auxiliary_candidates", video, condition, rep)
        required = [annotation, detections]
        if method == "e19_fi40":
            required.append(auxiliary)
        missing = [str(item) for item in required if not item.is_file()]
        if missing:
            raise FileNotFoundError("Missing E24 inputs:\n" + "\n".join(missing))
        if args.stage == "prepare":
            continue
        metadata = run(method, video, rep, condition, args.output_root, annotation,
                       detections, HERE / "methods.json", HERE / "protocol.json",
                       str(auxiliary) if method == "e19_fi40" else "", None, args.resume, 30,
                       OCSortAdapter if method == "ocsort" else CommitGuardTracker)
        with open(metadata, encoding="utf-8") as handle:
            data = json.load(handle)
        data["e24_scope"] = scope
        data["ocsort_adapter_sha256"] = sha256(HERE / "ocsort_adapter.py") if method == "ocsort" else ""
        data["ocsort_official_commit"] = PROTOCOL["source"]["commit"] if method == "ocsort" else ""
        with open(metadata, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        print("E24_DONE", scope, method, condition, video, rep, flush=True)
    if args.stage == "prepare":
        gt = {path(args.input_root, "ground_truth", video)
              for scope in PROTOCOL["scopes"].values() for video in scope["videos"]}
        missing = [str(item) for item in gt if not item.is_file()]
        if missing:
            raise FileNotFoundError("Missing E24 GT:\n" + "\n".join(missing))
        print(f"Validated inputs for {len(selected)} E24 runs")


if __name__ == "__main__":
    main()
