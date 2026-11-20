#!/usr/bin/env python3
"""Validate frozen inputs or run one of the formal replay matrices."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from commitguard.runner import run, sha256


def load_protocol(experiment):
    directory = REPO / "experiments" / experiment
    with open(directory / "protocol.json", encoding="utf-8") as handle:
        return directory, json.load(handle)


def input_path(protocol, input_root, kind, video, rep=1, condition="A"):
    values = {
        "video": video,
        "rep": rep,
        "condition": condition,
        "annotation": protocol.get("annotations", {}).get(video, ""),
    }
    return input_root / protocol["paths"][kind].format(**values)


def selected_jobs(protocol, args):
    for video in protocol["videos"]:
        if args.video and video != args.video:
            continue
        for rep in protocol["repeats"]:
            if args.rep and rep != args.rep:
                continue
            for group in protocol["groups"]:
                if args.method and group["method"] != args.method:
                    continue
                if args.condition and group["condition"] != args.condition:
                    continue
                yield video, rep, group


def prepare(protocol, input_root, jobs, output_root):
    records = []
    missing = []
    seen = set()
    for video, rep, group in jobs:
        condition = group["condition"]
        for kind in ("annotation", "detections"):
            path = input_path(protocol, input_root, kind, video, rep, condition)
            key = (kind, str(path))
            if key in seen:
                continue
            seen.add(key)
            if not path.is_file():
                missing.append(str(path))
                continue
            records.append({"kind": kind, "video": video, "rep": rep,
                            "condition": condition, "path": str(path.relative_to(input_root)),
                            "sha256": sha256(path), "bytes": path.stat().st_size})
        if group["method"].startswith("e19_"):
            path = input_path(protocol, input_root, "auxiliary_candidates", video, rep, condition)
            key = ("auxiliary_candidates", str(path))
            if key not in seen:
                seen.add(key)
                if not path.is_file():
                    missing.append(str(path))
                else:
                    records.append({"kind": "auxiliary_candidates", "condition": condition,
                                    "path": str(path.relative_to(input_root)),
                                    "sha256": sha256(path), "bytes": path.stat().st_size})
    if missing:
        raise FileNotFoundError("Missing frozen inputs:\n" + "\n".join(missing))
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "input_manifest.json"
    with open(manifest, "w", encoding="utf-8") as handle:
        json.dump({"experiment": protocol["experiment"], "historical_id": protocol["historical_id"],
                   "files": records}, handle, indent=2)
    print(f"Validated {len(records)} files; manifest: {manifest}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["mot17_development", "mot17_additional",
                                               "dancetrack_main"])
    parser.add_argument("stage", choices=["prepare", "run"])
    parser.add_argument("--input-root", type=Path, required=True,
                        help="External directory following protocol.json paths")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--video", default="")
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--method", default="")
    parser.add_argument("--condition", choices=["clean", "A"], default="")
    parser.add_argument("--only-segment", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    directory, protocol = load_protocol(args.experiment)
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    jobs = list(selected_jobs(protocol, args))
    if not jobs:
        raise ValueError("No jobs match the requested filters")
    if args.stage == "prepare":
        prepare(protocol, args.input_root, jobs, args.output_root)
        return
    for video, rep, group in jobs:
        condition = group["condition"]
        aux = ""
        if group["method"].startswith("e19_"):
            aux = str(input_path(protocol, args.input_root, "auxiliary_candidates",
                                 video, rep, condition))
        run(group["method"], video, rep, condition, args.output_root,
            input_path(protocol, args.input_root, "annotation", video, rep, condition),
            input_path(protocol, args.input_root, "detections", video, rep, condition),
            directory / "methods.json", directory / "protocol.json", aux,
            args.only_segment or None, args.resume, protocol["segment_length"])


if __name__ == "__main__":
    main()
