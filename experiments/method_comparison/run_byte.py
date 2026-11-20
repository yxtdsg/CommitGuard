#!/usr/bin/env python3
"""Run C0, BG, CG, TrackShield, or BUSCA on one frozen comparison input."""

import argparse
import json
import sys
from pathlib import Path

import common

REPO = common.REPO
sys.path.insert(0, str(REPO / "src"))
from commitguard.runner import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["c0", "g3", "e19_fi40", "a1", "a2", "busca"])
    parser.add_argument("--video", required=True, choices=common.VIDEOS)
    parser.add_argument("--rep", required=True, type=int, choices=common.REPS)
    parser.add_argument("--condition", default="A", choices=["A"])
    parser.add_argument("--input-root", default=str(common.INPUT_ROOT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--only-segment", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    common.configure(args.input_root)
    required = [common.annotation_path(args.video), common.detection_path(args.video, args.rep)]
    if args.method == "e19_fi40":
        required.append(common.auxiliary_path())
    if args.method in ("a1", "a2"):
        required.append(common.E29 / "calibration.json")
    if args.method == "busca":
        required.append(common.image_root())
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing comparison inputs:\n" + "\n".join(missing))
    if args.dry_run:
        print(f"READY {args.method} {args.video} r{args.rep} -> {args.output_root}")
        return
    metadata = run(args.method, args.video, args.rep, "A", Path(args.output_root),
                   common.annotation_path(args.video), common.detection_path(args.video, args.rep),
                   common.E29 / "methods.json", common.E29 / "protocol.json",
                   str(common.auxiliary_path()) if args.method == "e19_fi40" else "",
                   args.only_segment or None, args.resume, 30,
                   image_root=common.image_root())
    data = json.loads(metadata.read_text(encoding="utf-8"))
    common.add_e29_provenance(data, vars(args))
    metadata.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
