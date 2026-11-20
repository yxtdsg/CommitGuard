#!/usr/bin/env python3
"""Run the official OC-SORT adapter on one frozen comparison input."""

import argparse
import json
import os
import sys
from pathlib import Path

import common

sys.path.insert(0, str(common.REPO / "src"))
sys.path.insert(0, str(common.REPO / "experiments" / "ocsort_recovery"))
from commitguard.runner import run
from ocsort_adapter import OCSortAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, choices=common.VIDEOS)
    parser.add_argument("--rep", required=True, type=int, choices=common.REPS)
    parser.add_argument("--condition", default="A", choices=["A"])
    parser.add_argument("--input-root", default=str(common.INPUT_ROOT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ocsort-root", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    common.configure(args.input_root)
    os.environ["OCSORT_ROOT"] = str(Path(args.ocsort_root).resolve())
    required = [common.annotation_path(args.video), common.detection_path(args.video, args.rep)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing comparison inputs:\n" + "\n".join(missing))
    if args.dry_run:
        print(f"READY ocsort {args.video} r{args.rep}")
        return
    run("ocsort", args.video, args.rep, "A", Path(args.output_root),
        common.annotation_path(args.video), common.detection_path(args.video, args.rep),
        common.E29 / "methods.json", common.E29 / "protocol.json", "", None, args.resume, 30,
        OCSortAdapter)


if __name__ == "__main__":
    main()
