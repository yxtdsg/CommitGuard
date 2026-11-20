#!/usr/bin/env python3
"""Evaluate complete TraSw runs, identity outcomes, and reactive audit timing."""

import argparse
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["all", "metrics", "identity"], default="all", nargs="?")
    parser.add_argument("--runs-root", type=Path, default=REPO / "runs" / "trasw",
                        help="TraSw work directory containing victim_runs and tensors_raw")
    parser.add_argument("--gt-root", type=Path, required=True,
                        help="Directory containing MOT17-XX-FRCNN/gt/gt.txt")
    parser.add_argument("--input-root", type=Path, default=REPO / "external_inputs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()
    runs = args.runs_root.resolve()
    output = args.output_dir.resolve()
    os.environ["COMMITGUARD_TRASW_RUNS"] = str(runs)
    os.environ["COMMITGUARD_TRASW_ANALYSIS"] = str(output)
    os.environ["COMMITGUARD_TRASW_EVAL"] = str((args.work_dir or runs / "eval").resolve())
    os.environ["COMMITGUARD_GT_ROOT"] = str(args.gt_root.resolve())
    os.environ["COMMITGUARD_INPUT_ROOT"] = str(args.input_root.resolve())
    if args.stage in ("all", "metrics"):
        import evaluate_trasw
        evaluate_trasw.main()
    if args.stage in ("all", "identity"):
        import analyze_trasw_identity
        analyze_trasw_identity.main()


if __name__ == "__main__":
    main()
