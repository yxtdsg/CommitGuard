#!/usr/bin/env python3
"""Execute the serial 96-run frozen method-comparison matrix."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import common

HERE = Path(__file__).resolve().parent
FAMILIES = {method: method for method in
            ("c0", "a1", "a2", "busca", "ocsort", "bot_sort_noreid", "g3", "e19_fi40")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(common.INPUT_ROOT))
    parser.add_argument("--output-root", default=str(common.RESULTS))
    parser.add_argument("--ocsort-root", default=os.environ.get("OCSORT_ROOT", ""))
    parser.add_argument("--bot-sort-root", default=os.environ.get("BOTSORT_ROOT", ""))
    parser.add_argument("--only-method", choices=list(FAMILIES), default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    env = dict(os.environ)
    env["COMMITGUARD_COMPARISON_INPUTS"] = str(Path(args.input_root).resolve())
    env["COMMITGUARD_COMPARISON_RUNS"] = str(output)
    if args.ocsort_root:
        env["OCSORT_ROOT"] = str(Path(args.ocsort_root).resolve())
    if args.bot_sort_root:
        env["BOTSORT_ROOT"] = str(Path(args.bot_sort_root).resolve())
    jobs = []
    for method, family in FAMILIES.items():
        if args.only_method and method != args.only_method:
            continue
        for video in common.VIDEOS:
            for rep in common.REPS:
                if method == "ocsort":
                    command = [sys.executable, str(HERE / "run_ocsort.py"),
                               "--ocsort-root", env.get("OCSORT_ROOT", "")]
                elif method == "bot_sort_noreid":
                    command = [sys.executable, str(HERE / "run_bot.py"), "--method", "c0"]
                else:
                    command = [sys.executable, str(HERE / "run_byte.py"), "--method", method]
                command += ["--video", video, "--rep", str(rep), "--condition", "A",
                            "--input-root", str(Path(args.input_root).resolve()),
                            "--output-root", str(output / family)]
                if args.dry_run:
                    command.append("--dry-run")
                if args.resume:
                    command.append("--resume")
                jobs.append((method, command, env.copy()))
    print(f"E29 comparison jobs: {len(jobs)}; serial execution; dry_run={args.dry_run}")
    if args.dry_run:
        for method, command, _ in jobs:
            print("READY", method, " ".join(command[2:]))
        return
    for method, command, job_env in jobs:
        print("E29_LAUNCH", method, " ".join(command[2:]), flush=True)
        subprocess.run(command, cwd=HERE, env=job_env, check=True)
    if not args.dry_run:
        subprocess.run([sys.executable, str(HERE / "build_manifest.py"),
                        "--output", str(output / "run_manifest.json")], check=True, env=env)


if __name__ == "__main__":
    main()
