#!/usr/bin/env python3
"""Evaluate a completed replay matrix with TrackEval 1.3.0."""

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from commitguard.runner import sha256


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def input_path(protocol, input_root, kind, video, rep=1, condition="A"):
    values = {"video": video, "rep": rep, "condition": condition,
              "annotation": protocol.get("annotations", {}).get(video, "")}
    return input_root / protocol["paths"][kind].format(**values)


def sequence_names(protocol, video):
    prefix = "dancetrack" if protocol["experiment"] == "dancetrack_main" else "MOT17-"
    suffix = "" if protocol["experiment"] == "dancetrack_main" else "-FRCNN"
    return [f"{prefix}{video}{suffix}-{segment:03d}"
            for segment in range(protocol["segments"][video])]


def prepare_gt(protocol, input_root, video, sequences, eval_root):
    annotation_path = input_path(protocol, input_root, "annotation", video)
    with open(annotation_path, encoding="utf-8") as handle:
        annotation = json.load(handle)
    first = sorted(annotation["images"], key=lambda item: item["id"])[0]
    source = input_path(protocol, input_root, "ground_truth", video)
    with open(source, encoding="utf-8") as handle:
        lines = handle.readlines()
    dance = protocol["experiment"] == "dancetrack_main"
    for segment, sequence in enumerate(sequences):
        gt_dir = eval_root / "gt" / "MOT17-val" / sequence / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for line in lines:
            row = line.rstrip("\n").split(",")
            if len(row) < 8:
                continue
            frame = int(row[0])
            if 30 * segment + 1 <= frame <= 30 * segment + 30 and \
                    int(row[6]) == 1 and int(row[7]) == 1:
                row[0] = str(frame - 30 * segment)
                if dance:
                    row[1] = str(int(row[1]) + 1)
                rows.append(",".join(row))
        (gt_dir / "gt.txt").write_text("\n".join(rows) + ("\n" if rows else ""),
                                       encoding="utf-8")
        (gt_dir.parent / "seqinfo.ini").write_text(
            f"[Sequence]\nname={sequence}\nseqLength=30\n"
            f"frameRate={protocol.get('frame_rate', 30)}\n"
            f"imWidth={first['width']}\nimHeight={first['height']}\nimDir=img1\n",
            encoding="utf-8")


def metric_values(values):
    hota = values["HOTA"]
    identity = values["Identity"]
    clear = values["CLEAR"]
    return {
        "MOTA": float(clear["MOTA"]), "IDF1": float(identity["IDF1"]),
        "IDP": float(identity["IDP"]), "IDR": float(identity["IDR"]),
        "HOTA": float(np.mean(hota["HOTA"])), "AssA": float(np.mean(hota["AssA"])),
        "DetA": float(np.mean(hota["DetA"])), "DetRe": float(np.mean(hota["DetRe"])),
        "DetPr": float(np.mean(hota["DetPr"])), "IDSW": int(clear["IDSW"]),
        "FP": int(clear["CLR_FP"]), "FN": int(clear["CLR_FN"]),
        "TP": int(clear["CLR_TP"]),
    }


def aggregate(rows, keys, metrics, count_name):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, members in groups.items():
        row = dict(zip(keys, key))
        row[count_name] = len(members)
        for metric in metrics:
            row[metric] = float(np.mean([float(member[metric]) for member in members]))
        output.append(row)
    return sorted(output, key=lambda row: tuple(str(row[key]) for key in keys))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["mot17_development", "mot17_additional",
                                               "dancetrack_main"])
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("runs/evaluation"))
    parser.add_argument("--trackeval-root", type=Path, default=None,
                        help="Optional TrackEval 1.3.0 checkout; otherwise use installed package")
    args = parser.parse_args(argv)
    if args.trackeval_root:
        sys.path.insert(0, str(args.trackeval_root.resolve()))
    import trackeval

    experiment_dir = REPO / "experiments" / args.experiment
    with open(experiment_dir / "protocol.json", encoding="utf-8") as handle:
        protocol = json.load(handle)
    args.input_root = args.input_root.resolve()
    args.runs_root = args.runs_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    evaluator = trackeval.Evaluator({"USE_PARALLEL": False, "PRINT_RESULTS": False,
                                     "PRINT_PROGRESS": False, "TIME_PROGRESS": False})
    segment_rows = []
    manifest = []
    for video in protocol["videos"]:
        sequences = sequence_names(protocol, video)
        prepare_gt(protocol, args.input_root, video, sequences, args.work_dir)
        names = []
        contexts = {}
        for group in protocol["groups"]:
            for rep in protocol["repeats"]:
                run_id = f"{args.experiment}_{group['method']}_{group['condition']}{video}_r{rep}"
                source = args.runs_root / f"{group['method']}_{group['condition']}{video}_r{rep}"
                with open(source / "runtime.json", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                names.append(run_id)
                contexts[run_id] = group, rep, metadata
                target = args.work_dir / "trackers" / "MOT17-val" / run_id / "data"
                target.mkdir(parents=True, exist_ok=True)
                for sequence in sequences:
                    track = source / f"{sequence}.txt"
                    if not track.is_file():
                        raise FileNotFoundError(track)
                    shutil.copyfile(track, target / f"{sequence}.txt")
                manifest.append({"run_id": run_id, "method": group["method"],
                                 "condition": group["condition"], "video": video, "rep": rep,
                                 "runtime_sha256": sha256(source / "runtime.json")})
        seqmap = args.work_dir / f"seqmap_{args.experiment}_{video}.txt"
        seqmap.write_text("name\n" + "\n".join(sequences) + "\n", encoding="utf-8")
        config = {"GT_FOLDER": str(args.work_dir / "gt"),
                  "TRACKERS_FOLDER": str(args.work_dir / "trackers"),
                  "TRACKERS_TO_EVAL": names, "BENCHMARK": "MOT17", "SPLIT_TO_EVAL": "val",
                  "SEQMAP_FILE": str(seqmap), "PRINT_CONFIG": False,
                  "OUTPUT_FOLDER": None, "TIME_PROGRESS": False}
        dataset = trackeval.datasets.MotChallenge2DBox(config)
        result, _ = evaluator.evaluate([dataset], [trackeval.metrics.HOTA(),
                                                   trackeval.metrics.Identity(),
                                                   trackeval.metrics.CLEAR()])
        for run_id in names:
            group, rep, metadata = contexts[run_id]
            runtimes = {item["seg"]: item["mean_update_ms"] for item in metadata["segments"]}
            for sequence in sequences:
                values = result["MotChallenge2DBox"][run_id][sequence]["pedestrian"]
                row = {"label": group.get("label", group["method"]),
                       "method": group["method"], "condition": group["condition"],
                       "video": video, "rep": rep, "seg": int(sequence.rsplit("-", 1)[1])}
                row.update(metric_values(values))
                row["mean_update_ms"] = runtimes[sequence]
                segment_rows.append(row)
        print(f"EVALUATED {args.experiment} {video}", flush=True)

    metrics = protocol["metrics"]
    write_csv(args.output_dir / "segment_metrics.csv" if args.experiment != "mot17_development"
              else args.output_dir / "paired_metrics.csv", segment_rows)
    video_rows = aggregate(segment_rows, ["label", "method", "condition", "video"],
                           metrics, "segments_reps")
    macro_rows = aggregate(video_rows, ["label", "method", "condition"], metrics, "videos")
    if args.experiment != "dancetrack_main":
        for row in video_rows:
            row.pop("label", None)
            row.pop("segments_reps", None)
    write_csv(args.output_dir / "video_summary.csv", video_rows)
    if args.experiment != "dancetrack_main":
        for row in macro_rows:
            row.pop("label", None)
    write_csv(args.output_dir / "macro_summary.csv", macro_rows)
    with open(args.output_dir / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"experiment": args.experiment, "trackeval": "1.3.0",
                   "runs": manifest}, handle, indent=2)


if __name__ == "__main__":
    main()
