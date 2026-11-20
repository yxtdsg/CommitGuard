import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

import common
from common import E29, load_json, segment_name, sha256, write_json

PROTOCOL = load_json(E29 / "protocol.json")
ANALYSIS = Path(os.environ.get("COMMITGUARD_COMPARISON_ANALYSIS", common.RESULTS / "analysis")).resolve()
CACHE = ANALYSIS / "cache"
METRICS = ("MOTA", "IDF1", "HOTA", "IDSW", "FN", "FP")
RUN_FIELDS = ("method", "label", "video", "rep", "condition", "segments", *METRICS)


def normalized(path):
    return str(Path(path).resolve()).lower()


def check_track(path):
    problems = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = line.rstrip("\n").split(",")
            if len(row) < 7:
                problems.append(f"{path}:{line_number}: fewer than 7 columns")
                continue
            try:
                frame = int(float(row[0]))
            except ValueError:
                problems.append(f"{path}:{line_number}: invalid frame")
                continue
            if not 1 <= frame <= 30:
                problems.append(f"{path}:{line_number}: non-local frame {frame}")
    return problems


def validate_entry(entry):
    problems = []
    runtime_path = Path(entry["runtime"])
    track_paths = [Path(path) for path in entry["tracks"]]
    if not runtime_path.exists():
        return None, ["missing runtime.json"]
    try:
        runtime = load_json(runtime_path)
    except Exception as error:
        return None, [f"invalid runtime.json: {error}"]
    for path in track_paths:
        if not path.exists():
            problems.append(f"missing track: {path}")
        else:
            problems.extend(check_track(path))
    for field, expected in (("method", entry["runner_method"]), ("video", entry["video"]),
                            ("rep", entry["rep"]), ("condition", entry["condition"])):
        if runtime.get(field) != expected:
            problems.append(f"runtime {field}={runtime.get(field)!r}, expected {expected!r}")
    expected_segments = {segment_name(entry["video"], seg) for seg in (0, 1)}
    runtime_segments = runtime.get("segments", [])
    if {item.get("seg") for item in runtime_segments} != expected_segments:
        problems.append("runtime does not contain exactly seg000/001")
    if any(item.get("frames") != 30 for item in runtime_segments):
        problems.append("runtime segment length is not 30")
    if normalized(runtime.get("annotation", "")) != normalized(entry["annotation"]):
        problems.append("annotation path differs from manifest")
    if normalized(runtime.get("detections", "")) != normalized(entry["detections"]):
        problems.append("detection path differs from manifest")
    try:
        actual_detection_hash = sha256(common.detection_path(entry["video"], entry["rep"]))
        if runtime.get("detections_sha256") != actual_detection_hash:
            problems.append("runtime detection hash differs from current input")
        if entry.get("detections_sha256") and entry["detections_sha256"] != actual_detection_hash:
            problems.append("manifest detection hash differs from current input")
    except (FileNotFoundError, ValueError) as error:
        problems.append(str(error))
    if sha256(entry["annotation"]) != entry["annotation_sha256"]:
        problems.append("annotation hash differs from manifest")
    if not entry["reused"] and entry["family"] not in ("ocsort", "bot_sort_noreid"):
        history = PROTOCOL.get("config_history", {})
        valid_protocol_hashes = [sha256(E29 / "protocol.json")] + history.get("previous_protocol_sha256", [])
        valid_methods_hashes = [sha256(E29 / "methods.json")] + history.get("previous_methods_sha256", [])
        if runtime.get("e29_protocol_sha256") not in valid_protocol_hashes:
            problems.append("missing or unregistered E29 protocol hash")
        if runtime.get("e29_method_config_sha256") not in valid_methods_hashes:
            problems.append("missing or unregistered E29 methods hash")
    return runtime, problems


def filtered_gt(video, segment):
    source = common.INPUT_ROOT / "gt" / f"MOT17-{video}-FRCNN" / "gt" / "gt.txt"
    start = segment * 30 + 1
    stop = start + 29
    rows = []
    with open(source, encoding="utf-8") as handle:
        for line in handle:
            row = line.rstrip("\n").split(",")
            if len(row) < 8:
                continue
            global_frame = int(row[0])
            if start <= global_frame <= stop and int(row[6]) == 1 and int(row[7]) == 1:
                shifted = list(row)
                shifted[0] = str(global_frame - segment * 30)
                rows.append(",".join(shifted))
    text = "\n".join(rows) + ("\n" if rows else "")
    annotation = load_json(common.annotation_path(video))
    images = sorted(annotation["images"], key=lambda item: item["id"])[segment * 30:(segment + 1) * 30]
    if len(images) != 30:
        raise ValueError(f"Annotation segment incomplete: {video} seg{segment:03d}")
    info = {
        "source": str(source),
        "source_global_frames": [start, stop],
        "source_original_first_image": images[0]["file_name"],
        "source_original_last_image": images[-1]["file_name"],
        "tracker_local_frames": [1, 30],
        "width": int(images[0]["width"]),
        "height": int(images[0]["height"]),
        "frame_rate": 25 if video in ("10", "13") else 30,
        "seqLength": 30,
        "filter": "conf==1 and class==1",
    }
    return text, info


def evaluator_hash():
    import trackeval
    root = Path(trackeval.__file__).resolve().parent
    sources = [root / "eval.py", root / "utils.py",
               root / "datasets" / "mot_challenge_2d_box.py",
               root / "metrics" / "hota.py", root / "metrics" / "identity.py",
               root / "metrics" / "clear.py", Path(__file__).resolve()]
    payload = {
        "sources": {str(path): sha256(path) for path in sources},
        "metric_classes": ["HOTA", "Identity", "CLEAR"],
        "hota": "mean over alphas",
        "gt_filter": "conf==1 and class==1",
        "seqLength": 30,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def digest_bytes(data):
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def cache_identity(entry, segment):
    gt_text, gt_info = filtered_gt(entry["video"], segment)
    track = Path(entry["tracks"][segment])
    track_hash = sha256(track)
    gt_hash = digest_bytes(gt_text)
    evaluation_hash = evaluator_hash()
    key_payload = {"track_sha256": track_hash, "gt_sha256": gt_hash,
                   "evaluator_sha256": evaluation_hash,
                   "video": entry["video"], "segment": segment,
                   "frame_rate": gt_info["frame_rate"], "seqLength": 30}
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()
    return key, key_payload, gt_text, gt_info


def evaluate_segment(entry, segment):
    key, key_payload, gt_text, gt_info = cache_identity(entry, segment)
    directory = CACHE / key
    result_path = directory / "metrics.json"
    if result_path.exists():
        cached = load_json(result_path)
        if all(cached.get(field) == key_payload[field] for field in
               ("track_sha256", "gt_sha256", "evaluator_sha256")):
            return cached["metrics"], key
    import trackeval

    seq = f"MOT17-{entry['video']}-E29-{segment:03d}"
    tracker = entry["run_name"]
    gt_dir = directory / "gt" / "MOT17-val" / seq / "gt"
    tracker_dir = directory / "trackers" / "MOT17-val" / tracker / "data"
    gt_dir.mkdir(parents=True, exist_ok=True)
    tracker_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / "gt.txt").write_text(gt_text, encoding="utf-8")
    (gt_dir.parent / "seqinfo.ini").write_text(
        f"[Sequence]\nname={seq}\nseqLength=30\nframeRate={gt_info['frame_rate']}\n"
        f"imWidth={gt_info['width']}\nimHeight={gt_info['height']}\nimDir=img1\n",
        encoding="utf-8")
    shutil.copyfile(entry["tracks"][segment], tracker_dir / f"{seq}.txt")
    config = {"GT_FOLDER": str(directory / "gt"), "TRACKERS_FOLDER": str(directory / "trackers"),
              "TRACKERS_TO_EVAL": [tracker], "BENCHMARK": "MOT17", "SPLIT_TO_EVAL": "val",
              "SEQ_INFO": {seq: 30}, "PRINT_CONFIG": False, "OUTPUT_FOLDER": None,
              "TIME_PROGRESS": False}
    dataset = trackeval.datasets.MotChallenge2DBox(config)
    evaluator = trackeval.Evaluator({"USE_PARALLEL": False, "PRINT_RESULTS": False,
                                     "PRINT_PROGRESS": False, "TIME_PROGRESS": False})
    result, _ = evaluator.evaluate([dataset], [trackeval.metrics.HOTA(),
                                               trackeval.metrics.Identity(),
                                               trackeval.metrics.CLEAR()])
    values = result["MotChallenge2DBox"][tracker][seq]["pedestrian"]
    metrics = {
        "HOTA": float(np.mean(values["HOTA"]["HOTA"])),
        "IDF1": float(values["Identity"]["IDF1"]),
        "MOTA": float(values["CLEAR"]["MOTA"]),
        "IDSW": int(values["CLEAR"]["IDSW"]),
        "FN": int(values["CLEAR"]["CLR_FN"]),
        "FP": int(values["CLEAR"]["CLR_FP"]),
    }
    write_json(result_path, {"cache_key": key, **key_payload, "gt_mapping": gt_info,
                             "metrics": metrics})
    return metrics, key


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(run_rows, method_order):
    lookup = {(row["method"], row["video"], row["rep"]): row for row in run_rows}
    video_rows = []
    for method in method_order:
        for video in PROTOCOL["videos"]:
            rows = [lookup[(method, video, rep)] for rep in (1, 2, 3)]
            row = {"method": method, "label": rows[0]["label"], "video": video, "reps": 3}
            for metric in METRICS:
                row[metric] = float(np.mean([float(item[metric]) for item in rows]))
            video_rows.append(row)
    macro_rows = []
    for method in method_order:
        rows = [row for row in video_rows if row["method"] == method]
        row = {"method": method, "label": rows[0]["label"], "videos": 4, "reps": 3, "segments": 2}
        for metric in METRICS:
            row[metric] = float(np.mean([float(item[metric]) for item in rows]))
        for metric in ("MOTA", "IDF1", "HOTA"):
            row[metric] *= 100.0
        macro_rows.append(row)
    return video_rows, macro_rows


def main():
    global ANALYSIS, CACHE
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default=str(common.INPUT_ROOT))
    parser.add_argument("--manifest", default=str(common.MANIFEST))
    parser.add_argument("--output-dir", default=str(ANALYSIS))
    parser.add_argument("--coverage-only", action="store_true", help="validate manifest/runs without TrackEval")
    parser.add_argument("--dry-run", action="store_true", help="print structural coverage and write nothing")
    args = parser.parse_args()
    common.configure(input_root=args.input_root)
    ANALYSIS = Path(args.output_dir).resolve()
    CACHE = ANALYSIS / "cache"
    manifest = load_json(args.manifest)
    entries = manifest["entries"]
    if len(entries) != 96:
        raise ValueError("E29 requires exactly 96 runs")
    valid = []
    missing = []
    invalid = []
    for entry in entries:
        runtime, problems = validate_entry(entry)
        identity = {"method": entry["method"], "video": entry["video"], "rep": entry["rep"],
                    "source_dir": entry["source_dir"]}
        if runtime is None and all(not Path(path).exists() for path in entry["tracks"]):
            missing.append(identity)
        elif problems:
            invalid.append({**identity, "problems": problems})
        else:
            valid.append(entry)
    coverage = {"expected_runs": 96, "valid_runs": len(valid), "missing_runs": missing,
                "invalid_runs": invalid, "expected_segments": 192,
                "valid_segments": len(valid) * 2, "evaluated_segments": 0,
                "run_metrics_rows": 0, "evaluation_errors": [], "complete": False,
                "paper_macro_written": False}
    if args.dry_run:
        print(json.dumps(coverage, indent=2))
        return
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    run_rows = []
    if not args.coverage_only:
        for entry in valid:
            segment_metrics = []
            cache_keys = []
            try:
                for segment in (0, 1):
                    values, key = evaluate_segment(entry, segment)
                    segment_metrics.append(values)
                    cache_keys.append(key)
            except Exception as error:
                coverage["evaluation_errors"].append({
                    "method": entry["method"], "video": entry["video"], "rep": entry["rep"],
                    "error": str(error)})
                continue
            row = {"method": entry["method"], "label": entry["label"], "video": entry["video"],
                   "rep": entry["rep"], "condition": entry["condition"], "segments": 2}
            for metric in METRICS:
                row[metric] = float(np.mean([float(values[metric]) for values in segment_metrics]))
            run_rows.append(row)
            coverage["evaluated_segments"] += 2
            print("EVALUATED", entry["method"], entry["video"], entry["rep"], *cache_keys, flush=True)
        write_csv(ANALYSIS / "run_metrics.csv", run_rows, RUN_FIELDS)
    coverage["run_metrics_rows"] = len(run_rows)
    complete = (len(valid) == 96 and coverage["evaluated_segments"] == 192
                and len(run_rows) == 96 and not coverage["evaluation_errors"])
    coverage["complete"] = complete
    paper = ANALYSIS / "paper_macro.csv"
    if complete:
        video_rows, macro_rows = aggregate(run_rows, manifest["method_order"])
        write_csv(ANALYSIS / "video_summary.csv", video_rows,
                  ("method", "label", "video", "reps", *METRICS))
        write_csv(paper, macro_rows,
                  ("method", "label", "videos", "reps", "segments", *METRICS))
        coverage["paper_macro_written"] = True
    else:
        if paper.exists():
            paper.unlink()
    write_json(ANALYSIS / "coverage.json", coverage)
    print("COVERAGE", json.dumps({key: value for key, value in coverage.items()
                                  if key not in ("missing_runs", "invalid_runs", "evaluation_errors")}))


if __name__ == "__main__":
    main()
