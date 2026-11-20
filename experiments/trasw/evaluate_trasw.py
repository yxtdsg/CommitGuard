import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import trackeval

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = Path(os.environ.get("COMMITGUARD_TRASW_RUNS", ROOT / "runs" / "trasw")).resolve()
RUNS = WORK / "victim_runs"
ANALYSIS = Path(os.environ.get("COMMITGUARD_TRASW_ANALYSIS", WORK / "analysis")).resolve()
EVAL = Path(os.environ.get("COMMITGUARD_TRASW_EVAL", WORK / "eval")).resolve()
GT_ROOT = Path(os.environ.get("COMMITGUARD_GT_ROOT", ROOT / "external_inputs" / "gt" / "mot17")).resolve()
INPUT_ROOT = Path(os.environ.get("COMMITGUARD_INPUT_ROOT", ROOT / "external_inputs")).resolve()
ARMS = ("clean_c0", "attack_c0", "clean_cg", "attack_cg")
REPS = ("r1", "r2", "r3")
RATIO_METRICS = ("IDF1", "IDP", "IDR", "HOTA", "AssA", "MOTA")
COUNT_METRICS = ("IDSW", "FP", "FN", "TP")
METRICS = RATIO_METRICS + COUNT_METRICS


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_segments():
    freeze = HERE / "freeze.json"
    with open(freeze, encoding="utf-8") as f:
        segments = json.load(f)["segments"]
    if len(segments) != 10:
        raise RuntimeError("E29 开发集必须为 10 段")
    return segments


def completed_run(seg_name, arm, rep):
    run_name = f"e29_{arm}_{seg_name}_{rep}"
    run_root = RUNS / run_name
    completion_path = run_root / "completion.json"
    if not completion_path.is_file():
        raise FileNotFoundError(f"未完成运行: {run_name}")
    with open(completion_path, encoding="utf-8") as f:
        completion = json.load(f)
    if completion.get("status") != "COMPLETE_VALIDATED":
        raise RuntimeError(f"运行未通过完整性验证: {run_name}")
    validation = completion["validation"]
    result = Path(validation["result"])
    if not result.is_file():
        raise FileNotFoundError(f"缺少轨迹: {result}")
    return run_name, result, validation


def prepare_segment_gt(seg, seq):
    annotation_path = INPUT_ROOT / seg["annotation"]
    with open(annotation_path, encoding="utf-8") as f:
        annotation = json.load(f)
    first_image = annotation["images"][0]
    gt_dir = EVAL / "gt" / "MOT17-val" / seq / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    source = GT_ROOT / f"{seg['video']}-FRCNN" / "gt" / "gt.txt"
    rows = []
    with open(source, encoding="utf-8") as f:
        for line in f:
            row = line.rstrip("\n").split(",")
            frame = int(row[0])
            if seg["val_half_frame_start"] <= frame <= seg["val_half_frame_end"] and \
                    int(row[6]) == 1 and int(row[7]) == 1:
                row[0] = str(frame - seg["val_half_frame_start"] + 1)
                rows.append(",".join(row))
    (gt_dir / "gt.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (gt_dir.parent / "seqinfo.ini").write_text(
        f"[Sequence]\nname={seq}\nseqLength=30\nframeRate=30\n"
        f"imWidth={first_image['width']}\nimHeight={first_image['height']}\nimDir=img1\n",
        encoding="utf-8")


def evaluate_segment(seg):
    seq = seg["seg_name"]
    prepare_segment_gt(seg, seq)
    names = []
    run_info = {}
    for arm in ARMS:
        for rep in REPS:
            name, result, validation = completed_run(seq, arm, rep)
            names.append(name)
            run_info[name] = (arm, rep, validation)
            target = EVAL / "trackers" / "MOT17-val" / name / "data"
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(result, target / f"{seq}.txt")
    seqmap = EVAL / f"seqmap_{seq}.txt"
    seqmap.write_text(f"name\n{seq}\n", encoding="utf-8")
    cfg = {"GT_FOLDER": str(EVAL / "gt"), "TRACKERS_FOLDER": str(EVAL / "trackers"),
           "TRACKERS_TO_EVAL": names, "BENCHMARK": "MOT17", "SPLIT_TO_EVAL": "val",
           "SEQMAP_FILE": str(seqmap), "PRINT_CONFIG": False, "OUTPUT_FOLDER": None,
           "TIME_PROGRESS": False}
    dataset = trackeval.datasets.MotChallenge2DBox(cfg)
    evaluator = trackeval.Evaluator({"USE_PARALLEL": False, "PRINT_RESULTS": False,
                                     "PRINT_PROGRESS": False, "TIME_PROGRESS": False})
    result, _ = evaluator.evaluate([dataset], [
        trackeval.metrics.HOTA(), trackeval.metrics.Identity(), trackeval.metrics.CLEAR()])
    rows = []
    for name in names:
        arm, rep, validation = run_info[name]
        values = result["MotChallenge2DBox"][name][seq]["pedestrian"]
        row = {"arm": arm, "condition": arm.split("_")[0],
               "method": arm.split("_")[1], "video": seg["video"].removeprefix("MOT17-"),
               "seg_index": seg["seg_index"], "seg_name": seq, "rep": int(rep[1:]),
               "IDF1": float(values["Identity"]["IDF1"]),
               "IDP": float(values["Identity"]["IDP"]),
               "IDR": float(values["Identity"]["IDR"]),
               "HOTA": float(np.mean(values["HOTA"]["HOTA"])),
               "AssA": float(np.mean(values["HOTA"]["AssA"])),
               "MOTA": float(values["CLEAR"]["MOTA"]),
               "IDSW": int(values["CLEAR"]["IDSW"]),
               "FP": int(values["CLEAR"]["CLR_FP"]),
               "FN": int(values["CLEAR"]["CLR_FN"]),
               "TP": int(values["CLEAR"]["CLR_TP"]),
               "alarm_frames": validation.get("n_alarm_frames", 0),
               "intervention_frames": validation.get("n_intervention_frames", 0),
               "aux_review_frames": validation.get("n_aux_review_frames", 0),
               "aux_filter_removed": validation.get("aux_filter_removed", 0),
               "aux_fn_added": validation.get("aux_fn_added", 0),
               "aux_birth_rejected": validation.get("aux_birth_rejected", 0)}
        rows.append(row)
    print("EVALUATED", seq, flush=True)
    return rows


def aggregate(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, members in groups.items():
        row = dict(zip(keys, key))
        row["runs"] = len(members)
        for metric in METRICS:
            row[metric] = float(np.mean([float(r[metric]) for r in members]))
        for field in ("alarm_frames", "intervention_frames", "aux_review_frames",
                      "aux_filter_removed", "aux_fn_added", "aux_birth_rejected"):
            row[field] = float(np.mean([float(r[field]) for r in members]))
        output.append(row)
    return sorted(output, key=lambda r: tuple(str(r[k]) for k in keys))


def paired_effects(video_rows, macro_rows):
    rows = []
    comparisons = (("attack_defense", "attack_cg", "attack_c0"),
                   ("attack_damage", "attack_c0", "clean_c0"),
                   ("clean_cost", "clean_cg", "clean_c0"),
                   ("residual_gap", "attack_cg", "clean_c0"))
    for scope, source in (("video", video_rows), ("macro", macro_rows)):
        groups = {(r["arm"], r.get("video", "all")): r for r in source}
        videos = sorted({r.get("video", "all") for r in source})
        for video in videos:
            for comparison, test_arm, base_arm in comparisons:
                test = groups[(test_arm, video)]
                base = groups[(base_arm, video)]
                out = {"scope": scope, "video": video, "comparison": comparison,
                       "test_arm": test_arm, "base_arm": base_arm}
                for metric in RATIO_METRICS:
                    out[f"delta_{metric}_pp"] = 100.0 * (test[metric] - base[metric])
                for metric in COUNT_METRICS:
                    out[f"delta_{metric}"] = test[metric] - base[metric]
                rows.append(out)
    return rows


def clean_regression(segments):
    rows = []
    for seg in segments:
        for rep in REPS:
            _, c0_path, _ = completed_run(seg["seg_name"], "clean_c0", rep)
            _, cg_path, cg_validation = completed_run(seg["seg_name"], "clean_cg", rep)
            rows.append({"seg_name": seg["seg_name"], "rep": int(rep[1:]),
                         "c0_result_sha256": __import__("hashlib").sha256(c0_path.read_bytes()).hexdigest(),
                         "cg_result_sha256": __import__("hashlib").sha256(cg_path.read_bytes()).hexdigest(),
                         "byte_identical": c0_path.read_bytes() == cg_path.read_bytes(),
                         "alarm_frames": cg_validation.get("n_alarm_frames", 0),
                         "aux_review_frames": cg_validation.get("n_aux_review_frames", 0)})
    return rows


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)
    segments = load_segments()
    segment_rows = []
    for seg in segments:
        segment_rows.extend(evaluate_segment(seg))
    write_csv(ANALYSIS / "segment_metrics.csv", segment_rows)
    video_rows = aggregate(segment_rows, ["arm", "condition", "method", "video"])
    macro_rows = aggregate(segment_rows, ["arm", "condition", "method"])
    write_csv(ANALYSIS / "video_summary.csv", video_rows)
    write_csv(ANALYSIS / "macro_summary.csv", macro_rows)
    write_csv(ANALYSIS / "paired_effects.csv", paired_effects(video_rows, macro_rows))
    regression_rows = clean_regression(segments)
    write_csv(ANALYSIS / "clean_regression.csv", regression_rows)
    checks = {"segments": len(segments), "segment_rows": len(segment_rows),
              "expected_segment_rows": 120, "video_rows": len(video_rows),
              "macro_rows": len(macro_rows), "clean_runs": len(regression_rows),
              "clean_byte_identical_runs": sum(r["byte_identical"] for r in regression_rows),
              "attack_changed_frames": sum(json.load(open(
                  WORK / "tensors_raw" / seg["seg_name"] / "composition.json", encoding="utf-8"))[i]["changed"]
                  for seg in segments for i in range(30))}
    if checks["segment_rows"] != 120 or len(video_rows) != 12 or len(macro_rows) != 4:
        raise RuntimeError("E29 评测行数不符合冻结矩阵")
    with open(ANALYSIS / "evaluation_checks.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)
    print("DONE", checks, flush=True)


if __name__ == "__main__":
    main()
