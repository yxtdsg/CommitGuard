import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import trackeval

ANALYSIS = Path("runs/ocsort_recovery/analysis")
EVAL = Path("runs/ocsort_recovery/eval")
RUNS = Path("runs/ocsort_recovery/results")
GT_ROOT = Path("external_inputs/gt/mot17")
SCOPES = {
    "development": {"videos": {"02": 6, "04": 2, "05": 2}, "reps": (1, 2, 3)},
    "additional": {"videos": {"09": 2, "10": 2, "11": 2, "13": 2}, "reps": (1, 2)},
}
WINDOWS = {"attack": (1, 8), "recovery": (9, 30)}
METRICS = ("IDF1", "HOTA", "AssA", "MOTA", "IDSW", "FP", "FN", "TP", "mean_update_ms")
COUNTS = {"IDSW", "FP", "FN", "TP"}
CG_FIELDS = (
    "trigger_frames", "alarm_state_frames", "intervention_frames", "window_frames",
    "suspect_parked", "suspect_confirmed", "suspect_discarded", "suspect_expired",
    "confirmed_priority_blocked_pairs", "local_assoc_blocked_pairs", "affected_tracks_marked",
    "fn_recovery_eligible_lost_track_frames", "fn_recovery_low_score_detections",
    "fn_recovery_candidate_pairs", "fn_recovery_active_conflict_detections", "fn_recovery_matches",
    "aux_review_frames", "aux_birth_review_frames", "aux_original_detections",
    "aux_filter_retained", "aux_filter_removed", "aux_filter_track_protected",
    "aux_fn_candidates", "aux_fn_added", "aux_fn_duplicate_suppressed", "aux_birth_rejected",
    "aux_fn_lost_score_floored", "reid_association_calls", "reid_candidate_pairs",
    "reid_detection_features", "reid_gallery_updates", "reid_gallery_skipped_window_frames",
)
GT_CACHE = {}


def method_label(method):
    return "OC-SORT (official, motion association)" if method == "ocsort" else method


def case_methods(scope):
    pairs = [("c0", "clean"), ("ocsort", "clean")]
    for arm in ("B", "A", "C"):
        pairs.extend((method, arm) for method in ("c0", "e19_fi40", "ocsort"))
    return pairs


def result_root(scope, method, arm):
    return RUNS


def build_runs():
    runs = []
    for scope, protocol in SCOPES.items():
        for method, arm in case_methods(scope):
            for video in protocol["videos"]:
                for rep in protocol["reps"]:
                    name = f"{method}_{arm}{video}_r{rep}"
                    runs.append({
                        "scope": scope, "method": method, "arm": arm,
                        "method_label": method_label(method), "video": video, "rep": rep,
                        "name": name, "root": result_root(scope, method, arm),
                    })
    return runs


def load_gt(video, seg):
    key = (video, seg)
    if key in GT_CACHE:
        return GT_CACHE[key]
    path = GT_ROOT / f"MOT17-{video}-FRCNN" / "gt" / "gt.txt"
    raw = []
    objects = defaultdict(list)
    by_frame = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = line.rstrip("\n").split(",")
            frame = int(row[0]) - 30 * seg
            if not 1 <= frame <= 30:
                continue
            row[0] = str(frame)
            raw.append(row)
            if int(row[6]) == 1 and int(row[7]) == 1:
                box = list(map(float, row[2:6]))
                objects[int(row[1])].append((frame, box))
                by_frame[frame].append(box)
    for observations in objects.values():
        observations.sort()
    GT_CACHE[key] = (raw, objects, by_frame)
    return GT_CACHE[key]


def load_tracks(path):
    tracks = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = line.rstrip("\n").split(",")
            tracks[int(row[0])][int(row[1])] = list(map(float, row[2:6]))
    return tracks


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = iw * ih
    return intersection / max(aw * ah + bw * bh - intersection, 1e-12)


def matched_ids(box, frame_tracks):
    return {track_id for track_id, prediction in frame_tracks.items() if iou(box, prediction) >= 0.5}


def write_csv(name, rows, fieldnames=None):
    path = ANALYSIS / name
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def preflight(runs):
    missing = []
    runtime_errors = []
    for scope, protocol in SCOPES.items():
        for video, n_segments in protocol["videos"].items():
            gt = GT_ROOT / f"MOT17-{video}-FRCNN" / "gt" / "gt.txt"
            if not gt.is_file():
                missing.append(str(gt))
            for seg in range(n_segments):
                try:
                    load_gt(video, seg)
                except Exception as exc:
                    runtime_errors.append(f"GT {video}/{seg}: {exc}")
    for run in runs:
        run_dir = run["root"] / run["name"]
        runtime_path = run_dir / "runtime.json"
        if not runtime_path.is_file():
            missing.append(str(runtime_path))
            continue
        try:
            metadata = json.load(open(runtime_path, encoding="utf-8"))
            run["runtime"] = metadata
        except Exception as exc:
            runtime_errors.append(f"{runtime_path}: {exc}")
            continue
        n_segments = SCOPES[run["scope"]]["videos"][run["video"]]
        for seg in range(n_segments):
            segment = f"MOT17-{run['video']}-FRCNN-{seg:03d}"
            track_path = run_dir / f"{segment}.txt"
            if not track_path.is_file():
                missing.append(str(track_path))
            frames = [f for f in metadata.get("frames", []) if f.get("seg") == segment]
            observed = {(f.get("f"), f.get("ms")) for f in frames}
            if len(frames) != 30 or {f.get("f") for f in frames} != set(range(1, 31)):
                runtime_errors.append(f"{runtime_path}: {segment} has {len(frames)} frame timings")
            elif any(not isinstance(ms, (int, float)) for _, ms in observed):
                runtime_errors.append(f"{runtime_path}: {segment} has non-numeric timings")
    if missing or runtime_errors:
        for path in missing:
            print("MISSING", path, flush=True)
        for error in runtime_errors:
            print("RUNTIME_ERROR", error, flush=True)
        raise FileNotFoundError(f"Preflight failed: {len(missing)} missing files, {len(runtime_errors)} runtime errors")
    print(f"PREFLIGHT_OK runs={len(runs)} missing=0 runtime_errors=0", flush=True)


def prepare_video_eval(scope, video, runs):
    n_segments = SCOPES[scope]["videos"][video]
    sequences = []
    for seg in range(n_segments):
        raw, _, _ = load_gt(video, seg)
        for window, (lo, hi) in WINDOWS.items():
            seq = f"MOT17-{video}-E24-{seg:03d}-{window}"
            sequences.append((seg, window, seq))
            gt_dir = EVAL / "gt" / "MOT17-val" / seq / "gt"
            gt_dir.mkdir(parents=True, exist_ok=True)
            gt_rows = []
            for source_row in raw:
                frame = int(source_row[0])
                if not lo <= frame <= hi:
                    continue
                if scope == "additional" and (int(source_row[6]) != 1 or int(source_row[7]) != 1):
                    continue
                row = source_row.copy()
                row[0] = str(frame - lo + 1)
                gt_rows.append(",".join(row))
            (gt_dir / "gt.txt").write_text("\n".join(gt_rows) + ("\n" if gt_rows else ""), encoding="utf-8")
            width, height = (640, 480) if video == "05" else (1920, 1080)
            (gt_dir.parent / "seqinfo.ini").write_text(
                f"[Sequence]\nname={seq}\nseqLength={hi-lo+1}\nframeRate=30\n"
                f"imWidth={width}\nimHeight={height}\nimDir=img1\n", encoding="utf-8")
    for run in runs:
        run_dir = run["root"] / run["name"]
        target = EVAL / "trackers" / "MOT17-val" / run["name"] / "data"
        target.mkdir(parents=True, exist_ok=True)
        for seg, window, seq in sequences:
            lo, hi = WINDOWS[window]
            source = run_dir / f"MOT17-{video}-FRCNN-{seg:03d}.txt"
            track_rows = []
            with open(source, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = line.rstrip("\n").split(",")
                    frame = int(row[0])
                    if lo <= frame <= hi:
                        row[0] = str(frame - lo + 1)
                        track_rows.append(",".join(row))
            (target / f"{seq}.txt").write_text(
                "\n".join(track_rows) + ("\n" if track_rows else ""), encoding="utf-8")
    seqmap = EVAL / f"seqmap_{scope}_{video}.txt"
    seqmap.write_text("name\n" + "\n".join(seq for _, _, seq in sequences) + "\n", encoding="utf-8")
    return sequences, seqmap


def evaluate_windows(runs):
    rows = []
    evaluator = trackeval.Evaluator({
        "USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_PROGRESS": False, "TIME_PROGRESS": False,
    })
    for scope, protocol in SCOPES.items():
        for video in protocol["videos"]:
            video_runs = [r for r in runs if r["scope"] == scope and r["video"] == video]
            sequences, seqmap = prepare_video_eval(scope, video, video_runs)
            config = {
                "GT_FOLDER": str(EVAL / "gt"), "TRACKERS_FOLDER": str(EVAL / "trackers"),
                "TRACKERS_TO_EVAL": [r["name"] for r in video_runs], "BENCHMARK": "MOT17",
                "SPLIT_TO_EVAL": "val", "SEQMAP_FILE": str(seqmap), "PRINT_CONFIG": False,
                "OUTPUT_FOLDER": None, "TIME_PROGRESS": False,
            }
            dataset = trackeval.datasets.MotChallenge2DBox(config)
            result, _ = evaluator.evaluate(
                [dataset], [trackeval.metrics.HOTA(), trackeval.metrics.Identity(), trackeval.metrics.CLEAR()])
            for run in video_runs:
                metadata = run["runtime"]
                for seg, window, seq in sequences:
                    values = result["MotChallenge2DBox"][run["name"]][seq]["pedestrian"]
                    segment = f"MOT17-{video}-FRCNN-{seg:03d}"
                    lo, hi = WINDOWS[window]
                    update = [f["ms"] for f in metadata["frames"]
                              if f["seg"] == segment and lo <= f["f"] <= hi]
                    if len(update) != hi - lo + 1:
                        raise ValueError(f"Missing window timings: {run['name']} {segment} {window}")
                    rows.append({
                        "scope": scope, "method": run["method"], "arm": run["arm"],
                        "method_label": run["method_label"], "video": video, "rep": run["rep"],
                        "seg": seg, "window": window,
                        "IDF1": float(values["Identity"]["IDF1"]),
                        "HOTA": float(np.mean(values["HOTA"]["HOTA"])),
                        "AssA": float(np.mean(values["HOTA"]["AssA"])),
                        "MOTA": float(values["CLEAR"]["MOTA"]),
                        "IDSW": int(values["CLEAR"]["IDSW"]),
                        "FP": int(values["CLEAR"]["CLR_FP"]),
                        "FN": int(values["CLEAR"]["CLR_FN"]),
                        "TP": int(values["CLEAR"]["CLR_TP"]),
                        "mean_update_ms": float(np.mean(update)),
                    })
            print("WINDOW_EVAL", scope, video, "cumulative_rows", len(rows), flush=True)
    return rows


def aggregate_windows(segment_rows):
    video_groups = defaultdict(list)
    for row in segment_rows:
        key = (row["scope"], row["method"], row["arm"], row["video"], row["window"])
        video_groups[key].append(row)
    video_rows = []
    for (scope, method, arm, video, window), group in video_groups.items():
        out = {"scope": scope, "method": method, "arm": arm, "method_label": method_label(method),
               "video": video, "window": window}
        reps = SCOPES[scope]["reps"]
        for field in METRICS:
            rep_values = []
            for rep in reps:
                selected = [r for r in group if r["rep"] == rep]
                rep_values.append(float(np.sum([r[field] for r in selected])) if field in COUNTS
                                  else float(np.mean([r[field] for r in selected])))
            out[field] = float(np.mean(rep_values))
        video_rows.append(out)
    macro_groups = defaultdict(list)
    for row in video_rows:
        macro_groups[(row["scope"], row["method"], row["arm"], row["window"])].append(row)
    macro_rows = []
    for (scope, method, arm, window), group in macro_groups.items():
        out = {"scope": scope, "method": method, "arm": arm, "method_label": method_label(method), "window": window}
        for field in METRICS:
            out[field] = float(np.mean([r[field] for r in group]))
        macro_rows.append(out)
    return video_rows, macro_rows


def identity_for_segment(run, seg, objects, tracks):
    rows = []
    for gt_id, observations in objects.items():
        obs = dict(observations)
        pre_votes = []
        for frame in range(1, 6):
            if frame in obs:
                pre_votes.extend(matched_ids(obs[frame], tracks.get(frame, {})))
        if not pre_votes:
            continue
        original_id = Counter(pre_votes).most_common(1)[0][0]
        matches = {frame: matched_ids(box, tracks.get(frame, {})) for frame, box in obs.items()}
        damaged = any(original_id not in matches.get(frame, set())
                      for frame in range(6, 9) if frame in obs)
        if not damaged:
            continue
        postattack_obs = sum(9 <= frame <= 30 for frame in obs)
        eligible = postattack_obs >= 3
        original_matches = {frame: {original_id} & ids for frame, ids in matches.items()}
        different_matches = {frame: set(ids) - {original_id} for frame, ids in matches.items()}
        relaxed = any(original_matches.get(frame) for frame in range(9, 31))
        start = None
        if eligible:
            for candidate in range(9, 29):
                if all(original_id in matches.get(frame, set()) and
                       not different_matches.get(frame, set())
                       for frame in range(candidate, candidate + 3)):
                    start = candidate
                    break
        replacement = False
        if start is None and eligible:
            other_ids = set().union(*different_matches.values()) if different_matches else set()
            for other_id in other_ids:
                for candidate in range(9, 29):
                    if all(other_id in matches.get(frame, set()) and
                           original_id not in matches.get(frame, set())
                           for frame in range(candidate, candidate + 3)):
                        replacement = True
                        break
                if replacement:
                    break
        rows.append({
            "scope": run["scope"], "method": run["method"], "arm": run["arm"],
            "method_label": run["method_label"], "video": run["video"], "rep": run["rep"], "seg": seg,
            "gt_id": gt_id, "original_track_id": original_id, "postattack_gt_obs": postattack_obs,
            "eligible": int(eligible), "relaxed_recovered": int(relaxed),
            "strict_recovered": int(start is not None), "strict_start": start if start is not None else "",
            "recovery_delay": start - 8 if start is not None else "",
            "identity_replacement": int(replacement), "right_censored": int(eligible and start is None),
        })
    return rows


def delay_stats(targets):
    delays = [float(r["recovery_delay"]) for r in targets if r["strict_recovered"]]
    return {
        "median_delay": float(np.median(delays)) if delays else "",
        "iqr_delay": float(np.percentile(delays, 75) - np.percentile(delays, 25)) if delays else "",
        "p90_delay": float(np.percentile(delays, 90)) if delays else "",
    }


def aggregate_identity(target_rows):
    groups = defaultdict(list)
    for row in target_rows:
        groups[(row["scope"], row["method"], row["arm"], row["video"])].append(row)
    video_rows = []
    for scope, protocol in SCOPES.items():
        for method, arm in case_methods(scope):
            for video in protocol["videos"]:
                group = groups.get((scope, method, arm, video), [])
                eligible = [r for r in group if r["eligible"]]
                recovered = [r for r in eligible if r["strict_recovered"]]
                out = {"scope": scope, "method": method, "arm": arm, "method_label": method_label(method),
                       "video": video, "damaged": len(group), "eligible": len(eligible)}
                if eligible:
                    out.update({
                        "relaxed_rate": len([r for r in eligible if r["relaxed_recovered"]]) / len(eligible),
                        "strict_rate": len(recovered) / len(eligible),
                        "unrecovered_rate": len([r for r in eligible if r["right_censored"]]) / len(eligible),
                        "replacement_rate": len([r for r in eligible if r["identity_replacement"]]) / len(eligible)})
                else:
                    out.update({name: "" for name in ("relaxed_rate", "strict_rate", "unrecovered_rate", "replacement_rate")})
                out.update(delay_stats(eligible))
                video_rows.append(out)
    macro_rows = []
    for scope in SCOPES:
        for method, arm in case_methods(scope):
            group = [r for r in video_rows if r["scope"] == scope and r["method"] == method and r["arm"] == arm]
            valid = [r for r in group if r["eligible"] > 0]
            targets = [r for r in target_rows if r["scope"] == scope and r["method"] == method and r["arm"] == arm]
            out = {"scope": scope, "method": method, "arm": arm, "method_label": method_label(method),
                   "damaged": sum(r["damaged"] for r in group), "eligible": sum(r["eligible"] for r in group),
                   "identity_videos": len(valid),
                   "strict_rate_macro": float(np.mean([r["strict_rate"] for r in valid])) if valid else "",
                   "relaxed_rate_macro": float(np.mean([r["relaxed_rate"] for r in valid])) if valid else "",
                   "unrecovered_rate_macro": float(np.mean([r["unrecovered_rate"] for r in valid])) if valid else "",
                   "replacement_rate_macro": float(np.mean([r["replacement_rate"] for r in valid])) if valid else ""}
            eligible_targets = [r for r in targets if r["eligible"]]
            n_eligible = len(eligible_targets)
            out.update({
                "strict_rate_pooled": sum(r["strict_recovered"] for r in eligible_targets) / n_eligible if n_eligible else "",
                "relaxed_rate_pooled": sum(r["relaxed_recovered"] for r in eligible_targets) / n_eligible if n_eligible else "",
                "unrecovered_rate_pooled": sum(r["right_censored"] for r in eligible_targets) / n_eligible if n_eligible else "",
                "replacement_rate_pooled": sum(r["identity_replacement"] for r in eligible_targets) / n_eligible if n_eligible else ""})
            stats = delay_stats(targets)
            out.update({"median_delay_pooled": stats["median_delay"],
                        "iqr_delay_pooled": stats["iqr_delay"], "p90_delay_pooled": stats["p90_delay"]})
            macro_rows.append(out)
    return video_rows, macro_rows


def fp_for_segment(run, seg, valid_by_frame, tracks):
    by_id = defaultdict(list)
    for frame, frame_tracks in tracks.items():
        for track_id, box in frame_tracks.items():
            by_id[track_id].append((frame, box))
    instances = []
    for track_id, observations in by_id.items():
        observations.sort()
        first_frame = observations[0][0]
        if not 6 <= first_frame <= 8:
            continue
        if any(any(iou(box, gt_box) >= 0.5 for gt_box in valid_by_frame.get(frame, []))
               for frame, box in observations):
            continue
        instances.append({
            "scope": run["scope"], "method": run["method"], "arm": run["arm"],
            "method_label": run["method_label"], "video": run["video"], "rep": run["rep"], "seg": seg,
            "track_id": track_id, "observed_lifetime_frames": len(observations),
            "first_frame": first_frame, "last_frame": observations[-1][0],
        })
    return instances


def median_lifetime(instances):
    values = [r["observed_lifetime_frames"] for r in instances]
    return float(np.median(values)) if values else ""


def aggregate_fp(segment_rows, instances):
    summary = []
    for row in segment_rows:
        selected = [r for r in instances if r["scope"] == row["scope"] and r["method"] == row["method"] and
                    r["arm"] == row["arm"] and r["video"] == row["video"] and
                    r["rep"] == row["rep"] and r["seg"] == row["seg"]]
        summary.append({**row, "level": "segment", "aggregation": "observed_total", "segments": 1,
                        "n_instances": len(selected), "median_observed_lifetime": median_lifetime(selected)})
    video_rows = []
    video_groups = defaultdict(list)
    for row in segment_rows:
        video_groups[(row["scope"], row["method"], row["arm"], row["video"])].append(row)
    for (scope, method, arm, video), group in video_groups.items():
        rep_counts = [sum(r["fp_track_count"] for r in group if r["rep"] == rep) for rep in SCOPES[scope]["reps"]]
        selected = [r for r in instances if r["scope"] == scope and r["method"] == method and
                    r["arm"] == arm and r["video"] == video]
        out = {"scope": scope, "method": method, "arm": arm, "method_label": method_label(method),
               "video": video, "rep": "", "seg": "", "fp_track_count": float(np.mean(rep_counts)),
               "level": "video", "aggregation": "sum_segments_per_rep_then_mean_reps",
               "segments": len(group), "n_instances": len(selected),
               "median_observed_lifetime": median_lifetime(selected)}
        video_rows.append(out)
        summary.append(out)
    macro_groups = defaultdict(list)
    for row in video_rows:
        macro_groups[(row["scope"], row["method"], row["arm"])].append(row)
    for (scope, method, arm), group in macro_groups.items():
        selected = [r for r in instances if r["scope"] == scope and r["method"] == method and r["arm"] == arm]
        summary.append({"scope": scope, "method": method, "arm": arm, "method_label": method_label(method),
                        "video": "", "rep": "", "seg": "",
                        "fp_track_count": float(np.mean([r["fp_track_count"] for r in group])),
                        "level": "macro", "aggregation": "equal_video_macro",
                        "segments": sum(r["segments"] for r in group), "n_instances": len(selected),
                        "median_observed_lifetime": median_lifetime(selected)})
    return summary


def identity_and_fp(runs):
    identity_rows = []
    fp_instances = []
    fp_segments = []
    for run in runs:
        n_segments = SCOPES[run["scope"]]["videos"][run["video"]]
        for seg in range(n_segments):
            _, objects, valid_by_frame = load_gt(run["video"], seg)
            path = run["root"] / run["name"] / f"MOT17-{run['video']}-FRCNN-{seg:03d}.txt"
            tracks = load_tracks(path)
            identity_rows.extend(identity_for_segment(run, seg, objects, tracks))
            if run["arm"] in ("A", "C"):
                found = fp_for_segment(run, seg, valid_by_frame, tracks)
                fp_instances.extend(found)
                fp_segments.append({"scope": run["scope"], "method": run["method"], "arm": run["arm"],
                                    "method_label": run["method_label"], "video": run["video"],
                                    "rep": run["rep"], "seg": seg, "fp_track_count": len(found)})
        print("IDENTITY_FP", run["name"], flush=True)
    fp_summary = aggregate_fp(fp_segments, fp_instances)
    return identity_rows, fp_instances, fp_summary


def intervention_rows(runs):
    rows = []
    for run in runs:
        if run["method"] != "ocsort":
            continue
        frames = run["runtime"]["frames"]
        durations = [float(frame["ms"]) for frame in frames]
        row = {"scope": run["scope"], "method": run["method"], "arm": run["arm"],
               "method_label": run["method_label"], "video": run["video"], "rep": run["rep"],
               "run": run["name"], "segments": len(run["runtime"].get("segments", [])),
               "mean_update_ms": float(np.mean(durations)),
               "median_update_ms": float(np.median(durations))}
        for field in CG_FIELDS:
            row[field] = ""
        rows.append(row)
    return rows


def make_recovery_summary(window_macro, identity_macro):
    rows = []
    for scope in SCOPES:
        for method, arm in case_methods(scope):
            attack = next(r for r in window_macro if r["scope"] == scope and r["method"] == method and
                          r["arm"] == arm and r["window"] == "attack")
            recovery = next(r for r in window_macro if r["scope"] == scope and r["method"] == method and
                            r["arm"] == arm and r["window"] == "recovery")
            identity = next(r for r in identity_macro if r["scope"] == scope and
                            r["method"] == method and r["arm"] == arm)
            row = {"scope": scope, "method": method, "arm": arm, "method_label": method_label(method)}
            for prefix, source in (("attack", attack), ("recovery", recovery)):
                for field in METRICS:
                    row[f"{prefix}_{field}"] = source[field]
            for field in ("damaged", "eligible", "identity_videos", "strict_rate_macro", "relaxed_rate_macro",
                          "unrecovered_rate_macro", "replacement_rate_macro", "strict_rate_pooled",
                          "relaxed_rate_pooled", "unrecovered_rate_pooled", "replacement_rate_pooled", "median_delay_pooled",
                          "iqr_delay_pooled", "p90_delay_pooled"):
                row[field] = identity[field]
            rows.append(row)
    return rows


def evaluate():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)
    runs = build_runs()
    preflight(runs)
    rows = evaluate_windows(runs)
    video_rows, macro_rows = aggregate_windows(rows)
    targets, fp_instances, fp_summary = identity_and_fp(runs)
    identity_video, identity_macro = aggregate_identity(targets)
    interventions = intervention_rows(runs)
    recovery = make_recovery_summary(macro_rows, identity_macro)
    assert len(runs) == 187, len(runs)
    assert len(rows) == 1012, len(rows)
    assert len(video_rows) == 154, len(video_rows)
    assert len(macro_rows) == 44, len(macro_rows)
    assert len(identity_video) == 77, len(identity_video)
    assert len(identity_macro) == 22, len(identity_macro)
    assert len(interventions) == 68, len(interventions)
    assert len(recovery) == 22, len(recovery)
    keys = [(r["scope"], r["method"], r["arm"], r["video"], r["rep"], r["seg"], r["window"])
            for r in rows]
    assert len(keys) == len(set(keys))
    for row in identity_macro:
        if row["eligible"]:
            assert abs(row["strict_rate_pooled"] + row["unrecovered_rate_pooled"] - 1) < 1e-9
    for filename, data in (
            ("window_segment_metrics.csv", rows), ("window_video_summary.csv", video_rows),
            ("window_macro_summary.csv", macro_rows), ("identity_target_recovery.csv", targets),
            ("identity_video_summary.csv", identity_video),
            ("identity_macro_summary.csv", identity_macro),
            ("intervention_summary.csv", interventions), ("fp_track_instances.csv", fp_instances),
            ("fp_track_summary.csv", fp_summary), ("recovery_summary.csv", recovery)):
        write_csv(filename, data)
    print("E24_EVAL_DONE", json.dumps({"runs": len(runs), "window_rows": len(rows),
          "identity_targets": len(targets), "fp_instances": len(fp_instances),
          "interventions": len(interventions), "recovery_rows": len(recovery)}, indent=2), flush=True)


def main():
    global ANALYSIS, EVAL, RUNS, GT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS)
    parser.add_argument("--gt-root", type=Path, default=GT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS)
    parser.add_argument("--work-dir", type=Path, default=EVAL)
    args = parser.parse_args()
    RUNS = args.runs_root.resolve()
    GT_ROOT = args.gt_root.resolve()
    ANALYSIS = args.output_dir.resolve()
    EVAL = args.work_dir.resolve()
    evaluate()


if __name__ == "__main__":
    main()
