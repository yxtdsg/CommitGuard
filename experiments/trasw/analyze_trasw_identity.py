import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from evaluate_trasw import ARMS, REPS, WORK, completed_run, load_segments

ANALYSIS = Path(os.environ.get("COMMITGUARD_TRASW_ANALYSIS", WORK / "analysis")).resolve()
GT_ROOT = Path(os.environ.get("COMMITGUARD_GT_ROOT", Path(__file__).resolve().parents[2] / "external_inputs" / "gt" / "mot17")).resolve()
ATTACK_START = 11


def write_csv(name, rows):
    with open(ANALYSIS / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    return inter / max(aw * ah + bw * bh - inter, 1e-12)


def load_gt(seg):
    objects = defaultdict(list)
    source = GT_ROOT / f"{seg['video']}-FRCNN" / "gt" / "gt.txt"
    with open(source, encoding="utf-8") as f:
        for line in f:
            row = line.rstrip("\n").split(",")
            frame = int(row[0])
            if not (seg["val_half_frame_start"] <= frame <= seg["val_half_frame_end"]):
                continue
            if int(row[6]) != 1 or int(row[7]) != 1:
                continue
            local = frame - seg["val_half_frame_start"] + 1
            objects[int(row[1])].append((local, list(map(float, row[2:6]))))
    return objects


def load_tracks(path):
    tracks = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = line.rstrip("\n").split(",")
            tracks[int(row[0])][int(row[1])] = list(map(float, row[2:6]))
    return tracks


def matched_ids(box, predictions):
    return {track_id for track_id, pred in predictions.items() if iou(box, pred) >= 0.5}


def persistent_switch_start(matches, original_id):
    other_ids = set()
    for frame in range(ATTACK_START, 31):
        other_ids.update(matches.get(frame, set()) - {original_id})
    starts = []
    for other_id in other_ids:
        for start in range(ATTACK_START, 29):
            if all(other_id in matches.get(frame, set()) and
                   original_id not in matches.get(frame, set())
                   for frame in range(start, start + 3)):
                starts.append(start)
                break
    return min(starts) if starts else None


def strict_original_start(matches, original_id):
    for start in range(ATTACK_START, 29):
        if all(original_id in matches.get(frame, set()) and
               not (matches.get(frame, set()) - {original_id})
               for frame in range(start, start + 3)):
            return start
    return None


def identity_rows_for_run(seg, arm, rep):
    _, result, _ = completed_run(seg["seg_name"], arm, rep)
    tracks = load_tracks(result)
    rows = []
    for gt_id, observations in load_gt(seg).items():
        obs = dict(observations)
        matches = {frame: matched_ids(box, tracks.get(frame, {}))
                   for frame, box in observations}
        pre_votes = []
        for frame in range(1, ATTACK_START):
            pre_votes.extend(matches.get(frame, set()))
        if not pre_votes:
            continue
        original_id, original_votes = Counter(pre_votes).most_common(1)[0]
        eligible = original_votes >= 3 and sum(
            ATTACK_START <= frame <= 30 for frame in obs) >= 3
        if not eligible:
            continue
        different = [frame for frame in range(ATTACK_START, 31)
                     if matches.get(frame, set()) - {original_id}]
        switch_start = persistent_switch_start(matches, original_id)
        original_start = strict_original_start(matches, original_id)
        rows.append({"seg_name": seg["seg_name"], "video": seg["video"].removeprefix("MOT17-"),
                     "rep": int(rep[1:]), "arm": arm, "gt_id": gt_id,
                     "original_track_id": original_id, "pre_votes": original_votes,
                     "any_different_id": int(bool(different)),
                     "first_different_id_frame": min(different) if different else "",
                     "persistent_switch": int(switch_start is not None),
                     "persistent_switch_frame": switch_start if switch_start is not None else "",
                     "ever_original_post_attack": int(any(
                         original_id in matches.get(frame, set()) for frame in range(ATTACK_START, 31))),
                     "strict_original_preserved": int(original_start is not None),
                     "no_post_attack_match": int(not any(
                         matches.get(frame, set()) for frame in range(ATTACK_START, 31)))})
    return rows


def audit_timing(segments):
    rows = []
    clean_alarm_frames = 0
    clean_aux_frames = 0
    for seg in segments:
        with open(WORK / "tensors_raw" / seg["seg_name"] / "composition.json", encoding="utf-8") as f:
            composition = json.load(f)
        changed = [r["frame"] for r in composition if r["changed"]]
        first_changed = min(changed) if changed else None
        for rep in REPS:
            _, _, validation = completed_run(seg["seg_name"], "attack_cg", rep)
            with open(validation["precommit_log"], encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            l1 = [r for r in events if r["t"] == "l1_state"]
            alarms = [r["f"] for r in l1 if r.get("state_out") == "ALARM"]
            attack_alarms = [f for f in alarms if f >= ATTACK_START]
            first_alarm = min(attack_alarms) if attack_alarms else None
            rows.append({"seg_name": seg["seg_name"], "rep": int(rep[1:]),
                         "first_changed_frame": first_changed,
                         "changed_frames": len(changed), "alarm_frames": len(alarms),
                         "attack_window_alarm_frames": len(attack_alarms),
                         "first_attack_alarm": first_alarm if first_alarm is not None else "",
                         "detected": int(first_alarm is not None),
                         "alarm_before_attack_start": int(any(f < ATTACK_START for f in alarms)),
                         "alarm_before_first_changed": int(first_alarm is not None and first_alarm < first_changed),
                         "alarm_same_as_first_changed": int(first_alarm == first_changed),
                         "delay_from_first_changed": first_alarm - first_changed if first_alarm is not None else "",
                         "delay_from_protocol_start": first_alarm - ATTACK_START if first_alarm is not None else ""})
            _, _, clean_validation = completed_run(seg["seg_name"], "clean_cg", rep)
            clean_alarm_frames += clean_validation.get("n_alarm_frames", 0)
            clean_aux_frames += clean_validation.get("n_aux_review_frames", 0)
    write_csv("audit_timing.csv", rows)
    detected = [r for r in rows if r["detected"]]
    per_segment = defaultdict(list)
    for row in rows:
        per_segment[row["seg_name"]].append(row)
    summary = {"attack_cg_runs": len(rows), "runs_with_attack_window_alarm": len(detected),
               "run_detection_rate": len(detected) / len(rows),
               "segments_with_alarm_in_any_rep": sum(any(r["detected"] for r in members)
                                                   for members in per_segment.values()),
               "segments_with_alarm_in_all_three_reps": sum(all(r["detected"] for r in members)
                                                         for members in per_segment.values()),
               "alarms_before_protocol_attack_start": sum(r["alarm_before_attack_start"] for r in rows),
               "alarms_before_first_changed_frame": sum(r["alarm_before_first_changed"] for r in rows),
               "alarms_same_as_first_changed_frame": sum(r["alarm_same_as_first_changed"] for r in rows),
               "median_delay_from_first_changed": float(np.median([
                   r["delay_from_first_changed"] for r in detected])) if detected else None,
               "delay_range_from_first_changed": [min(r["delay_from_first_changed"] for r in detected),
                                                   max(r["delay_from_first_changed"] for r in detected)] if detected else None,
               "clean_cg_runs": 30, "clean_alarm_frames": clean_alarm_frames,
               "clean_aux_review_frames": clean_aux_frames,
               "interpretation": "Run-level detection means at least one audit ALARM at frame 11 or later. It is reactive evidence from tracker births, not prediction or direct TraSw classification."}
    with open(ANALYSIS / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def aggregate_identities(target_rows):
    groups = defaultdict(list)
    for row in target_rows:
        groups[(row["arm"], row["video"])].append(row)
    video_rows = []
    fields = ("any_different_id", "persistent_switch", "ever_original_post_attack",
              "strict_original_preserved", "no_post_attack_match")
    for (arm, video), rows in sorted(groups.items()):
        out = {"arm": arm, "video": video, "eligible_target_runs": len(rows)}
        for field in fields:
            out[field + "_rate"] = float(np.mean([r[field] for r in rows]))
        video_rows.append(out)
    write_csv("identity_video_summary.csv", video_rows)
    macro = []
    for arm in ARMS:
        rows = [r for r in video_rows if r["arm"] == arm]
        out = {"arm": arm, "videos": len(rows),
               "eligible_target_runs_pooled": sum(r["eligible_target_runs"] for r in rows)}
        for field in fields:
            out[field + "_rate_macro"] = float(np.mean([r[field + "_rate"] for r in rows]))
            out[field + "_rate_pooled"] = float(np.mean([
                r[field] for r in target_rows if r["arm"] == arm]))
        macro.append(out)
    write_csv("identity_macro_summary.csv", macro)
    return video_rows, macro


def paired_identity(target_rows):
    index = {(r["seg_name"], r["rep"], r["gt_id"], r["arm"]): r for r in target_rows}
    rows = []
    for key, cg in index.items():
        if key[3] != "attack_cg":
            continue
        base_key = (*key[:3], "attack_c0")
        if base_key not in index:
            continue
        c0 = index[base_key]
        rows.append({"seg_name": cg["seg_name"], "video": cg["video"], "rep": cg["rep"],
                     "gt_id": cg["gt_id"],
                     "c0_persistent_switch": c0["persistent_switch"],
                     "cg_persistent_switch": cg["persistent_switch"],
                     "c0_strict_original_preserved": c0["strict_original_preserved"],
                     "cg_strict_original_preserved": cg["strict_original_preserved"]})
    write_csv("identity_paired_targets.csv", rows)
    summaries = []
    for video in ("02", "04", "05", "all"):
        members = rows if video == "all" else [r for r in rows if r["video"] == video]
        if not members:
            continue
        summaries.append({"video": video, "common_eligible_target_runs": len(members),
                          "c0_persistent_switch_rate": float(np.mean([r["c0_persistent_switch"] for r in members])),
                          "cg_persistent_switch_rate": float(np.mean([r["cg_persistent_switch"] for r in members])),
                          "delta_persistent_switch_pp": 100 * float(np.mean([
                              r["cg_persistent_switch"] - r["c0_persistent_switch"] for r in members])),
                          "c0_strict_original_rate": float(np.mean([r["c0_strict_original_preserved"] for r in members])),
                          "cg_strict_original_rate": float(np.mean([r["cg_strict_original_preserved"] for r in members])),
                          "delta_strict_original_pp": 100 * float(np.mean([
                              r["cg_strict_original_preserved"] - r["c0_strict_original_preserved"] for r in members]))})
    write_csv("identity_paired_summary.csv", summaries)
    return summaries


def main():
    segments = load_segments()
    audit = audit_timing(segments)
    targets = []
    for seg in segments:
        for arm in ARMS:
            for rep in REPS:
                targets.extend(identity_rows_for_run(seg, arm, rep))
        print("IDENTITY_AUDITED", seg["seg_name"], flush=True)
    write_csv("identity_targets.csv", targets)
    aggregate_identities(targets)
    paired = paired_identity(targets)
    print("DONE", {"audit": audit, "identity_target_runs": len(targets),
                   "paired": paired}, flush=True)


if __name__ == "__main__":
    main()
