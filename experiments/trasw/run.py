# -*- coding: utf-8 -*-
"""Strictly serial E29 full-dev-matrix executor with validated resume."""
import argparse
import json
import os
import subprocess
import time

from trasw_common import (ARMS, CODE, E29, FIRST_SEG_NAME, FNF, PROTOCOL, PY, RAW, ROOT,
                          REPLAY, REPS, TRASA, WORK, attack_validation_path,
                          aux_path, compose_segment, file_sha256, ledger,
                          load_and_verify_freeze, load_protocol, now,
                          prepare_annotation, selected_segments, validate_aux,
                          validate_composition)

FULL_SCOPE = "FULL_MATRIX_10_SEGMENTS_R1_R3"


def run_logged(cmd, cwd, env, log_path):
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
    return process.returncode, time.time() - start


def event_base():
    return {"protocol": str(PROTOCOL), "protocol_sha256": file_sha256(PROTOCOL)}


def source_hashes():
    return {str(path): file_sha256(path) for path in (
        CODE / "run.py", CODE / "trasw_common.py", CODE / "generate_aux.py",
        CODE / "attack_entry.py", CODE / "attack_driver.py", CODE / "tensor_ops.py",
        CODE / "victim_entry.py", CODE / "tracker.py",
        ROOT / "src" / "commitguard" / "_vendor" / "byte_tracker.py",
        ROOT / "src" / "commitguard" / "tracker.py")}


def read_attack_validation(seg, allow_legacy=True):
    validation = validate_composition(seg)
    path = attack_validation_path(seg["seg_name"])
    if not path.exists() and allow_legacy and seg["seg_name"] == FIRST_SEG_NAME:
        path = WORK / "attack_validation.json"
    with open(path, encoding="utf-8") as f:
        recorded = json.load(f)
    if recorded != validation:
        raise RuntimeError("攻击验证与当前张量不符: " + seg["seg_name"])
    return validation


def attack_one(seg, frozen):
    seg_name = seg["seg_name"]
    validation_path = attack_validation_path(seg_name)
    if (RAW / seg_name / "composition.json").exists():
        validation = read_attack_validation(seg)
        if not validation_path.exists():
            validation_path.parent.mkdir(parents=True, exist_ok=True)
            with open(validation_path, "x", encoding="utf-8") as f:
                json.dump(validation, f, indent=2)
        ledger({**event_base(), "kind": "attack", "seg": seg_name,
                "status": "skipped_complete", "validation": validation})
        return validation
    if (RAW / seg_name).exists() or (REPLAY / "clean" / seg_name).exists() or \
            (REPLAY / "attack" / seg_name).exists():
        raise RuntimeError("E29 攻击/合成现场不完整，保留现场并拒绝自动重试: " + seg_name)
    run_root = WORK / "attack_runs" / seg_name
    if run_root.exists():
        raise RuntimeError("E29 攻击运行目录已存在但无完整合成，拒绝覆盖: " + str(run_root))
    run_root.mkdir(parents=True)
    values = {"video_id": str(seg["video_id"]),
              "attack_output": str(run_root / "tracks"),
              "attack_stats": str(run_root / "stats")}
    official = [x.format(**values) for x in frozen["attack_generation"]["command_template"]]
    cmd = [PY, str(CODE / "attack_entry.py")] + official[2:]
    env = os.environ.copy()
    env.update({"E14_EXPORT_ROOT": str(RAW), "E14_EXPORT_SEG": seg_name,
                "E14_DATA_NUM_WORKERS": "0", "E14_PIN_MEMORY": "0",
                "PRECOMMIT_LOG_DIR": str(run_root / "precommit"),
                "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(TRASA),
                "PYTHONDONTWRITEBYTECODE": "1",
                "E29_TRASA_SEG_ANN": str(run_root / "filtered_annotation.json")})
    event = {**event_base(), "kind": "attack", "seg": seg_name,
             "video_id": seg["video_id"], "command": cmd, "cwd": str(TRASA),
             "official_driver": official[1],
             "environment": {k: env[k] for k in (
                 "E14_EXPORT_ROOT", "E14_EXPORT_SEG", "E14_DATA_NUM_WORKERS",
                 "E14_PIN_MEMORY", "PRECOMMIT_LOG_DIR", "CUDA_VISIBLE_DEVICES",
                 "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "E29_TRASA_SEG_ANN")}}
    ledger({**event, "status": "started"})
    rc, elapsed = run_logged(cmd, TRASA, env, run_root / "stdout.log")
    ledger({**event, "status": "process_finished" if rc == 0 else "failed",
            "returncode": rc, "elapsed_s": elapsed})
    if rc:
        raise RuntimeError("E29 官方 TraSw 工程失败，保留现场且不自动重试: " + seg_name)
    try:
        validation = compose_segment(seg)
    except Exception as exc:
        ledger({**event, "kind": "composition", "status": "failed", "error": str(exc)})
        raise
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    with open(validation_path, "x", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)
    ledger({**event, "kind": "composition", "status": "finished",
            "validation": validation})
    return validation


def attack_all(segments, frozen):
    results = {}
    for index, seg in enumerate(segments, 1):
        print("ATTACK_STAGE %d/10 %s" % (index, seg["seg_name"]), flush=True)
        results[seg["seg_name"]] = attack_one(seg, frozen)
        print("ATTACK_OK %s changed_frames=%d" %
              (seg["seg_name"], results[seg["seg_name"]]["changed_frames"]), flush=True)
    return results


def aux_one(seg, condition):
    seg_name = seg["seg_name"]
    try:
        validation = validate_aux(condition, seg_name)
        ledger({**event_base(), "kind": "aux", "condition": condition,
                "seg": seg_name, "status": "skipped_complete",
                "validation": validation})
        return validation
    except (FileNotFoundError, RuntimeError):
        path = aux_path(condition, seg_name)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                partial = [json.loads(line) for line in f if line.strip()]
            if len(partial) > 25:
                raise RuntimeError("E29 aux 记录数异常，拒绝续跑: " + str(path))
    run_root = WORK / "aux_runs" / seg_name / condition
    run_root.mkdir(parents=True, exist_ok=True)
    aux_path(condition, seg_name).parent.mkdir(parents=True, exist_ok=True)
    log_path = run_root / ("stdout_%d.log" % (len(list(run_root.glob("stdout_*.log"))) + 1))
    cmd = [PY, str(CODE / "generate_aux.py"), "--condition", condition,
           "--seg", seg_name]
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "0", "PYTHONDONTWRITEBYTECODE": "1"})
    event = {**event_base(), "kind": "aux", "condition": condition,
             "seg": seg_name, "command": cmd, "cwd": str(CODE)}
    ledger({**event, "status": "started", "log": str(log_path)})
    rc, elapsed = run_logged(cmd, CODE, env, log_path)
    validation = None
    error = None
    if rc == 0:
        try:
            validation = validate_aux(condition, seg_name)
        except Exception as exc:
            error = str(exc)
    ledger({**event, "status": "finished" if rc == 0 and validation else "failed",
            "returncode": rc, "elapsed_s": elapsed,
            "validation": validation, "error": error})
    if rc or validation is None:
        raise RuntimeError("E29 aux 工程/完整性失败，保留现场: %s %s %s" %
                           (seg_name, condition, error or rc))
    return validation


def aux_all(segments):
    results = {}
    for seg in segments:
        seg_name = seg["seg_name"]
        validate_composition(seg)
        results[seg_name] = {}
        for condition in ("clean", "attack"):
            print("AUX_STAGE %s %s" % (seg_name, condition), flush=True)
            results[seg_name][condition] = aux_one(seg, condition)
            print("AUX_OK %s %s detections=%d" %
                  (seg_name, condition, results[seg_name][condition]["detections"]), flush=True)
    return results


def victim_command(seg, arm, rep, cfg):
    spec = cfg["victim"]["arms"][arm]
    run_name = "e29_%s_%s_%s" % (arm, seg["seg_name"], rep)
    run_root = WORK / "victim_runs" / run_name
    values = {"seg_name": seg["seg_name"],
              "tensor_root": str(REPLAY / spec["input"]),
              "annotation": str(WORK / "annotations" / (seg["seg_name"] + ".json")),
              "run_name": run_name, "results_root": str(run_root / "results"),
              "l1_mode": spec["l1_mode"], "l1_intervene": str(spec["l1_intervene"])}
    official = [x.format(**values) for x in cfg["victim"]["official_command_template"]]
    cmd = [PY, str(CODE / "victim_entry.py"), "--arm", arm,
           "--aux-file", str(aux_path(spec["input"], seg["seg_name"])), "--"] + official[2:]
    env = os.environ.copy()
    env.update({"E14_DATA_NUM_WORKERS": "0", "E14_PIN_MEMORY": "0",
                "PRECOMMIT_LOG_DIR": str(run_root / "precommit"),
                "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(FNF),
                "PYTHONDONTWRITEBYTECODE": "1"})
    return cmd, env, run_name, run_root


def validate_victim(seg, arm, rep, run_root, cfg):
    results_root = run_root / "results"
    experiments = os.listdir(results_root)
    expected_name = "e29_%s_%s_%s" % (arm, seg["seg_name"], rep)
    if experiments != [expected_name]:
        raise RuntimeError("E29 结果目录不唯一或名称错误: " + str(experiments))
    result = results_root / expected_name / (seg["seg_name"] + ".txt")
    if not result.is_file():
        raise RuntimeError("E29 缺少轨迹文件，不能当作空轨迹")
    log_path = run_root / "precommit" / ("clean__" + seg["seg_name"] + ".jsonl")
    with open(log_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    def events(kind):
        return [r for r in rows if r["t"] == kind]

    forwards = events("model_forward")
    inits = events("tracker_init")
    frames = events("frame")
    replays = events("strict_replay")
    configs = events("e14_replay_config")
    if len(configs) != 1 or configs[0].get("seg") != seg["seg_name"] or \
            configs[0].get("data_num_workers") != 0 or \
            configs[0].get("pin_memory") is not False or \
            configs[0].get("attack_enabled") != 0 or \
            configs[0].get("det_input_jsonl") or configs[0].get("resume_results"):
        raise RuntimeError("E29 replay 配置台账不完整")
    if len(replays) != 30 or [r["f"] for r in replays] != list(range(1, 31)):
        raise RuntimeError("E29 严格 replay 次数/顺序失败")
    arm_input = cfg["victim"]["arms"][arm]["input"]
    with open(REPLAY / arm_input / seg["seg_name"] / "frame_hashes.json", encoding="utf-8") as f:
        manifest = json.load(f)
    if any(r["seg"] != seg["seg_name"] or r["replay_seg"] != seg["seg_name"] or
           r["data_sha256"] != manifest["frames"][str(r["f"])]["data_sha256"]
           for r in replays):
        raise RuntimeError("E29 replay 段名或逐帧数据哈希不符")
    if len(forwards) != 30 or [r["f"] for r in forwards] != list(range(1, 31)):
        raise RuntimeError("E29 真实模型 forward 完整性失败")
    if len(inits) != 1 or inits[0]["f"] != 1:
        raise RuntimeError("E29 单受害者冷启动失败")
    if len(frames) != 30 or [r["f"] for r in frames] != list(range(1, 31)):
        raise RuntimeError("E29 帧日志完整性失败")
    l1_states = events("l1_state")
    cg_states = events("e29_cg_state")
    aux_reviews = events("aux_review")
    aux_verified = events("e29_aux_verified")
    tracker_configs = events("e29_tracker_config")
    if arm.endswith("_cg"):
        if len(l1_states) != 30 or len(cg_states) != 30 or \
                [r["f"] for r in cg_states] != list(range(1, 31)):
            raise RuntimeError("E29 CG/L1 状态日志不完整")
        if len(tracker_configs) != 1:
            raise RuntimeError("E29 CG tracker 配置日志不唯一")
        expected_aux = aux_path(arm_input, seg["seg_name"])
        if tracker_configs[0].get("aux_candidate_sha256") != file_sha256(expected_aux):
            raise RuntimeError("E29 CG 实际 aux 文件哈希改变")
        for r in aux_verified:
            if not 6 <= r["f"] <= 30 or r["seg"] != seg["seg_name"] or \
                    r["replay_seg"] != seg["seg_name"] or \
                    r["input_data_sha256"] != manifest["frames"][str(r["f"])]["data_sha256"]:
                raise RuntimeError("E29 aux 使用事件验证失败")
        if any(not any(v["f"] == r["f"] for v in aux_verified) for r in aux_reviews):
            raise RuntimeError("E29 aux review 缺少同帧哈希验证事件")
        n_interventions = sum(bool(r["alarm"]) for r in cg_states)
        n_window_frames = sum(bool(r["window_active"]) for r in cg_states)
        final_stats = cg_states[-1]
    else:
        if l1_states or cg_states or aux_reviews or aux_verified or tracker_configs:
            raise RuntimeError("E29 C0 出现意外 CG/aux 日志")
        n_interventions = 0
        n_window_frames = 0
        final_stats = {}
    trajectory_rows = 0
    track_ids = set()
    with open(result, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(",")
            if len(parts) != 10 or not 1 <= int(parts[0]) <= 30:
                raise RuntimeError("E29 轨迹行格式/帧号错误")
            trajectory_rows += 1
            track_ids.add(int(parts[1]))
    return {"arm": arm, "rep": rep, "seg": seg["seg_name"],
            "result": str(result), "result_sha256": file_sha256(result),
            "precommit_log": str(log_path), "precommit_sha256": file_sha256(log_path),
            "n_model_forward": len(forwards), "n_tracker_init": len(inits),
            "n_frames": len(frames), "n_strict_replay": len(replays),
            "trajectory_rows": trajectory_rows, "unique_track_ids": len(track_ids),
            "n_alarm_frames": sum(r.get("state_out") == "ALARM" for r in l1_states),
            "n_intervention_frames": n_interventions,
            "n_window_frames": n_window_frames,
            "n_aux_review_frames": len(aux_reviews),
            "n_aux_verification_events": len(aux_verified),
            "aux_filter_removed": sum(int(r.get("removed", 0)) for r in aux_reviews),
            "aux_fn_added": sum(int(r.get("added", 0)) for r in aux_reviews),
            "aux_birth_rejected": int(final_stats.get("aux_birth_rejected", 0)),
            "parked_candidate_events": sum(int(r.get("parked", 0)) for r in cg_states)}


def read_validated_completion(seg, arm, rep, run_root, cfg):
    completion_path = run_root / "completion.json"
    with open(completion_path, encoding="utf-8") as f:
        completed = json.load(f)
    validation = validate_victim(seg, arm, rep, run_root, cfg)
    if completed.get("status") != "COMPLETE_VALIDATED" or \
            completed.get("validation") != validation:
        raise RuntimeError("E29 completion 与当前结果/日志不一致: " + run_root.name)
    legacy = seg["seg_name"] == FIRST_SEG_NAME and rep == "r1" and \
        completed.get("scope") == "FIRST_SEGMENT_R1_ONLY"
    if not legacy:
        if completed.get("scope") != FULL_SCOPE or \
                completed.get("protocol_sha256") != file_sha256(PROTOCOL) or \
                completed.get("source_sha256") != source_hashes():
            raise RuntimeError("E29 全矩阵 completion 的代码/协议哈希不匹配: " + run_root.name)
    return validation


def victim_one(seg, arm, rep, cfg):
    cmd, env, run_name, run_root = victim_command(seg, arm, rep, cfg)
    if run_root.exists():
        if not (run_root / "completion.json").exists():
            raise RuntimeError("E29 不完整 victim 现场拒绝覆盖/重跑: " + str(run_root))
        validation = read_validated_completion(seg, arm, rep, run_root, cfg)
        ledger({**event_base(), "kind": "victim", "seg": seg["seg_name"],
                "arm": arm, "rep": rep, "run_name": run_name,
                "status": "skipped_complete", "validation": validation})
        return validation
    run_root.mkdir(parents=True)
    event = {**event_base(), "kind": "victim", "seg": seg["seg_name"],
             "arm": arm, "rep": rep, "run_name": run_name, "command": cmd,
             "cwd": str(FNF),
             "environment": {k: env[k] for k in (
                 "E14_DATA_NUM_WORKERS", "E14_PIN_MEMORY", "PRECOMMIT_LOG_DIR",
                 "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE")}}
    ledger({**event, "status": "started"})
    rc, elapsed = run_logged(cmd, FNF, env, run_root / "stdout.log")
    validation = None
    error = None
    if rc == 0:
        try:
            validation = validate_victim(seg, arm, rep, run_root, cfg)
        except Exception as exc:
            error = str(exc)
    ledger({**event, "status": "finished" if rc == 0 and validation else "failed",
            "returncode": rc, "elapsed_s": elapsed,
            "validation": validation, "error": error})
    if rc or validation is None:
        raise RuntimeError("E29 victim 工程/完整性失败，保留现场且不自动重试: %s %s" %
                           (run_root, error or rc))
    with open(run_root / "completion.json", "x", encoding="utf-8") as f:
        json.dump({"time_utc": now(), "status": "COMPLETE_VALIDATED",
                   "scope": FULL_SCOPE, "command": cmd,
                   "protocol_sha256": file_sha256(PROTOCOL),
                   "source_sha256": source_hashes(), "validation": validation}, f, indent=2)
    return validation


def victims_all(segments, cfg, selected_reps):
    for seg in segments:
        read_attack_validation(seg)
        for condition in ("clean", "attack"):
            validate_aux(condition, seg["seg_name"])
    total = len(selected_reps) * len(segments) * len(ARMS)
    index = 0
    for rep in selected_reps:
        for seg in segments:
            for arm in ARMS:
                index += 1
                run_name = "e29_%s_%s_%s" % (arm, seg["seg_name"], rep)
                print("VICTIM_STAGE %d/%d %s" % (index, total, run_name), flush=True)
                validation = victim_one(seg, arm, rep, cfg)
                print("VICTIM_OK %s rows=%d alarms=%d" %
                      (run_name, validation["trajectory_rows"],
                       validation["n_alarm_frames"]), flush=True)


def summarize(segments, cfg):
    attacks = []
    aux_records = []
    valid_attacks = 0
    valid_aux = 0
    failures = []
    incomplete = []
    for seg in segments:
        seg_name = seg["seg_name"]
        try:
            validation = read_attack_validation(seg)
            attacks.append({"seg": seg_name, "status": "VALID", "validation": validation})
            valid_attacks += 1
        except Exception as exc:
            item = {"kind": "attack", "seg": seg_name, "error": str(exc)}
            attacks.append({"seg": seg_name, "status": "INCOMPLETE", "error": str(exc)})
            incomplete.append(item)
        for condition in ("clean", "attack"):
            try:
                validation = validate_aux(condition, seg_name)
                aux_records.append({"seg": seg_name, "condition": condition,
                                    "status": "VALID", "validation": validation})
                valid_aux += 1
            except Exception as exc:
                item = {"kind": "aux", "seg": seg_name,
                        "condition": condition, "error": str(exc)}
                aux_records.append({"seg": seg_name, "condition": condition,
                                    "status": "INCOMPLETE", "error": str(exc)})
                incomplete.append(item)
    runs = []
    valid_runs = 0
    for rep in REPS:
        for seg in segments:
            for arm in ARMS:
                run_name = "e29_%s_%s_%s" % (arm, seg["seg_name"], rep)
                run_root = WORK / "victim_runs" / run_name
                record = {"run_name": run_name, "seg": seg["seg_name"],
                          "arm": arm, "rep": rep}
                try:
                    validation = read_validated_completion(seg, arm, rep, run_root, cfg)
                    record.update({"status": "VALID", "validation": validation})
                    valid_runs += 1
                except Exception as exc:
                    record.update({"status": "FAILED" if run_root.exists() else "MISSING",
                                   "error": str(exc)})
                    item = {"kind": "victim", "run_name": run_name, "error": str(exc)}
                    (failures if run_root.exists() else incomplete).append(item)
                runs.append(record)
    complete = valid_runs == 120 and valid_attacks == 10 and valid_aux == 20
    status = "FULL_MATRIX_COMPLETE" if complete else (
        "FAILED_INCOMPLETE" if failures else "INCOMPLETE")
    summary = {
        "experiment": "E29", "status": status, "trackeval_performed": False,
        "protocol": str(PROTOCOL), "protocol_sha256": file_sha256(PROTOCOL),
        "segments": [s["seg_name"] for s in segments], "repetitions": list(REPS),
        "planned_victim_runs": 120, "valid_victim_runs": valid_runs,
        "new_valid_victim_runs": valid_runs,
        "valid_attacks": valid_attacks, "planned_attacks": 10,
        "valid_aux_files": valid_aux, "planned_aux_files": 20,
        "attack_validation": attacks, "aux_validation": aux_records,
        "victim_validation": runs, "failures": failures, "incomplete": incomplete,
        "resume_commands": [
            PY + " " + str(CODE / "run.py") + " attack",
            PY + " " + str(CODE / "run.py") + " aux",
            PY + " " + str(CODE / "run.py") + " victims --rep all",
            PY + " " + str(CODE / "run.py") + " summarize"]}
    with open(WORK / "summary_full.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "attack", "aux", "victims", "all", "summarize"))
    parser.add_argument("--rep", default="all", choices=("r1", "r2", "r3", "all"))
    args = parser.parse_args()
    cfg = load_protocol()
    frozen = load_and_verify_freeze()
    segments = selected_segments(frozen)
    for seg in segments:
        prepare_annotation(seg)
    if args.command == "check":
        print(json.dumps({"status": "STATIC_CHECK_OK", "segments": 10,
                          "planned_runs": 120, "repetitions": list(REPS),
                          "preserved_smoke_runs_valid": 0,
                          "protocol": str(PROTOCOL)}, indent=2))
        return
    if args.command in ("attack", "all"):
        attack_all(segments, frozen)
    if args.command in ("aux", "all"):
        aux_all(segments)
    if args.command in ("victims", "all"):
        victims_all(segments, cfg, list(REPS) if args.rep == "all" else [args.rep])
    if args.command in ("summarize", "all"):
        summary = summarize(segments, cfg)
        print(json.dumps({"status": summary["status"],
                          "valid_attacks": summary["valid_attacks"],
                          "valid_aux_files": summary["valid_aux_files"],
                          "valid_victim_runs": summary["valid_victim_runs"],
                          "failures": len(summary["failures"]),
                          "incomplete": len(summary["incomplete"]),
                          "summary": str(WORK / "summary_full.json")}, indent=2))


if __name__ == "__main__":
    main()
