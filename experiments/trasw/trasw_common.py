# -*- coding: utf-8 -*-
"""E29 TraSw helpers; frozen E14 numerical routines are reused read-only."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
E29 = CODE
WORK = Path(os.environ.get("COMMITGUARD_TRASW_RUNS", ROOT / "runs" / "trasw")).resolve()
INPUT_ROOT = Path(os.environ.get("COMMITGUARD_INPUT_ROOT", ROOT / "external_inputs")).resolve()
TRASA = Path(os.environ.get("TRASA_ROOT", ROOT / "external" / "TraSA")).resolve()
FNF = Path(os.environ.get("FNF_ROOT", ROOT / "external" / "FnF")).resolve()
RAW = WORK / "tensors_raw"
REPLAY = WORK / "replay_tensors"
LEDGER = WORK / "execution_ledger.jsonl"
PROTOCOL = CODE / "protocol.json"
PY = os.environ.get("TRASW_PYTHON", sys.executable)
FIRST_SEG_NAME = "MOT17-02-FRCNN-000"
ARMS = ("clean_c0", "attack_c0", "clean_cg", "attack_cg")
REPS = ("r1", "r2", "r3")
sys.path.insert(0, str(ROOT / "src"))
from tensor_ops import compose_frame, data_sha256, file_sha256, save_new_npy, tensor_record


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def ledger(event):
    WORK.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time_utc": now(), **event}, sort_keys=True) + "\n")


def load_protocol():
    with open(PROTOCOL, encoding="utf-8") as f:
        cfg = json.load(f)
    scope = cfg["scope"]
    assert scope["dev_segment_count"] == 10
    assert scope["planned_matrix_runs"] == 120
    assert scope["execution_segments"] == "all E14 freeze segments"
    assert scope["execution_repetitions"] == list(REPS)
    assert scope["precompleted_smoke_runs"] == 0
    assert tuple(cfg["victim"]["arms"]) == ARMS
    return cfg


def load_and_verify_freeze():
    with open(CODE / "freeze.json", encoding="utf-8") as handle:
        frozen = json.load(handle)
    for relative, expected in load_protocol()["runtime_source_sha256"].items():
        if relative.startswith("FNF/"):
            path = FNF / relative[4:]
        elif relative.startswith("TRASA/"):
            path = TRASA / relative[6:]
        else:
            raise ValueError("Unknown external source root: " + relative)
        if file_sha256(path) != expected:
            raise RuntimeError("Frozen external source changed: " + str(path))
    for relative, expected in load_protocol()["checkpoint_sha256"].items():
        family, checkpoint = relative.split("/", 1)
        path = (FNF if family == "FNF" else TRASA) / checkpoint
        if file_sha256(path) != expected:
            raise RuntimeError("Frozen checkpoint changed or missing: " + str(path))
    return frozen


def tree_sha256(root):
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            relative = str(path.relative_to(root)).replace("\\", "/")
            digest.update((relative + "\0" + file_sha256(path) + "\n").encode())
    return digest.hexdigest()


def verify_preserved_smoke(cfg):
    preserved = cfg["preserved_smoke"]
    for path, expected in preserved["files"].items():
        if file_sha256(path) != expected:
            raise RuntimeError("首段不可变文件已改变: " + path)
    for path, expected in preserved["trees"].items():
        if tree_sha256(path) != expected:
            raise RuntimeError("首段不可变目录树已改变: " + path)


def selected_segments(frozen):
    segments = list(frozen["segments"])
    names = [s["seg_name"] for s in segments]
    if len(segments) != 10 or len(set(names)) != 10:
        raise RuntimeError("E14 dev 段集合不是 10 个唯一段")
    if names[0] != FIRST_SEG_NAME or segments[0]["video_id"] != 1:
        raise RuntimeError("E14 首段不再是已完成的 MOT17-02-FRCNN-000")
    return segments


def prepare_annotation(seg):
    source = INPUT_ROOT / seg["annotation"]
    destination = WORK / "annotations" / (seg["seg_name"] + ".json")
    if file_sha256(source) != seg["annotation_sha256"]:
        raise RuntimeError("E14 段标注哈希不符: " + seg["seg_name"])
    if destination.exists():
        if file_sha256(destination) != seg["annotation_sha256"]:
            raise RuntimeError("E29 已有标注与 E14 源标注不一致，拒绝覆盖")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    with open(destination, encoding="utf-8") as f:
        if len(json.load(f)["images"]) != 30:
            raise RuntimeError("E29 段标注不是 30 帧: " + seg["seg_name"])
    return destination


def compose_segment(seg):
    seg_name = seg["seg_name"]
    raw_seg = RAW / seg_name
    composition = []
    clean_records = []
    attack_records = []
    for frame in range(1, 31):
        clean_path = raw_seg / "clean" / ("frame_%d.npy" % frame)
        clean = np.load(clean_path, allow_pickle=False)
        clean_record = tensor_record(seg_name, "clean", frame, str(clean_path))
        candidate_records = []
        candidate_dir = raw_seg / "candidates"
        if candidate_dir.is_dir():
            prefix = "frame_%d_target_" % frame
            for name in os.listdir(candidate_dir):
                if name.startswith(prefix) and name.endswith(".npy"):
                    target_id = int(name[len(prefix):-4])
                    candidate_records.append(tensor_record(
                        seg_name, "candidate", frame, str(candidate_dir / name), target_id))
        candidate_records.sort(key=lambda x: x["target_id"])
        candidates = [(r["target_id"], np.load(r["path"], allow_pickle=False))
                      for r in candidate_records]
        changed_target_ids = [tid for tid, candidate in candidates
                              if not np.array_equal(candidate, clean)]
        attack = compose_frame(clean, candidates)
        attack_path = REPLAY / "attack" / seg_name / ("frame_%d.npy" % frame)
        save_new_npy(str(attack_path), attack)
        attack_record = tensor_record(seg_name, "attack", frame, str(attack_path))
        attack_record["target_ids"] = [r["target_id"] for r in candidate_records]
        attack_records.append(attack_record)
        clean_dest = REPLAY / "clean" / seg_name / ("frame_%d.npy" % frame)
        save_new_npy(str(clean_dest), clean)
        copied = tensor_record(seg_name, "clean", frame, str(clean_dest))
        if copied["data_sha256"] != clean_record["data_sha256"]:
            raise RuntimeError("clean replay 数值改变")
        clean_records.append(copied)
        composition.append({
            "seg": seg_name, "frame": frame,
            "target_ids": attack_record["target_ids"],
            "changed_target_ids": changed_target_ids,
            "candidate_data_sha256": [r["data_sha256"] for r in candidate_records],
            "changed": not np.array_equal(attack, clean),
            "max_abs_delta": float(np.max(np.abs(
                attack.astype(np.float64) - clean.astype(np.float64))))})
    for kind, records in (("clean", clean_records), ("attack", attack_records)):
        directory = REPLAY / kind / seg_name
        manifest = {"seg": seg_name, "n_frames": 30, "dtype": "float32",
                    "shape": [3, 800, 1440], "fp16_values": True,
                    "frames": {str(r["frame"]): {
                        "file_sha256": r["file_sha256"],
                        "data_sha256": r["data_sha256"]} for r in records}}
        with open(directory / "frame_hashes.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
    with open(raw_seg / "composition.json", "w", encoding="utf-8") as f:
        json.dump(composition, f, indent=2)
    return validate_composition(seg)


def validate_composition(seg):
    seg_name = seg["seg_name"]
    raw_seg = RAW / seg_name
    with open(raw_seg / "composition.json", encoding="utf-8") as f:
        rows = json.load(f)
    if len(rows) != 30 or [r["frame"] for r in rows] != list(range(1, 31)):
        raise RuntimeError("E29 composition 帧清单不完整")
    manifests = {}
    for kind in ("clean", "attack"):
        with open(REPLAY / kind / seg_name / "frame_hashes.json", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest["seg"] != seg_name or manifest["n_frames"] != 30:
            raise RuntimeError("E29 replay manifest 段名/帧数错误")
        manifests[kind] = manifest
    for row in rows:
        frame = row["frame"]
        clean = np.load(raw_seg / "clean" / ("frame_%d.npy" % frame), allow_pickle=False)
        candidates = []
        changed_target_ids = []
        for target_id in row["target_ids"]:
            path = raw_seg / "candidates" / ("frame_%d_target_%d.npy" % (frame, target_id))
            candidate = np.load(path, allow_pickle=False)
            candidates.append((target_id, candidate))
            if not np.array_equal(candidate, clean):
                changed_target_ids.append(target_id)
        if row["target_ids"] != sorted(set(row["target_ids"])):
            raise RuntimeError("candidate target ID 非唯一升序")
        if changed_target_ids != row["changed_target_ids"]:
            raise RuntimeError("changed candidate 清单不符")
        expected_attack = compose_frame(clean, candidates)
        for kind, expected in (("clean", clean), ("attack", expected_attack)):
            path = REPLAY / kind / seg_name / ("frame_%d.npy" % frame)
            record = tensor_record(seg_name, kind, frame, str(path))
            expected_hash = manifests[kind]["frames"][str(frame)]
            if record["file_sha256"] != expected_hash["file_sha256"] or \
                    record["data_sha256"] != expected_hash["data_sha256"] or \
                    record["data_sha256"] != data_sha256(expected):
                raise RuntimeError("E29 replay 与冻结合成/哈希不符")
        if (not np.array_equal(expected_attack, clean)) != row["changed"]:
            raise RuntimeError("changed 帧标记不符")
    return {"seg": seg_name, "clean_frames": 30, "attack_frames": 30,
            "changed_frames": sum(r["changed"] for r in rows),
            "candidate_targets": sum(len(r["target_ids"]) for r in rows),
            "changed_candidate_targets": sum(len(r["changed_target_ids"]) for r in rows),
            "changed_frame_ids": [r["frame"] for r in rows if r["changed"]],
            "composition": str(raw_seg / "composition.json")}


def aux_path(condition, seg_name):
    suffix = "clean" if condition == "clean" else "A"
    return WORK / "auxiliary" / ("aux_yolov7_1280_%s_%s.jsonl" % (suffix, seg_name))


def attack_validation_path(seg_name):
    return WORK / "attack_validations" / (seg_name + ".json")


def validate_aux(condition, seg_name):
    path = aux_path(condition, seg_name)
    expected_kind = "trasw_%s_tensor" % condition
    with open(REPLAY / condition / seg_name / "frame_hashes.json", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if len(rows) != 25 or [r["f"] for r in rows] != list(range(6, 31)):
        raise RuntimeError("E29 aux 必须恰好含帧 6..30: " + str(path))
    for row in rows:
        expected = manifest["frames"][str(row["f"])]
        if row["seg"] != seg_name or row["input_kind"] != expected_kind or \
                row["input_data_sha256"] != expected["data_sha256"] or \
                row["input_file_sha256"] != expected["file_sha256"]:
            raise RuntimeError("E29 aux 段名/kind/输入哈希错误")
        np.asarray(row["dets"], dtype=np.float64).reshape(-1, 5)
        if not isinstance(row.get("cost"), dict):
            raise RuntimeError("E29 aux 缺少逐帧成本")
    return {"path": str(path), "frames": len(rows),
            "detections": sum(len(r["dets"]) for r in rows),
            "file_sha256": file_sha256(path)}
