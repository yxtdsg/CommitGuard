import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
INPUT_ROOT = Path(os.environ.get("COMMITGUARD_COMPARISON_INPUTS", REPO / "external_inputs" / "method_comparison")).resolve()
FNF = Path(os.environ.get("BOTSORT_ROOT", REPO / "external" / "BoT-SORT")).resolve()
sys.path.insert(0, str(FNF))

MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[None, None, :]
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[None, None, :]
IMG_SIZE = (800, 1440)
PROTOCOL = json.load(open(HERE / "bot_sort_protocol.json", encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_annotation(video):
    path = INPUT_ROOT / PROTOCOL["videos"][video]["annotation"]
    data = json.load(open(path, encoding="utf-8"))
    images = sorted(data["images"], key=lambda x: x["id"])
    base = data.get("video_name") or images[0]["file_name"].split("/")[0]
    length = PROTOCOL["segment_length"]
    if len(images) != length * PROTOCOL["videos"][video]["segments"]:
        raise ValueError(f"Unexpected image count in {path}: {len(images)}")
    segments = {f"{base}-{i // length:03d}": images[i:i + length]
                for i in range(0, len(images), length)}
    return path, segments


def load_detections(video, rep, condition):
    path = INPUT_ROOT / PROTOCOL["videos"][video]["detection_template"].format(
        condition=condition, rep=rep)
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records[(rec["seg"], int(rec["f"]))] = rec["dets"]
    return path, records


def detection_array(dets):
    if not dets:
        return np.zeros((0, 7), dtype=np.float32)
    return np.asarray([[x, y, x + w, y + h, s, 1.0, 0.0]
                       for x, y, w, h, s in dets], dtype=np.float32)


def method_args(method, segment, video, condition, run_name):
    cfg = PROTOCOL["bot_sort"]
    args = SimpleNamespace(
        track_high_thresh=cfg["track_high_thresh"],
        track_low_thresh=cfg["track_low_thresh"],
        new_track_thresh=cfg["new_track_thresh"],
        match_thresh=cfg["match_thresh"],
        track_buffer=cfg["track_buffer"],
        proximity_thresh=cfg["proximity_thresh"],
        appearance_thresh=cfg["appearance_thresh"],
        with_reid=cfg["with_reid"],
        fast_reid_config="",
        fast_reid_weights="",
        device="cpu",
        cmc_method=cfg["cmc_method"],
        ablation=False,
        name=run_name,
        mot20=cfg["mot20"],
        min_box_area=cfg["min_box_area"],
        current_seg=segment,
        current_gt=[],
        l1_mode="off",
        l1_intervene=0,
        quarantine_mode="off",
        aux_review_mode=False,
        aux_filter_mode=False,
        aux_fn_mode=False,
        aux_candidate_file="",
        aux_condition=condition,
        review_noq=False,
        review_trigger="alarm",
    )
    if method in ("audit", "g3", "e19_fi40"):
        args.l1_mode = "fixed8"
    if method in ("g3", "e19_fi40"):
        args.l1_intervene = 1
    if method == "e19_fi40":
        for key, value in PROTOCOL["cg"].items():
            setattr(args, key, value)
        args.aux_candidate_file = str(INPUT_ROOT / PROTOCOL["videos"][video]["aux_template"].format(
            condition=condition))
        args.aux_condition = condition
    return args


def frame_image(info, segment, local_frame, condition, video, frozen_hashes):
    if condition == "A" and local_frame in PROTOCOL["attack_local_frames"]:
        path = INPUT_ROOT / PROTOCOL["videos"][video]["frozen_root"] / segment / f"frame_{local_frame}.npy"
        key = f"{segment}/frame_{local_frame}.npy"
        if key not in frozen_hashes or sha256(path) != frozen_hashes[key]:
            raise RuntimeError(f"Frozen tensor hash mismatch: {path}")
        tensor = np.load(path)
        if tensor.shape != (3, *IMG_SIZE):
            raise ValueError(f"Unexpected frozen tensor shape {tensor.shape}: {path}")
    else:
        path = INPUT_ROOT / PROTOCOL["image_root"] / info["file_name"]
        original = cv2.imread(str(path))
        if original is None:
            raise FileNotFoundError(path)
        from yolox.data.data_augment import preproc
        tensor, _ = preproc(original, IMG_SIZE, MEAN[:, 0, 0], STD[:, 0, 0])
    rgb = np.clip((tensor.transpose(1, 2, 0) * STD + MEAN) * 255.0,
                  0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def write_tracks(path, rows, scale, min_box_area):
    with open(path, "w", encoding="utf-8") as f:
        for frame_id, targets in rows:
            for track_id, tlwh, score in targets:
                if tlwh[2] * tlwh[3] <= min_box_area or tlwh[2] / tlwh[3] > 1.6:
                    continue
                x, y, w, h = tlwh / scale
                f.write(f"{frame_id},{track_id},{x:.1f},{y:.1f},{w:.1f},{h:.1f},{score:.2f},-1,-1,-1\n")


def run(method, video, rep, condition, output_root, only_segment=None, resume=False):
    from yolox.tracker_bot.bot_sort import BoTSORT
    ann_path, segments = load_annotation(video)
    det_path, det_records = load_detections(video, rep, condition)
    run_name = f"{method}_{condition}{video}_r{rep}"
    out_dir = output_root / run_name
    log_dir = output_root / "logs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if only_segment:
        segments = {only_segment: segments[only_segment]}
    metadata_path = out_dir / "runtime.json"
    if resume and metadata_path.exists() and all((out_dir / f"{seg}.txt").exists() for seg in segments):
        print(f"SKIP {run_name}", flush=True)
        return

    frozen_hashes = {}
    if condition == "A":
        frozen_hashes = json.load(open(INPUT_ROOT / PROTOCOL["videos"][video]["frozen_hashes"], encoding="utf-8"))
    runtime = []
    segment_meta = []
    for segment, frames in segments.items():
        args = method_args(method, segment, video, condition, run_name)
        init_start = time.perf_counter()
        tracker = BoTSORT(args, frame_rate=PROTOCOL["bot_sort"]["frame_rate"])
        init_sec = time.perf_counter() - init_start
        rows = []
        frame_times = []
        first_image = frames[0]
        scale = min(IMG_SIZE[0] / float(first_image["height"]),
                    IMG_SIZE[1] / float(first_image["width"]))
        for local_frame, image in enumerate(frames, start=1):
            key = (segment, local_frame)
            if key not in det_records:
                raise KeyError(f"Missing detection record: {key}")
            output = detection_array(det_records[key])
            current_frame = frame_image(image, segment, local_frame, condition, video, frozen_hashes)
            start = time.perf_counter()
            targets = tracker.update(output, current_frame)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            frame_times.append(elapsed_ms)
            event = getattr(tracker, "last_g3", {"b_t": 0, "alarm": False, "parked": 0})
            runtime.append({"seg": segment, "f": local_frame, "ms": elapsed_ms,
                            "b_t": int(event.get("b_t", 0)),
                            "g3_alarm": bool(event.get("alarm", False)),
                            "g3_parked": int(event.get("parked", 0))})
            snapshot = [(int(track.track_id), np.asarray(track.tlwh, dtype=float).copy(),
                         float(track.score)) for track in targets]
            rows.append((local_frame, snapshot))
        output_path = out_dir / f"{segment}.txt"
        write_tracks(output_path, rows, scale, PROTOCOL["bot_sort"]["min_box_area"])
        with open(output_path, encoding="utf-8") as f:
            written_rows = sum(1 for _ in f)
        segment_meta.append({
            "seg": segment,
            "frames": len(frames),
            "output_rows": written_rows,
            "tracker_init_sec": init_sec,
            "mean_update_ms": float(np.mean(frame_times)),
            "median_update_ms": float(np.median(frame_times)),
            "cg_stats": copy.deepcopy(getattr(tracker, "cg_stats", {})),
        })
        print(f"DONE bot {run_name} {segment} mean_ms={np.mean(frame_times):.3f}", flush=True)
        del tracker

    source_files = [Path(__file__), FNF / "yolox" / "tracker_bot" / "bot_sort.py",
                    FNF / "yolox" / "tracker_bot" / "gmc.py",
                    FNF / "yolox" / "tracker_bot" / "matching.py"]
    aux_file = str(INPUT_ROOT / PROTOCOL["videos"][video]["aux_template"].format(condition=condition)) \
        if method == "e19_fi40" else ""
    metadata = {
        "tracker": "BoT-SORT",
        "tracker_config": copy.deepcopy(PROTOCOL["bot_sort"]),
        "method": method,
        "video": video,
        "rep": rep,
        "condition": condition,
        "annotation": str(ann_path),
        "annotation_sha256": sha256(ann_path),
        "detections": str(det_path),
        "detections_sha256": sha256(det_path),
        "protocol": str(HERE / "bot_sort_protocol.json"),
        "protocol_sha256": sha256(HERE / "bot_sort_protocol.json"),
        "method_config": copy.deepcopy(PROTOCOL["cg"] if method == "e19_fi40" else {}),
        "aux_candidate_file": aux_file,
        "aux_candidate_sha256": sha256(aux_file) if aux_file else "",
        "frozen_hashes_file": PROTOCOL["videos"][video]["frozen_hashes"] if condition == "A" else "",
        "source_sha256": {str(p): sha256(p) for p in source_files},
        "segments": segment_meta,
        "frames": runtime,
        "mean_update_ms": float(np.mean([x["ms"] for x in runtime])),
        "median_update_ms": float(np.median([x["ms"] for x in runtime])),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main():
    global INPUT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["c0", "audit", "g3", "e19_fi40"])
    parser.add_argument("--video", required=True, choices=list(PROTOCOL["videos"]))
    parser.add_argument("--rep", required=True, type=int, choices=PROTOCOL["repeats"])
    parser.add_argument("--condition", required=True, choices=["clean", "A"])
    parser.add_argument("--input-root", default=str(INPUT_ROOT))
    parser.add_argument("--output-root", default=str(REPO / "runs" / "method_comparison" / "bot_sort_noreid"))
    parser.add_argument("--only-segment", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    INPUT_ROOT = Path(args.input_root).resolve()
    if args.dry_run:
        print(f"READY bot_sort_noreid {args.video} r{args.rep}")
        return
    cv2.setNumThreads(1)
    run(args.method, args.video, args.rep, args.condition, Path(args.output_root),
        args.only_segment or None, args.resume)


if __name__ == "__main__":
    main()
