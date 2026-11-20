"""Frozen-detection replay runner for ByteTrack, Audit, BG, and CommitGuard."""

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .tracker import CommitGuardTracker

IMG_SIZE = (800, 1440)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_annotation(annotation_path, frames_per_chunk=30):
    """Load one COCO-style video annotation and split it into complete segments."""
    if not annotation_path:
        raise ValueError("An explicit annotation path is required")
    path = Path(annotation_path)
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    images = sorted(data["images"], key=lambda item: item["id"])
    if not images:
        raise ValueError(f"No images in {path}")
    base = data.get("video_name")
    if not base:
        bases = {item["file_name"].split("/")[0] for item in images}
        if len(bases) != 1:
            raise ValueError(f"Expected one source video in {path}")
        base = next(iter(bases))
    chunks = [images[i:i + frames_per_chunk]
              for i in range(0, len(images), frames_per_chunk)]
    if any(len(chunk) != frames_per_chunk for chunk in chunks):
        raise ValueError(f"Incomplete segment in {path}")
    segments = {f"{base}-{index:03d}": chunk for index, chunk in enumerate(chunks)}
    return path, segments


def load_detections(detection_file):
    """Load frozen per-frame TLWH detections from JSONL."""
    if not detection_file:
        raise ValueError("An explicit detection file is required")
    path = Path(detection_file)
    records = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            key = (record["seg"], int(record["f"]))
            if key in records:
                raise ValueError(f"Duplicate detection record: {key}")
            records[key] = record["dets"]
    return path, records


def namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{key: namespace(value) for key, value in obj.items()})
    if isinstance(obj, list):
        return [namespace(value) for value in obj]
    return obj


def method_args(method, segment, methods, condition, protocol,
                config_dir, aux_candidate_file="", log_dir=None):
    config = methods.get(method)
    if config is None:
        raise ValueError(f"Unknown method: {method}")
    base_method = config.get("base", method)
    if base_method not in {"c0", "g3", "ts_cold", "ts_warm", "ts_audit", "ts_calibrate", "busca"}:
        raise ValueError(f"Unsupported public base method: {base_method}")
    args = SimpleNamespace(
        track_thresh=0.4,
        track_buffer=14 if segment.startswith("MOT17-05") else 30,
        match_thresh=0.9,
        det_thresh=0.5,
        min_box_area=10,
        mot20=False,
        disable_strict_birth=False,
        current_seg=segment,
        current_gt=[],
        l1_mode="off",
        l1_intervene=0,
        quarantine_mode="off",
        birth_scorer="",
        randq_quota_json="",
        randq_seed=0,
        gated_sc=0,
        s1_thr=0.0,
        use_busca=False,
        trackshield_mode="off",
        trackshield_calibration="",
        trackshield_calibration_key=segment.split("-")[1] if "-" in segment else segment,
        external_log_dir=str(log_dir or Path("runs") / "logs"),
        aux_condition=condition,
    )
    if base_method == "g3":
        args.l1_mode = "fixed8"
        args.l1_intervene = 1
    elif base_method in {"ts_cold", "ts_warm", "ts_audit", "ts_calibrate"}:
        args.trackshield_mode = {"ts_audit": "audit", "ts_calibrate": "calibrate"}.get(
            base_method, base_method)
    elif base_method == "busca":
        root_value = os.environ.get("BUSCA_ROOT")
        if not root_value:
            raise RuntimeError("BUSCA requires BUSCA_ROOT for the external official checkout")
        import yaml
        busca_root = Path(root_value).resolve()
        with open(busca_root / "config" / "ByteTrack" / "MOT17" /
                  "config_bytetrack_mot17.yml", encoding="utf-8") as handle:
            config_yaml = yaml.safe_load(handle)
        tracker_config = config_yaml["tracker"]
        args.use_busca = True
        args.device = tracker_config["device"]
        args.busca_ckpt = str(busca_root / "models" / "BUSCA" / "motsynth" / "model_busca.pth")
        args.transformer = namespace(config_yaml["transformer"])
        for key in ("busca_thresh", "use_camera_motion_compensation", "seq_len",
                    "num_candidates", "transformer_update_mems_only_first_round",
                    "reliable_thresh", "use_broader_memory", "select_highest_candidate"):
            setattr(args, key, tracker_config[key])
        args.highest_candidate_minimum_thresh = None
        args.ignore_custom_byte_thresholds = True
        args.track_thresh = 0.6
        args.det_thresh = 0.7
        args.track_buffer = 30
        args.seed = tracker_config["seed"]
    for key, value in protocol.get("base", {}).items():
        if key not in {"track_thresh", "det_thresh", "match_thresh",
                       "min_box_area", "track_buffer", "test_conf", "nms_thresh"}:
            raise ValueError(f"Unsupported tracker override: {key}")
        setattr(args, key, value)
    for key, value in config.get("args", {}).items():
        if key == "aux_candidate_file":
            continue
        if key.startswith("reid_") or key in {"use_busca", "trackshield_mode"}:
            raise ValueError(f"Unsupported public method override: {key}")
        if key == "trackshield_calibration":
            calibration = Path(value)
            if not calibration.is_absolute():
                calibration = config_dir / calibration
            value = str(calibration.resolve())
        setattr(args, key, value)
    aux_path = aux_candidate_file or config.get("args", {}).get("aux_candidate_file", "")
    if aux_path:
        aux_path = str(aux_path).format(condition=condition)
        candidate = Path(aux_path)
        if not candidate.is_absolute():
            candidate = config_dir / candidate
        args.aux_candidate_file = str(candidate.resolve())
    elif getattr(args, "aux_review_mode", False):
        raise ValueError(f"{method} requires --aux-candidate-file")
    return args


def detection_tensor(detections):
    if not detections:
        return None
    rows = [[x, y, x + w, y + h, score, 1.0, 0.0]
            for x, y, w, h, score in detections]
    return torch.tensor(rows, dtype=torch.float32)


def busca_frame(image_path):
    """Reproduce the frozen YOLOX preprocessing used by the BUSCA adapter."""
    import cv2
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = IMG_SIZE
    padded = np.ones((height, width, 3), dtype=np.float32) * 114.0
    ratio = min(height / image.shape[0], width / image.shape[1])
    resized = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
                         interpolation=cv2.INTER_LINEAR).astype(np.float32)
    padded[:int(image.shape[0] * ratio), :int(image.shape[1] * ratio)] = resized
    padded = padded[:, :, ::-1] / 255.0
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    transformed = ((padded - mean) / std).transpose(2, 0, 1)
    mean = mean[:, None, None]
    std = std[:, None, None]
    frame = np.clip(transformed * std + mean, 0.0, 1.0)
    frame = frame.transpose(1, 2, 0)[..., ::-1]
    return np.ascontiguousarray((frame * 255.0).astype(np.uint8))


def write_tracks(path, rows, scale):
    with open(path, "w", encoding="utf-8") as handle:
        for frame_id, targets in rows:
            for track_id, tlwh, score in targets:
                if tlwh[2] * tlwh[3] <= 10 or tlwh[2] / tlwh[3] > 1.6:
                    continue
                x, y, w, h = tlwh / scale
                handle.write(
                    f"{frame_id},{track_id},{x:.1f},{y:.1f},{w:.1f},{h:.1f},"
                    f"{score:.2f},-1,-1,-1\n")


def run(method, video, rep, condition, output_root, annotation_path,
        detection_file, method_config, protocol_path, aux_candidate_file="",
        only_segment=None, resume=False, frames_per_chunk=30,
        tracker_class=CommitGuardTracker, image_root=None):
    """Run one frozen-detection video and write MOT rows plus runtime metadata."""
    method_path = Path(method_config)
    protocol_file = Path(protocol_path)
    with open(method_path, encoding="utf-8") as handle:
        methods = json.load(handle)
    with open(protocol_file, encoding="utf-8") as handle:
        protocol = json.load(handle)
    ann_path, segments = load_annotation(annotation_path, frames_per_chunk)
    det_path, det_records = load_detections(detection_file)
    run_name = f"{method}_{condition}{video}_r{rep}"
    out_dir = Path(output_root) / run_name
    log_dir = Path(output_root) / "logs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if only_segment:
        if only_segment not in segments:
            raise KeyError(f"Unknown segment: {only_segment}")
        segments = {only_segment: segments[only_segment]}
    metadata_path = out_dir / "runtime.json"
    if resume and metadata_path.exists() and all(
            (out_dir / f"{segment}.txt").exists() for segment in segments):
        print(f"SKIP {run_name}", flush=True)
        return metadata_path

    runtime = []
    segment_meta = []
    for segment, frames in segments.items():
        args = method_args(method, segment, methods, condition, protocol,
                           method_path.parent, aux_candidate_file, log_dir)
        init_start = time.perf_counter()
        tracker = tracker_class(args)
        init_sec = time.perf_counter() - init_start
        rows = []
        frame_times = []
        scale = min(IMG_SIZE[0] / float(frames[0]["height"]),
                    IMG_SIZE[1] / float(frames[0]["width"]))
        for local_frame in range(1, len(frames) + 1):
            key = (segment, local_frame)
            if key not in det_records:
                raise KeyError(f"Missing detection record: {key}")
            output = detection_tensor(det_records[key])
            current_frame = None
            if method == "busca":
                if image_root is None:
                    raise ValueError("BUSCA requires image_root")
                current_frame = busca_frame(Path(image_root) / frames[local_frame - 1]["file_name"])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            targets = tracker.update(output, IMG_SIZE, IMG_SIZE, current_frame=current_frame)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            frame_times.append(elapsed_ms)
            event = getattr(tracker, "last_g3", {"b_t": 0, "alarm": False, "parked": 0})
            runtime.append({"seg": segment, "f": local_frame, "ms": elapsed_ms,
                            "b_t": event["b_t"], "g3_alarm": bool(event["alarm"]),
                            "g3_parked": int(event["parked"])})
            snapshot = [(int(track.track_id), np.asarray(track.tlwh, dtype=float).copy(),
                         float(track.score)) for track in targets]
            rows.append((local_frame, snapshot))
        output_path = out_dir / f"{segment}.txt"
        write_tracks(output_path, rows, scale)
        with open(output_path, encoding="utf-8") as handle:
            written_rows = sum(1 for _ in handle)
        segment_meta.append({
            "seg": segment,
            "frames": len(frames),
            "output_rows": written_rows,
            "tracker_init_sec": init_sec,
            "mean_update_ms": float(np.mean(frame_times)),
            "median_update_ms": float(np.median(frame_times)),
            "cg_stats": copy.deepcopy(getattr(tracker, "cg_stats", {})),
        })
        if method == "ts_calibrate":
            controller = tracker.trackshield
            segment_meta[-1]["trackshield_state"] = {
                "mu": controller.mu, "n": controller.n,
                "history": {signal: list(controller.hist[signal]) for signal in controller.SIGNALS},
            }
        print(f"DONE {run_name} {segment} mean_ms={np.mean(frame_times):.3f}", flush=True)
        del tracker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    source_files = [Path(__file__), Path(__file__).with_name("tracker.py"),
                    Path(__file__).parent / "_vendor" / "byte_tracker.py",
                    Path(__file__).parent / "_vendor" / "matching.py"]
    metadata = {
        "method": method,
        "video": video,
        "rep": rep,
        "condition": condition,
        "annotation": str(ann_path),
        "detections": str(det_path),
        "detections_sha256": sha256(det_path),
        "protocol": str(protocol_file),
        "protocol_sha256": sha256(protocol_file),
        "method_config_path": str(method_path),
        "method_config_sha256": sha256(method_path),
        "method_config": copy.deepcopy(methods.get(method, {})),
        "aux_candidate_file": aux_candidate_file,
        "aux_candidate_sha256": sha256(aux_candidate_file) if aux_candidate_file else "",
        "source_sha256": {path.name: sha256(path) for path in source_files},
        "segments": segment_meta,
        "frames": runtime,
        "mean_update_ms": float(np.mean([item["ms"] for item in runtime])),
        "median_update_ms": float(np.median([item["ms"] for item in runtime])),
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--rep", required=True, type=int)
    parser.add_argument("--condition", required=True, choices=["clean", "A"])
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--method-config", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--annotation-path", required=True)
    parser.add_argument("--detection-file", required=True)
    parser.add_argument("--aux-candidate-file", default="")
    parser.add_argument("--only-segment", default="")
    parser.add_argument("--frames-per-chunk", type=int, default=30)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run(args.method, args.video, args.rep, args.condition, args.output_root,
        args.annotation_path, args.detection_file, args.method_config,
        args.protocol, args.aux_candidate_file, args.only_segment or None,
        args.resume, args.frames_per_chunk, image_root=args.image_root)


if __name__ == "__main__":
    main()
