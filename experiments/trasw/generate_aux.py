# -*- coding: utf-8 -*-
"""Resumable YOLOv7 candidates exclusively from E29 replay tensors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
from trasw_common import WORK

YOLO_ROOT = Path(os.environ.get("YOLOV7_ROOT", Path(__file__).resolve().parents[2] / "external" / "yolov7")).resolve()
WEIGHTS = Path(os.environ.get("YOLOV7_WEIGHTS", YOLO_ROOT / "yolov7.pt")).resolve()
MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]
STDS = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :]

from trasw_common import aux_path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_tensor(path, expected_file_hash, expected_data_hash):
    if sha256(path) != expected_file_hash:
        raise RuntimeError("E29 aux 输入文件哈希不匹配")
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.float32 or array.shape != (3, 800, 1440) or \
            not np.isfinite(array).all():
        raise RuntimeError("E29 aux 输入张量非法")
    actual = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    if actual != expected_data_hash:
        raise RuntimeError("E29 aux 输入数据哈希不匹配")
    rgb = np.clip((array.transpose(1, 2, 0) * STDS + MEANS) * 255.0,
                  0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=("clean", "attack"))
    parser.add_argument("--seg", required=True)
    args = parser.parse_args()
    condition = args.condition
    output = aux_path(condition, args.seg)
    kind = "trasw_%s_tensor" % condition
    seg_root = WORK / "replay_tensors" / condition / args.seg
    with open(seg_root / "frame_hashes.json", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest["seg"] != args.seg or manifest["n_frames"] != 30:
        raise RuntimeError("E29 aux replay manifest 错误")
    existing = {}
    if output.exists():
        with open(output, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                frame = int(rec["f"])
                if frame in existing or not 6 <= frame <= 30:
                    raise RuntimeError("E29 aux 续跑文件重复或帧越界")
                expected = manifest["frames"][str(frame)]
                if rec["seg"] != args.seg or rec["input_kind"] != kind or \
                        rec["input_data_sha256"] != expected["data_sha256"] or \
                        rec["input_file_sha256"] != expected["file_sha256"]:
                    raise RuntimeError("E29 aux 续跑记录与当前输入不符")
                existing[frame] = rec
        if list(existing) != sorted(existing):
            raise RuntimeError("E29 aux 续跑记录顺序异常")
    missing = [frame for frame in range(6, 31) if frame not in existing]
    if not missing:
        print("AUX_SKIP_COMPLETE", condition, str(output), flush=True)
        return
    with open(CODE / "protocol.json", encoding="utf-8") as handle:
        expected_weights = json.load(handle)["yolov7_weights_sha256"]
    if sha256(WEIGHTS) != expected_weights:
        raise RuntimeError("YOLOv7 weights do not match the frozen SHA256")
    sys.path.insert(0, str(YOLO_ROOT))
    from models.experimental import attempt_load
    from utils.datasets import letterbox
    from utils.general import non_max_suppression, scale_coords
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    original_load = torch.load

    def trusted_local_load(*load_args, **load_kwargs):
        load_kwargs["weights_only"] = False
        return original_load(*load_args, **load_kwargs)

    start = time.perf_counter()
    torch.load = trusted_local_load
    try:
        model = attempt_load(str(WEIGHTS), map_location=device)
    finally:
        torch.load = original_load
    model = model.half().eval()
    torch.cuda.synchronize()
    load_sec = time.perf_counter() - start
    stride = int(model.stride.max())
    warmup_image = decode_tensor(
        seg_root / "frame_6.npy", manifest["frames"]["6"]["file_sha256"],
        manifest["frames"]["6"]["data_sha256"])
    transformed = letterbox(warmup_image, 1280, stride=stride)[0]
    transformed = transformed[:, :, ::-1].transpose(2, 0, 1)
    warmup_tensor = torch.from_numpy(np.ascontiguousarray(transformed)).to(device).half()
    warmup_tensor = warmup_tensor.unsqueeze(0) / 255.0
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(3):
        with torch.no_grad():
            prediction = model(warmup_tensor)[0]
        non_max_suppression(prediction, 0.01, 0.45, classes=[0])
    torch.cuda.synchronize()
    warmup_sec = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    produced = []
    with open(output, "a", encoding="utf-8") as out:
        for frame in missing:
            expected = manifest["frames"][str(frame)]
            path = seg_root / ("frame_%d.npy" % frame)
            torch.cuda.synchronize()
            total_start = time.perf_counter()
            image = decode_tensor(path, expected["file_sha256"], expected["data_sha256"])
            input_end = time.perf_counter()
            transformed = letterbox(image, 1280, stride=stride)[0]
            transformed = transformed[:, :, ::-1].transpose(2, 0, 1)
            tensor = torch.from_numpy(np.ascontiguousarray(transformed)).to(device).half()
            tensor = tensor.unsqueeze(0) / 255.0
            torch.cuda.synchronize()
            preprocess_end = time.perf_counter()
            with torch.no_grad():
                prediction = model(tensor)[0]
            torch.cuda.synchronize()
            inference_end = time.perf_counter()
            detections = non_max_suppression(prediction, 0.01, 0.45, classes=[0])[0]
            if len(detections):
                detections[:, :4] = scale_coords(
                    tensor.shape[2:], detections[:, :4], image.shape).round()
            rows = []
            for det in detections.detach().cpu().numpy():
                x1, y1, x2, y2, score = det[:5]
                rows.append([float(x1), float(y1), float(x2 - x1),
                             float(y2 - y1), float(score)])
            torch.cuda.synchronize()
            total_end = time.perf_counter()
            record = {
                "seg": args.seg, "f": frame, "input_kind": kind,
                "input_path": str(path), "input_file_sha256": expected["file_sha256"],
                "input_data_sha256": expected["data_sha256"],
                "width": 1440, "height": 800, "ratio": 1.0, "dets": rows,
                "cost": {"input_decode_ms": (input_end - total_start) * 1000.0,
                         "preprocess_h2d_ms": (preprocess_end - input_end) * 1000.0,
                         "inference_ms": (inference_end - preprocess_end) * 1000.0,
                         "nms_post_ms": (total_end - inference_end) * 1000.0,
                         "total_aux_ms": (total_end - total_start) * 1000.0}}
            out.write(json.dumps(record) + "\n")
            out.flush()
            produced.append(record)
            print("AUX", condition, args.seg, frame, len(rows), flush=True)
    print("AUX_DONE", json.dumps({
        "condition": condition, "seg": args.seg, "new_frames": len(produced),
        "resumed_existing_frames": len(existing), "weights": str(WEIGHTS),
        "weights_sha256": sha256(WEIGHTS), "img_size": 1280,
        "conf": 0.01, "nms_iou": 0.45, "classes": [0],
        "device": torch.cuda.get_device_name(0), "dtype": "float16",
        "model_load_sec_excluded": load_sec, "warmup_sec_excluded": warmup_sec,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 1024 ** 2,
        "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
