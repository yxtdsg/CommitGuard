"""Frozen numerical tensor composition and hash validation used by TraSw."""

import hashlib
import os
from pathlib import Path

import numpy as np

MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float64)
STDS = np.array([0.229, 0.224, 0.225], dtype=np.float64)
LOW = (-MEANS / STDS)[:, None, None]
HIGH = ((1.0 - MEANS) / STDS)[:, None, None]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def tensor_record(seg, kind, frame, path, target_id=None):
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.float32 or array.shape != (3, 800, 1440) or not np.isfinite(array).all():
        raise ValueError("Invalid frozen tensor: " + str(path))
    if not np.array_equal(array.astype(np.float16).astype(np.float32), array):
        raise ValueError("Tensor is not a lossless float32 representation of fp16 values: " + str(path))
    return {"seg": seg, "kind": kind, "frame": frame, "target_id": target_id,
            "path": str(path), "dtype": "float32", "shape": [3, 800, 1440],
            "file_sha256": file_sha256(path), "data_sha256": data_sha256(array)}


def compose_frame(clean, candidates):
    """Compose all candidates in sorted unique target-ID order, then clip once."""
    if clean.dtype != np.float32 or clean.shape != (3, 800, 1440):
        raise ValueError("Clean tensor has the wrong frozen shape or dtype")
    ids = [target for target, _ in candidates]
    if ids != sorted(set(ids)):
        raise ValueError("Candidate target IDs must be unique and ascending")
    accumulated = clean.astype(np.float64)
    for _, candidate in candidates:
        if candidate.shape != clean.shape or candidate.dtype != np.float32:
            raise ValueError("Candidate tensor shape or dtype mismatch")
        accumulated += candidate.astype(np.float64) - clean.astype(np.float64)
    return np.clip(accumulated, LOW, HIGH).astype(np.float16).astype(np.float32)


def save_new_npy(path, array):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite a frozen tensor: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
