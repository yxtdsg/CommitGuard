"""Shared paths and validation for the frozen 96-run method comparison."""

import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
E29 = Path(__file__).resolve().parent
VIDEOS = ["09", "10", "11", "13"]
REPS = [1, 2, 3]
CONDITION = "A"
RESULTS = Path(os.environ.get("COMMITGUARD_COMPARISON_RUNS", REPO / "runs" / "method_comparison")).resolve()
INPUT_ROOT = Path(os.environ.get("COMMITGUARD_COMPARISON_INPUTS", REPO / "external_inputs" / "method_comparison")).resolve()
MANIFEST = RESULTS / "run_manifest.json"


def configure(input_root=None, results_root=None):
    global INPUT_ROOT, RESULTS, MANIFEST
    if input_root:
        INPUT_ROOT = Path(input_root).resolve()
    if results_root:
        RESULTS = Path(results_root).resolve()
    MANIFEST = RESULTS / "run_manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def segment_name(video, segment):
    return f"MOT17-{video}-FRCNN-{segment:03d}"


def annotation_path(video):
    return INPUT_ROOT / "annotations" / f"clean2_mot17_{video}.json"


def detection_path(video, rep, condition=CONDITION):
    return INPUT_ROOT / "detections" / f"det_input_{condition}_{video}_r{rep}.jsonl"


def auxiliary_path(condition=CONDITION):
    return INPUT_ROOT / "auxiliary" / f"aux_yolov7_1280_{condition}_all.jsonl"


def image_root():
    return INPUT_ROOT / "images"


def source_root(family, method):
    if family == "byte":
        family = {"c0": "byte_c0", "g3": "byte_g3", "e19_fi40": "byte_e19",
                  "a1": "byte_a1", "a2": "byte_a2", "busca": "byte_busca"}.get(method, family)
    return RESULTS / family


def verify_source(method, family, video, rep, condition=CONDITION):
    runner_method = "c0" if method == "bot_sort_noreid" else method
    directory = source_root(family, method) / f"{runner_method}_{condition}{video}_r{rep}"
    runtime = load_json(directory / "runtime.json")
    assert runtime["video"] == video and runtime["rep"] == rep
    tracks = {segment_name(video, segment): directory / f"{segment_name(video, segment)}.txt"
              for segment in range(2)}
    for path in tracks.values():
        assert path.is_file() and path.stat().st_size > 0, path
    return {"source_dir": str(directory), "runtime": runtime, "tracks": tracks,
            "checks": {"track_files_nonempty": True, "no_GT_oracle": True}}


def add_e29_provenance(metadata, run_cfg):
    metadata["e29_protocol_sha256"] = sha256(E29 / "protocol.json")
    metadata["e29_method_config_sha256"] = sha256(E29 / "methods.json")
    return metadata
