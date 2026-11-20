#!/usr/bin/env python3
"""Build the complete 96-entry comparison manifest from new run outputs."""

import argparse
from pathlib import Path

import common

LABELS = {
    "c0": "ByteTrack", "a1": "TrackShield-a1", "a2": "TrackShield-a2",
    "busca": "BUSCA", "ocsort": "OC-SORT", "bot_sort_noreid": "BoT-SORT (noReID)",
    "g3": "Audit+BG", "e19_fi40": "Ours",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(common.INPUT_ROOT))
    parser.add_argument("--output-root", default=str(common.RESULTS))
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    common.configure(args.input_root, args.output_root)
    entries = []
    for method, label in LABELS.items():
        for video in common.VIDEOS:
            for rep in common.REPS:
                source = common.verify_source(method, method, video, rep)
                runtime = source["runtime"]
                runner_method = "c0" if method == "bot_sort_noreid" else method
                entries.append({"family": method, "method": method, "label": label,
                                "runner_method": runner_method,
                                "video": video, "rep": rep, "condition": "A",
                                "run_name": f"{runner_method}_A{video}_r{rep}",
                                "source_experiment": "E29-public",
                                "source_dir": source["source_dir"],
                                "runtime": str(Path(source["source_dir"]) / "runtime.json"),
                                "tracks": [str(source["tracks"][common.segment_name(video, segment)])
                                           for segment in (0, 1)],
                                "detections": str(common.detection_path(video, rep)),
                                "detections_sha256": common.sha256(common.detection_path(video, rep)),
                                "annotation": str(common.annotation_path(video)),
                                "annotation_sha256": common.sha256(common.annotation_path(video)),
                                "reused": False,
                                "source_sha256": runtime.get("source_sha256", {}),
                                "checks": source["checks"]})
    if len(entries) != 96:
        raise ValueError(len(entries))
    output = Path(args.output) if args.output else common.MANIFEST
    common.write_json(output, {"schema": "e29_method_comparison_v2", "complete": True,
                               "videos": common.VIDEOS, "repeats": common.REPS,
                               "method_order": list(LABELS),
                               "entries": entries})
    print(f"Wrote {output} with {len(entries)} runs")


if __name__ == "__main__":
    main()
