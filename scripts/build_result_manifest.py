#!/usr/bin/env python3
"""Record provenance and hashes for the curated, redistributed paper results."""

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "results" / "paper"

SOURCES = {
    "mot17_development": ("E19", "TIFD_E0-1A/E19/analysis"),
    "mot17_additional": ("E20", "TIFD_E0-1A/E20/analysis"),
    "ocsort_recovery": ("E24", "TIFD_E0-1A/E24/analysis"),
    "dancetrack_main": ("E26", "TIFD_E0-1A/E26/analysis"),
    "trasw": ("E29", "TIFD_E0-1A/E29/analysis"),
    "method_comparison": ("E29", "TIFD_E0-1A/E29/analysis"),
    "detectability": ("E4_4", "TIFD_E0-1A/E4_4"),
    "figures": ("paper", "TIFD_E0-1A/figures"),
    "figures/mot17": ("E19/E20", "TIFD_E0-1A/mot17_visualizations"),
    "figures/dancetrack": ("E26", "TIFD_E0-1A/dancetrack_visualizations"),
}
COST_FILES = {"primary_runtime.csv", "cost_summary.csv", "cost_by_video.csv",
              "frame_cost.csv", "aux_runtime.csv"}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    records = []
    for path in sorted(PAPER.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(PAPER).as_posix()
        parent = path.parent.relative_to(PAPER).as_posix()
        parent = "" if parent == "." else parent
        historical, source_dir = SOURCES[parent]
        if parent == "mot17_additional" and path.name in COST_FILES:
            source_dir = "TIFD_E0-1A/E20"
        records.append({"path": relative, "historical_experiment": historical,
                        "source": f"{source_dir}/{path.name}",
                        "sha256": sha256(path), "bytes": path.stat().st_size})
    manifest = {
        "schema": 1,
        "description": "Curated final paper results; no raw tracks, tensors, caches, or logs.",
        "scopes": {
            "E19": {"videos": ["02", "04", "05"], "repeats": 3},
            "E20": {"videos": ["09", "10", "11", "13"], "repeats": 2},
            "E24": {"role": "OC-SORT and identity/window recovery"},
            "E26": {"videos": ["0004", "0005", "0007", "0010"],
                    "segments_per_video": 4, "repeats": 3},
            "E29": {"roles": ["complete TraSw", "multi-method comparison"]},
            "E4_4": {"role": "final detectability figure source only"}
        },
        "files": records,
    }
    output = PAPER / "manifest.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {output} with {len(records)} files")

    supplementary = REPO / "results" / "supplementary"
    supplementary_records = []
    for path in sorted(supplementary.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(supplementary).as_posix()
        historical = path.relative_to(supplementary).parts[0]
        if historical == "E30":
            source = f"TIFD_E0-1A/E30/analysis/{path.name}"
        elif historical == "E31":
            source = f"TIFD_E0-1A/E31/{path.name}"
        elif path.suffix == ".pdf":
            source = f"TIFD_E0-1A/dancetrack_visualizations/{path.name}"
        else:
            source = "public sanitized summary"
        supplementary_records.append({"path": relative, "historical_experiment": historical,
                                      "source": source, "sha256": sha256(path),
                                      "bytes": path.stat().st_size})
    supplementary_manifest = {
        "schema": 1,
        "formal_statistics_included": False,
        "notes": {
            "E30": "Non-significant sensitivity analyses; not merged into the formal E19/E20/E26 statistics.",
            "E31": "Clean MOT17 test-sequence timing only; not a defense evaluation.",
            "E32": "DanceTrack-0014 visualization only; never merged into the four-video E26 statistics."
        },
        "files": supplementary_records,
    }
    supplementary_output = supplementary / "manifest.json"
    with open(supplementary_output, "w", encoding="utf-8") as handle:
        json.dump(supplementary_manifest, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {supplementary_output} with {len(supplementary_records)} files")


if __name__ == "__main__":
    main()
