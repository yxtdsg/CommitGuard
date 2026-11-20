#!/usr/bin/env python3
"""Install the small frozen FnF/TraSA source overlays after an explicit review."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fnf-root", type=Path, required=True)
    parser.add_argument("--trasa-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="copy missing files")
    parser.add_argument("--replace", action="store_true",
                        help="with --execute, also replace files whose hashes differ")
    args = parser.parse_args()
    with open(HERE / "protocol.json", encoding="utf-8") as handle:
        protocol = json.load(handle)
    for relative, expected in protocol["runtime_source_sha256"].items():
        family, source_relative = relative.split("/", 1)
        overlay = HERE / "overlays" / family.lower() / source_relative
        root = args.fnf_root if family == "FNF" else args.trasa_root
        target = root / source_relative
        current = sha256(target) if target.is_file() else "MISSING"
        status = "OK" if current == expected else "REPLACE" if current != "MISSING" else "COPY"
        print(status, relative, "current=", current, "expected=", expected)
        if status == "OK" or not args.execute:
            continue
        if status == "REPLACE" and not args.replace:
            raise RuntimeError(f"Refusing to replace {target}; review and pass --replace")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(overlay, target)
        actual = sha256(target)
        if actual != expected:
            raise RuntimeError(f"Overlay hash mismatch for {target}: {actual}")


if __name__ == "__main__":
    main()
