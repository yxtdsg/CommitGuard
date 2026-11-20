#!/usr/bin/env python3
"""Scan the release tree for private paths, credentials, caches and large files."""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv", "runs"}
TEXT_SUFFIXES = {".py", ".json", ".md", ".txt", ".toml", ".csv", ".gitignore", ".ini", ".yml", ".yaml"}
PRIVATE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in (
        r"[A-Za-z]:[\\/]sensor", r"[A-Za-z]:[\\/]MOTAttack", r"C:[\\/]Users[\\/]yxt",
        r"/home/" + "yxt", r"bytetrack" + "5090", r"_e29" + "_execution", r"_tmp[\\/]e14_common",
    )
]
SECRET_PATTERNS = [
    re.compile(pattern) for pattern in (
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"\bAKIA[0-9A-Z]{16}\b", r"\bghp_[A-Za-z0-9]{30,}\b",
        r"\bgithub_pat_[A-Za-z0-9_]{30,}\b", r"\bsk-[A-Za-z0-9]{30,}\b",
    )
]
REQUIRED = ["README.md", "LICENSE", "NOTICE.md", "pyproject.toml", "requirements.txt",
            "results/paper/manifest.json", "results/supplementary/manifest.json"]


def main():
    problems = []
    for relative in REQUIRED:
        if not (REPO / relative).is_file():
            problems.append(f"missing required file: {relative}")
    files = []
    largest = []
    for path in REPO.rglob("*"):
        relative = path.relative_to(REPO)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if path.is_symlink():
            problems.append(f"symlink not allowed: {relative}")
            continue
        if not path.is_file():
            continue
        files.append(relative)
        size = path.stat().st_size
        largest.append((size, relative))
        if size > 20 * 1024 * 1024:
            problems.append(f"file over 20 MiB: {relative} ({size})")
        if path.suffix.lower() in {".npy", ".npz", ".pt", ".pth", ".ckpt", ".tex", ".wsl"}:
            problems.append(f"excluded artifact type: {relative}")
        if path.suffix.lower() == ".pem" or path.name in {".env", "id_rsa", "credentials.json"}:
            problems.append(f"credential-like file: {relative}")
        if path.suffix not in TEXT_SUFFIXES and path.name != ".gitignore":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in PRIVATE_PATTERNS:
            match = pattern.search(text)
            if match:
                problems.append(f"private path in {relative}: {match.group(0)}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                problems.append(f"credential-like content in {relative}")
    with open(REPO / "results" / "paper" / "manifest.json", encoding="utf-8") as handle:
        paper = json.load(handle)
    with open(REPO / "results" / "supplementary" / "manifest.json", encoding="utf-8") as handle:
        supplementary = json.load(handle)
    if problems:
        raise SystemExit("RELEASE SCAN FAILED\n" + "\n".join(sorted(set(problems))))
    largest.sort(reverse=True)
    print(json.dumps({
        "status": "ok", "files_scanned": len(files),
        "paper_files": len(paper["files"]), "supplementary_files": len(supplementary["files"]),
        "largest_files": [{"path": path.as_posix(), "bytes": size} for size, path in largest[:5]],
        "private_paths": 0, "credential_matches": 0, "oversized_files": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
