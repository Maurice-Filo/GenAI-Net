"""Snapshot explicitly selected scientific sources, including dirty worktree edits."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "comparisons/search_exploration/baseline"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root=DEFAULT):
    manifest = json.loads((root / "manifest.json").read_text())
    changed = [p for p, expected in manifest["files"].items()
               if not (ROOT / p).exists() or digest(ROOT / p) != expected]
    if digest(root / "scientific_source.tar.gz") != manifest["archive_sha256"]:
        raise ValueError("Baseline archive changed")
    if changed:
        raise ValueError(f"Frozen sources changed: {changed}")
    return len(manifest["files"])


def freeze(root=DEFAULT):
    root.mkdir(parents=True, exist_ok=False)
    paths = set()
    for folder in ("RL4CRN", "comparisons/llm_crn_generation", "comparisons/llm_seeded_evolution",
                   "comparisons/rpa_search/src", "comparisons/rpa_search/configs", "tests"):
        for path in (ROOT / folder).rglob("*"):
            if path.suffix not in (".py", ".json", ".toml", ".yaml", ".yml"):
                continue
            if any(part in {"search_exploration", "campaigns", "data", "generated", "reports", "baseline", "__pycache__"}
                   for part in path.relative_to(ROOT).parts):
                continue
            if path.name.startswith("test_search_exploration"):
                continue
            paths.add(path)
    for path in ROOT.iterdir():
        if path.name in {"pyproject.toml", "uv.lock", "requirements.txt", "requirements-dev.txt"}:
            paths.add(path)
    files = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    archive = root / "scientific_source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name in files:
            stream.add(ROOT / name, arcname=name, recursive=False)
    manifest = {"schema": 1, "selection": "Scientific source/config files only; no homes, credentials or run outputs",
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "files": files, "archive_sha256": digest(archive)}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Frozen {verify(root)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(verify()) if args.verify else freeze()
