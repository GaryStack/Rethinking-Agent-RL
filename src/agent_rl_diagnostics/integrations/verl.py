"""Verify an external upstream checkout. No vendoring or optional imports at import time."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

from ..artifacts.io import canonical_hash, file_hash
from ..contracts.schema import SchemaError

VERL_COMMIT = "7dc46e834209948cf1cdd8a04d83f82b4a7efd24"


def verify_verl_source(root: Path) -> dict:
    lock = json.loads(
        files("agent_rl_diagnostics").joinpath("resources/verl-7dc46e-lock.json").read_text()
    )
    for name, expected in lock["files"].items():
        path = root / name
        if not path.is_file() or path.is_symlink() or file_hash(path) != expected:
            raise SchemaError(f"verl source differs from pinned commit at {name}")
    actual = {
        str(p.relative_to(root))
        for p in (root / "verl").rglob("*")
        if p.is_file() and p.suffix in {".py", ".yaml"}
    }
    if actual != set(lock["files"]):
        raise SchemaError("unexpected/missing Python or YAML files in pinned verl package")
    return {
        "commit": VERL_COMMIT,
        "source_lock_hash": canonical_hash(lock),
        "verified_files": len(actual),
    }
