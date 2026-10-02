from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=lambda value: _invalid_constant(value))


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


@contextmanager
def new_output_directory(output: Path) -> Iterator[Path]:
    """Publish only complete runs; refuse to replace any existing output."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}; choose a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ard-", dir=output.parent) as directory:
        staging = Path(directory)
        yield staging
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"output appeared during evaluation: {output}")
        os.rename(staging, output)


def git_identity(root: Path) -> dict[str, Any]:
    """Do not accidentally record a parent repository's commit."""
    if not (root / ".git").exists():
        return {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def verify_artifacts(run_directory: Path) -> dict[str, Any]:
    manifest = read_json(run_directory / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "ard-manifest-v1":
        raise ValueError("unsupported or malformed manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("manifest requires artifact hashes")
    for name, expected_hash in artifacts.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path.name != name:
            raise ValueError(f"unsafe artifact name: {name}")
        target = run_directory / path
        if target.is_symlink() or not target.is_file() or file_hash(target) != expected_hash:
            raise ValueError(f"artifact missing or hash mismatch: {name}")
    return manifest
