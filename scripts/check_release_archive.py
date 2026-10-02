"""Inspect the actual wheel/sdist payload without opening any private research tree."""

import argparse
import json
import re
import stat
import tarfile
import zipfile
from pathlib import Path


def check(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            entries = []
            for entry in archive.infolist():
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError("release archive must not contain links")
                if entry.file_size > 2_000_000:
                    raise ValueError(f"large archive member: {entry.filename}")
                if not entry.is_dir():
                    entries.append((entry.filename, archive.read(entry)))
    else:
        with tarfile.open(path) as archive:
            entries = []
            for entry in archive.getmembers():
                if entry.issym() or entry.islnk():
                    raise ValueError("release archive must not contain links")
                if entry.size > 2_000_000:
                    raise ValueError(f"large archive member: {entry.name}")
                if entry.isfile():
                    entries.append((entry.name, archive.extractfile(entry).read()))
    forbidden = {"runs", "wandb", ".venv", ".git", "__pycache__", "data", ".env", ".aws", ".ssh"}
    for name, content in entries:
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or forbidden.intersection(Path(name).parts)
        ):
            raise ValueError(f"forbidden archive member: {name}")
        if (
            name.endswith((".safetensors", ".bin", ".pt", ".pth", ".pyc", ".log"))
            or len(content) > 2_000_000
        ):
            raise ValueError(f"binary/large runtime artifact: {name}")
        text = content.decode("utf-8", errors="replace")
        if re.search(
            r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{32,}",
            text,
        ):
            raise ValueError(f"credential-like content in: {name}")
        if (
            "agent_rl_diagnostics" in Path(name).parts
            and name.endswith(".py")
            and ("/netdisk/" in text or re.search(r"/home/[^/\s\"']+/", text))
        ):
            raise ValueError(f"hardcoded research path in package code: {name}")
    return {
        "status": "verified",
        "archive": path.name,
        "files": len(entries),
        "scope": "payload_paths_sizes_known_secret_patterns",
        "manual_review_still_required": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    print(json.dumps(check(parser.parse_args().archive), indent=2))
