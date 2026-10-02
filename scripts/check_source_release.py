"""Check selected public source files, local document links and version consistency."""

import argparse
import ast
import json
import re
import tomllib
from pathlib import Path

PUBLIC_DIRS = (
    ".github",
    "src",
    "tests",
    "scripts",
    "configs",
    "docs",
    "paper",
    "examples",
    "requirements",
)
PUBLIC_FILES = (
    ".gitignore",
    "AGENTS.md",
    "README.md",
    "README.zh-CN.md",
    "CHANGELOG.md",
    "STATUS.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "NOTICE",
    "MANIFEST.in",
    "pyproject.toml",
)
SKIP = {"__pycache__", ".ruff_cache", ".pytest_cache"}


def check(root):
    root = root.resolve()
    errors, paths = [], []
    for name in PUBLIC_FILES:
        path = root / name
        if not path.is_file():
            errors.append(f"missing public file: {name}")
        else:
            paths.append(path)
    for name in PUBLIC_DIRS:
        directory = root / name
        if not directory.is_dir() or directory.is_symlink():
            errors.append(f"missing or linked public directory: {name}")
            continue
        for path in directory.rglob("*"):
            if SKIP.intersection(path.parts) or any(p.endswith(".egg-info") for p in path.parts):
                continue
            if path.is_symlink():
                errors.append(f"linked public path: {path.relative_to(root)}")
            elif path.is_file():
                paths.append(path)
    for path in paths:
        name = str(path.relative_to(root))
        if path.is_symlink() or path.stat().st_size > 2_000_000:
            errors.append(f"linked or large public file: {name}")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError:
            errors.append(f"non-text public file: {name}")
            continue
        if re.search(r"/netdisk/[A-Za-z0-9_.-]+|/home/[A-Za-z0-9_.-]+/", content):
            errors.append(f"personal absolute path: {name}")
        if re.search(
            r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{32,}",
            content,
        ):
            errors.append(f"credential-like content: {name}")
        if path.suffix == ".md":
            prose = re.sub(r"```.*?```", "", content, flags=re.S)
            for match in re.finditer(r"\[[^\]\n]+\]\(([^)\n]+)\)", prose):
                target = match[1].split("#", 1)[0].strip("<>")
                if not target or re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:|//", target):
                    continue
                destination = (path.parent / target).resolve()
                if not destination.is_relative_to(root) or not destination.exists():
                    errors.append(f"missing or external local link: {name} -> {target}")
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    tree = ast.parse((root / "src/agent_rl_diagnostics/__init__.py").read_text())
    runtime = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets)
    )
    citation = (root / "CITATION.cff").read_text()
    citation_version = re.search(r"^version:\s*[\"']?([^\"'\s]+)", citation, re.M)
    if runtime != version or not citation_version or citation_version[1] != version:
        errors.append("project/runtime/citation version mismatch")
    return {
        "status": "verified_preview" if not errors else "failed",
        "version": version,
        "public_files_checked": len(paths),
        "errors": errors,
        "citation_uses_contributors_placeholder": "Agent RL Diagnostics contributors" in citation,
        "publishes_artifacts": False,
        "scope": "selected_text_payload_paths_local_links_version_known_secret_patterns",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = check(args.project)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not result["errors"] else 1)
