import shutil

import pytest

from agent_rl_diagnostics.artifacts.io import read_json
from agent_rl_diagnostics.cli import _audit_run, main


def test_installed_walkthrough_relocates_and_detects_raw_tampering(tmp_path, capsys):
    original = tmp_path / "original"
    assert main(["toolkit-demo", "--output", str(original)]) == 0
    assert read_json(original / "walkthrough.json")["trained_models"] is False
    moved = tmp_path / "moved"
    shutil.move(original, moved)
    for kind, pool in (
        ("budget", None),
        ("step", "pool"),
        ("error-step", "errors"),
        ("recovery", "errors"),
        ("monitor-step", "base-monitor"),
    ):
        for label in ("Base", "RL"):
            _audit_run(moved / f"{kind}-{label}", moved / pool if pool else None)
        assert (moved / f"compare-{kind}" / "report.html").is_file()
    raw = moved / "step-Base" / "actions.jsonl"
    raw.write_text(raw.read_text() + "{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        _audit_run(moved / "step-Base", moved / "pool")
    assert main(["toolkit-demo", "--output", str(moved)]) == 2
