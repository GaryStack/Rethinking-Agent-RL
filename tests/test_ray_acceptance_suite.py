"""Admission checks only, not model or GPU training evidence."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("visible", ["", "0,1", "GPU-not-a-numeric-index"])
def test_suite_rejects_missing_or_multiple_devices_before_output(tmp_path, visible):
    repo = Path(__file__).parents[1]
    output = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/run_ray_acceptance_suite.py"),
            "--model-path",
            str(tmp_path / "not-loaded"),
            "--verl-source",
            str(tmp_path / "not-imported"),
            "--output",
            str(output),
        ],
        env={**os.environ, "PYTHONPATH": str(repo / "src"), "CUDA_VISIBLE_DEVICES": visible},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "select one physical" in result.stderr
    assert not output.exists()


def test_suite_help_is_available_without_training_dependencies():
    repo = Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, str(repo / "scripts/run_ray_acceptance_suite.py"), "--help"],
        env={**os.environ, "PYTHONPATH": str(repo / "src")},
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert "--verl-source" in result.stdout
