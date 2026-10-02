import os
import subprocess
import sys
from pathlib import Path


def test_optional_ray_smoke_help_does_not_require_training_libraries():
    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    result = subprocess.run(
        [sys.executable, str(root / "scripts/run_verl_ray_smoke.py"), "--help"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert "--train-budget" in result.stdout
    assert "--group-size" in result.stdout
    assert "--preflight-only" in result.stdout


def test_process_options_reject_missing_coefficient_or_policy_before_optional_imports():
    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    for extra in (["--process-coefficient", "0.5"], ["--process-failure-policy", "error"]):
        result = subprocess.run(
            [
                sys.executable,
                str(root / "scripts/run_verl_ray_smoke.py"),
                "--model-path",
                "unused",
                "--verl-source",
                "unused",
                "--monitor-config",
                "unused",
                "--pool",
                "unused",
                "--output",
                "unused",
                *extra,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 2
        assert "supplied together" in result.stderr
