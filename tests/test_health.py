"""Real bounded Python probes; these do not establish GPU training acceptance."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.integrations.health import run_probe


def test_probe_success_and_failure_capture_actual_exit():
    result = run_probe("success", [sys.executable, "-c", "print('ready')"])
    assert result["status"] == "passed"
    assert result["returncode"] == 0 and result["stdout"].strip() == "ready"
    result = run_probe("failure", [sys.executable, "-c", "raise SystemExit(7)"])
    assert result["status"] == "failed" and result["returncode"] == 7


def test_probe_timeout_does_not_hang():
    result = run_probe(
        "blocked", [sys.executable, "-c", "import time; time.sleep(60)"], timeout_seconds=0.2
    )
    assert result["status"] == "timeout" and result["process_reaped"]
    assert result["elapsed_seconds"] < 5


def test_probe_missing_executable_is_infrastructure_failure():
    result = run_probe("absent", ["/does-not-exist/ard-python"])
    assert result["status"] == "launch_error" and result["returncode"] is None


@pytest.mark.parametrize("timeout", [True, 0, float("nan"), float("inf"), 301])
def test_invalid_timeouts_rejected(timeout):
    with pytest.raises(SchemaError, match="timeout"):
        run_probe("invalid", [sys.executable], timeout_seconds=timeout)


def test_cpu_health_cli_is_immutable_and_does_not_claim_training(tmp_path):
    script = Path(__file__).parents[1] / "scripts/check_runtime.py"
    output = tmp_path / "health.json"
    environment = dict(os.environ, PYTHONPATH=str(script.parent.parent / "src"))
    command = [sys.executable, str(script), "--output", str(output)]
    subprocess.run(command, env=environment, capture_output=True, check=True, timeout=15)
    report = json.loads(output.read_text())
    assert report["requested_checks_passed"]
    assert not report["inference_executed"] and not report["training_executed"]
    assert {p["name"] for p in report["probes"]} == {"python", "package_metadata"}
    before = output.read_bytes()
    repeated = subprocess.run(command, env=environment, capture_output=True, timeout=15)
    assert repeated.returncode == 2 and output.read_bytes() == before
