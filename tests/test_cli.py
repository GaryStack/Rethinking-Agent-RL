import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from agent_rl_diagnostics.artifacts.io import file_hash, read_json
from agent_rl_diagnostics.cli import main
from agent_rl_diagnostics.config import load_config
from agent_rl_diagnostics.contracts import SchemaError, Trajectory
from agent_rl_diagnostics.evaluators.budget import generate_trajectories

ROOT = Path(__file__).resolve().parents[1]


def test_offline_end_to_end_and_recomputation(tmp_path, capsys):
    output = tmp_path / "run"
    assert main(["demo", "--offline", "--output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["evidence_kind"] == "demo_synthetic"
    assert result["trajectories"] == 1152
    assert (output / "report.html").is_file()
    manifest = read_json(output / "manifest.json")
    assert manifest["cost"]["network_requests"] == 0
    assert manifest["cost"]["gpu_required"] is False
    assert manifest["environment"]["official_benchmark"] is False
    assert main(["audit", "--input", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    report = tmp_path / "report"
    assert main(["report", "--input", str(output), "--output", str(report)]) == 0
    assert "Synthetic demonstration" in (report / "report.html").read_text()
    assert read_json(report / "summary.json") == read_json(output / "summary.json")


def test_repeatable_demo_and_no_overwrite(tmp_path, capsys):
    one, two = tmp_path / "one", tmp_path / "two"
    assert main(["demo", "--offline", "--output", str(one)]) == 0
    assert main(["demo", "--offline", "--output", str(two)]) == 0
    assert file_hash(one / "trajectories.jsonl") == file_hash(two / "trajectories.jsonl")
    assert file_hash(one / "summary.json") == file_hash(two / "summary.json")
    original = file_hash(one / "manifest.json")
    assert main(["demo", "--offline", "--output", str(one)]) == 2
    assert file_hash(one / "manifest.json") == original
    assert "already exists" in capsys.readouterr().err


def test_dry_run_does_not_create_output(tmp_path, capsys):
    output = tmp_path / "absent"
    assert main(["demo", "--offline", "--output", str(output), "--dry-run"]) == 0
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["action_upper_bound"] == 4608


def test_config_relative_output_does_not_depend_on_cwd(tmp_path, monkeypatch):
    config_path = ROOT / "configs/examples/frozenlake_budget.json"
    monkeypatch.chdir(tmp_path)
    config = load_config(config_path)
    assert config.output == ROOT / "runs/frozenlake-budget"


def test_heavy_modules_not_imported_and_network_never_used():
    code = """
import socket, sys, tempfile
def denied(*args, **kwargs):
    raise AssertionError('network used')
socket.socket = denied
from agent_rl_diagnostics.cli import main
with tempfile.TemporaryDirectory() as directory:
    assert main(['demo', '--offline', '--output', directory + '/run']) == 0
assert not any(name.split('.')[0] in {'torch', 'ray', 'vllm', 'verl', 'wandb', 'numpy'}
               for name in sys.modules)
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_audit_detects_modified_raw_file(tmp_path, capsys):
    output = tmp_path / "run"
    assert main(["demo", "--offline", "--output", str(output)]) == 0
    with (output / "trajectories.jsonl").open("a") as stream:
        stream.write("{}\n")
    assert main(["audit", "--input", str(output)]) == 2
    assert "hash mismatch" in capsys.readouterr().err


def test_audit_replays_environment_even_if_manifest_hashes_are_updated(tmp_path, capsys):
    from agent_rl_diagnostics.artifacts.io import write_json

    output = tmp_path / "run"
    assert main(["demo", "--offline", "--output", str(output)]) == 0
    path = output / "trajectories.jsonl"
    lines = path.read_text().splitlines()
    record = json.loads(lines[0])
    record["steps"][0]["state"] = 1  # Internally shaped like a valid record, wrong initial state.
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n")
    manifest = read_json(output / "manifest.json")
    manifest["artifacts"]["trajectories.jsonl"] = file_hash(path)
    write_json(output / "manifest.json", manifest)
    assert main(["audit", "--input", str(output)]) == 2
    assert "environment initial/current state" in capsys.readouterr().err


def test_schema_roundtrip_and_remaining_budget_guard():
    config = load_config(ROOT / "configs/examples/frozenlake_budget.json")
    trajectory = next(generate_trajectories(config, "run"))
    assert Trajectory.from_dict(json.loads(json.dumps(trajectory.to_dict()))) == trajectory
    with pytest.raises(SchemaError, match="remaining budget"):
        replace(trajectory, steps=(replace(trajectory.steps[0], remaining_budget=2),))


def test_infra_failure_cannot_claim_boolean_result():
    config = load_config(ROOT / "configs/examples/frozenlake_budget.json")
    trajectory = next(generate_trajectories(config, "run"))
    with pytest.raises(SchemaError, match="success=null"):
        replace(trajectory, status="infrastructure_failure", success=False)


def test_invalid_command_is_structured_error(capsys):
    assert main(["eval", "step"]) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "SchemaError"


def test_evaluation_does_not_consume_global_rng(tmp_path):
    import random

    random.seed(42)
    before = random.getstate()
    assert main(["demo", "--offline", "--output", str(tmp_path / "run")]) == 0
    assert random.getstate() == before
