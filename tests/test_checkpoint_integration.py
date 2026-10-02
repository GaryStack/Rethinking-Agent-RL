import copy
import json
from contextlib import nullcontext
from pathlib import Path

import pytest

from agent_rl_diagnostics.agent_config import parse_agent_config
from agent_rl_diagnostics.artifacts.io import file_hash, read_json, write_json
from agent_rl_diagnostics.cli import main
from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.evaluators.agent import run_agent_budget
from agent_rl_diagnostics.integrations.checkpoint import (
    audit_checkpoint_validation,
    checkpoint_identity,
    run_checkpoint_validation,
)
from agent_rl_diagnostics.integrations.tracking import DiagnosticTracker
from agent_rl_diagnostics.pools.frozen import build_pool, load_pool
from agent_rl_diagnostics.pools.witness import build_witness_pool

from .agent_helpers import FixtureModel, agent_config


def fake_checkpoint(root: Path):
    root.mkdir()
    for name in ("config.json", "tokenizer_config.json"):
        (root / name).write_text("{}")
    (root / "model.safetensors").write_bytes(b"authored-hash-only-fixture-not-real-weights")
    return root


def test_checkpoint_hash_changes_with_weights(tmp_path):
    root = fake_checkpoint(tmp_path / "checkpoint")
    before = checkpoint_identity(root)
    (root / "model.safetensors").write_bytes(b"different-fixture")
    assert checkpoint_identity(root)["fingerprint"] != before["fingerprint"]


def test_checkpoint_missing_shard_and_unsafe_names_fail(tmp_path):
    root = fake_checkpoint(tmp_path / "checkpoint")
    index = root / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"p": "missing.safetensors"}}))
    with pytest.raises(SchemaError, match="incomplete"):
        checkpoint_identity(root)
    index.write_text(json.dumps({"weight_map": {"p": "../escape.safetensors"}}))
    with pytest.raises(SchemaError, match="unsafe"):
        checkpoint_identity(root)


def test_dual_validation_binds_both_evaluators_to_checkpoint(tmp_path, monkeypatch):
    config = agent_config(tmp_path)
    source, pool = tmp_path / "source", tmp_path / "pool"
    run_agent_budget(config, source, FixtureModel())
    build_pool(
        [source], pool, kind="base_monitor", size=3, seed=7, base_model_id="base", max_per_task=3
    )
    root = fake_checkpoint(tmp_path / "checkpoint")
    raw = copy.deepcopy(config.raw)
    raw["split"] = "smoke"
    raw["model"] = {
        "backend": "transformers",
        "model_id": "trained-fixture",
        "revision": "placeholder",
        "path": str(root),
    }
    config = parse_agent_config(raw, base=tmp_path)

    def create(spec, sampling):
        model = FixtureModel()
        model.metadata = {"kind": "test_fixture", "revision": spec["revision"]}
        return model

    monkeypatch.setattr("agent_rl_diagnostics.integrations.checkpoint.create_model", create)
    monkeypatch.setattr(
        "agent_rl_diagnostics.integrations.checkpoint.preserve_runtime_rng",
        lambda device: nullcontext(),
    )
    output = tmp_path / "dual"
    point = run_checkpoint_validation(config, pool, root, output, checkpoint_step=3, samples=2)
    assert point["checkpoint_step"] == 3
    assert point["checkpoint_id"].startswith("hf-sha256:")
    for child in ("budget", "step"):
        assert (
            read_json(output / child / "manifest.json")["checkpoint_id"] == point["checkpoint_id"]
        )
    assert point["metrics"]["diag/health/incomplete"] == 0
    assert audit_checkpoint_validation(output, pool) == point
    report = tmp_path / "dual-report"
    assert (
        main(["report", "--input", str(output), "--pool", str(pool), "--output", str(report)]) == 0
    )
    assert read_json(report / "report-source.json")["source_summary_hash"] == file_hash(
        output / "metrics.json"
    )
    (root / "model.safetensors").write_bytes(b"changed after evaluation")
    with pytest.raises(SchemaError, match="weights changed"):
        audit_checkpoint_validation(output, pool, verify_weights=True)
    altered = read_json(output / "metrics.json")
    altered["metrics"]["diag/cost/actions"] += 1
    write_json(output / "metrics.json", altered)
    manifest = read_json(output / "manifest.json")
    manifest["artifacts"]["metrics.json"] = file_hash(output / "metrics.json")
    write_json(output / "manifest.json", manifest)
    with pytest.raises(SchemaError, match="metrics mismatch"):
        audit_checkpoint_validation(output, pool)


def point(step=1):
    return {
        "event_id": str(step),
        "checkpoint_step": step,
        "checkpoint_id": "hash" + str(step),
        "metrics": {"diag/score": 0.25, "diag/missing": None},
    }


def test_tracking_resume_idempotence_monotonicity_and_rank(tmp_path):
    tracker = DiagnosticTracker(tmp_path)
    assert tracker.log(point())
    assert not DiagnosticTracker(tmp_path).log(point())
    with pytest.raises(SchemaError, match="conflicting"):
        tracker.log({**point(), "checkpoint_id": "other"})
    assert tracker.log(point(2))
    with pytest.raises(SchemaError, match="monotonic"):
        tracker.log(point(0))
    with pytest.raises(SchemaError, match="rank 0"):
        DiagnosticTracker(tmp_path, rank=1)
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2


def test_witness_prefixes_are_explicitly_not_base_collected(tmp_path):
    config = agent_config(tmp_path)
    pool, errors = tmp_path / "witness", tmp_path / "errors"
    build_witness_pool(config, pool, budget=8, size=1)
    manifest, states, _ = load_pool(pool)
    assert manifest["selection"]["kind"] == "oracle_reference"
    assert states[0]["source"]["kind"] == "independent_exact_geometry_witness"
    from agent_rl_diagnostics.pools.errors import build_error_pool

    assert build_error_pool(pool, errors, require_recoverable=True)["pairs"] == 1
