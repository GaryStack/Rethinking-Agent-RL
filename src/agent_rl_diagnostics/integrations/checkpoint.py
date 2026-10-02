"""Immutable HF checkpoint dual validation, independent of trainer processes."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ..agent_config import AgentConfig, load_recorded_agent_config, parse_agent_config
from ..artifacts.io import (
    canonical_hash,
    file_hash,
    new_output_directory,
    read_json,
    verify_artifacts,
    write_json,
)
from ..contracts.schema import SchemaError, integer
from ..evaluators.agent import audit_agent_budget, read_records, run_agent_budget
from ..evaluators.audit import audit_fixed_or_recovery
from ..evaluators.fixed import run_fixed
from ..models import create_model
from ..pools.frozen import load_pool
from .runtime import preserve_runtime_rng


def checkpoint_identity(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    required = ["config.json", "tokenizer_config.json"]
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        index_data = read_json(index)
        if not isinstance(index_data, dict):
            raise SchemaError("checkpoint index must be an object")
        mapping = index_data.get("weight_map", {})
        if not isinstance(mapping, dict) or not mapping:
            raise SchemaError("checkpoint has no sharded weight map")
        if any(
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
            for name in mapping.values()
        ):
            raise SchemaError("unsafe checkpoint shard name")
        shards = set(mapping.values())
        required += [index.name, *sorted(shards)]
    else:
        required.append("model.safetensors")
    optional = (
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "generation_config.json",
    )
    required += [name for name in optional if (directory / name).is_file()]
    if any(not (directory / name).is_file() for name in required):
        raise SchemaError(
            "incomplete HF checkpoint; requires config, tokenizer metadata and all safetensors"
        )
    hashes = {name: file_hash(directory / name) for name in sorted(required)}
    return {
        "schema_version": "ard-checkpoint-v1",
        "path": str(directory),
        "files": hashes,
        "fingerprint": canonical_hash(hashes),
        "full_weight_hash_verified": True,
    }


def diagnostic_metrics(budget: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    namespace = "diag/" + budget["environment"]
    result: dict[str, Any] = {}
    for cell in budget["cells"]:
        key = f"{namespace}/multi/pass_k{cell['k']}_t{cell['total_budget']}"
        result[key] = cell["pass_at_k"]
        result[key + "_missing_slots"] = cell["missing_slots"]
        result[key + "_infra_failures"] = cell["infrastructure_failures"]
    scorer_key = (
        "q_star_mean"
        if budget["environment"] == "frozenlake_deterministic"
        else "test_fraction_mean"
    )
    result[f"{namespace}/step/{scorer_key}"] = step["r_single"]
    result[f"{namespace}/step/valid_samples"] = step["valid_samples"]
    result[f"{namespace}/step/expected_samples"] = step["state_count"] * step["samples_per_state"]
    result["diag/health/incomplete"] = int(
        not step["complete"] or any(c["status"] != "complete" for c in budget["cells"])
    )
    return result


def _run_checkpoint_validation(
    config: AgentConfig,
    pool: Path,
    checkpoint: Path,
    output: Path,
    *,
    checkpoint_step: int,
    samples: int = 8,
) -> dict[str, Any]:
    integer(checkpoint_step, "checkpoint_step")
    integer(samples, "samples", 1)
    if config.raw["split"] not in {"validation", "smoke"}:
        raise SchemaError("training diagnostics require an explicit validation or smoke split")
    if config.model["backend"] != "transformers":
        raise SchemaError(
            "checkpoint validation requires local Transformers, not an unattested endpoint"
        )
    pool_manifest, _, _ = load_pool(pool)
    if pool_manifest["selection"].get("kind") != "base_monitor":
        raise SchemaError("training dynamics require a Base-only monitor pool")
    before = checkpoint_identity(checkpoint)
    raw = copy.deepcopy(config.raw)
    raw.pop("output", None)
    raw["model"].update(
        path=str(checkpoint.resolve()), revision="hf-sha256:" + before["fingerprint"]
    )
    bound = parse_agent_config(raw, base=Path.cwd())
    job_id = canonical_hash(
        [
            checkpoint_step,
            before["fingerprint"],
            bound.config_hash,
            pool_manifest["pool_hash"],
            samples,
        ]
    )
    with new_output_directory(output) as staging:
        backend = create_model(bound.model, bound.sampling)
        if backend.metadata.get("revision") != bound.checkpoint_id:
            raise SchemaError("loaded backend is not bound to this checkpoint")
        run_agent_budget(bound, staging / "budget", backend)
        run_fixed(bound, pool, staging / "step", samples=samples, model=backend)
        budget = audit_agent_budget(
            staging / "budget", verify_artifacts(staging / "budget"), replay_code=True
        )
        step = audit_fixed_or_recovery(
            staging / "step", verify_artifacts(staging / "step"), pool, replay_code=True
        )
        if checkpoint_identity(checkpoint)["fingerprint"] != before["fingerprint"]:
            raise SchemaError("checkpoint changed while validation was running")
        metrics = diagnostic_metrics(budget, step)
        actions, tokens, invalid = 0, 0, 0
        generations = [
            s for r in read_records(staging / "budget" / "trajectories.jsonl") for s in r["steps"]
        ]
        generations += [
            r["step"] for r in read_records(staging / "step" / "actions.jsonl") if r["step"]
        ]
        for record in generations:
            actions += 1
            invalid += record["action"]["kind"] == "invalid"
            tokens += record["generation"].get("usage", {}).get("completion_tokens", 0)
        metrics.update(
            {
                "diag/cost/actions": actions,
                "diag/cost/completion_tokens": tokens,
                "diag/health/invalid_actions": invalid,
            }
        )
        point = {
            "schema_version": "ard-diagnostic-point-v1",
            "event_id": job_id,
            "checkpoint_step": checkpoint_step,
            "checkpoint_id": bound.checkpoint_id,
            "pool_hash": pool_manifest["pool_hash"],
            "config_hash": bound.config_hash,
            "split": bound.raw["split"],
            "heldout_task_id_check": pool_manifest["selection"]["heldout_task_id_check"],
            "metrics": metrics,
        }
        write_json(staging / "checkpoint.json", before)
        write_json(staging / "metrics.json", point)
        write_json(
            staging / "manifest.json",
            {
                "schema_version": "ard-manifest-v1",
                "kind": "checkpoint_validation",
                "status": "complete" if not metrics["diag/health/incomplete"] else "incomplete",
                "event_id": job_id,
                "checkpoint_step": checkpoint_step,
                "checkpoint_id": bound.checkpoint_id,
                "pool_hash": pool_manifest["pool_hash"],
                "artifacts": {
                    name: file_hash(staging / name) for name in ("checkpoint.json", "metrics.json")
                },
                "child_manifest_hashes": {
                    name: file_hash(staging / name / "manifest.json") for name in ("budget", "step")
                },
            },
        )
    return point


def run_checkpoint_validation(
    config: AgentConfig,
    pool: Path,
    checkpoint: Path,
    output: Path,
    *,
    checkpoint_step: int,
    samples: int = 8,
) -> dict[str, Any]:
    # Loading a fresh model can consume RNG too: protect the entire lifecycle.
    with preserve_runtime_rng(config.model.get("device", "cpu")):
        return _run_checkpoint_validation(
            config, pool, checkpoint, output, checkpoint_step=checkpoint_step, samples=samples
        )


def audit_checkpoint_validation(
    directory: Path, pool: Path, *, replay_code: bool = False, verify_weights: bool = False
) -> dict[str, Any]:
    """Recompute dual validation from raw records; optionally rehash local weights."""
    manifest = verify_artifacts(directory)
    if manifest.get("kind") != "checkpoint_validation":
        raise SchemaError("not a checkpoint validation")
    identity = read_json(directory / "checkpoint.json")
    point = read_json(directory / "metrics.json")
    digest = canonical_hash(identity["files"])
    if identity["fingerprint"] != digest or point["checkpoint_id"] != "hf-sha256:" + digest:
        raise SchemaError("checkpoint identity mismatch")
    if verify_weights and checkpoint_identity(Path(identity["path"]))["fingerprint"] != digest:
        raise SchemaError("checkpoint weights changed")
    pool_manifest, _, _ = load_pool(pool)
    if point["pool_hash"] != pool_manifest["pool_hash"]:
        raise SchemaError("checkpoint pool mismatch")
    children = {}
    for name in ("budget", "step"):
        child = directory / name
        if file_hash(child / "manifest.json") != manifest["child_manifest_hashes"][name]:
            raise SchemaError("child manifest changed")
        children[name] = verify_artifacts(child)
        config = load_recorded_agent_config(child / "config.json")
        if (
            config.config_hash != point["config_hash"]
            or config.checkpoint_id != point["checkpoint_id"]
        ):
            raise SchemaError("child checkpoint/config mismatch")
    budget = audit_agent_budget(directory / "budget", children["budget"], replay_code=replay_code)
    step = audit_fixed_or_recovery(
        directory / "step", children["step"], pool, replay_code=replay_code
    )
    expected_event = canonical_hash(
        [
            point["checkpoint_step"],
            digest,
            point["config_hash"],
            point["pool_hash"],
            step["samples_per_state"],
        ]
    )
    if (
        point["event_id"] != expected_event
        or manifest["event_id"] != expected_event
        or manifest["checkpoint_id"] != point["checkpoint_id"]
        or manifest["checkpoint_step"] != point["checkpoint_step"]
    ):
        raise SchemaError("checkpoint event identity mismatch")
    metrics = diagnostic_metrics(budget, step)
    generations = [
        s for r in read_records(directory / "budget" / "trajectories.jsonl") for s in r["steps"]
    ]
    generations += [
        r["step"] for r in read_records(directory / "step" / "actions.jsonl") if r["step"]
    ]
    metrics.update(
        {
            "diag/cost/actions": len(generations),
            "diag/cost/completion_tokens": sum(
                s["generation"].get("usage", {}).get("completion_tokens", 0) for s in generations
            ),
            "diag/health/invalid_actions": sum(
                s["action"]["kind"] == "invalid" for s in generations
            ),
        }
    )
    if metrics != point["metrics"]:
        raise SchemaError("checkpoint diagnostic metrics mismatch")
    return point
