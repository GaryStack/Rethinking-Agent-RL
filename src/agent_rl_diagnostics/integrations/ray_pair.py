"""Compare observed one-rank actor state and complete authored training rollouts.

This intentionally does not certify controller/server RNG or full-trainer parity.
"""

from pathlib import Path

from ..artifacts.io import canonical_hash, read_json
from ..contracts.schema import SchemaError, integer
from ..pools.frozen import load_pool
from .grid_fixture import TRAIN_GRID_TASKS
from .grid_trace import audit_grid_trace
from .training_state import compare_training_states


def compare_ray_pair(left: Path, right: Path):
    if left.resolve() == right.resolve():
        raise SchemaError("distinct runs required")
    runs = [read_json(root / "trainer-returned.json") for root in (left, right)]
    if {run.get("checkpoint_diagnostics_enabled") for run in runs} != {False, True}:
        raise SchemaError("one diagnostics-off and one diagnostics-on run required")
    for run in runs:
        if type(run.get("checkpoint_diagnostics_enabled")) is not bool:
            raise SchemaError("explicit boolean diagnostics mode required")
        if run.get("actor_state_capture_enabled") is not True or run.get(
            "budgetmix_enabled", False
        ):
            raise SchemaError("captured fixed-budget runs required for this paired acceptance")
        if run.get("scope") != "actual_single_rank_RayPPOTrainer_vllm_FSDP":
            raise SchemaError("unsupported trainer pair scope")
        integer(run.get("steps_requested"), "steps", 2)
        integer(run.get("samples_per_training_task"), "group_size", 2)
    for field in (
        "steps_requested",
        "train_action_budget",
        "samples_per_training_task",
        "algorithm",
        "actor_initialization_seed",
    ):
        if runs[0].get(field) != runs[1].get(field):
            raise SchemaError("paired training settings differ")
    steps, size = runs[0]["steps_requested"], runs[0]["samples_per_training_task"]
    seed = runs[0].get("actor_initialization_seed")
    seeded = seed is not None
    if seeded:
        integer(seed, "actor_initialization_seed")
        if seed > 2**32 - 1:
            raise SchemaError("actor initialization seed out of range")
        expected = {
            "schema_version": "ard-worker-initialization-v1",
            "protocol": "one_time_before_native_init_model",
            "seed": seed,
            "cuda_device": 0,
            "rng_domains": ["python", "numpy", "torch_cpu", "torch_cuda_local"],
            "full_trainer_parity_verified": False,
        }
        for root in (left, right):
            if read_json(root / "state-proofs/initialization.json") != expected:
                raise SchemaError("missing or invalid one-time actor initialization record")

    def normalized_config(root):
        stored = read_json(root / "verl-config.json")
        checkpoint_dir = Path(stored["trainer"]["default_local_dir"])
        if not checkpoint_dir.is_absolute() or checkpoint_dir.name != "checkpoints":
            raise SchemaError("recorded trainer checkpoint namespace is invalid")
        recorded_root = checkpoint_dir.parent
        if stored["trainer"]["experiment_name"] != recorded_root.name:
            raise SchemaError("recorded trainer run name and output namespace differ")

        def replace(value):
            if isinstance(value, dict):
                return {k: replace(v) for k, v in value.items()}
            if isinstance(value, list):
                return [replace(v) for v in value]
            if isinstance(value, str):
                return value.replace(str(recorded_root), "$RUN")
            return value

        config = replace(stored)
        if seeded and config.get("ard_actor_initialization") != {
            "protocol": "one_time_before_native_init_model",
            "seed": seed,
        }:
            raise SchemaError("resolved actor initialization protocol differs from run record")
        trace = config.get("actor_rollout_ref", {}).get("rollout", {}).get("trace", {})
        if "experiment_name" in trace:
            if trace["experiment_name"] != config["trainer"]["experiment_name"]:
                raise SchemaError("rollout trace name must equal its owning trainer run name")
            trace["experiment_name"] = "$RUN_NAME"
        config["trainer"]["experiment_name"] = "$RUN_NAME"
        return config

    if normalized_config(left) != normalized_config(right):
        raise SchemaError("resolved configurations differ beyond output paths/run names")
    for filename in ("source-lock.json", "environment.json", "monitor-config.json"):
        if read_json(left / filename) != read_json(right / filename):
            raise SchemaError("paired source/environment/monitor identities differ")
    left_pool, _, _ = load_pool(left / "pool")
    right_pool, _, _ = load_pool(right / "pool")
    if left_pool != right_pool:
        raise SchemaError("paired frozen pool identities differ")
    state_report = compare_training_states(left / "state-proofs", right / "state-proofs")
    if (
        state_report["scope"] != "single_rank_actor_worker_boundary"
        or state_report["updates"] != steps
    ):
        raise SchemaError("incomplete actual actor state capture")
    tasks = {t["task_id"]: t for t in TRAIN_GRID_TASKS}

    def training(root):
        grouped = {(version, task): [] for version in range(steps) for task in tasks}
        seen = set()
        for path in sorted((root / "rollouts/grid-traces").glob("*.json")):
            trace = read_json(path)
            if trace.get("task", {}).get("task_id") not in tasks:
                continue  # native validation is a separately scoped observation
            audit_grid_trace(trace)
            task = trace["task"]["task_id"]
            request = trace.get("request_id")
            if not isinstance(request, str) or not request or request in seen:
                raise SchemaError("missing/duplicate actual training trace identity")
            seen.add(request)
            if trace["task"] != tasks[task] or trace["budget"] != runs[0]["train_action_budget"]:
                raise SchemaError("training map or action budget differs")
            key = (trace["weight_version"], task)
            if key not in grouped:
                raise SchemaError("unexpected training weight version")
            normalized = {k: v for k, v in trace.items() if k != "request_id"}
            grouped[key].append(canonical_hash(normalized))
        if any(len(values) != size for values in grouped.values()):
            raise SchemaError("missing or duplicate training completion slots")
        return {f"{version}/{task}": sorted(values) for (version, task), values in grouped.items()}

    a, b = training(left), training(right)
    differing = [key for key in a if a[key] != b[key]]
    rng_match = not any(
        field.endswith("_rng")
        for difference in state_report["differences"]
        for field in difference["fields"]
    )
    return {
        "schema_version": "ard-ray-pair-comparison-v1",
        "status": "matched_observed_actor_and_rollouts"
        if not differing and not state_report["differences"]
        else "different",
        "scope": "single_rank_actor_boundaries_and_complete_training_rollouts",
        "actor_state": state_report,
        "training_groups": len(a),
        "training_completions_per_run": len(a) * size,
        "left_training_hash": canonical_hash(a),
        "right_training_hash": canonical_hash(b),
        "differing_training_groups": differing,
        "request_ids_excluded": True,
        "actor_rng_initialization_recorded": seeded,
        "observed_actor_rng_match": rng_match,
        "controller_rng_verified": False,
        "rollout_server_rng_verified": False,
        "full_trainer_parity_verified": False,
        "distributed_parity_verified": False,
        "paper_reproduction": False,
    }
