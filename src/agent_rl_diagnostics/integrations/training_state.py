"""Optional read-only actor-boundary fingerprints; comparison stays stdlib-only.

These proofs cover observed actor state, not complete trainer/distributed parity.
"""

import hashlib
import math
import pickle
import random
from pathlib import Path

from ..artifacts.io import canonical_hash, read_json
from ..contracts.schema import SchemaError

STATE_KEYS = (
    "parameters",
    "buffers",
    "optimizer",
    "scheduler",
    "training_modes",
    "batch_tensors",
    "python_rng",
    "numpy_rng",
    "torch_cpu_rng",
    "torch_cuda_rng",
)


def capture_training_state(module, optimizer, scheduler, batch, *, cuda_device=None):
    import numpy as np
    import torch

    def tensor(value, *, allow_small_device_copy=False):
        if value.layout != torch.strided:
            raise SchemaError("state capture requires ordinary strided tensors")
        if value.device.type != "cpu":
            if not allow_small_device_copy or value.numel() * value.element_size() > 64 * 1024**2:
                raise SchemaError(
                    "actor parameters/optimizer/batch must be CPU-offloaded for capture"
                )
        view = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        raw = memoryview(view).cast("B")
        digest = hashlib.sha256()
        for offset in range(0, len(raw), 1024**2):
            digest.update(raw[offset : offset + 1024**2])
        return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": digest.hexdigest()}

    def tree(value):
        # The pinned VeRL passes OmegaConf betas directly into AdamW. Preserve
        # this container's type and resolved contents; do not mutate live state.
        if type(value).__module__.startswith("omegaconf."):
            from omegaconf import DictConfig, ListConfig, OmegaConf

            if isinstance(value, (DictConfig, ListConfig)):
                return {
                    type(value).__name__: tree(
                        OmegaConf.to_container(value, resolve=True, throw_on_missing=True)
                    )
                }
        if torch.is_tensor(value):
            return {"tensor": tensor(value)}
        if isinstance(value, dict):
            if any(type(k) not in (int, str) for k in value):
                raise SchemaError("unsupported state dictionary key")
            return {f"{type(k).__name__}:{k}": tree(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return {type(value).__name__: [tree(v) for v in value]}
        if value is None or type(value) in (bool, int, str):
            return value
        if type(value) is float and math.isfinite(value):
            return value
        raise SchemaError(
            "unsupported or nonfinite optimizer/scheduler state: "
            f"{type(value).__module__}.{type(value).__name__}"
        )

    if cuda_device is not None and not torch.cuda.is_initialized():
        raise SchemaError("CUDA RNG capture requires an already initialized actor")
    identity = {
        "parameters": canonical_hash({name: tensor(p) for name, p in module.named_parameters()}),
        "buffers": canonical_hash(
            {name: tensor(b, allow_small_device_copy=True) for name, b in module.named_buffers()}
        ),
        "optimizer": canonical_hash(tree(optimizer.state_dict())),
        "scheduler": canonical_hash(tree(scheduler.state_dict()))
        if scheduler is not None
        else None,
        "training_modes": canonical_hash(
            {name: child.training for name, child in module.named_modules()}
        ),
        "batch_tensors": canonical_hash({name: tensor(value) for name, value in batch.items()}),
        "python_rng": hashlib.sha256(pickle.dumps(random.getstate(), protocol=5)).hexdigest(),
        "numpy_rng": hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=5)).hexdigest(),
        "torch_cpu_rng": tensor(torch.get_rng_state())["sha256"],
        "torch_cuda_rng": tensor(torch.cuda.get_rng_state(cuda_device))["sha256"]
        if cuda_device is not None
        else None,
    }
    return identity


def compare_training_states(left: Path, right: Path):
    if left.resolve() == right.resolve():
        raise SchemaError("training state comparison requires distinct directories")

    def load(directory):
        paths = sorted(directory.glob("update-*.json"))
        records = [read_json(path) for path in paths]
        if not records:
            raise SchemaError("no completed actor state proofs")
        for expected, record in enumerate(records, start=1):
            if (
                record.get("schema_version") != "ard-training-state-v1"
                or type(record.get("update")) is not int
                or record["update"] != expected
                or paths[expected - 1].name != f"update-{expected:06d}.json"
            ):
                raise SchemaError("invalid or noncontiguous actor state proof")
            if record.get("scope") not in (
                "single_rank_actor_worker_boundary",
                "cpu_optimizer_fixture",
            ):
                raise SchemaError("unsupported actor state proof scope")
            versions = record.get("framework_versions", {})
            if set(versions) != {"torch", "numpy"} or not all(
                isinstance(v, str) and v for v in versions.values()
            ):
                raise SchemaError("missing actor framework versions")
            for phase in ("before", "after"):
                state = record.get(phase, {})
                if set(state) != set(STATE_KEYS):
                    raise SchemaError("incomplete actor state fingerprint")
                for key, digest in state.items():
                    if digest is None and key in ("scheduler", "torch_cuda_rng"):
                        continue
                    if (
                        not isinstance(digest, str)
                        or len(digest) != 64
                        or any(c not in "0123456789abcdef" for c in digest)
                    ):
                        raise SchemaError("invalid actor state fingerprint")
                if (
                    record["scope"] == "single_rank_actor_worker_boundary"
                    and state["torch_cuda_rng"] is None
                ):
                    raise SchemaError("actual actor CUDA RNG was not observed")
            norms = record.get("actor_grad_norms", [])
            if (
                not isinstance(norms, list)
                or not norms
                or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in norms)
            ):
                raise SchemaError("missing/nonfinite/zero actor gradients")
            if record["before"]["parameters"] == record["after"]["parameters"]:
                raise SchemaError("actor parameter contents did not change")
            if record["before"]["batch_tensors"] != record["after"]["batch_tensors"]:
                raise SchemaError("actor input tensors mutated during the update")
        return records

    first, second = load(left), load(right)
    if len(first) != len(second):
        raise SchemaError("actor update counts differ")
    differences = []
    for a, b in zip(first, second, strict=True):
        fields = [
            key
            for key in ("scope", "framework_versions", "actor_grad_norms")
            if a.get(key) != b.get(key)
        ]
        fields += [
            f"{phase}/{key}"
            for phase in ("before", "after")
            for key in STATE_KEYS
            if a[phase][key] != b[phase][key]
        ]
        if fields:
            differences.append({"update": a["update"], "fields": fields})
    return {
        "schema_version": "ard-training-state-comparison-v1",
        "status": "matched_observed_actor_states" if not differences else "different",
        "updates": len(first),
        "scope": first[0].get("scope"),
        "differences": differences,
        "cuda_rng_observed": all(
            record[phase]["torch_cuda_rng"] is not None
            for record in first + second
            for phase in ("before", "after")
        ),
        "full_trainer_parity_verified": False,
        "distributed_parity_verified": False,
    }
