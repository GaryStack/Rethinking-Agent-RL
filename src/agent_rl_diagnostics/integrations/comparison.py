"""Audited exact comparison of checkpoint validation artifacts, with no model loading."""

from pathlib import Path

from ..artifacts.io import file_hash
from ..contracts.schema import SchemaError
from .checkpoint import audit_checkpoint_validation


def compare_checkpoint_validations(left: Path, right: Path, pool: Path, *, verify_weights=False):
    if left.resolve() == right.resolve():
        raise SchemaError("comparison requires two distinct validation directories")
    points = [
        audit_checkpoint_validation(directory, pool, verify_weights=verify_weights)
        for directory in (left, right)
    ]
    if any(p["metrics"]["diag/health/incomplete"] for p in points):
        raise SchemaError("cannot establish parity from incomplete diagnostic points")
    mismatches = {
        key: {"left": points[0].get(key), "right": points[1].get(key)}
        for key in sorted(set(points[0]) | set(points[1]))
        if points[0].get(key) != points[1].get(key)
    }
    hashes = {
        name: {"left": file_hash(left / name), "right": file_hash(right / name)}
        for name in ("budget/trajectories.jsonl", "step/actions.jsonl")
    }
    raw_equal = {name: value["left"] == value["right"] for name, value in hashes.items()}
    matched = not mismatches and all(raw_equal.values())
    return {
        "schema_version": "ard-checkpoint-comparison-v1",
        "status": "matched" if matched else "different",
        "left": str(left.resolve()),
        "right": str(right.resolve()),
        "checkpoint_step": points[0]["checkpoint_step"],
        "checkpoint_id": points[0]["checkpoint_id"],
        "point_mismatches": mismatches,
        "raw_hashes": hashes,
        "raw_records_equal": raw_equal,
        "weights_rehashed": verify_weights,
        "training_state_parity_verified": False,
    }
