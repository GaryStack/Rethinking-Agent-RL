"""Optional Inspect reader feeding the stdlib recorded-trace workflow."""

import hashlib
import io
import json
import tempfile
from pathlib import Path

from ..contracts.schema import SchemaError, integer, text
from ..external_traces import import_traces, normalize, validate_design


def _binary(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value in {"C", "I"}:
        return value == "C"
    if type(value) in (int, float) and value in (0, 1):
        return value == 1
    raise SchemaError(
        "Inspect scorer must supply binary C/I, boolean, or numeric 0/1; "
        "partial and composite values need explicit preprocessing"
    )


def convert_inspect(data, *, scorer, design, source_sha256, inspect_version):
    """Convert an SDK JSON dump, retaining every exported sample and evaluation spec."""
    design = validate_design(design)
    text(scorer, "scorer")
    if not isinstance(data, dict) or not isinstance(data.get("eval"), dict):
        raise SchemaError("Inspect log requires an evaluation specification")
    samples = data.get("samples") or []
    if not isinstance(samples, list) or any(not isinstance(s, dict) for s in samples):
        raise SchemaError("Inspect samples must be a list of objects")
    spec = data["eval"]
    if data.get("status") not in {"success", "error", "cancelled"}:
        raise SchemaError("Inspect log must have a finished or interrupted status")
    configured = spec.get("config", {}).get("epochs")
    if configured is not None and configured != design["samples_per_task"]:
        raise SchemaError("Inspect epochs disagree with declared samples_per_task")
    declared = {s.get("name") for s in (spec.get("scorers") or [])}
    available = {name for s in samples for name in (s.get("scores") or {})}
    if scorer not in declared | available:
        raise SchemaError(f"Inspect scorer {scorer!r} not found; available: {sorted(available)}")
    rows, id_types = [], {}
    for sample in samples:
        identity = sample.get("id")
        if type(identity) not in (int, str):
            raise SchemaError("Inspect sample IDs must be integers or strings")
        task_id = str(identity)
        if task_id in id_types and id_types[task_id] is not type(identity):
            raise SchemaError("Inspect integer/string IDs collide after conversion")
        id_types[task_id] = type(identity)
        epoch = sample.get("epoch")
        integer(epoch, "Inspect epoch", 1)
        scores = sample.get("scores") or {}
        success = None
        if sample.get("error") is not None:
            status = "infrastructure_failure"
        elif (
            sample.get("invalidation") is not None
            or scorer not in scores
            or scores[scorer].get("value") is None
            or scores[scorer].get("value") == "U"
        ):
            status = "missing"
        else:
            success = _binary(scores[scorer]["value"])
            limit_type = (sample.get("limit") or {}).get("type")
            budget_limits = {"context", "time", "working", "message", "token", "turn", "cost"}
            status = (
                "budget_exhausted" if limit_type in budget_limits and not success else "completed"
            )
        usage = sample.get("model_usage") or {}
        tokens = sum(u["total_tokens"] for u in usage.values()) if usage else None
        # Inspect total_cost is USD; never reinterpret it as another currency or unit.
        costs = [u.get("total_cost") for u in usage.values()]
        cost = (
            sum(costs)
            if costs and all(c is not None for c in costs) and design["cost_unit"] == "USD"
            else None
        )
        rows.append(
            {
                "task_id": task_id,
                "sample_id": epoch - 1,
                "status": status,
                "success": success,
                "tokens": tokens,
                "cost": cost,
                "latency_seconds": sample.get("total_time"),
                "inspect_sample": sample,
                "inspect_source": {
                    "sha256": source_sha256,
                    "inspect_version": inspect_version,
                    "adapter_version": "ard-inspect-v1",
                    "scorer": scorer,
                    "status": data["status"],
                    "eval": spec,
                },
            }
        )
    normalize(rows, {}, design)
    return rows


def import_inspect(source, *, scorer, design, label, output):
    """Read a local .eval/.json through the public SDK; no model execution or network."""
    design = validate_design(design)
    if source.suffix not in {".eval", ".json"}:
        raise SchemaError("Inspect input must be a local .eval or .json log")
    try:
        from importlib.metadata import version

        from inspect_ai.log import read_eval_log
    except ImportError as exc:
        raise ImportError(
            "Inspect import requires the optional extra: pip install '.[inspect]'"
        ) from exc
    raw = source.read_bytes()
    with tempfile.TemporaryDirectory(prefix="ard-inspect-") as directory:
        directory = Path(directory)
        try:
            log = read_eval_log(io.BytesIO(raw), format=source.suffix[1:], resolve_attachments=True)
        except Exception as exc:
            raise SchemaError(f"Inspect log could not be decoded: {exc}") from exc
        rows = convert_inspect(
            log.model_dump(mode="json", exclude_none=True),
            scorer=scorer,
            design=design,
            source_sha256=hashlib.sha256(raw).hexdigest(),
            inspect_version=version("inspect-ai"),
        )
        exported = directory / "export.jsonl"
        exported.write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8"
        )
        return import_traces(exported, design, {}, label, output)
