"""Driver-only durable metrics and opt-in strictly offline W&B."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..contracts.schema import SchemaError, integer


class DiagnosticTracker:
    def __init__(
        self,
        directory: Path,
        *,
        wandb_offline: bool = False,
        project: str = "agent-rl-diagnostics",
        rank: int = 0,
    ) -> None:
        if rank != 0:
            raise SchemaError("only driver/rank 0 may initialize diagnostic tracking")
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "metrics.jsonl"
        self.previous = []
        if self.path.exists():
            self.previous = [
                json.loads(line) for line in self.path.read_text().splitlines() if line
            ]
        self.run = None
        if wandb_offline:
            import wandb

            self.run = wandb.init(
                project=project,
                dir=str(directory),
                mode="offline",
                settings=wandb.Settings(console="off"),
            )
            self.run.define_metric("checkpoint_step")
            self.run.define_metric("diag/*", step_metric="checkpoint_step")

    def log(self, point: dict[str, Any]) -> bool:
        integer(point["checkpoint_step"], "checkpoint_step")
        same_step = [p for p in self.previous if p["checkpoint_step"] == point["checkpoint_step"]]
        if same_step:
            if same_step[0] == point:
                return False
            raise SchemaError("conflicting diagnostic event/checkpoint at an already logged step")
        if self.previous and point["checkpoint_step"] <= self.previous[-1]["checkpoint_step"]:
            raise SchemaError("diagnostic points must have monotonic checkpoint steps")
        numeric = {k: v for k, v in point["metrics"].items() if v is not None}
        if any(type(v) not in {int, float} or not math.isfinite(v) for v in numeric.values()):
            raise SchemaError("tracking metrics must be finite numeric values or explicit null")
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(point, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
        self.previous.append(point)
        if self.run:
            self.run.log(
                {
                    **numeric,
                    "checkpoint_step": point["checkpoint_step"],
                    "checkpoint_id": point["checkpoint_id"],
                }
            )
        return True

    def close(self) -> None:
        if self.run:
            self.run.finish()
