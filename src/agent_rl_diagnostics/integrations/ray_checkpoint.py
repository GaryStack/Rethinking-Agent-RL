"""Optional concrete adapter for the locked upstream RayPPOTrainer checkpoint API."""

import os
import subprocess
import sys
import time
from pathlib import Path

from ..artifacts.io import read_json, write_json
from ..contracts.schema import SchemaError
from .tracking import DiagnosticTracker
from .verl import verify_verl_source


def diagnostic_trainer_class(source: Path):
    verify_verl_source(source)
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    class CheckpointDiagnosticTrainer(RayPPOTrainer):
        def _save_checkpoint(self):
            settings = self.config.ard_diagnostics
            cadence = settings.every_n_steps
            if type(cadence) is not int or cadence < 1:
                raise SchemaError("diagnostic cadence must be a positive integer")
            if self.config.actor_rollout_ref.actor.checkpoint.async_save:
                raise SchemaError("diagnostic barrier requires synchronous checkpoint saving")
            if self.config.actor_rollout_ref.model.lora_rank:
                raise SchemaError("this adapter requires full HF exports, not unmerged LoRA")
            if not self.config.actor_rollout_ref.actor.fsdp_config.param_offload:
                raise SchemaError("same-GPU sidecar requires actor parameter offload")
            started = time.perf_counter()
            super()._save_checkpoint()
            step = self.global_steps
            root = Path(self.config.trainer.default_local_dir)
            # Upstream commits this marker after blocking worker saves and dataloader save.
            if (root / "latest_checkpointed_iteration.txt").read_text().strip() != str(step):
                raise SchemaError("upstream checkpoint commit marker is missing or stale")
            if step % cadence:
                return
            checkpoint = root / f"global_step_{step}" / "actor" / "huggingface"
            output = Path(settings.output) / f"validation-{step}"
            if output.exists():
                raise SchemaError("diagnostic output already exists; no silent overwrite")
            export_seconds = time.perf_counter() - started
            child_env = dict(os.environ)
            child_env["CUDA_VISIBLE_DEVICES"] = str(settings.cuda_visible_devices)
            command = [
                sys.executable,
                "-m",
                "agent_rl_diagnostics",
                "eval",
                "checkpoint",
                "--config",
                str(settings.config),
                "--pool",
                str(settings.pool),
                "--checkpoint",
                str(checkpoint),
                "--checkpoint-step",
                str(step),
                "--samples",
                str(settings.samples),
                "--output",
                str(output),
            ]
            logs = Path(settings.output) / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            with (
                (logs / f"step-{step}.stdout").open("x") as stdout,
                (logs / f"step-{step}.stderr").open("x") as stderr,
            ):
                subprocess.run(
                    command,
                    env=child_env,
                    stdout=stdout,
                    stderr=stderr,
                    check=True,
                    timeout=int(settings.timeout_seconds),
                )
            point = read_json(output / "metrics.json")
            point["metrics"]["diag/cost/export_seconds"] = export_seconds
            point["metrics"]["diag/cost/barrier_seconds"] = time.perf_counter() - started
            tracker = DiagnosticTracker(Path(settings.tracking), wandb_offline=False)
            tracker.log(point)
            # Upstream fit() owns its driver logger. Supplement its normal step metrics.
            self._ard_pending_metrics = point["metrics"]
            write_json(
                output / "trainer-barrier.json",
                {
                    "trainer_step": step,
                    "checkpoint_id": point["checkpoint_id"],
                    "export_seconds": export_seconds,
                    "barrier_seconds": point["metrics"]["diag/cost/barrier_seconds"],
                    "upstream_commit_marker": str(root / "latest_checkpointed_iteration.txt"),
                },
            )

        def _validate(self, *args, **kwargs):
            metrics = super()._validate(*args, **kwargs)
            metrics["checkpoint_step"] = self.global_steps
            if self.global_steps == 0:
                metrics.update(
                    read_json(Path(self.config.ard_diagnostics.initial_output) / "metrics.json")[
                        "metrics"
                    ]
                )
            metrics.update(getattr(self, "_ard_pending_metrics", {}))
            self._ard_pending_metrics = {}
            return metrics

    return CheckpointDiagnosticTrainer
