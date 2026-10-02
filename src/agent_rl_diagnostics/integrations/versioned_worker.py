"""Optional one-rank adapter for the locked legacy worker's omitted version tag."""

import json
import os
import time
from pathlib import Path

from ..contracts.schema import SchemaError, integer
from .verl import verify_verl_source


def versioned_worker_class(source: Path):
    verify_verl_source(source)
    from verl.single_controller.base.decorator import (
        Dispatch,
        make_nd_compute_dataproto_dispatch_fn,
        register,
    )
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

    class VersionedAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def init_model(self):
            raw_seed = os.environ.get("ARD_ACTOR_INIT_SEED", "")
            if not raw_seed:
                return super().init_model()
            if getattr(self, "_ard_seed_initialization_started", False):
                raise SchemaError("seeded actor initialization may run only once")
            if self.world_size != 1 or not raw_seed.isascii() or not raw_seed.isdecimal():
                raise SchemaError("actor initialization needs one rank and a decimal seed")
            from .worker_seed import initialize_actor_rng

            proof_root = os.environ.get("ARD_TRAIN_STATE_DIR")
            destination = Path(proof_root) / "initialization.json" if proof_root else None
            if destination is not None and (destination.exists() or destination.is_symlink()):
                raise SchemaError("actor initialization proof exists; refuse overwrite")
            self._ard_seed_initialization_started = True
            record = initialize_actor_rng(int(raw_seed), cuda_device=0)
            result = super().init_model()
            self._ard_initialization = record
            if destination is not None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("x") as stream:
                    json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
                    stream.write("\n")
            return result

        @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
        def update_actor(self, data):
            advantages, mask = data.batch.get("advantages"), data.batch.get("response_mask")
            if advantages is None or mask is None or advantages.shape != mask.shape:
                raise SchemaError("missing or mismatched actual training signal/mask")
            active = advantages[mask.bool()]
            if (
                not active.numel()
                or not active.isfinite().all().item()
                or active.count_nonzero().item() == 0
            ):
                raise SchemaError(
                    "actual generated-token advantages are nonfinite or all zero; no smoke update"
                )
            proof_root = os.environ.get("ARD_TRAIN_STATE_DIR")
            if not proof_root:
                return super().update_actor(data)
            if self.world_size != 1:
                raise SchemaError("actor state capture currently requires one rank")
            import numpy as np
            import torch

            from .training_state import capture_training_state

            update = getattr(self, "_ard_state_update", 0) + 1
            destination = Path(proof_root) / f"update-{update:06d}.json"
            if destination.exists():
                raise SchemaError("actor state proof exists; refuse overwrite")
            started = time.perf_counter()
            before = capture_training_state(
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                data.batch,
                cuda_device=0,
            )
            before_seconds = time.perf_counter() - started
            output = super().update_actor(data)
            started = time.perf_counter()
            after = capture_training_state(
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                data.batch,
                cuda_device=0,
            )
            norms = output.meta_info["metrics"]["actor/grad_norm"]
            if not isinstance(norms, (list, tuple)):
                norms = [norms]
            proof = {
                "schema_version": "ard-training-state-v1",
                "update": update,
                "scope": "single_rank_actor_worker_boundary",
                "framework_versions": {"torch": torch.__version__, "numpy": np.__version__},
                "before": before,
                "after": after,
                "actor_grad_norms": [float(norm) for norm in norms],
                "capture_seconds": {
                    "before": before_seconds,
                    "after": time.perf_counter() - started,
                },
            }
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x") as stream:
                json.dump(proof, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
            self._ard_state_update = update
            return output

        @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
        async def update_weights(self, global_steps=None):
            integer(global_steps, "global_steps")
            if self.world_size != 1:
                raise SchemaError("version adapter is accepted only for the one-rank smoke")
            # Pinned parent performs/awaits the real IPC transfer but drops its step
            # argument in rollout_mode(). Never tag the server before that returns.
            result = await super().update_weights(global_steps=global_steps)
            if result is not True or self.rollout.server_handle is None:
                raise SchemaError("actual upstream weight transfer did not complete")
            await self.rollout.server_handle.set_global_steps.remote(global_steps)
            return result

    return VersionedAsyncActorRolloutRefWorker
