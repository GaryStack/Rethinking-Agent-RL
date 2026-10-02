import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from agent_rl_diagnostics.artifacts.io import read_json, write_json
from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.integrations import ray_checkpoint


def trainer(tmp_path, monkeypatch, *, marker=True):
    class FixtureTrainer:
        def _save_checkpoint(self):
            root = Path(self.config.trainer.default_local_dir)
            root.mkdir(exist_ok=True)
            (root / "latest_checkpointed_iteration.txt").write_text("1" if marker else "0")

        def _validate(self):
            return {"native/validation": 0.5}

    module = types.ModuleType("verl.trainer.ppo.ray_trainer")
    module.RayPPOTrainer = FixtureTrainer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(ray_checkpoint, "verify_verl_source", lambda source: None)
    instance = ray_checkpoint.diagnostic_trainer_class(tmp_path)()
    instance.global_steps = 1
    instance.config = NS(
        trainer=NS(default_local_dir=str(tmp_path / "checkpoints")),
        actor_rollout_ref=NS(
            actor=NS(checkpoint=NS(async_save=False), fsdp_config=NS(param_offload=True)),
            model=NS(lora_rank=0),
        ),
        ard_diagnostics=NS(
            every_n_steps=1,
            config="validation.json",
            pool="pool",
            output=str(tmp_path / "diag"),
            tracking=str(tmp_path / "tracking"),
            cuda_visible_devices="2",
            samples=2,
            timeout_seconds=30,
        ),
    )
    return instance


def test_concrete_save_barrier_invokes_public_cli_and_driver_metrics(tmp_path, monkeypatch):
    instance = trainer(tmp_path, monkeypatch)
    calls = []

    def child(command, **kwargs):
        calls.append((command, kwargs))
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        write_json(
            output / "metrics.json",
            {
                "event_id": "fixture",
                "checkpoint_id": "fixture",
                "checkpoint_step": 1,
                "metrics": {"diag/grid/step/q_star_mean": 0.25},
            },
        )

    monkeypatch.setattr(ray_checkpoint.subprocess, "run", child)
    instance._save_checkpoint()
    assert calls[0][0][1:4] == ["-m", "agent_rl_diagnostics", "eval"]
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert calls[0][1]["check"] is True
    assert calls[0][1]["timeout"] == 30
    point = read_json(tmp_path / "diag/validation-1/trainer-barrier.json")
    assert point["trainer_step"] == 1
    assert instance._validate()["diag/grid/step/q_star_mean"] == 0.25
    assert instance._validate() == {"native/validation": 0.5, "checkpoint_step": 1}


def test_stale_commit_marker_never_launches_eval(tmp_path, monkeypatch):
    instance = trainer(tmp_path, monkeypatch, marker=False)
    monkeypatch.setattr(
        ray_checkpoint.subprocess, "run", lambda *a, **kw: pytest.fail("must not launch")
    )
    with pytest.raises(SchemaError, match="stale"):
        instance._save_checkpoint()


@pytest.mark.parametrize("cadence", [0, -1, True, 1.5])
def test_invalid_diagnostic_cadence_rejected(tmp_path, monkeypatch, cadence):
    instance = trainer(tmp_path, monkeypatch)
    instance.config.ard_diagnostics.every_n_steps = cadence
    with pytest.raises(SchemaError, match="cadence"):
        instance._save_checkpoint()


@pytest.mark.parametrize(
    "change,error", [("async", "synchronous"), ("lora", "unmerged"), ("offload", "offload")]
)
def test_unsafe_export_modes_rejected(tmp_path, monkeypatch, change, error):
    instance = trainer(tmp_path, monkeypatch)
    if change == "async":
        instance.config.actor_rollout_ref.actor.checkpoint.async_save = True
    elif change == "lora":
        instance.config.actor_rollout_ref.model.lora_rank = 4
    else:
        instance.config.actor_rollout_ref.actor.fsdp_config.param_offload = False
    with pytest.raises(SchemaError, match=error):
        instance._save_checkpoint()
