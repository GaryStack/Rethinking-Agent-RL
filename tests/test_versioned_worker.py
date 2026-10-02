"""Contract fixtures, not real Ray/FSDP acceptance."""

import asyncio
import sys
import types
from types import SimpleNamespace as NS

import pytest

from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.integrations import versioned_worker


def fixture_worker(tmp_path, monkeypatch, *, complete=True, fail=False):
    calls = []

    class Parent:
        def init_model(self):
            calls.append(("init_model", None))
            return "native_result"

        def update_actor(self, data):
            calls.append(("update", data))
            return NS(meta_info={"metrics": {"actor/grad_norm": [1.0]}})

        async def update_weights(self, global_steps=None):
            calls.append(("transfer", global_steps))
            await asyncio.sleep(0)
            if fail:
                raise RuntimeError("transfer failed")
            calls.append(("complete", global_steps))
            return complete

    async def tag(step):
        calls.append(("tag", step))

    worker_module = types.ModuleType("verl.workers.fsdp_workers")
    worker_module.AsyncActorRolloutRefWorker = Parent
    decorator_module = types.ModuleType("verl.single_controller.base.decorator")
    decorator_module.Dispatch = NS(ONE_TO_ALL="fixture")
    decorator_module.register = lambda **kwargs: lambda function: function
    decorator_module.make_nd_compute_dataproto_dispatch_fn = lambda **kwargs: "fixture"
    monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)
    monkeypatch.setitem(sys.modules, decorator_module.__name__, decorator_module)
    monkeypatch.setattr(versioned_worker, "verify_verl_source", lambda _: None)
    worker = versioned_worker.versioned_worker_class(tmp_path)()
    worker.world_size = 1
    worker.rollout = NS(server_handle=NS(set_global_steps=NS(remote=tag)))
    return worker, calls


def test_version_is_tagged_only_after_completed_actual_parent_barrier(tmp_path, monkeypatch):
    worker, calls = fixture_worker(tmp_path, monkeypatch)
    assert asyncio.run(worker.update_weights(0)) is True
    assert calls == [("transfer", 0), ("complete", 0), ("tag", 0)]


@pytest.mark.parametrize("mode", ["incomplete", "failure", "multi_rank", "missing_step"])
def test_failed_or_unverified_transfer_never_tags(tmp_path, monkeypatch, mode):
    worker, calls = fixture_worker(
        tmp_path, monkeypatch, complete=mode != "incomplete", fail=mode == "failure"
    )
    worker.world_size = 2 if mode == "multi_rank" else 1
    with pytest.raises((SchemaError, RuntimeError)):
        asyncio.run(worker.update_weights(None if mode == "missing_step" else 1))
    assert not any(kind == "tag" for kind, _ in calls)


def test_missing_training_signal_rejected_before_parent_update(tmp_path, monkeypatch):
    worker, calls = fixture_worker(tmp_path, monkeypatch)
    with pytest.raises(SchemaError, match="training signal"):
        worker.update_actor(NS(batch={}))
    assert calls == []


def valid_actor_input():
    active = NS(
        numel=lambda: 1,
        isfinite=lambda: NS(all=lambda: NS(item=lambda: True)),
        count_nonzero=lambda: NS(item=lambda: 1),
    )

    class FixtureArray:
        shape = (1,)

        def __getitem__(self, index):
            return active

    return NS(
        batch={"advantages": FixtureArray(), "response_mask": NS(shape=(1,), bool=lambda: True)}
    )


def test_capture_disabled_delegates_native_update(tmp_path, monkeypatch):
    monkeypatch.delenv("ARD_TRAIN_STATE_DIR", raising=False)
    worker, calls = fixture_worker(tmp_path, monkeypatch)
    data = valid_actor_input()
    output = worker.update_actor(data)
    assert calls == [("update", data)]
    assert output.meta_info["metrics"]["actor/grad_norm"] == [1.0]


def test_default_initialization_delegates_without_optional_imports(tmp_path, monkeypatch):
    monkeypatch.delenv("ARD_ACTOR_INIT_SEED", raising=False)
    worker, calls = fixture_worker(tmp_path, monkeypatch)
    assert worker.init_model() == "native_result"
    assert calls == [("init_model", None)]


def test_seed_initialization_once_before_native_model_not_updates(tmp_path, monkeypatch):
    from agent_rl_diagnostics.artifacts.io import read_json
    from agent_rl_diagnostics.integrations import worker_seed

    monkeypatch.setenv("ARD_ACTOR_INIT_SEED", "5422")
    monkeypatch.setenv("ARD_TRAIN_STATE_DIR", str(tmp_path / "proofs"))
    worker, calls = fixture_worker(tmp_path, monkeypatch)

    def initialize(seed, **kwargs):
        assert seed == 5422 and kwargs == {"cuda_device": 0}
        calls.append(("seed", seed))
        return {"fixture": True}

    monkeypatch.setattr(worker_seed, "initialize_actor_rng", initialize)
    assert worker.init_model() == "native_result"
    assert calls == [("seed", 5422), ("init_model", None)]
    assert read_json(tmp_path / "proofs/initialization.json") == {"fixture": True}
    with pytest.raises(SchemaError, match="only once"):
        worker.init_model()
    assert len(calls) == 2
    monkeypatch.delenv("ARD_TRAIN_STATE_DIR")
    worker.update_actor(valid_actor_input())
    assert [kind for kind, _ in calls] == ["seed", "init_model", "update"]


@pytest.mark.parametrize("seed", ["-1", "1.0", "１２", "True"])
def test_invalid_seed_rejected_before_native_initialization(tmp_path, monkeypatch, seed):
    monkeypatch.setenv("ARD_ACTOR_INIT_SEED", seed)
    worker, calls = fixture_worker(tmp_path, monkeypatch)
    with pytest.raises(SchemaError):
        worker.init_model()
    assert calls == []


def test_opt_in_capture_surrounds_native_update_and_refuses_collision(tmp_path, monkeypatch):
    from agent_rl_diagnostics.artifacts.io import read_json
    from agent_rl_diagnostics.integrations import training_state

    monkeypatch.setenv("ARD_TRAIN_STATE_DIR", str(tmp_path / "proofs"))
    for package in ("numpy", "torch"):
        monkeypatch.setitem(sys.modules, package, NS(__version__="fixture"))
    worker, calls = fixture_worker(tmp_path, monkeypatch)
    worker.actor_module_fsdp = worker.actor_optimizer = worker.actor_lr_scheduler = None

    def capture(*args, **kwargs):
        assert kwargs == {"cuda_device": 0}
        calls.append(("capture", len(calls)))
        return {"fixture": len(calls)}

    monkeypatch.setattr(training_state, "capture_training_state", capture)
    data = valid_actor_input()
    output = worker.update_actor(data)
    assert [kind for kind, _ in calls] == ["capture", "update", "capture"]
    assert output.meta_info["metrics"]["actor/grad_norm"] == [1.0]
    proof = read_json(tmp_path / "proofs/update-000001.json")
    assert proof["before"] == {"fixture": 1} and proof["after"] == {"fixture": 3}
    worker._ard_state_update = 0
    with pytest.raises(SchemaError, match="refuse overwrite"):
        worker.update_actor(data)
    assert len(calls) == 3


@pytest.mark.parametrize("finite,count", [(True, 0), (False, 1)])
def test_zero_or_nonfinite_training_signal_rejected(tmp_path, monkeypatch, finite, count):
    worker, calls = fixture_worker(tmp_path, monkeypatch)
    active = NS(
        numel=lambda: 1,
        isfinite=lambda: NS(all=lambda: NS(item=lambda: finite)),
        count_nonzero=lambda: NS(item=lambda: count),
    )

    class FixtureArray:
        shape = (1,)

        def __getitem__(self, index):
            return active

    with pytest.raises(SchemaError, match="no smoke update"):
        worker.update_actor(
            NS(
                batch={
                    "advantages": FixtureArray(),
                    "response_mask": NS(shape=(1,), bool=lambda: True),
                }
            )
        )
    assert calls == []
