"""Trainer-hook fixtures, not real trainer or GPU acceptance."""

import sys
from types import SimpleNamespace as NS

import pytest

from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.environments.text_grid import TextGrid
from agent_rl_diagnostics.integrations import verl_budgetmix
from agent_rl_diagnostics.training.budgetmix import validate_assignment


def fixture_trainer(tmp_path, monkeypatch):
    monkeypatch.setattr(verl_budgetmix, "verify_verl_source", lambda _: None)
    monkeypatch.setitem(sys.modules, "numpy", NS(empty=lambda n, dtype: [None] * n))
    calls = []

    class Parent:
        def _get_gen_batch(self, batch):
            calls.append("native_generation_hook")
            return batch

        def _validate(self, batch):
            return self._get_gen_batch(batch)

    cls = verl_budgetmix.budgetmix_trainer_class(tmp_path, base_class=Parent)
    trainer = cls()
    trainer.global_steps = 1
    trainer.config = NS(
        ard_budgetmix=NS(tmax=2, seed=5422, output=str(tmp_path / "plans")),
        trainer=NS(resume_mode="disable"),
        actor_rollout_ref=NS(rollout=NS(n=4)),
    )
    task = {"task_id": "authored-infeasible", "desc": ["SFFG"]}
    prompt = TextGrid(task, prompt_profile="action_only_v2").initial_messages(2)
    batch = NS(
        batch={"dummy_tensor": "fixture"},
        non_tensor_batch={
            "extra_info": [{"task": task, "budget": 2}, {"task": task, "budget": 2}],
            "raw_prompt": [prompt, prompt],
        },
    )
    return trainer, batch, calls


def test_pre_repeat_shared_budget_rebuilds_prompt_and_retains_infeasible(tmp_path, monkeypatch):
    trainer, batch, calls = fixture_trainer(tmp_path, monkeypatch)
    assert trainer._get_gen_batch(batch) is batch
    records = batch.non_tensor_batch["extra_info"]
    assert (
        records[0]["budgetmix"]["task_occurrence_id"]
        != records[1]["budgetmix"]["task_occurrence_id"]
    )
    for index, entry in enumerate(records):
        validate_assignment(entry["budgetmix"])
        assert entry["budget"] < 3  # all below the true structural minimum
        assert batch.non_tensor_batch["raw_prompt"][index] == TextGrid(
            entry["task"], prompt_profile="action_only_v2"
        ).initial_messages(entry["budget"])
    assert calls == ["native_generation_hook"]


def test_native_validation_budget_unchanged(tmp_path, monkeypatch):
    trainer, batch, calls = fixture_trainer(tmp_path, monkeypatch)
    assert trainer._validate(batch) is batch
    assert "budgetmix" not in batch.non_tensor_batch["extra_info"][0]
    assert not (tmp_path / "plans").exists()
    assert trainer._ard_budgetmix_validation is False
    assert calls == ["native_generation_hook"]


def test_assignment_metrics_join_native_validation_once(tmp_path, monkeypatch):
    trainer, batch, _ = fixture_trainer(tmp_path, monkeypatch)
    trainer._get_gen_batch(batch)
    native_base = type(trainer).__bases__[0]
    monkeypatch.setattr(native_base, "_validate", lambda self: {"native": 1})
    metrics = trainer._validate()
    assert metrics["native"] == 1
    assert metrics["train/budgetmix/groups"] == 2
    assert (
        sum(value for key, value in metrics.items() if key.startswith("train/budgetmix/groups_t"))
        == 2
    )
    assert trainer._validate() == {"native": 1}


@pytest.mark.parametrize("mode", ["tokenized", "resume", "prompt"])
def test_unsafe_inputs_rejected_before_native_hook(tmp_path, monkeypatch, mode):
    trainer, batch, calls = fixture_trainer(tmp_path, monkeypatch)
    if mode == "tokenized":
        batch.batch["input_ids"] = "stale"
    if mode == "resume":
        trainer.config.trainer.resume_mode = "auto"
    if mode == "prompt":
        batch.non_tensor_batch["raw_prompt"][0] = []
    with pytest.raises(SchemaError):
        trainer._get_gen_batch(batch)
    assert calls == []
