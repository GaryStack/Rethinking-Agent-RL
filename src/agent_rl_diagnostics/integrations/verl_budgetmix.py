"""Opt-in text-grid adapter at the locked trainer's pre-repeat generation hook."""

import copy
from pathlib import Path

from ..artifacts.io import new_output_directory, write_json
from ..contracts.schema import SchemaError
from ..environments.text_grid import TextGrid
from ..training.budgetmix import audit_groups, plan_occurrences
from .verl import verify_verl_source


def budgetmix_trainer_class(source: Path, *, base_class=None):
    verify_verl_source(source)
    if base_class is None:
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        base_class = RayPPOTrainer

    class BudgetMixTrainer(base_class):
        def _validate(self, *args, **kwargs):
            previous = getattr(self, "_ard_budgetmix_validation", False)
            self._ard_budgetmix_validation = True
            try:
                result = super()._validate(*args, **kwargs)
            finally:
                self._ard_budgetmix_validation = previous
            pending = getattr(self, "_ard_budgetmix_metrics", {})
            if pending:
                result.update(pending)
                self._ard_budgetmix_metrics = {}
            return result

        def _get_gen_batch(self, batch):
            # Native validation budgets must never be resampled.
            if getattr(self, "_ard_budgetmix_validation", False):
                return super()._get_gen_batch(batch)
            import numpy as np

            settings = self.config.ard_budgetmix
            if self.config.trainer.resume_mode != "disable":
                raise SchemaError("BudgetMix resume has not been accepted; require disable")
            # This locked dataset emits raw prompts only before rollout. Refuse
            # stale tokenized/multimodal inputs instead of retokenizing partially.
            if batch.batch is not None and set(batch.batch.keys()) - {"dummy_tensor"}:
                raise SchemaError("BudgetMix adapter requires raw-prompt-only grid inputs")
            info = copy.deepcopy(list(batch.non_tensor_batch["extra_info"]))
            prompts = list(batch.non_tensor_batch["raw_prompt"])
            if len(info) != len(prompts) or not info:
                raise SchemaError("missing/misaligned raw grid prompts")
            assignments = plan_occurrences(
                [entry["task"]["task_id"] for entry in info],
                tmax=settings.tmax,
                seed=settings.seed,
                update=self.global_steps,
            )
            for index, (entry, assignment) in enumerate(zip(info, assignments, strict=True)):
                env = TextGrid(entry["task"], prompt_profile="action_only_v2")
                if prompts[index] != env.initial_messages(entry["budget"]):
                    raise SchemaError("unexpected original grid prompt")
                entry["budget"] = assignment["budget"]
                entry["budgetmix"] = assignment
                prompts[index] = env.initial_messages(entry["budget"])
            size = self.config.actor_rollout_ref.rollout.n
            report = audit_groups([r for r in assignments for _ in range(size)], group_size=size)
            # Immutable occurrence plan: require fresh outputs; repeated hooks
            # for the same update must fail before mutating the native batch.
            with new_output_directory(
                Path(settings.output) / f"update-{self.global_steps:06d}"
            ) as root:
                write_json(root / "assignments.json", assignments)
                write_json(root / "metrics.json", report)
            self._ard_budgetmix_metrics = {
                "train/budgetmix/budget_mean": report["budget_mean"],
                "train/budgetmix/groups": report["groups"],
                **{
                    f"train/budgetmix/groups_t{budget}": count
                    for budget, count in report["budget_histogram"].items()
                },
            }
            for key, values in (("extra_info", info), ("raw_prompt", prompts)):
                array = np.empty(len(values), dtype=object)
                array[:] = values
                batch.non_tensor_batch[key] = array
            return super()._get_gen_batch(batch)

    return BudgetMixTrainer
