"""Optional pinned-verl agent loop: real grid actions, masked environment tokens.

Import only inside the optional Ray/vLLM environment, never in the core CLI.
"""

import time
from pathlib import Path
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput

from ..artifacts.io import canonical_hash, write_json
from ..contracts.schema import SchemaError
from ..environments.text_grid import TextGrid
from ..training.budgetmix import validate_assignment
from .grid_trace import audit_grid_trace


class GridAgentLoop(AgentLoopBase):
    # Smoke uses one agent worker and resume=disable. Give repeated samples distinct,
    # recorded seeds without tying sampling to random request UUIDs or wall time.
    _sample_counts = {}

    async def run(self, sampling_params, **kwargs):
        info = kwargs["extra_info"]
        task = {"task_id": str(info["task"]["task_id"]), "desc": list(info["task"]["desc"])}
        budget = int(info["budget"])
        process = self.config.get("ard_process_signal")
        process_enabled = process is not None and process.get("enabled", False)
        phase = None
        if process_enabled:
            from ..training.process_signals import process_signal_settings, process_training_phase

            process = process_signal_settings(process["coefficient"], process["failure_policy"])
            phase = process_training_phase(info)
        assignment = info.get("budgetmix")
        if assignment is not None:
            validate_assignment(assignment)
            if assignment["budget"] != budget or assignment["task_id"] != task["task_id"]:
                raise SchemaError("rollout differs from assigned BudgetMix occurrence")
        sample_key = (task["task_id"], float(sampling_params["temperature"]))
        if assignment is not None:
            sample_key += (assignment["task_occurrence_id"],)
        sample = self._sample_counts.get(sample_key, 0)
        self._sample_counts[sample_key] = sample + 1
        env = TextGrid(task, prompt_profile="action_only_v2")
        messages = env.initial_messages(budget)
        if list(kwargs["raw_prompt"]) != messages:
            raise SchemaError("dataset prompt differs from actual budget-visible grid input")
        initial = await self.apply_chat_template(messages)
        prompt_ids = list(initial)
        response_ids, masks, logprobs, steps = [], [], [], []
        version, success, reason, elapsed = None, False, "budget_exhausted", 0.0
        request = uuid4().hex
        for turn in range(budget):
            remaining = budget - turn
            cap = min(32, int(self.rollout_config.response_length) - len(response_ids))
            if cap <= 0:
                raise SchemaError("token cap prevents completing declared action budget")
            params = {**sampling_params, "max_tokens": cap}
            seed = int(canonical_hash([5422, task["task_id"], sample, turn])[:8], 16) % (2**31)
            if assignment is not None:
                seed = int(
                    canonical_hash([assignment["task_occurrence_id"], sample, turn])[:8], 16
                ) % (2**31)
            params["seed"] = seed
            params.pop("max_new_tokens", None)
            before = env.snapshot()
            started = time.perf_counter()
            output = await self.server_manager.generate(
                request_id=request, prompt_ids=prompt_ids, sampling_params=params
            )
            elapsed += time.perf_counter() - started
            if output.stop_reason != "completed":
                raise SchemaError("rollout backend did not complete the action")
            current = output.extra_fields.get("global_steps")
            if (
                type(current) is not int
                or current < 0
                or (version is not None and current != version)
            ):
                raise SchemaError("missing or mixed rollout weight versions")
            version = current
            tokens = list(output.token_ids)
            if not tokens or len(tokens) > cap:
                raise SchemaError("empty/oversized generated action")
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            outcome = env.apply(text, remaining)
            steps.append(
                {
                    "turn": turn,
                    "remaining_budget": remaining,
                    "prompt_ids": list(prompt_ids),
                    "response_ids": tokens,
                    "response": text,
                    "snapshot_before": before,
                    "snapshot_after": env.snapshot(),
                    "outcome": outcome,
                    "weight_version": current,
                    "seed": seed,
                    "sampling_params": params,
                }
            )
            response_ids.extend(tokens)
            masks.extend([1] * len(tokens))
            logprobs.extend(
                output.log_probs if output.log_probs is not None else [0.0] * len(tokens)
            )
            prompt_ids.extend(tokens)
            if outcome["terminated"]:
                success = outcome["success"]
                reason = "success" if success else "environment_terminal"
                break
            if remaining > 1:
                feedback = [
                    {
                        "role": "user",
                        "content": env.budget_observation(
                            budget, remaining - 1, outcome["observation"]
                        ),
                    }
                ]
                env_ids = await self.apply_chat_template(feedback, remove_system_prompt=True)
                if len(response_ids) + len(env_ids) >= int(self.rollout_config.response_length):
                    raise SchemaError("environment tokens exceed configured response cap")
                prompt_ids.extend(env_ids)
                response_ids.extend(env_ids)
                masks.extend([0] * len(env_ids))
                logprobs.extend([0.0] * len(env_ids))
                steps[-1]["feedback_messages"] = feedback
                steps[-1]["feedback_ids"] = env_ids
        trace = {
            "schema_version": "ard-verl-grid-rollout-v1",
            "request_id": request,
            "task": task,
            "budget": budget,
            "initial_messages": messages,
            "initial_prompt_ids": initial,
            "response_ids": response_ids,
            "response_mask": masks,
            "steps": steps,
            "success": success,
            "terminal_reward": float(success),
            "termination_reason": reason,
            "weight_version": version,
        }
        if assignment is not None:
            trace["budgetmix"] = assignment
        audit_grid_trace(trace)
        reward = float(success)
        if phase is not None:
            trace["ard_phase"] = phase
        if process_enabled and phase == "train":
            from ..training.process_signals import grid_process_reward

            trace["process_signal"] = grid_process_reward(
                trace,
                coefficient=process["coefficient"],
                failure_policy=process["failure_policy"],
            )
            reward = trace["process_signal"]["reward"]
            audit_grid_trace(trace)
        destination = Path(self.config.trainer.rollout_data_dir) / "grid-traces"
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / f"{request}.json", trace)
        return AgentLoopOutput(
            prompt_ids=initial,
            response_ids=response_ids,
            response_mask=masks,
            response_logprobs=logprobs if sampling_params.get("logprobs") else None,
            reward_score=reward,
            num_turns=2 * len(steps),
            metrics=AgentLoopMetrics(generate_sequences=elapsed),
            extra_fields={
                "global_steps": version,
                # Pinned upstream JSON dumping cannot serialize its np.int64 columns.
                # Log count as a float; raw trace still records exact integer actions.
                "reward_extra_info": {
                    "grid_success": float(success),
                    "composed_reward": float(reward),
                    "process_component": float(reward - float(success)),
                    "grid_actions": float(len(steps)),
                    "action_budget": float(budget),
                    "budget_exhausted": float(reason == "budget_exhausted"),
                },
            },
        )
