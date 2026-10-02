"""Raw-distribution teacher-forcing records and strict two-policy pairing."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import fmean

from ..artifacts.io import canonical_hash
from ..contracts.schema import SchemaError, finite, integer


def score_record(event, *, model_id, prompt_ids, response_ids, logprobs, offsets):
    """Offsets are response-relative character intervals; no token may cross its boundary."""
    if not isinstance(event, dict):
        raise SchemaError("likelihood event must be an object")
    for name in ("event_id", "task_id", "trajectory_id", "source", "prompt", "response"):
        if not isinstance(event.get(name), str) or not event[name]:
            raise SchemaError(f"event requires nonempty {name}")
    if event["source"] not in {"base", "rl"} or type(event.get("success")) is not bool:
        raise SchemaError("source must be base/rl and success boolean")
    if (
        not prompt_ids
        or not response_ids
        or not (len(response_ids) == len(logprobs) == len(offsets))
    ):
        raise SchemaError("empty context/response or misaligned token scores")
    for token in prompt_ids + response_ids:
        integer(token, "token_id")
    if any(not math.isfinite(x) or x > 0 for x in logprobs):
        raise SchemaError("logprobs must be finite and nonpositive")
    span = event.get("action_span")
    if span is not None:
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(type(x) is not int for x in span)
            or not 0 <= span[0] < span[1] <= len(event["response"])
        ):
            raise SchemaError("invalid response-relative action_span")
    mask = []
    for left, right in offsets:
        if not 0 <= left < right <= len(event["response"]):
            raise SchemaError("invalid response token offset")
        mask.append(span is not None and max(left, span[0]) < min(right, span[1]))
    total = math.fsum(logprobs)
    action = [-lp for lp, yes in zip(logprobs, mask, strict=True) if yes]
    return {
        "schema_version": "ard-likelihood-v1",
        "status": "ok",
        "model_id": model_id,
        **{
            name: event[name]
            for name in ("event_id", "task_id", "trajectory_id", "source", "success")
        },
        "input_hash": canonical_hash([event["prompt"], event["response"], span]),
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "offsets": [list(x) for x in offsets],
        "action_mask": mask,
        "token_logprobs": logprobs,
        "response_tokens": len(response_ids),
        "response_logprob_sum": total,
        "response_nll_sum": -total,
        "response_nll_per_token": -total / len(response_ids),
        "action_tokens": len(action),
        "action_nll_sum": math.fsum(action) if action else None,
        "action_nll_per_token": fmean(action) if action else None,
        "action_span_convention": "overlapping_tokens_conditioned_on_full_prior_response",
    }


def pair_scores(base, rl):
    """LLR = log P_RL(response|prompt) - log P_Base(response|prompt)."""
    indexes = []
    for rows in (base, rl):
        if not isinstance(rows, list):
            raise SchemaError("policy scores must be JSON arrays")
        index = {}
        for row in rows:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("event_id"), str)
                or not row["event_id"]
            ):
                raise SchemaError("policy score requires a nonempty event_id")
            if row.get("status") not in {"ok", "scoring_failure"}:
                raise SchemaError("unsupported scoring status; migrate legacy records explicitly")
            if row["status"] == "ok":
                if row.get("schema_version") != "ard-likelihood-v1":
                    raise SchemaError("unsupported likelihood score schema")
                for name in ("model_id", "task_id", "trajectory_id", "input_hash"):
                    if not isinstance(row.get(name), str) or not row[name]:
                        raise SchemaError(f"score requires {name}")
                if row.get("source") not in {"base", "rl"} or type(row.get("success")) is not bool:
                    raise SchemaError("invalid score source/outcome")
                for name in (
                    "prompt_ids",
                    "response_ids",
                    "token_logprobs",
                    "offsets",
                    "action_mask",
                ):
                    if not isinstance(row.get(name), list) or not row[name]:
                        raise SchemaError(f"missing token array: {name}")
                for token in row["prompt_ids"] + row["response_ids"]:
                    integer(token, "token_id")
            key = row["event_id"]
            if key in index:
                raise SchemaError("duplicate scoring event")
            index[key] = row
        indexes.append(index)
    a, b = indexes
    paired, excluded = [], []
    for key in sorted(a.keys() | b.keys()):
        x, y = a.get(key), b.get(key)
        reason = None
        if x is None or y is None:
            reason = "missing_policy_score"
        elif x.get("status") != "ok" or y.get("status") != "ok":
            reason = "scoring_failure"
        elif any(
            x.get(name) != y.get(name)
            for name in (
                "input_hash",
                "task_id",
                "trajectory_id",
                "source",
                "success",
                "prompt_ids",
                "response_ids",
                "offsets",
                "action_mask",
            )
        ):
            raise SchemaError("LLR requires identical inputs, targets and token alignment")
        if reason:
            provenance = x if x is not None and x.get("trajectory_id") else y or {}
            excluded.append(
                {
                    "event_id": key,
                    "reason": reason,
                    **{
                        name: provenance.get(name)
                        for name in ("task_id", "trajectory_id", "source", "success")
                    },
                }
            )
            continue
        # Recompute aggregates from token scores; never trust stored NLL/LLR totals.
        n = len(x["response_ids"])
        if (
            not n
            or any(len(row["token_logprobs"]) != n for row in (x, y))
            or len(x["action_mask"]) != n
            or any(type(v) is not bool for v in x["action_mask"])
        ):
            raise SchemaError("invalid token score lengths")
        for row in (x, y):
            if any(finite(lp, "token logprob") > 0 for lp in row["token_logprobs"]):
                raise SchemaError("invalid token logprobs")
        bx, ry = math.fsum(x["token_logprobs"]), math.fsum(y["token_logprobs"])
        action = [-lp for lp, yes in zip(x["token_logprobs"], x["action_mask"], strict=True) if yes]
        paired.append(
            {
                **{
                    name: x[name]
                    for name in ("event_id", "task_id", "trajectory_id", "source", "success")
                },
                "response_tokens": n,
                "base_nll_sum": -bx,
                "base_nll_per_token": -bx / n,
                "llr_sum": ry - bx,
                "llr_per_token": (ry - bx) / n,
                "base_action_nll_sum": math.fsum(action) if action else None,
                "base_action_nll_per_token": fmean(action) if action else None,
            }
        )
    return {
        "schema_version": "ard-llr-v1",
        "llr_sign": "rl_minus_base",
        "paired": paired,
        "excluded": excluded,
        "full_estimate_available": not excluded and bool(paired),
        "completeness_scope": "supplied_event_union_only_not_original_trajectory_manifest",
    }


def length_control(rows, *, token_caliper, step_caliper, excluded_events=()):
    """Greedy within-task/outcome trajectory matching; descriptive toolkit control."""
    if not math.isfinite(token_caliper) or not 0 <= token_caliper <= 1:
        raise SchemaError("token_caliper must be in [0,1]")
    integer(step_caliper, "step_caliper")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task_id"], row["trajectory_id"], row["source"], row["success"])].append(row)
    contaminated = {
        (item.get("task_id"), item.get("trajectory_id"), item.get("source"))
        for item in excluded_events
    }
    if any(None in key for key in contaminated):
        return {
            "available": False,
            "reason": "excluded_events_lack_trajectory_identity",
            "input_trajectories": len(groups),
            "matched_pairs": 0,
            "pairs": [],
            "excluded_events": len(excluded_events),
        }
    excluded_trajectories = 0
    trajectories = []
    for (task, trajectory, source, success), events in sorted(groups.items()):
        if (task, trajectory, source) in contaminated:
            excluded_trajectories += 1
            continue
        count = sum(x["response_tokens"] for x in events)
        trajectories.append(
            {
                "task_id": task,
                "trajectory_id": trajectory,
                "source": source,
                "success": success,
                "tokens": count,
                "steps": len(events),
                "base_nll_per_token": sum(x["base_nll_sum"] for x in events) / count,
                "llr_per_token": sum(x["llr_sum"] for x in events) / count,
            }
        )
    candidates = []
    for i, x in enumerate(trajectories):
        for j, y in enumerate(trajectories):
            if (x["source"], y["source"]) != ("base", "rl") or (x["task_id"], x["success"]) != (
                y["task_id"],
                y["success"],
            ):
                continue
            gap = abs(x["tokens"] - y["tokens"]) / max(x["tokens"], y["tokens"])
            steps = abs(x["steps"] - y["steps"])
            if gap <= token_caliper and steps <= step_caliper:
                candidates.append((gap, steps, i, j))
    used, pairs = set(), []
    for gap, steps, i, j in sorted(candidates):
        if i in used or j in used:
            continue
        used.update((i, j))
        x, y = trajectories[i], trajectories[j]
        pairs.append(
            {
                "task_id": x["task_id"],
                "success": x["success"],
                "base_trajectory_id": x["trajectory_id"],
                "rl_trajectory_id": y["trajectory_id"],
                "relative_token_gap": gap,
                "step_gap": steps,
                "delta_base_nll_per_token": y["base_nll_per_token"] - x["base_nll_per_token"],
                "delta_llr_per_token": y["llr_per_token"] - x["llr_per_token"],
            }
        )
    return {
        "token_caliper": token_caliper,
        "step_caliper": step_caliper,
        "available": True,
        "input_trajectories": len(groups),
        "eligible_trajectories": len(trajectories),
        "excluded_incomplete_trajectories": excluded_trajectories,
        "excluded_event_trajectories": len(contaminated),
        "completeness_scope": "supplied_event_union_only",
        "matched_pairs": len(pairs),
        "unmatched_trajectories": len(trajectories) - 2 * len(pairs),
        "pairs": pairs,
        "aggregation": "descriptive_pairs_no_causal_claim",
    }
