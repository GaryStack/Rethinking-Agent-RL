# Remaining preview work

The current release scope follows the 2026-10-01 handoff, not the old GPU-parity
roadmap. Major evaluator/BudgetMix/dynamics code is reused; likelihood analysis,
process mixing, comparison reports and an external scorer adapter are added.

1. Extend the verified fixed-response CPU HF smoke only when new scoring
   changes require it; multi-task numerical reproduction remains deferred.
2. Locked process reward → advantage → actor update has tiny-model CPU and
   two-update real Qwen3-1.7B one-rank Ray/vLLM/FSDP evidence. No further GPU job
   is required for this preview; PPO/GiGPO, multi-rank and resume remain deferred.
3. Confirm authors, paper URL and release metadata before external publication.
4. Optional host namespace, multi-rank/resume and noninterference investigation
   remain follow-up evidence; they do not block the companion-code preview.

No full data pipeline or all-task/model training sweep is planned. Preserve old
negative results; see `verification/history/dev8-status.md` and `dev8-next.md`.
