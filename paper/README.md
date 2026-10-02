# Paper companion

This directory maps the major methods of *What Does Multi-Turn Reinforcement
Learning Really Improve in LLM Agents?* to runnable toolkit code. Start with
[coverage](coverage.md), then [inputs](../docs/inputs.md). Formal numerical
reproduction is unverified; paper presets remain reserved. Authored examples use
`toolkit_v1`, never `paper_v1`.

Run the CPU K/T demo, audit and compare from README; run likelihood fixtures via
`ard likelihood analyze`; run `python examples/process_signal_training.py
--coefficient 0.5` for an actual synthetic reward-to-update path. The four tool
tutorials link real evaluators and the pinned integrations. No full data pipeline,
all-model training sweep or complete trainer-state parity is required for this
code companion. Original prompts, pools, coefficients and figure settings that
remain unresolved are limitations rather than invented parameters.
