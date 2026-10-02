# User-supplied inputs

No complete data download/cleaning pipeline is required by this preview.

- Model: compatible chat endpoint or existing local checkpoint, explicit model ID
  and revision, decoding configuration and supported seed behavior. Secrets use an
  environment-variable reference, never config literals.
- Tasks: JSON array via `tasks_file`, or embedded `tasks`. Grid: `task_id`, `desc`
  string rows. Coding: `task_id`, `question`, `public_tests`, `hidden_tests` with
  string `input/output` pairs. These are your own licensed assets.
- Frozen pools: built with `ard pool build` from toolkit runs; retain complete
  messages, snapshots, task identity, depth and remaining budget. Use the same
  pool for matched comparisons and a separate Base-only pool for monitoring.
- External trajectories: migrate explicitly to toolkit schemas; no implicit import
  of historical research formats. See contracts in `docs/protocols.md`.
- Likelihood: JSON fixed events with exact rendered prompt/response and optional
  response-relative action span; see `docs/likelihood.md`.
- Process reward: step oracle/judge score with explicit scoring identity,
  coefficient and failure policy; see `docs/process-signals.md`.
- Dynamics: exported checkpoint series plus immutable validation inputs and raw
  child results. See `docs/training-dynamics.md` for the manifest schema.

Scorer units, infra failures and missing slots remain separate. None of the small
fixtures establishes original benchmark splits or model training provenance.
