# Major-method coverage

`code_available` means executable code exists; `smoke_verified` refers only to the
scope stated below. Every `numerical_reproduction_verified` value is **false**.

| Method | Code | Runnable entry | code_available | smoke_verified / scope |
|---|---|---|---|---|
| Pass@(K,T), tight budget | `metrics/pass_at_kt.py`, `evaluators/agent.py` | `ard demo`, `ard eval budget`, `ard compare` | true | CPU + earlier real model engineering smoke |
| Single-step/depth | `evaluators/fixed.py`, `reporting/comparison.py` | `ard eval step`, `ard compare` | true | CPU + earlier fixed-state model smoke |
| Error-prefix/recovery | `pools/errors.py`, `evaluators/recovery.py` | `ard pool errors`, `ard eval recovery` | true | CPU + earlier model smoke; derived injector |
| Dual validation | `integrations/checkpoint.py`, `reporting/dynamics.py` | `ard eval checkpoint`, `ard dynamics` | true | Earlier one-rank 1.7B engineering smoke |
| Likelihood/NLL/LLR | `analysis/likelihood.py`, `models/teacher_forcing.py` | `ard likelihood score/analyze` | true | Deterministic fixtures + tiny HF + local 1.7B fixed-response CPU smoke |
| BudgetMix | `training/budgetmix.py`, `integrations/verl_budgetmix.py` | `ard budgetmix plan/audit` | true | CPU + earlier two one-rank updates |
| Process signal | `training/process_signals.py`, `integrations/verl_grid.py` | `python examples/process_signal_training.py --coefficient 0.5` | true | CPU toy + locked tiny-model actor update + two real 1.7B one-rank Ray/vLLM/FSDP REINFORCE++ updates |

Limits: coding uses a derived checker and syntax mutation; BFCL/SQL/Terminal are
external-analysis inputs only. Formal coefficient/figure conventions are not
inferred from toolkit choices. Full PPO/GiGPO integrations are deferred.
No paper numbers are represented as recomputed, and seeded trainer off/on divergence
remains recorded. The historical asset ledger is evidence, not a release gate.
