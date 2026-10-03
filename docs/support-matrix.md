# Support matrix

| Component | Available | Experimental / deferred |
|---|---|---|
| Deterministic FrozenLake | Rollout, exact solver, frozen quality, recovery | Slippery runtime |
| Derived stdin coding | Namespace execution, test fraction, syntax intervention | Official DeepCoder checker/injector |
| BFCL / SQL / Terminal | External frozen response or scorer analysis | Bundled runnable environments |
| Models | HTTP chat; optional local Transformers | Other backend adapters |
| Mechanism | Token NLL/LLR aggregation, length control, HTML/SVG | Real 1.7B fixed-response CPU smoke; formal matrix deferred |
| Dynamics | Checkpoint subprocess; pinned one-rank smoke; optional offline W&B | Full noninterference, multi-rank, resume |
| BudgetMix | Sampler/audit; pinned one-rank REINFORCE++ smoke | PPO critic, GiGPO trainer, resume |
| Process signal | Oracle/judge cache/mixing; locked tiny-model CPU update; two real 1.7B one-rank Ray/vLLM/FSDP updates | PPO/GiGPO training, multi-rank/resume |
| Inspect logs | Optional pinned SDK reader, binary scores, native tool-loop case | Live-model acceptance, broad version compatibility |
| Extension | Public external single-step scoring API | General built-in environment registration |

No formal paper numerical reproduction is certified. `paper_v1` remains reserved.
Namespace tests are opt-in on a suitable Linux host. Optional dependencies are not
imported by the core. HTTP server compatibility and seed behavior must be recorded.
