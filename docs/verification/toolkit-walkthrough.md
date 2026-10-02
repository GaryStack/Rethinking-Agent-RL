# CPU toolkit walkthrough verification

2026-10-02, local unpublished `0.1.0.dev10` candidate refresh.

`ard toolkit-demo --output /tmp/ard-toolkit-pilot` completed with ten audited
agent-v2 budget, step, error-step, recovery and Base-monitor step evaluations,
five HTML/SVG comparisons and an audited BudgetMix plan. No network endpoint,
model weights or optional inference/training dependency is used. Script labels
are synthetic comparison roles, not trained Base/RL policies.

The regression moves the entire output directory and reaudits every raw child.
It then modifies an action file and confirms that artifact hash verification
rejects it. Existing output is rejected rather than overwritten.

A likelihood CLI regression includes one missing step in a supplied trajectory.
The whole trajectory is excluded from length control, leaving zero matched
trajectory pairs rather than matching its shortened surviving fragment.
Completeness covers the supplied event union only, not an unprovided expected
trajectory manifest.

Default CPU validation: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest
-q -p no:cacheprovider` — 270 passed, 3 optional namespace tests skipped.
Ruff checks and formatting pass. The refreshed wheel is tested outside the
checkout in a clean no-dependency environment; its walkthrough and a separate
recovery audit complete. Wheel/source archive inspection also passes.

This update does not add actual checkpoint training, GPU process acceptance,
numerical reproduction or external uploads. Existing dev10 native CPU and
historical real-model smoke limitations remain in force.
