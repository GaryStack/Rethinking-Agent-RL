# Source publication and local preview release

Current candidate: **0.1.0.dev11**, prepared on 2026-10-03. This version includes
external JSONL analysis and optional native Inspect ingestion. It is a paper
companion and lightweight diagnostic toolkit; formal numerical reproduction is
unverified. See [current status](../STATUS.md) and [verification](verification/dev11.md).

## Candidate files

The current `dist/release-manifest.json` names the source commit and hashes of:

- `agent-rl-diagnostics-0.1.0.dev11.bundle`: committed public Git source/history.
- `agent_rl_diagnostics-0.1.0.dev11-py3-none-any.whl`: stdlib core and library modules.
- `agent_rl_diagnostics-0.1.0.dev11.tar.gz`: source, examples, tutorials and tests.

`dist/SHA256SUMS` includes those artifacts and the manifest. Older dev9/dev10 files
are historical and are not listed as current candidates. Large private traces,
weights, W&B binaries and execution outputs stay outside the packages.

Clone the bundle to review the actual source:

```bash
git clone /path/to/agent-rl-diagnostics-0.1.0.dev11.bundle agent-rl-diagnostics
cd agent-rl-diagnostics
python -m pip install -e .
ard toolkit-demo --output runs/toolkit-walkthrough
```

For your own logs, follow the [JSONL tutorial](tutorials/external-traces.md).
Inspect users can install `.[inspect]` and follow the
[native log and tool-case tutorial](tutorials/inspect-agent.md).

## Build and check

Run from the committed source with setuptools>=68 and wheel installed:

```bash
python scripts/check_source_release.py
python -m pytest -q
ruff check .
ruff format --check src tests scripts examples
python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
python -c 'from setuptools.build_meta import build_sdist; build_sdist("dist")'
python scripts/check_release_archive.py dist/agent_rl_diagnostics-0.1.0.dev11-py3-none-any.whl
python scripts/check_release_archive.py dist/agent_rl_diagnostics-0.1.0.dev11.tar.gz
git bundle create dist/agent-rl-diagnostics-0.1.0.dev11.bundle main HEAD
git bundle verify dist/agent-rl-diagnostics-0.1.0.dev11.bundle
```

Install the wheel in a fresh environment without dependencies and exercise the
CLI outside the checkout. Optionally install the pinned Inspect extra separately.
Artifact scans check selected paths, sizes and known credential patterns; manual
review remains necessary. Do not scan/copy the original private research trees.

## External publication handoff

Author names and the paper link are intentionally pending. `CITATION.cff` keeps
the explicit collective placeholder; only its software version was updated.
The source destination is [GaryStack/Rethinking-Agent-RL](https://github.com/GaryStack/Rethinking-Agent-RL).
The final public package version remains pending. Before a package release, update
metadata, rerun the source check and rebuild packages from that exact commit.

The October 3 local acceptance predates GitHub publication. Its files in `dist/`
remain historical local artifacts; the GitHub source includes subsequent README
and publication metadata updates. These local archives are not uploaded by a Git
push. GitHub Actions results must be checked separately after pushing; PyPI
publication is not part of this source handoff. CI configuration
includes the stdlib Python matrix, isolated wheel checks and a separate pinned
Inspect native-runtime job. Their local equivalents are recorded in verification;
remote results must be observed after an authorized push.

Retain the known trainer divergence, narrow one-rank training evidence, missing
PPO/GiGPO/multi-rank/resume acceptance and synthetic example labels in release notes.
Imported logs are caller-attested; integrity checks do not establish their truth
or counterfactual recovery. Historical verification remains available and does not
replace the current candidate manifest.
