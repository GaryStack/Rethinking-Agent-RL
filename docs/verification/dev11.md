# Dev11 local release verification

2026-10-03. This candidate brings the external trace and Inspect additions into
committed source and distributable packages. It adds no new research experiments.
Formal paper numerical reproduction and live-model Inspect efficacy remain
unverified. Author/paper metadata are deliberately deferred.

Commands run for this candidate:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider
ruff check .
ruff format --check src tests scripts examples
python scripts/check_source_release.py
python -c 'from setuptools.build_meta import build_wheel, build_sdist; build_wheel("dist"); build_sdist("dist")'
python scripts/check_release_archive.py dist/agent_rl_diagnostics-0.1.0.dev11-py3-none-any.whl
python scripts/check_release_archive.py dist/agent_rl_diagnostics-0.1.0.dev11.tar.gz
```

Core regression: 333 passed, 4 skipped. Skips are three opt-in namespace tests and
one optional Inspect-native test. Prior pinned Inspect native acceptance was
27 passed; it is recorded in [its verification](inspect-adapter.md) and is not
counted as newly rerun here. All formatter/linter and selected-source checks pass.

The built wheel is installed without dependencies in a fresh local environment.
The checks run outside the source checkout: version, doctor, toolkit-demo, legacy
demo/audit, JSONL import/audit/compare, BudgetMix plan/audit and likelihood analysis.
The wheel contains both new public modules. The source archive includes their
examples and tutorials. The final manifest lists actual artifact hashes, commit
identity, public file count and checks performed, rather than previous dev10 values.

A complete main/HEAD Git bundle is verified and cloned independently; its commit
matches the current candidate. Wheel payloads rebuilt from the independently
unpacked sdist are byte-identical. These checks cover local handoff only: no remote
repository, push, hosted CI, model inference or package upload is claimed.

The process-enabled real 1.7B engineering evidence and known trainer off/on divergence
remain unchanged. Metadata and a destination remote can be supplied later; updating
metadata requires a fresh commit/package build before public publication.
