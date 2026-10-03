# Inspect adapter acceptance

Verified locally on 2026-10-02 with Python 3.12.4 and inspect-ai 0.3.275. Changes
are unreleased and do not alter historical dev10 bundles. The pinned optional
extra does not add dependencies to the core.

Executed from the staged source checkout:

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src /tmp/ard-inspect-env/bin/python -m pytest tests/test_inspect_logs.py -q
ruff check .
ruff format --check src tests scripts examples
PYTHONPATH=src python scripts/check_source_release.py
```

Core regression: **333 passed, 4 skipped** (three opt-in namespace tests and the
optional native Inspect test). In the isolated Inspect environment: **27 passed**,
including the native tool-loop test. That test executes four task attempts, checks
original .eval file hashes, per-sample tool messages and retry counts, and imports
the SDK JSON export through `ard traces inspect`. It confirms both success rates
and unknown cost handling without claiming model efficacy.

Also executed:

```bash
XDG_DATA_HOME=/tmp/ard-inspect-data XDG_CACHE_HOME=/tmp/ard-inspect-cache PYTHONPATH=src /tmp/ard-inspect-env/bin/python examples/inspect_inventory.py --output /tmp/ard-inspect-case-v6
python -c 'from setuptools.build_meta import build_wheel, build_sdist; build_wheel("dist"); build_sdist("dist")'
python scripts/check_release_archive.py dist/agent_rl_diagnostics-0.1.0.dev10-py3-none-any.whl
python scripts/check_release_archive.py dist/agent_rl_diagnostics-0.1.0.dev10.tar.gz
```

The case's real SQLite tools run via Inspect with scripted model responses. The
base pen query stops after the injected temporary error; the retry wrapper makes
two backend attempts and returns 7. The notebook query returns 4 in both runs.
Synthetic terminal success rates are 0.5 and 1.0. Scripted tokens are explicitly
zero; monetary usage is unknown. This is an authored integration example, not a
business benchmark, controlled model recovery experiment or numerical replication.

Installed the built wheel without dependencies in `/tmp/ard-inspect-env` (which
already contained the optional SDK). From `/tmp` in Python isolated mode (`-I`), without the checkout on
PYTHONPATH, its CLI imported the native base .eval, audited the exported records,
and compared them with the candidate. Output:
`/tmp/ard-inspect-wheel-isolated-comparison/report.html`. Archive scans passed for 78 wheel
and 229 sdist members; scope is known paths/sizes/secret patterns, not a security
certification. Temporary paths record execution, not user prerequisites.

Initial environment probes found a read-only SDK data directory and a tokenizer
fetch in the mock provider. SDK data/cache paths were isolated under /tmp and
scripted usage made explicit. Native path-based log reading stalled; the adapter
uses the public reader's supported `IO[bytes]` input with explicit format, which
passed .eval and JSON acceptance. No upstream internals are imported by the adapter.

The live-model entry point, other SDK versions and real-company tasks remain
unverified. No model inference, GPU use or publication occurred. The optional
GitHub Actions job has been added but has not run remotely.
