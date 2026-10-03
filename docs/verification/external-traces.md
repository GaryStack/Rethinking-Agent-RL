# External trace workflow verification

Verified locally on 2026-10-02, Python 3.12.4. Changes are unreleased; historical
dev10 bundles are unchanged. This is an offline recorded-data workflow, not real
agent execution or enterprise acceptance.

Commands executed from the staged source checkout:

```bash
PYTHONPATH=src python -m pytest -q
ruff check .
ruff format --check src tests scripts examples
PYTHONPATH=src python scripts/check_source_release.py
python -c 'from setuptools.build_meta import build_wheel, build_sdist; build_wheel("dist"); build_sdist("dist")'
python scripts/check_release_archive.py dist/agent_rl_diagnostics-0.1.0.dev10-py3-none-any.whl
python scripts/check_release_archive.py dist/agent_rl_diagnostics-0.1.0.dev10.tar.gz
```

Full regression: **307 passed, 3 skipped**. Namespace executor tests remain opt-in.
The 23 new tests cover mapped imports, missing slots, failure classification,
resource coverage, invalid values, duplicate samples, incompatible protocols,
hash/recomputation checks, escaped trace HTML, paired differences and CLI gates.
After final report metadata changes, all 23 new tests passed again.

Ran the [tutorial](../tutorials/external-traces.md) import/compare commands with
outputs under `/tmp/ard-product-example-*`. Gate passed; synthetic success delta
was +0.5, with a two-task bootstrap interval [0, 1]. This is not efficacy evidence.

Built the wheel again after report changes, installed it without dependencies in
`/tmp/ard-product-wheel-env`, and ran `ard traces audit` and `ard traces compare`
from `/tmp`, outside the checkout. Report output:
`/tmp/ard-product-wheel-comparison/report.html`. Temporary local paths are execution
records, not distributable artifacts or prerequisites for users.

No GPU runs, external uploads, publication, or framework-adapter acceptance were
performed. Imported source rows are retained in local reports; users must choose
appropriate data before sharing them. Hash checks do not authenticate outcomes.
