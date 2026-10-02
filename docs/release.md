# Local preview release

Version: `0.1.0.dev10`. The project is a paper companion / diagnostic toolkit,
not full numerical reproduction. A local wheel and source distribution are prepared
under `dist/`; neither has been uploaded. See `docs/verification/dev10.md` for the
executed checks. The source package carries tutorials, fixtures and static reports;
the wheel carries only the dependency-free library and packaged demo resource.
The unpublished candidate has a [CPU walkthrough refresh](verification/toolkit-walkthrough.md):
`ard toolkit-demo` is available directly from the wheel; likelihood CLI length
control retains failed/missing event exclusions. No version was advanced for
this local packaging refresh.

Check public source, then build and inspect from a clean checkout:

```bash
python scripts/check_source_release.py
python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
python -c 'from setuptools.build_meta import build_sdist; print(build_sdist("dist"))'
python scripts/check_release_archive.py dist/agent_rl_diagnostics-0.1.0.dev10-py3-none-any.whl
python scripts/check_release_archive.py dist/agent_rl_diagnostics-0.1.0.dev10.tar.gz
```

The build backend must already satisfy `setuptools>=68`; use an isolated build
environment if it does not. Archive inspection checks payload names, runtime/binary
assets, sizes and known credential patterns. It is not a guarantee that arbitrary
text contains no sensitive data. Personal model/interpreter paths in the public historical verification narrative
use portable placeholders; unmodified records remain in ignored private evidence.
Source, scripts and configs have no hardcoded personal research paths. Do not scan or copy the original private experiment trees.

The following remain explicit before an external release:

- Confirm citation authors, paper URL and final version. `CITATION.cff` retains a
  clearly identified contributors placeholder; no person or paper URL is invented.
- Retain the documented process CPU and narrow real 1.7B one-rank GPU scope,
  deferred PPO/GiGPO and known trainer divergence in the release notes. Source tests and
  engineering smoke do not establish formal numerical reproduction.
- Review selected public artifacts and permissions. Large results, weights,
  private tests, original logs, W&B binaries and caches are excluded. New examples
  are authored fixtures; real fixed-response smoke output stays under ignored runs.
- Request explicit author instruction for remote repository creation, push,
  package upload or online W&B. These actions have not been performed.

CI runs CPU tests/format checks, demo/audit, BudgetMix, comparison, likelihood,
external scorer and process/dynamics fixtures. Remote CI has not run because the
repository is local. Namespace checks remain opt-in; optional HF and native VeRL
checks are separately recorded rather than counted as default CPU tests.

The local candidate also includes the [real process GPU verification](verification/process-gpu.md)
and native score-export audit. Private raw evidence remains under ignored `runs/`;
no weights, offline W&B binaries or remote credentials are added to distributions.

The public source includes [Chinese quickstart](../README.zh-CN.md),
[changes](../CHANGELOG.md) and locally checked relative document links.
The initial local commit and `dist/agent-rl-diagnostics-0.1.0.dev10.bundle`
provide a reviewable Git handoff. Clone it without a remote account:

```bash
git clone /path/to/agent-rl-diagnostics-0.1.0.dev10.bundle agent-rl-diagnostics
cd agent-rl-diagnostics
python -m pip install -e .
ard toolkit-demo --output runs/toolkit-walkthrough
```

The bundle contains selected public source only, with no `runs/`, distributions
or credentials in its commit. Its origin points to the local bundle and must be
replaced with the chosen destination before a later authorized push. Citation
uses the stated collective contributors placeholder until the author supplies
final metadata; no paper link, repository URL or individual author was invented.
