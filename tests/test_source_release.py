import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "source_release", ROOT / "scripts/check_source_release.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_public_source_has_no_broken_links_and_reports_metadata_placeholder():
    result = module.check(ROOT)
    assert result["status"] == "verified_preview"
    assert isinstance(result["citation_uses_contributors_placeholder"], bool)
    assert not result["publishes_artifacts"]


@pytest.mark.parametrize("change", ["link", "private-path", "credential", "version", "symlink"])
def test_release_source_rejects_actual_public_payload_problems(tmp_path, change):
    project = tmp_path / "project"
    project.mkdir()
    for name in module.PUBLIC_FILES:
        shutil.copy2(ROOT / name, project / name)
    for name in module.PUBLIC_DIRS:
        shutil.copytree(
            ROOT / name, project / name, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info")
        )
    if change == "link":
        (project / "docs/broken.md").write_text("[Missing](nonexistent.md)")
    elif change == "private-path":
        (project / "docs/private.md").write_text("/" + "netdisk/test-user/model")
    elif change == "credential":
        (project / "configs/credential.txt").write_text("sk" + "-" + "a" * 40)
    elif change == "version":
        citation = project / "CITATION.cff"
        citation.write_text(
            citation.read_text().replace(
                f'version: "{module.check(ROOT)["version"]}"', 'version: "9.9.9"'
            )
        )
    else:
        external = tmp_path / "external.md"
        external.write_text("external file")
        (project / "docs/linked.md").symlink_to(external)
    result = module.check(project)
    assert result["status"] == "failed"
    assert result["errors"]
