import importlib.util
import zipfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "release_check", Path(__file__).parents[1] / "scripts/check_release_archive.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("member", ["runs/raw.json", "models/weights.safetensors", "../escape"])
def test_archive_rejects_runtime_assets_and_traversal(tmp_path, member):
    path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, "{}")
    with pytest.raises(ValueError):
        module.check(path)


def test_wheel_code_rejects_private_paths(tmp_path):
    path = tmp_path / "fixture.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("agent_rl_diagnostics/models/private.py", 'root = "/' + 'netdisk/private"')
    with pytest.raises(ValueError, match="research path"):
        module.check(path)


def test_archive_rejects_hidden_credentials_and_zip_links(tmp_path):
    for member in (".env", ".aws/credentials", ".ssh/id_rsa"):
        path = tmp_path / "fixture.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(member, "private")
        with pytest.raises(ValueError):
            module.check(path)
    with zipfile.ZipFile(path, "w") as archive:
        entry = zipfile.ZipInfo("link")
        entry.create_system = 3
        entry.external_attr = 0o120777 << 16
        archive.writestr(entry, "target")
    with pytest.raises(ValueError, match="links"):
        module.check(path)
