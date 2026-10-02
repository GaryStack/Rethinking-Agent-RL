from pathlib import Path

from agent_rl_diagnostics.artifacts.io import read_json
from agent_rl_diagnostics.integrations.ray_paths import create_ray_runtime_directory


def test_deep_result_path_uses_private_short_runtime_without_deleting_logs(tmp_path):
    output = tmp_path / ("deep" * 30)
    output.mkdir()
    runtime = create_ray_runtime_directory(output)
    assert runtime.is_dir() and runtime.parent == Path("/tmp")
    socket = runtime / "session_2026-10-01_12-41-10_490376_1234567/sockets/plasma_store"
    assert len(str(socket).encode()) <= 107
    record = read_json(output / "ray-runtime.json")
    assert record["temp_dir"] == str(runtime) and record["retained_after_shutdown"]
    # Remove only this empty test-created directory; actual run logs are retained.
    runtime.rmdir()
