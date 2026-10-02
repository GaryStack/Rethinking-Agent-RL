import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.integrations.grid_fixture import TRAIN_GRID_TASKS, check_grid_holdout


def test_authored_fixture_holdout_ids_and_map_contents_checked():
    check_grid_holdout([{"task_id": "heldout", "desc": ["SFG"]}])
    for field in ("task_id", "desc"):
        task = {"task_id": "heldout", "desc": ["SFG"]}
        task[field] = copy.deepcopy(TRAIN_GRID_TASKS[0][field])
        with pytest.raises(SchemaError):
            check_grid_holdout([task])


def test_monitor_dry_run_no_output_no_optional_imports(tmp_path):
    root = Path(__file__).parents[1]
    code = """
import runpy,sys
sys.argv = ['prepare_ray_monitor.py', '--model-path', sys.argv[1], '--template',
            sys.argv[2], '--output', sys.argv[3], '--dry-run']
runpy.run_path(sys.argv[0],run_name='__main__')
assert not any(k.split('.')[0] in {'torch','ray','vllm','verl','numpy','wandb'} for k in sys.modules)
"""
    # Run the script's dry-run in a clean core interpreter.
    path = root / "scripts/prepare_ray_monitor.py"
    code = code.replace("runpy.run_path(sys.argv[0]", f"runpy.run_path({str(path)!r}")
    output = tmp_path / "absent"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(tmp_path),
            str(root / "configs/examples/frozenlake_ray_monitor.json"),
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not output.exists() and not json.loads(result.stdout)["inference_executed"]
