import os
import subprocess
import sys
from pathlib import Path


def test_native_process_help_is_available_without_optional_libraries():
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/verify_process_signal_cpu.py"), "--help"],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert "--verl-source" in result.stdout
    assert "No Ray cluster" in result.stdout
