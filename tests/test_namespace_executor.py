"""Explicit opt-in: namespace setup is unavailable inside some CI sandboxes."""

import os

import pytest

from agent_rl_diagnostics.environments.execution import NamespaceExecutor

pytestmark = pytest.mark.skipif(
    os.environ.get("ARD_RUN_NAMESPACE_TESTS") != "1",
    reason="set ARD_RUN_NAMESPACE_TESTS=1 on a namespace-enabled Linux host",
)


def test_real_stdin_execution_and_program_error():
    executor = NamespaceExecutor()
    assert executor.run("print(sum(map(int,input().split())))", "17 29\n")["stdout"] == "46\n"
    assert executor.run("this is not Python !!!")["status"] == "program_error"


def test_host_paths_and_network_are_not_exposed():
    executor = NamespaceExecutor()
    code = "from pathlib import Path\nprint(Path('/netdisk').exists(), Path('/home').exists())\nimport socket\ns=socket.socket()\ns.settimeout(0.1)\ntry:\n s.connect(('1.1.1.1',80))\n print('NETWORK_EXPOSED')\nexcept OSError:\n print('isolated')"
    result = executor.run(code)
    assert result["status"] == "ok"
    assert result["stdout"] == "False False\nisolated\n"


def test_timeout_and_output_bound():
    executor = NamespaceExecutor(timeout_seconds=0.5, output_bytes=1024)
    assert executor.run("while True: pass")["status"] == "timeout"
    result = executor.run("print('x'*100000)")
    assert len(result["stdout"].encode()) <= 1024
    assert result["status"] == "program_error"
