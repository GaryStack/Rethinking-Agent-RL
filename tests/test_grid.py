from dataclasses import replace

import pytest

from agent_rl_diagnostics.contracts import SchemaError
from agent_rl_diagnostics.environments.frozenlake import GridEnvironment, GridTask, finite_horizon_q


def test_bfs_exact_minimum_and_boundary_horizon():
    task = GridTask("corridor", ("SFG",))
    assert task.shortest_distance() == 2
    certificate = task.certificate()
    assert certificate.kind == "exact_minimum"
    assert certificate.value == 2
    assert not certificate.unreachable
    q = finite_horizon_q(task, 2, gamma=1)
    assert max(q[1][task.start]) == 0
    assert q[2][task.start][2] == 1


def test_unreachable_is_proven_not_unknown():
    task = GridTask("blocked", ("SHG",))
    certificate = task.certificate()
    assert certificate.value is None
    assert certificate.unreachable
    assert certificate.kind == "exact_minimum"
    assert max(finite_horizon_q(task, 6, gamma=1)[6][task.start]) == 0


def test_success_probability_is_not_discounted_q():
    task = GridTask("corridor", ("SFG",))
    assert finite_horizon_q(task, 2, gamma=0.95)[2][0][2] == pytest.approx(0.95)
    assert finite_horizon_q(task, 2, gamma=1)[2][0][2] == 1
    # Intended RIGHT has only 1/3 probability at each step under slippery dynamics.
    assert finite_horizon_q(task, 2, gamma=1, slippery=True)[2][0][2] == pytest.approx(1 / 9)


def test_goal_reward_is_counted_once():
    task = GridTask("adjacent", ("SG",))
    table = finite_horizon_q(task, 8, gamma=0.95)
    assert table[1][0][2] == 1
    assert table[8][0][2] == 1
    assert table[8][1] == (0, 0, 0, 0)


def test_snapshot_restores_state_and_action_count_without_leakage():
    task = GridTask("corridor", ("SFG",))
    env = GridEnvironment(task)
    before = env.export_snapshot()
    env.step(2)
    env.step(2)
    env.restore_snapshot(before)
    assert env.state == task.start
    assert env.action_count == 0
    assert env.step(2) == (1, 0, False, False)
    with pytest.raises(SchemaError, match="mismatch"):
        GridEnvironment(GridTask("other", ("SG",))).restore_snapshot(before)


@pytest.mark.parametrize("desc", [("SSG",), ("SFX",), ("SF", "G"), ("SGG",)])
def test_invalid_maps_rejected(desc):
    with pytest.raises(SchemaError):
        GridTask("invalid", desc)


def test_inconsistent_certificate_rejected():
    certificate = GridTask("corridor", ("SFG",)).certificate()
    with pytest.raises(SchemaError):
        replace(certificate, value=None)
    with pytest.raises(SchemaError):
        replace(certificate, unreachable=True)
