"""Admission validation needs no Torch/NumPy imports."""

import pytest

from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.integrations.worker_seed import initialize_actor_rng


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5])
def test_invalid_seed_rejected_before_optional_libraries(seed):
    with pytest.raises(SchemaError):
        initialize_actor_rng(seed, cuda_device=None)
