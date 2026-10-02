"""Explicit one-time initialization, never a reset at capture/update boundaries."""

import random

from ..contracts.schema import SchemaError, integer


def initialize_actor_rng(seed, *, cuda_device=0):
    integer(seed, "actor_seed")
    if seed > 2**32 - 1:
        raise SchemaError("actor_seed must fit NumPy's uint32 seed range")
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_device is not None:
        with torch.cuda.device(cuda_device):
            torch.cuda.manual_seed(seed)
    return {
        "schema_version": "ard-worker-initialization-v1",
        "protocol": "one_time_before_native_init_model",
        "seed": seed,
        "cuda_device": cuda_device,
        "rng_domains": ["python", "numpy", "torch_cpu"]
        + (["torch_cuda_local"] if cuda_device is not None else []),
        "full_trainer_parity_verified": False,
    }
