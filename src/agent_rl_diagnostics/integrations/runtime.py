"""Optional in-process guard around the entire diagnostic job, not just sampling."""

import random
from contextlib import contextmanager


@contextmanager
def preserve_runtime_rng(device: str):
    import numpy as np
    import torch

    cpu = random.getstate()
    numpy = np.random.get_state()
    target = torch.device(device)
    devices = [target.index or 0] if target.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(cpu)
        np.random.set_state(numpy)
