from types import SimpleNamespace

import numpy as np
import torch


def get_seed(rng = None):
    if rng is None:
        return get_seed(np.random.default_rng())
    return int(rng.integers(1_000_000))


def set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)


def ns_to_dict(ns):
    if not isinstance(ns, SimpleNamespace):
        return ns

    return {
        k: ns_to_dict(v)
        for k, v in ns.__dict__.items()
    }
