import numpy as np
from torch import nn


def make_layers(
        name, input_size, layers, out_logits=False, activation=nn.SiLU,
        print_module=True
):
    modules = []
    for output_size in layers:
        modules.extend((
            nn.Linear(input_size, output_size, dtype=float),
            activation()
        ))
        input_size = output_size

    if len(modules) > 0:
        if out_logits:
            modules.pop()
        modules = nn.Sequential(*modules)

    if print_module:
        print(f'{name}: ', modules)

    if len(modules) == 0:
        modules = None
    return input_size, modules


def concat_obs_parts(obs_parts):
    obs = [o.reshape(o.shape[0], -1) for o in obs_parts]
    obs = np.hstack(obs, dtype=float)
    return obs
