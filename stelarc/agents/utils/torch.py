import numpy as np
import torch
from torch import nn, optim


def get_device(device: str = None):
    if device is None:
        device = (
            "cuda:0" if torch.cuda.is_available() else
            # "mps" if torch.mps.is_available() else
            "cpu"
        )
    return torch.device(device)


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


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


def get_ema_step_fn(weights_ema_lr):
    def ema_step_fn(avg_model_parameter, model_parameter, _):
        lr = weights_ema_lr
        p, avg_p = model_parameter, avg_model_parameter
        return (1.0 - lr) * avg_p + lr * p

    return ema_step_fn


class DynamicLearningRateScaler:
    def __init__(self, optimiser, learning_rate, decay, max_decay_scale):
        self.lr = learning_rate
        self._min_lr = self.lr / max_decay_scale

        self.lr_scheduler = optim.lr_scheduler.MultiplicativeLR(
            optimiser, lr_lambda=self.get_decay,
        )
        self._lr_epoch_step = 0
        self._lr_epoch_steps = self.get_steps_in_epoch(learning_rate)

    @staticmethod
    def get_steps_in_epoch(lr):
        return int(7.0 / lr)

    @staticmethod
    def get_decay(_):
        return 0.8
