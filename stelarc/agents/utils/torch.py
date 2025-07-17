from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Distribution, Categorical


def get_device(device: str = None):
    if device is None:
        device = (
            "cuda:0" if torch.cuda.is_available() else
            "mps" if torch.mps.is_available() else
            "cpu"
        )
    return torch.device(device)


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def make_layers(
        name, input_size, layers, out_logits=False, activation=nn.SiLU,
        dtype=torch.float32,
        print_module=True,
):
    modules = []
    for output_size in layers:
        modules.extend((
            nn.Linear(input_size, output_size, dtype=dtype),
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
    base_decay: float = 0.5
    base_decay_schedule: float = 2.0

    def __init__(
            self, optimiser, learning_rate, *,

            # min LR can be defined either with a value
            #   < 1.0: abs min LR value
            #   >= 1.0: rel to initial, i.e. "init LR" / "min LR"
            #           (e.g. 10.0 means min LR ten times smaller)
            min_lr: float,

            # normalised params that determine the strength
            # of LR decay (higher slope — stronger decay)
            # and how quickly it reaches min LR (higher — faster)
            slope: float = 1.0, speed: float = 1.0
    ):
        self.lr = learning_rate
        if min_lr < 1.0:
            self.min_lr = min_lr
        else:
            self.min_lr = self.lr / min_lr

        self.decay = self.base_decay ** slope
        self.speed = self.base_decay_schedule / speed

        self.lr_scheduler = optim.lr_scheduler.MultiplicativeLR(
            optimiser, lr_lambda=self.get_decay,
        )
        self.epoch_step = 0
        self.n_epoch_steps = self.get_steps_in_epoch()
        print(f'Init LR: {self.lr:.5f} for {self.n_epoch_steps} steps')

    def step(self):
        self.epoch_step += 1

        enough = self.lr < self.min_lr
        early = self.epoch_step < self.n_epoch_steps
        if enough or early:
            return

        self.lr_scheduler.step()
        self.lr *= self.decay
        self.epoch_step = 0
        self.n_epoch_steps = self.get_steps_in_epoch()
        print(f'New LR: {self.lr:.5f} for {self.n_epoch_steps} steps')

    def get_steps_in_epoch(self):
        return int(self.speed / self.lr ** 1.2)

    def get_decay(self, _):
        return self.decay


class MultiCategorical(Distribution):
    """
    Provide a compact way to represent and work with multi-categorical distribution,
    i.e. with a list of Categorical distributions, such that you can work with it
    as if it was vectorised (under the hood it is not).
    """

    def __init__(self, dists: Sequence[Categorical]):
        super().__init__(validate_args=False)
        self.dists = dists

    def log_prob(self, values):
        res = [
            dist.log_prob(value)
            for dist, value in zip(self.dists, values.unbind(-1))
        ]
        return torch.stack(res, dim=-1)

    def entropy(self):
        return torch.stack([d.entropy() for d in self.dists], dim=-1)

    def normalised_entropy(self):
        def _normalise(h, size):
            return h / torch.log(h.new([size]))

        return torch.stack([
            _normalise(d.entropy(), d.probs.size(-1))
            for d in self.dists
        ], dim=-1)

    def sample(self, sample_shape=torch.Size(), greedy=False):
        # TODO: check if it correctly support sample_shape for both options
        if greedy:
            sampled = [d.mode.expand(d.mode.shape + sample_shape) for d in self.dists]
        else:
            sampled = [d.sample(sample_shape) for d in self.dists]

        return torch.stack(sampled, dim=-1)

    @staticmethod
    def from_logits(logits: torch.Tensor, spec: Sequence[int]):
        split_logits = torch.split(logits, list(spec), dim=-1)
        return MultiCategorical([
            Categorical(logits=sl) for sl in split_logits
        ])
