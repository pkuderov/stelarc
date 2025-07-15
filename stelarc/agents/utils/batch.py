from types import SimpleNamespace

import torch


class Batch(SimpleNamespace):
    def __init__(self, n_envs, n_steps, obs_size, action_size):
        super().__init__()
        self.shape = (n_steps, n_envs)
        self.n_steps, self.n_envs = n_steps, n_envs
        self.size = n_steps * n_envs

        extra_step_shape = (n_steps + 1, n_envs)
        action_shape = (n_steps, n_envs, action_size)

        self.obs = self.make_tensor(extra_step_shape + (obs_size, ))

        self.a = self.make_tensor(action_shape, dtype=torch.int)
        self.logprob = self.make_tensor(action_shape)
        self.v = self.make_tensor(extra_step_shape)

        self.r = self.make_tensor()
        self.term = self.make_tensor(extra_step_shape, dtype=torch.bool)
        self.trunc = self.make_tensor(extra_step_shape, dtype=torch.bool)
        self.reset = self.make_tensor(dtype=torch.bool)

        self.lambda_ret = self.make_tensor()

    def put(self, i_step, **kwargs):
        i = i_step
        for k, v in kwargs.items():
            self.__dict__[k][i] = v

    def split_ixs(self, mini_batch_size):
        order = torch.randperm(self.size)
        return torch.tensor_split(order, self.size // mini_batch_size)

    def flatten(self):
        def _flatten(a):
            return a.view(a.shape[0] * a.shape[1], *a.shape[2:])

        return SimpleNamespace(
            **{
                k: _flatten(v[:self.n_steps])
                for k, v in self.__dict__.items()
                if k in {'obs', 'a', 'logprob', 'trunc', 'lambda_ret', 'z'}
            }
        )

    def make_tensor(
            self, shape=None, *,
            dtype=torch.float, requires_grad=False, device=None
    ):
        if shape is None:
            shape = self.shape
        return torch.empty(
            shape, dtype=dtype, device=device,
            requires_grad=requires_grad
        )


class RnnBatch(Batch):
    def __init__(self, n_envs, n_steps, obs_size, action_size):
        super().__init__(n_envs, n_steps, obs_size, action_size)

        self.initial_state = None
        self.final_state = None

    def split_ixs(self, mini_batch_size):
        # split over "envs" axis in chunks of full rollouts
        order = torch.randperm(self.n_envs)

        n_rollouts = mini_batch_size // self.n_steps
        return torch.tensor_split(order, self.n_envs // n_rollouts)

    def flatten(self):
        def _flatten(a):
            return a.view(a.shape[0] * a.shape[1], *a.shape[2:])

        return SimpleNamespace(
            **{
                k: _flatten(v[:self.n_steps])
                for k, v in self.__dict__.items()
                if k in {'obs', 'a', 'logprob', 'trunc', 'lambda_ret', 'z'}
            }
        )
