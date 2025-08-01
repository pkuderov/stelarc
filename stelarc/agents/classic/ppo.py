from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim.swa_utils import AveragedModel

from stelarc.agents.utils.lambda_return import LambdaReturn
from stelarc.agents.utils.soft_clip import SoftClip
from stelarc.agents.utils.torch import get_device


class Batch(SimpleNamespace):
    def __init__(self, n_envs, n_steps, obs_size):
        super().__init__()
        self.shape = (n_steps, n_envs)
        self.n_steps, self.n_envs = n_steps, n_envs
        self.size = n_steps * n_envs

        extra_step_shape = (n_steps + 1, n_envs)

        self.obs = self.make_tensor(extra_step_shape + (obs_size, ))

        self.a = self.make_tensor(dtype=torch.int)
        self.logprob = self.make_tensor()
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


class ActorCritic(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions, *args, **kwargs):
        super().__init__()

        # NB: for pi, expecting raw logits as outputs instead of probs
        self.actor = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs):
        pi_logits = self.actor(obs)
        pi = Categorical(logits=pi_logits)
        v = self.critic(obs).squeeze(-1)
        return pi, v


class SharedActorCritic(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super().__init__()

        # notice the same total number of layers for Separate and Shared ACs
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs, pi_f=None, v_f=None, dyn_f=None):
        z = self.encoder(obs)
        pi_logits = self.actor(z)
        pi = Categorical(logits=pi_logits)
        v = self.critic(z)
        return pi, v.squeeze(-1)


class Ppo:
    def __init__(
            self, ac_type, obs_size, hidden_size, n_actions,
            *, lr, betas,
            gamma, gae_lambda,
            K_epochs, mini_batch_size,
            eps_clip, v_clip=False,
            v_loss_alpha=None, ent_loss_alpha=0.1, ent_loss_alpha_decay=0.997,
            min_ent_loss_ratio=1 / 50.0,
            ema_lr=0.,
            device=None,
    ):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.lambda_return = LambdaReturn(gamma, gae_lambda)
        self.eps_clip = eps_clip
        self.v_clip_enabled = v_clip
        self.v_clip = SoftClip(eps=self.eps_clip, start_from=0.75)
        self.K_epochs = K_epochs
        self.mini_batch_size = mini_batch_size
        self.v_loss_alpha = v_loss_alpha if v_loss_alpha is not None else 1.0
        self.ent_loss_alpha_init = self.ent_loss_alpha = ent_loss_alpha
        self.ent_loss_alpha_decay = ent_loss_alpha_decay
        self.min_ent_loss_ratio = min_ent_loss_ratio

        device = get_device(device)
        self.policy = ac_type(obs_size, hidden_size, n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=betas)

        self.ema_lr = ema_lr
        self.ema_policy_enabled = 0.0 < ema_lr < 1.0
        if self.ema_policy_enabled:
            ema_step_fn = lambda avg_model_parameter, model_parameter, _: (
                    (1.0 - ema_lr) * avg_model_parameter + ema_lr * model_parameter
            )
            self.target_policy = AveragedModel(self.policy, avg_fn=ema_step_fn)
        else:
            # in case it's turned off, expose the same policy
            self.target_policy = self.policy

    def update(self, batch: Batch):
        # NB: data in batch is already no-grad
        if self.ema_policy_enabled:
            with torch.no_grad():
                # replace target policy distr logprobs with policy logprobs
                # for the entire batch
                pi, _ = self.policy(batch.obs[:-1])
                batch.logprob = pi.log_prob(batch.a)

        fl_batch = batch.flatten()
        clip = self.eps_clip
        stats_hist = defaultdict(list)

        for i_epoch in range(1, self.K_epochs + 1):
            for ixs in batch.split_ixs(self.mini_batch_size):
                obs, a, orig_logprob = fl_batch.obs[ixs], fl_batch.a[ixs], fl_batch.logprob[ixs]
                G, trunc = fl_batch.lambda_ret[ixs], fl_batch.trunc[ixs]

                pi, v = self.policy(obs)
                logprob = pi.log_prob(a)
                log_ratio = logprob - orig_logprob
                ratio = torch.exp(log_ratio)

                # term: G=0 -> correct td err —> learn V(s_term) = 0
                # trunc: zero out td err to turn off learning for this pseudo-step
                td_err = torch.where(trunc, 0., G - v)
                A = td_err.detach()

                # find surrogate loss:
                surr1 = ratio * A
                surr2 = ratio.clamp(1 - clip, 1 + clip) * A

                pi_loss = -torch.minimum(surr1, surr2).mean()

                # take squared TD (MC) err,..
                v_lr = self.v_loss_alpha
                sq_td_err = torch.pow(td_err, 2)

                # .. then soft clip it via log ratios from policy:
                # such clipping doesn't affect gradient flow, but rescales
                # learning rate resulting to fast vanishing gradients for "should-be-clipped" (s,a) pairs
                v_soft_clip_alpha, _ = self.v_clip(log_ratio.detach())
                if self.v_clip_enabled:
                    v_lr = v_lr * v_soft_clip_alpha
                # NB: apply lr before taking mean because learning rate is now sample-based
                v_loss = torch.mean(v_lr * sq_td_err)

                # calc entropy loss
                h_lr = self.ent_loss_alpha
                h_loss = -h_lr * pi.entropy().mean()

                loss = pi_loss + v_loss + h_loss

                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.optimizer.step()

                if i_epoch != self.K_epochs:
                    continue
                # save stats for logging
                with torch.no_grad():
                    abs_td_err = torch.abs(td_err)
                    clip_ratio = (torch.abs(ratio - 1.0) > clip).float()
                    kl_div = torch.abs(-log_ratio)

                    stats_keys = [
                        'pi', '|pi|', 'pi_clip', 'v', 'v_clip',
                        'h', '|kl|', 'A', '|TDErr|'
                    ]
                    stats_vals = [
                        pi_loss, torch.abs(pi_loss), clip_ratio,
                        v_loss, v_soft_clip_alpha, h_loss, kl_div,
                        A, torch.abs(td_err),
                    ]
                    for k, v in zip(stats_keys, stats_vals):
                        stats_hist[k].append(torch.mean(v).item())

        self.ent_loss_alpha *= self.ent_loss_alpha_decay
        self.ent_loss_alpha = max(
            self.ent_loss_alpha, self.ent_loss_alpha_init * self.min_ent_loss_ratio
        )

        # apply target policy slow update if enabled
        if self.ema_policy_enabled:
            self.target_policy.update_parameters(self.policy)
        return {k: np.mean(v) for k, v in stats_hist.items()}
