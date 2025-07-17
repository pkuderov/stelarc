from collections import defaultdict

import numpy as np
import torch
from torch import nn

from stelarc.agents.utils.lambda_return import LambdaReturn, RlFlags
from stelarc.agents.utils.torch import DynamicLearningRateScaler
from stelarc.model import Model


class Agent(nn.Module):
    def __init__(
            self, *,
            model: Model,

            # learning
            learning_rate: float,
            lr_min: float = 20.0,
            lr_decay_slope: float = 1.0,
            lr_decay_speed: float = 1.0,
            adamw_betas: tuple[float, float] = (0.9, 0.999),

            # RL/Losses
            discount_gamma: float = 0.99,
            gae_lambda: float = None,
            val_loss_scale: float = 0.25,

            ent_loss_scale: float = 0.1,
            ent_loss_heads_scale: tuple[float] = (1.0, 0.2, 0.5, 0.5),
            ent_loss_scale_decay=1.0,
            ent_loss_scale_max_decay: float = 1.0,

            # general
            device=None,
            seed: int = None,
            dtype=torch.float32,
    ):
        super().__init__()
        # TODO: support random seeding

        self.device = device
        self.obs_size = model.obs_size
        self.action_size = model.action_size

        self.action_names = model.action_names
        self.model = model.to(device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, betas=adamw_betas
        )
        self.lr_scheduler = DynamicLearningRateScaler(
            self.optimizer, learning_rate,
            min_lr=lr_min, slope=lr_decay_slope, speed=lr_decay_speed
        )

        self.discount_gamma = discount_gamma
        self.lambda_return = LambdaReturn(discount_gamma, gae_lambda)

        self.val_loss_scale = val_loss_scale

        self.ent_loss_scale_init = self.ent_loss_scale = ent_loss_scale
        self.ent_loss_scale_decay = ent_loss_scale_decay
        self.ent_loss_scale_max_decay = ent_loss_scale_max_decay

        ent_loss_heads_scale = np.array(ent_loss_heads_scale)
        self.ent_loss_heads_scale = torch.from_numpy(ent_loss_heads_scale).to(dtype=dtype)

        self._loss = 0.
        self._last_loss = 0.
        self._batch_cnt = 0

    def forward(self, x, state=None):
        # outer no_grads have higher priority: to have an ability to
        # calc in no_grad mode something during training
        enable_grads = torch.is_grad_enabled() and self.training
        with torch.set_grad_enabled(enable_grads):
            return self.model(x, state)

    @torch.no_grad()
    def evaluate(self, x, state=None):
        return self.model.evaluate(x, state)

    # noinspection PyPep8Naming
    def update(self, batch):
        stats_hist = defaultdict(list)

        def _slice(ix, mask, *args):
            return (x[ix][mask] for x in args)

        loss = 0.
        for t in range(batch.n_steps):
            flags_tn = batch.flags[t]
            reset_tn = flags_tn & RlFlags.RESET
            # we exclude T -> 0 between-two-episodes transitions
            enabled = torch.logical_not(reset_tn.bool())

            a, logprob, entropy, V, G = _slice(
                t, enabled, batch.a, batch.logprob, batch.entropy, batch.v, batch.lambda_ret
            )

            assert not G.requires_grad and V.requires_grad
            td_err = G - V
            A = td_err.detach()

            act_type = a[..., 0]
            _, move_mask, ans_mask = self.model.split_action_type(act_type, move=True, answer=True)
            log_pi = self.model.get_log_pi(logprob, move_mask=move_mask, ans_mask=ans_mask)

            pi_loss = -torch.mean(log_pi * A)

            # print(act_type)
            # print('M', move_mask)
            # print('A', ans_mask)
            # print(logprob)
            # print(log_pi)
            # print('=======')

            v_scale = self.val_loss_scale
            v_loss = v_scale * torch.square(td_err).mean()

            # entropy loss
            h_scale = self.ent_loss_scale
            # weighted sum over the action heads
            h_aggregated = torch.mv(entropy, self.ent_loss_heads_scale)
            h_loss = -h_scale * h_aggregated.mean()

            loss += pi_loss + v_loss + h_loss

        loss /= batch.n_steps
        self.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # save stats for logging
        with torch.no_grad():
            # abs_td_err = torch.abs(td_err)
            abs_td_err = v_loss / v_scale

            stats_keys = [
                'pi',
                # '|pi|',
                'v', 'h',
                # 'A',
                # '|TDErr|'
            ]
            stats_vals = [
                pi_loss,
                # torch.abs(pi_loss),
                v_loss, h_loss,
                # A,
                # abs_td_err,
            ]
            for k, v in zip(stats_keys, stats_vals):
                stats_hist[k].append(torch.mean(v).item())

        self.decay_ent_los_scale()
        self.lr_scheduler.step()

        return {k: np.mean(v) for k, v in stats_hist.items()}

    def decay_ent_los_scale(self):
        self.ent_loss_scale *= self.ent_loss_scale_decay
        self.ent_loss_scale = max(
            self.ent_loss_scale, self.ent_loss_scale_init / self.ent_loss_scale_max_decay
        )

    def reset_state(self, state, reset_mask):
        if isinstance(reset_mask, np.ndarray):
            reset_mask = torch.from_numpy(reset_mask)
        return self.model.reset_state(state, reset_mask)

    def detach_state(self, state):
        return self.model.detach_state(state)


def _normalize_entropy(h, size):
    return h / torch.log(h.new([size]))


def _get_entropy(p, dim=-1, keepdim=False, normalize=False):
    single_val = p.shape[dim] == 1
    if normalize and not single_val:
        p = p / (p.sum(-1, keepdim=True) + 1e-6)

    def _entr(p):
        return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

    H = _entr(p) if not single_val else _entr(p) + _entr(1.0 - p)
    n = max(2, p.shape[dim])
    return H / torch.log(H.new([n]))
