from collections import defaultdict

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel

from stelarc.agents.utils.lambda_return import LambdaReturn
from stelarc.agents.utils.soft_clip import SoftClip
from stelarc.agents.utils.torch import (
    get_ema_step_fn
)
from stelarc.model import Model


class Agent:
    def __init__(
            self, *,
            model: Model,

            # learning
            learning_rate: float,
            lr_decay: float,
            lr_max_decay: float = 20.0,
            adamw_betas: tuple[float, float],
            mini_batch_size: int,
            batch_epochs: int = 4,

            # RL/PPO
            discount_gamma: float = 0.99,
            gae_lambda: float = 0.95,
            ppo_pi_clip: float,
            ppo_v_clip_enabled: bool = False,
            val_loss_scale: float = 0.25,

            ent_loss_scale: float = 0.1,
            ent_loss_heads_scale: tuple[float] = (1.0, 0.2, 0.5, 0.5),
            ent_loss_scale_decay=0.997,
            ent_loss_scale_max_decay: float = 40.0,

            weights_ema_lr: float = 0.,

            # general
            device=None,
            seed: int = None,
            dtype=torch.float32
    ):
        # TODO: support random seeding
        self.obs_size = model.obs_size
        self.action_size = model.action_size

        self.action_names = model.action_names
        self.model = model.to(device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, betas=adamw_betas
        )

        self.mini_batch_size = mini_batch_size
        self.batch_epochs = batch_epochs

        self.discount_gamma = discount_gamma
        self.lambda_return = LambdaReturn(discount_gamma, gae_lambda)

        self.ppo_pi_clip = ppo_pi_clip
        self.v_clip_enabled = ppo_v_clip_enabled
        self.ppo_v_clip = SoftClip(eps=self.ppo_pi_clip, start_from=0.75)
        self.val_loss_scale = val_loss_scale

        self.ent_loss_heads_scale = ent_loss_heads_scale
        self.ent_loss_scale_init = self.ent_loss_scale = ent_loss_scale
        self.ent_loss_scale_decay = ent_loss_scale_decay
        self.ent_loss_scale_max_decay = ent_loss_scale_max_decay
        self._ent_scales = torch.from_numpy(np.array(ent_loss_heads_scale)).to(dtype=dtype)
        self._ent_scales.requires_grad_(False)

        self.ema_policy_enabled = 0.0 < weights_ema_lr < 1.0
        if self.ema_policy_enabled:
            ema_step_fn = get_ema_step_fn(weights_ema_lr)
            self.target_model = AveragedModel(self.model, avg_fn=ema_step_fn)
        else:
            # in case it's turned off, expose the same policy
            self.target_model = self.model

    def update(self, batch):
        clip = self.ppo_pi_clip
        stats_hist = defaultdict(list)
        final_state = None

        for i_epoch in range(1, self.batch_epochs + 1):
            obs, a, orig_logprob = batch.obs, batch.a, batch.logprob
            G, trunc, reset = batch.lambda_ret, batch.trunc, batch.reset
            state = batch.initial_state if batch.initial_state is not None else None
            loss = 0.
            for i_step in range(batch.n_steps):
                pi, v, state = self.model(obs[i_step], state)
                enabled = torch.logical_not(reset[i_step])

                move_mask = a[i_step, ..., 0] == 1
                ans_mask = a[i_step, ..., 0] == 2
                log_pi = self.model.get_log_pi(
                    pi.log_prob(a[i_step]), ans_mask=ans_mask, move_mask=move_mask
                )
                log_pi_old = self.model.get_log_pi(
                    orig_logprob[i_step], ans_mask=ans_mask, move_mask=move_mask
                )

                log_ratio = log_pi - log_pi_old
                ratio = torch.exp(log_ratio)

                # term: G=0 -> correct td err —> learn V(s_term) = 0
                # trunc: zero out td err to turn off learning for this pseudo-step
                td_err = torch.where(trunc[i_step], 0., G[i_step] - v)
                A = td_err.detach()
                # print(A.shape, ratio.shape)

                # find surrogate loss:
                surr1 = ratio * A
                surr2 = ratio.clamp(1 - clip, 1 + clip) * A

                pi_loss = -torch.minimum(surr1, surr2)[enabled].mean()

                # take squared TD (MC) err,...
                v_lr = self.val_loss_scale
                sq_td_err = torch.pow(td_err, 2)

                # ... then soft clip it via log ratios from policy:
                # such clipping doesn't affect gradient flow, but rescales
                # learning rate resulting to fast vanishing gradients for
                # "should-be-clipped" (s,a) pairs
                v_soft_clip_alpha, _ = self.ppo_v_clip(log_ratio.detach())
                if self.v_clip_enabled:
                    v_lr = v_lr * v_soft_clip_alpha
                # NB: apply lr before taking mean because learning rate is now sample-based
                v_loss = v_lr * sq_td_err
                v_loss = torch.mean(v_loss[enabled])

                # calc entropy loss
                h_lr = self.ent_loss_scale
                h_loss = -h_lr * (pi.normalised_entropy() @ self._ent_scales)[enabled].mean()

                loss += pi_loss + v_loss + h_loss

                state = self.reset_state(state, reset[i_step])
                if i_step == batch.n_steps - 1 and i_epoch == self.batch_epochs:
                    final_state = state.detach()

            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if i_epoch != self.batch_epochs:
                continue
            # save stats for logging
            with torch.no_grad():
                abs_td_err = torch.abs(td_err)
                clip_ratio = (torch.abs(ratio - 1.0) > clip).float()
                kl_div = torch.abs(torch.mean(log_pi - log_pi_old))

                stats_keys = ['pi', '|pi|', 'pi_clip', 'v', 'v_clip', 'h', 'kl', 'A', '|TDErr|']
                stats_vals = [
                    pi_loss, torch.abs(pi_loss), clip_ratio,
                    v_loss, v_soft_clip_alpha, h_loss, kl_div,
                    A, abs_td_err,
                ]
                for k, v in zip(stats_keys, stats_vals):
                    stats_hist[k].append(torch.mean(v).item())

        batch.final_state = final_state

        self.decay_ent_los_scale()
        # apply target policy slow update if enabled
        if self.ema_policy_enabled:
            self.target_model.update_parameters(self.model)

        return {k: np.mean(v) for k, v in stats_hist.items()}

    def decay_ent_los_scale(self):
        self.ent_loss_scale *= self.ent_loss_scale_decay
        self.ent_loss_scale = max(
            self.ent_loss_scale, self.ent_loss_scale_init / self.ent_loss_scale_max_decay
        )

    @staticmethod
    def reset_state(state, reset_mask):
        # TODO: use torch.where + init_state fixed random vector
        if isinstance(reset_mask, np.ndarray):
            reset_mask = torch.from_numpy(reset_mask)
        not_done = torch.logical_not(reset_mask).view(-1, 1)
        return state * not_done


def _normalize_entropy(h, size):
    return h / torch.log(h.new([size]))
