from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel

from stelarc.agents.utils.gae import GAE
from stelarc.agents.utils.soft_clip import SoftClip
from stelarc.agents.utils.torch import (
    make_layers, get_ema_step_fn,
    MultiCategorical
)


class Agent:
    def __init__(
            self, *,
            obs_size: int,
            layer_topology,
            policy_heads: list[tuple[str, int]],

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
        self.obs_size = obs_size
        self.action_size = len(policy_heads)

        print(policy_heads)
        self.action_space_names, pi_head_sizes = _split_policy_heads_declaration(policy_heads)
        # noinspection PyTypeChecker
        self.model = PpoModule(
            obs_size=obs_size,
            pi_heads=pi_head_sizes,
            dtype=dtype,
            **layer_topology
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, betas=adamw_betas
        )

        self.mini_batch_size = mini_batch_size
        self.batch_epochs = batch_epochs

        self.discount_gamma = discount_gamma
        self.gae = GAE(discount_gamma, gae_lambda)

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

    def update_old(self, batch):
        clip = self.ppo_pi_clip
        stats_hist = defaultdict(list)
        final_state = torch.empty_like(batch.final_state)

        for i_epoch in range(1, self.batch_epochs + 1):
            for ixs in batch.split_ixs(self.mini_batch_size):
                obs, a, orig_logprob = batch.obs[:, ixs], batch.a[:, ixs], batch.logprob[:, ixs]
                G, trunc = batch.lambda_ret[:, ixs], batch.trunc[:, ixs]
                state = batch.initial_state[ixs] if batch.initial_state is not None else None

                loss = 0.
                for i_step in range(batch.n_steps):
                    pi, v, state = self.model(obs[i_step], state)

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
                    td_err = torch.where(trunc, 0., G[i_step] - v)
                    A = td_err.detach()
                    # print(A.shape, ratio.shape)

                    # find surrogate loss:
                    surr1 = ratio * A
                    surr2 = ratio.clamp(1 - clip, 1 + clip) * A

                    pi_loss = -torch.minimum(surr1, surr2).mean()

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
                    v_loss = torch.mean(v_lr * sq_td_err)

                    # calc entropy loss
                    h_lr = self.ent_loss_scale
                    h_loss = -h_lr * pi.entropy().mean()

                    loss += pi_loss + v_loss + h_loss
                    if i_step == batch.n_steps - 1 and i_epoch == self.batch_epochs:
                        final_state[ixs] = state.detach()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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


class PpoModule(nn.Module):
    def __init__(
            self,
            obs_size: int,
            obs_encoder: list[int],
            mem_hidden_size: int,
            mem_skip_connection: bool,
            # TODO: split into shared_body + split_body
            body: list[int],
            pi_heads: tuple[int],
            dtype=torch.float32
    ):
        super().__init__()
        # TODO: support seeded initialisation
        self.n_classes = pi_heads[1]
        self._pi_heads = pi_heads
        self._pi_head_shifts = np.cumsum(pi_heads)

        # [optional] input_size -> enc_size
        enc_size, self.encoder = make_layers(
            name='Encoder', input_size=obs_size, layers=obs_encoder, dtype=dtype
        )

        # enc_size -> hid_size
        # self.rnn = nn.LSTMCell(enc_size, hid_size, dtype=float)
        self.rnn = nn.GRUCell(enc_size, mem_hidden_size, dtype=dtype)
        print('Memory: ', self.rnn)

        # body input is the input state for RL part,
        #   it aggregates observation + rnn state
        body_input_size = mem_hidden_size

        # [optional] skip connection from input to body bypassing rnn,
        #   and is added to rnn output
        self.skip_conn = None
        self.skip_conn_option = None
        if mem_skip_connection:
            if self.encoder is None:
                # no encoder: input_size -> hid_size (+ rnn output)
                self.skip_conn_option = 0
                _, self.skip_conn = make_layers(
                    name='Mem skip connection', input_size=obs_size, layers=[mem_hidden_size],
                    dtype=dtype
                )

            elif enc_size != mem_hidden_size:
                # has encoder, rnn input and hidden has diff size.
                #   I expect hidden > input ==> add projection layer to rnn and skip conn
                #   from enc to the output of the projection layer
                self.skip_conn_option = 1
                body_input_size, self.skip_conn = make_layers(
                    name='Mem output projection', input_size=mem_hidden_size, layers=[enc_size],
                    dtype=dtype
                )
            else:
                # has encoder, enc_size == hid_size ==> can pass enc via skip connection
                #   w/o any additional layers
                self.skip_conn_option = 2

        # Body + heads
        pi_enc_size, self.pi_body = make_layers(
            name='Policy body', input_size=body_input_size, layers=body, dtype=dtype
        )
        val_enc_size, self.val_body = make_layers(
            name='Value body', input_size=body_input_size, layers=body, dtype=dtype
        )

        _, self.pi_head = make_layers(
            name='Policy heads', input_size=pi_enc_size, layers=[sum(pi_heads)],
            out_logits=True, dtype=dtype
        )
        _, self.val_head = make_layers(
            name='Value head', input_size=val_enc_size, layers=[1],
            out_logits=True, dtype=dtype
        )

        # TODO: check if should be configurable
        self.body_skip = pi_enc_size == val_enc_size == body_input_size and len(body) > 0

    def forward(self, x, state=None):
        return self._predict(x, state)

    def _encode(self, x, state):
        e = self.encoder(x) if self.encoder is not None else x
        # h, _ = state = self.rnn(e, state)
        h = state = self.rnn(e, state)

        if self.skip_conn_option is None:
            # no skip connection
            u = h

        elif self.skip_conn_option == 0:
            # add mlp(input) to rnn hidden
            u = h + self.skip_conn(x)

        elif self.skip_conn_option == 1:
            # add enc to proj(rnn hidden)
            rnn_proj = self.skip_conn
            u = rnn_proj(h) + e

        elif self.skip_conn_option == 2:
            # enc_size == hid_size ==> simple skip conn from enc to rnn hidden
            u = h + e

        # noinspection PyUnboundLocalVariable
        return u, state

    def _predict(self, x, state):
        z, state = self._encode(x, state)

        u = self.pi_body(z) if self.pi_body is not None else z
        if self.body_skip:
            u = u + z
        pi = self.pi_head(u)

        v = self.val_body(z) if self.val_body is not None else z
        if self.body_skip:
            v = v + z
        val = self.val_head(v).squeeze()

        pi_distr = MultiCategorical.from_logits(logits=pi, spec=self._pi_heads)
        return pi_distr, val, state

    def _evaluate(self, x, state):
        z, state = self._encode(x, state)

        u = self.val_body(z) if self.val_body is not None else z
        if self.body_skip:
            u = u + z

        val = self.val_head(u).squeeze()
        return val, state

    @staticmethod
    def get_log_pi(log_prob, ans_mask, move_mask):
        log_pi = (
                log_prob[..., 0]
                + torch.where(ans_mask, log_prob[..., 1], 0.0)
                + torch.where(move_mask, log_prob[..., 2:].sum(-1), 0.0)
        )
        return log_pi


def _split_policy_heads_declaration(pi_heads: list[tuple[str, int]]) -> tuple[list[str], list[int]]:
    head_sizes, head_names = zip(*pi_heads)
    return list(head_names), list(head_sizes)


def _normalize_entropy(h, size):
    return h / torch.log(h.new([size]))
