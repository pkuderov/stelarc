from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim.swa_utils import AveragedModel

from stelarc.agents.utils.gae import GAE
from stelarc.agents.utils.soft_clip import SoftClip
from stelarc.agents.utils.torch import make_layers, concat_obs_parts, get_ema_step_fn


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
    ):
        # TODO: support random seeding

        self.action_space_names, pi_head_sizes = _split_policy_heads_declaration(policy_heads)
        print(self.action_space_names, pi_head_sizes)
        # noinspection PyTypeChecker
        self.model = PpoModule(
            obs_size=obs_size,
            pi_heads=pi_head_sizes,
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

        self.ema_policy_enabled = 0.0 < weights_ema_lr < 1.0
        if self.ema_policy_enabled:
            ema_step_fn = get_ema_step_fn(weights_ema_lr)
            self.target_model = AveragedModel(self.model, avg_fn=ema_step_fn)
        else:
            # in case it's turned off, expose the same policy
            self.target_model = self.model

    def update(self, batch):
        # NB: data in batch is already no-grad
        if self.ema_policy_enabled:
            with torch.no_grad():
                # replace target policy distr logprobs with policy logprobs
                # for the entire batch
                pi, _ = self.policy(batch.obs[:-1])
                batch.logprob = pi.log_prob(batch.a)

        fl_batch = batch.flatten()
        clip = self.ppo_pi_clip
        stats_hist = defaultdict(list)

        for i_epoch in range(1, self.batch_epochs + 1):
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
                v_lr = self.val_loss_scale
                sq_td_err = torch.pow(td_err, 2)

                # .. then soft clip it via log ratios from policy:
                # such clipping doesn't affect gradient flow, but rescales
                # learning rate resulting to fast vanishing gradients for "should-be-clipped" (s,a) pairs
                v_soft_clip_alpha, _ = self.ppo_v_clip(log_ratio.detach())
                if self.v_clip_enabled:
                    v_lr = v_lr * v_soft_clip_alpha
                # NB: apply lr before taking mean because learning rate is now sample-based
                v_loss = torch.mean(v_lr * sq_td_err)

                # calc entropy loss
                h_lr = self.ent_loss_scale
                h_loss = -h_lr * pi.entropy().mean()

                loss = pi_loss + v_loss + h_loss

                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.optimizer.step()

                if i_epoch != self.batch_epochs:
                    continue
                # save stats for logging
                with torch.no_grad():
                    abs_td_err = torch.abs(td_err)
                    clip_ratio = (torch.abs(ratio - 1.0) > clip).float()
                    kl_div = torch.abs(torch.exp(logprob) * log_ratio)

                    stats_keys = ['pi', '|pi|', 'pi_clip', 'v', 'v_clip', 'h', 'kl', 'A', '|TDErr|']
                    stats_vals = [
                        pi_loss, torch.abs(pi_loss), clip_ratio,
                        v_loss, v_soft_clip_alpha, h_loss, kl_div,
                        A, torch.abs(td_err),
                    ]
                    for k, v in zip(stats_keys, stats_vals):
                        stats_hist[k].append(torch.mean(v).item())

        self.ent_loss_scale *= self.ent_loss_scale_decay
        self.ent_loss_scale = max(
            self.ent_loss_scale, self.ent_loss_scale_init * self.ent_loss_scale_max_decay
        )

        # apply target policy slow update if enabled
        if self.ema_policy_enabled:
            self.target_model.update_parameters(self.policy)
        return {k: np.mean(v) for k, v in stats_hist.items()}


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
    ):
        super().__init__()
        # TODO: support seeded initialisation
        self.n_classes = pi_heads[1]
        self._pi_heads = pi_heads
        self._pi_head_shifts = np.cumsum(pi_heads)

        # [optional] input_size -> enc_size
        enc_size, self.encoder = make_layers('Encoder', obs_size, obs_encoder)

        # enc_size -> hid_size
        # self.rnn = nn.LSTMCell(enc_size, hid_size, dtype=float)
        self.rnn = nn.GRUCell(enc_size, mem_hidden_size, dtype=float)
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
                    'Mem skip connection', obs_size, [mem_hidden_size]
                )

            elif enc_size != mem_hidden_size:
                # has encoder, rnn input and hidden has diff size.
                #   I expect hidden > input ==> add projection layer to rnn and skip conn
                #   from enc to the output of the projection layer
                self.skip_conn_option = 1
                body_input_size, self.skip_conn = make_layers(
                    'Mem output projection', mem_hidden_size, [enc_size]
                )
            else:
                # has encoder, enc_size == hid_size ==> can pass enc via skip connection
                #   w/o any additional layers
                self.skip_conn_option = 2

        pi_enc_size, self.pi_body = make_layers('Policy body', body_input_size, body)
        val_enc_size, self.val_body = make_layers('Value body', body_input_size, body)

        _, self.pi_head = make_layers(
            'Policy heads', pi_enc_size, [sum(pi_heads)], out_logits=True
        )
        _, self.val_head = make_layers(
            'Value head', val_enc_size, [1], out_logits=True
        )

        # TODO: check if should be configurable
        self.body_skip = pi_enc_size == val_enc_size == body_input_size and len(body) > 0

    def _encode(self, x, state):
        x = concat_obs_parts(x)
        x = torch.from_numpy(x)

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

    def _split_pi(self, pi):
        mv, cl, r, c = self._pi_head_shifts
        return (
            pi[..., :mv], pi[..., mv:cl],
            pi[..., cl:r], pi[..., r:c],
        )

    def _predict(self, x, state, greedy=False):
        z, state = self._encode(x, state)

        u = self.pi_body(z) if self.pi_body is not None else z
        if self.body_skip:
            u = u + z
        pi = self.pi_head(u)

        v = self.val_body(z) if self.val_body is not None else z
        if self.body_skip:
            v = v + z
        val = self.val_head(v).squeeze()

        # zma == zoom out|move|answer
        zma, cl, row, col = self._split_pi(pi)
        zma_distr = Categorical(logits=zma)
        class_distr = Categorical(logits=cl)
        row_distr = Categorical(logits=row)
        col_distr = Categorical(logits=col)

        a_zma = torch.argmax(zma, -1) if greedy else zma_distr.sample()
        a_cl = torch.argmax(cl, -1) if greedy else class_distr.sample()
        a_row = torch.argmax(row, -1) if greedy else row_distr.sample()
        a_col = torch.argmax(col, -1) if greedy else col_distr.sample()

        result = dict(
            state=state,
            val=val,
            a_zma=a_zma, a_class=a_cl,
            a_row=a_row, a_col=a_col,

            zma_log_prob=zma_distr.log_prob(a_zma),
            class_log_prob=class_distr.log_prob(a_cl),
            row_log_prob=row_distr.log_prob(a_row),
            col_log_prob=col_distr.log_prob(a_col),

            zma_entropy=zma_distr.entropy(),
            class_entropy=class_distr.entropy(),
            row_entropy=row_distr.entropy(),
            col_entropy=col_distr.entropy(),
        )

        return result

    def _evaluate(self, x, state):
        z, state = self._encode(x, state)

        u = self.val_body(z) if self.val_body is not None else z
        if self.body_skip:
            u = u + z

        val = self.val_head(u).squeeze()
        return val, state

    def forward(self, x, state=None):
        return self._predict(x, state)

    def predict(self, x, state=None, train=True, greedy=False):
        if train:
            return self._predict(x, state)

        with torch.no_grad():
            return self._predict(x, state, greedy=greedy)

    def learn(self, obs, r, a, obs_next, done, reset_mask):
        batch_size = len(done)
        gamma = self.gamma
        done = torch.from_numpy(done).float()
        # done -> reset transition samples are disabled
        enabled = torch.logical_not(torch.from_numpy(reset_mask))
        r = torch.from_numpy(r)
        state = a['state']
        zma = a['a_zma']
        v_s = a['val']

        move_mask = zma == 1
        ans_mask = zma == 2
        log_pi = (
                a['zma_log_prob']
                + torch.where(move_mask, a['row_log_prob'] + a['col_log_prob'], 0.0)
                + torch.where(ans_mask, a['class_log_prob'], 0.0)
        )
        log_pi = log_pi[enabled]

        with torch.no_grad():
            v_sn, _ = self._evaluate(obs_next, state)
            td_tar = r + gamma * (1 - done) * v_sn

        td_err = td_tar - v_s
        td_err = td_err[enabled]

        pi_heads_entropy = [
            a['zma_entropy'], a['class_entropy'],
            a['row_entropy'], a['col_entropy'],
        ]

        v_loss = torch.mean(torch.pow(td_err, 2))
        pi_loss = -torch.mean(log_pi * td_err.detach())
        ent_loss = -sum(
            scale * _normalize_entropy(torch.mean(entropy[enabled]), size)
            for scale, size, entropy in zip(
                self.ent_scales, self._pi_heads, pi_heads_entropy
            )
        )
        loss = pi_loss + self.val_loss_scale * v_loss + self.ent_loss_scale * ent_loss

        ret_loss = pi_loss.detach().item(), v_loss.detach().item()

        self._batch_cnt += batch_size
        self._loss += loss

        if self._batch_cnt >= self.batch_size:
            self._loss /= round(self._batch_cnt / batch_size)
            self._last_loss += 0.05 * (self._loss.detach().item() - self._last_loss)
            self._loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            self._batch_cnt = 0
            self._loss = 0.

            # h_st, c_st = state
            # a['state'] = h_st.detach(), c_st.detach()
            a['state'] = state.detach()

            self._lr_epoch_step += 1
            if self.lr > self._min_lr and self._lr_epoch_step >= self._lr_epoch_steps:
                self.lr *= self._lr_scaler(True)
                self._lr_epoch_step = 0
                self._lr_epoch_steps = self._lr_epoch(self.lr)
                # print(f'New LR: {self.lr:.5f} for {self._lr_epoch_steps} steps')
                self.lr_scheduler.step()

        return ret_loss


def _split_policy_heads_declaration(pi_heads: list[tuple[str, int]]) -> tuple[list[str], list[int]]:
    head_sizes, head_names = zip(*pi_heads)
    return list(head_names), list(head_sizes)
