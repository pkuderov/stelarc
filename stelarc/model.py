from typing import Sequence

import numpy as np
import torch
from torch import nn

from stelarc.agents.utils.torch import make_layers, MultiCategorical


class Model(nn.Module):
    def __init__(
            self, *,
            obs_size: int,
            obs_encoder: Sequence[int] = (),

            mem_hidden_size: int,
            mem_skip_connection: bool = False,
            is_lstm: bool = False,

            body_shared: Sequence[int] = (),
            body_separate: Sequence[int] = (),
            body_skip_connection: bool = False,

            policy_heads: Sequence[tuple[str, int]],
            action_types: dict[str, int],

            dtype=torch.float32
    ):
        super().__init__()
        # TODO: support seeded initialisation

        self.obs_size = obs_size
        self.action_size = len(policy_heads)

        print(policy_heads)
        self.action_names, policy_head_sizes = _split_policy_heads_declaration(policy_heads)
        self._act_zoom = action_types['zoom']
        self._act_move = action_types['move']
        self._act_answer = action_types['answer']

        self.n_classes = policy_head_sizes[1]
        self._pi_heads = policy_head_sizes
        self._pi_head_shifts = np.cumsum(policy_head_sizes)

        # [optional] input_size -> enc_size
        enc_size, self.encoder = make_layers(
            name='Encoder', input_size=obs_size, layers=obs_encoder, dtype=dtype
        )

        # enc_size -> hid_size
        self.is_lstm = is_lstm
        if self.is_lstm:
            self.rnn = nn.LSTMCell(enc_size, mem_hidden_size, dtype=dtype)
        else:
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

        body_output_size, self.shared_body = make_layers(
            name="Shared body", input_size=body_input_size, layers=body_shared, dtype=dtype
        )

        # Separate heads (with their own bodies)
        pi_enc_size, self.pi_body = make_layers(
            name='Policy body', input_size=body_output_size, layers=body_separate, dtype=dtype
        )
        val_enc_size, self.val_body = make_layers(
            name='Value body', input_size=body_output_size, layers=body_separate, dtype=dtype
        )

        _, self.pi_head = make_layers(
            name='Policy heads', input_size=pi_enc_size, layers=[sum(policy_head_sizes)],
            out_logits=True, dtype=dtype
        )
        _, self.val_head = make_layers(
            name='Value head', input_size=val_enc_size, layers=[1],
            out_logits=True, dtype=dtype
        )

        self.body_skip = False
        if body_skip_connection:
            body_skip_compatible = pi_enc_size == val_enc_size == body_input_size
            body_non_empty = len(body_shared) + len(body_separate) > 0
            self.body_skip = body_skip_compatible and body_non_empty

    def forward(self, x, state=None):
        return self._predict(x, state)

    def _predict(self, x, state):
        z, state = self._encode(x, state)
        s = _fw(self.shared_body, z)

        u = _fw(self.pi_body, s)
        if self.body_skip:
            u = u + z
        pi = self.pi_head(u)

        v = _fw(self.val_body, s)
        if self.body_skip:
            v = v + z
        val = self.val_head(v).squeeze()

        pi_distr = MultiCategorical.from_logits(logits=pi, spec=self._pi_heads)
        return pi_distr, val, state

    def _encode(self, x, state):
        e = _fw(self.encoder, x)

        state = self.rnn(e, state)
        h = state[0] if self.is_lstm else state

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

    @torch.no_grad()
    def evaluate(self, x, state=None):
        z, state = self._encode(x, state)
        s = _fw(self.shared_body, z)

        v = _fw(self.val_body, s)
        if self.body_skip:
            v = v + z
        val = self.val_head(v).squeeze()
        return val

    @staticmethod
    def get_log_pi(log_prob, *, move_mask, ans_mask):
        return (
            log_prob[..., 0]
            + torch.where(ans_mask, log_prob[..., 1], 0.0)
            + torch.where(move_mask, log_prob[..., 2] + log_prob[..., 3], 0.0)
        )

    def split_action_type(self, act_type, zoom=False, move=False, answer=False):
        zoom_mask, move_mask, answer_mask = None, None, None
        if zoom:
            zoom_mask = act_type == self._act_zoom
        if move:
            move_mask = act_type == self._act_move
        if answer:
            answer_mask = act_type == self._act_answer
        return zoom_mask, move_mask, answer_mask

    def reset_state(self, state, reset_mask):
        reset_mask = reset_mask.view(-1, 1)

        def _reset(h):
            return torch.where(reset_mask, 0.0, h)

        return (_reset(state[0]), _reset(state[1])) if self.is_lstm else _reset(state)

    def detach_state(self, state):
        return (state[0].detach(), state[1].detach()) if self.is_lstm else state.detach()


def _fw(module, x):
    """Call module if it's not None or return input untouched."""
    return module(x) if module is not None else x


def _split_policy_heads_declaration(
        pi_heads: Sequence[tuple[str, int]]
) -> tuple[list[str], list[int]]:
    head_sizes, head_names = zip(*pi_heads)
    return list(head_names), list(head_sizes)
