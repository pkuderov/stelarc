import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical

from stelarc.agents.utils.torch import make_layers, concat_obs_parts


class RlClassifier(nn.Module):
    def __init__(
            self,
            obs_size: int,
            obs_encoder: list[int],
            mem_hidden_size: int,
            mem_skip_connection: bool,
            body: list[int],
            pi_heads: tuple[int],
            learning_rate: float,
            gamma: float,
            batch_size: int = 64,
            val_loss_scale: float = 0.25,
            ent_scales: tuple[float] = (1.0, 1.0, 1.0, 1.0),
            ent_loss_scale: float = 0.001,
            seed: int = None,
            min_lr_scale: float = 20.0
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.lr = learning_rate
        self.n_classes = pi_heads[1]
        self.gamma = gamma
        self.val_loss_scale = val_loss_scale
        self.ent_loss_scale = ent_loss_scale
        self.ent_scales = ent_scales
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

        self.optim = optim.AdamW(self.parameters(), lr=self.lr)
        self.mse = nn.MSELoss()

        self.batch_size = batch_size
        self._loss = 0.
        self._last_loss = 0.
        self._batch_cnt = 0

        self._min_lr = self.lr / min_lr_scale
        self._lr_epoch = lambda lr: int(7.0 / lr)
        self._lr_scaler = lambda _: 0.8
        self.lr_scheduler = optim.lr_scheduler.MultiplicativeLR(
            self.optim, lr_lambda=self._lr_scaler,
        )
        self._lr_epoch_step = 0
        self._lr_epoch_steps = self._lr_epoch(self.lr)

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
            u = e + rnn_proj(h)

        elif self.skip_conn_option == 2:
            # enc_size == hid_size ==> simple skip conn from enc to rnn hidden
            u = e + h

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


def test_ac_compilation():
    from functools import partial
    import numpy as np

    from oculr.dataset import Dataset
    from oculr.env import ImageEnvironment
    from oculr.image.buffer import PrefetchedImageBuffer

    seed = 8041990
    ds = Dataset( 'mnist', grayscale=True, lp_norm=None, seed=seed)
    env = ImageEnvironment(
        ds, num_envs=2, obs_hw_shape=7, max_time_steps=20, seed=42,
        answer_reward=(1.0, -0.3), step_reward=-0.01,
        img_buffer_fn=partial(PrefetchedImageBuffer, prefetch_size=256),
    )
    o, info = env.reset()
    o, r, terminated, truncated, info = env.step(
        np.array([
            [1, 0, 5, 10],
            [0, 10, 4, 6],
        ])
    )

    agent = RlClassifier(
        obs_size=env.total_obs_size,
        obs_encoder=[32], mem_hidden_size=64, mem_skip_connection=True, body=[32],
        pi_heads=(
            3, env.n_classes, *env.pos_range[1]
        ),
        learning_rate=0.05,
        gamma=0.95,
        batch_size=64,
        seed=seed,
    )
    _ = agent.predict(o)


if __name__ == '__main__':
    test_ac_compilation()
