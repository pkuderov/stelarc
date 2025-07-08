from collections import defaultdict
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers.vector import (
    TransformReward, FlattenObservation, NormalizeObservation,
    RecordEpisodeStatistics
)
from gymnasium.wrappers.vector.numpy_to_torch import NumpyToTorch
from torch import nn
from torch.distributions import Categorical
from torch.optim.swa_utils import AveragedModel

from stelarc.compare_envs import compare_envs


device = torch.device(
    "cuda:0" if torch.cuda.is_available() else
    "mps" if torch.mps.is_available() else
    "cpu"
)


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)
def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class ActorCritic(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions, *args, **kwargs):
        super().__init__()

        # NB: на выходе далее будем ожидать логиты, а не вероятности!
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


class Batch(SimpleNamespace):
    def __init__(self, n_envs, n_steps, obs_size):
        self.shape = shape = (n_steps, n_envs)
        self.n_steps, self.n_envs = n_steps, n_envs
        self.size = n_steps * n_envs
        self.obs = self.make_tensor((n_steps + 1, n_envs, obs_size))

        self.a = self.make_tensor(dtype=torch.int)
        self.logprob = self.make_tensor()
        self.v = self.make_tensor((n_steps + 1, n_envs))

        self.r = self.make_tensor()
        self.term = self.make_tensor((n_steps + 1, n_envs), dtype=torch.bool)
        self.trunc = self.make_tensor((n_steps + 1, n_envs), dtype=torch.bool)

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
            self, shape=None, *, dtype=torch.float, requires_grad=False,
            device=device
    ):
        if shape is None:
            shape = self.shape
        return torch.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)


class GAE:
    def __init__(self, gamma: float, lambda_: float = 0.95):
        self.lambda_ = lambda_
        self.gamma = gamma

    def __call__(self, batch: Batch):
        gamma, lambda_ = self.gamma, self.lambda_
        V, r, term, trunc = batch.v, batch.r, batch.term, batch.trunc
        G = batch.lambda_ret
        t = batch.n_steps
        G_tn = V[t]

        while t > 0:
            V_tn = V[t]
            t -= 1

            backup = (1.0 - lambda_) * V_tn + lambda_ * G_tn
            # done: r = 0. | gamma * ... = <invalid>
            # trunc: G[t] = <any> | G_tn = V[t]
            # term: G[t] = 0. | G_tn = 0.
            G_tn = G[t] = torch.where(
                term[t], 0., torch.where(
                trunc[t], V[t], r[t] + gamma * backup
            ))

        return G


class SoftClip:
    """
    Empirically constructed soft-clip func.
    It starts the clipping a bit before the threshold and decays very fast.
    """

    def __init__(self, eps, start_from):
        self.eps = eps
        self.start_from = start_from
        self.max_hardness = 24
        self.hardness = 1.0 + self.max_hardness * (1.0 - self.eps)

    def __call__(self, x):
        eps, alpha, beta = self.eps, self.start_from, self.hardness
        z = torch.clamp(alpha * eps + torch.exp(-torch.abs(x)), max=1.0)
        return torch.pow(z, beta), z

    def plot(self):
        import matplotlib.pyplot as plt

        xs = np.linspace(0.00001, 1.0, 100)
        ys, zs = self(torch.log(torch.from_numpy(xs)))
        plt.plot(xs, ys)
        plt.plot(xs, zs)
        plt.show()


# TODO
# SoftClip(eps=0.1, start_from=0.75).plot()


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
    ):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.gae = GAE(gamma, gae_lambda)
        self.eps_clip = eps_clip
        self.v_clip_enabled = v_clip
        self.v_clip = SoftClip(eps=self.eps_clip, start_from=0.75)
        self.K_epochs = K_epochs
        self.mini_batch_size = mini_batch_size
        self.v_loss_alpha = v_loss_alpha if v_loss_alpha is not None else 1.0
        self.ent_loss_alpha_init = self.ent_loss_alpha = ent_loss_alpha
        self.ent_loss_alpha_decay = ent_loss_alpha_decay
        self.min_ent_loss_ratio = min_ent_loss_ratio

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
                    kl_div = torch.abs(torch.exp(logprob) * log_ratio)

                    stats_keys = ['pi', '|pi|', 'pi_clip', 'v', 'v_clip', 'h', 'kl', 'A', '|TDErr|']
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


def ns_to_dict(ns):
    if not isinstance(ns, SimpleNamespace):
        return ns

    return {
        k: ns_to_dict(v)
        for k, v in ns.__dict__.items()
    }

config = SimpleNamespace()
ns_to_dict(config)


def start_wandb_run(config):
    if not config.log.wandb:
        return None
    config.ac = config.ppo.ac_type.__name__
    import wandb
    return wandb.init(
        project=config.log.project,
        tags=config.log.tags,
        config=ns_to_dict(config),
    )


def get_stats_wrapper(env):
    while not hasattr(env, 'episode_count'):
        env = env.env
    return env


def log_results(
        run_data, config, env
):
    loss_stats = run_data.loss_stats
    env = get_stats_wrapper(env)
    run_data.next_log += config.log.schedule

    step = run_data.step
    ep = run_data.ep = env.episode_count

    avg_ep_len, avg_ep_t, avg_ret = 0., 0., 0.
    if ep > 0:
        avg_ep_len = np.mean(env.length_queue)
        avg_ep_t = np.mean(env.time_queue)
        avg_ret = np.mean(env.return_queue)

    avg_sps = avg_ep_len * config.env.n_envs / (avg_ep_t + 1e-9)

    _loss_stats = {k: np.mean(v) for k, v in loss_stats.items()}
    loss_stats.clear()
    loss_stats = _loss_stats

    print(
        f'{step // 1_000:5d}k  [{run_data.ep:3d}] {avg_sps / 1000.0:.2f} ksps'
        f'  Len: {avg_ep_len:.1f}'
        f'  Ret: {avg_ret:.1f}',
        end=' |'
    )
    for k, v in loss_stats.items():
        print(f'  {k}: {v:.5f}', end='')
    print()

    run_data.is_task_solved = avg_ret >= config.env.solved_reward

    if run_data.wandb_run is not None:
        metrics = {
            'Episode': ep,
            'Global Step': step,
            'avgEpLen': avg_ep_len,
            'avgRet': avg_ret,
            'avgSPS': avg_sps,
        } | loss_stats
        run_data.wandb_run.log(metrics)
    return avg_ret


def sample_batch(env, ppo, batch: Batch):
    n_steps, n_envs = batch.shape
    # move last items from "previous" batch to the beginning of the current one
    obs, term, trunc = batch.obs[-1], batch.term[-1], batch.trunc[-1]

    for i in range(n_steps):
        # use target policy for acting
        pi, v = ppo.target_policy(obs)
        a = pi.sample()
        logprob = pi.log_prob(a)

        obs_, reward, term_, trunc_, info = env.step(a)
        batch.put(
            i, obs=obs, a=a, logprob=logprob, v=v,
            r=reward, term=term, trunc=trunc
        )
        # append_episode_stats(stats, info)
        obs, term, trunc = obs_, term_, trunc_

    # put extra
    _, v = ppo.target_policy(obs)
    batch.put(-1, v=v, obs=obs, term=term, trunc=trunc)

    ppo.gae(batch)
    return obs


def train_batch(ppo, batch, loss_stats):
    new_loss_stats = ppo.update(batch)
    for k, v in new_loss_stats.items():
        loss_stats[k].append(v)


def run_experiment(config, env, ppo):
    if not isinstance(config.run.steps_range, tuple):
        config.run.steps_range = (config.run.steps_range, config.run.steps_range)
    min_steps, max_steps = config.run.steps_range
    batch = Batch(
        n_envs=config.env.n_envs, n_steps=config.run.n_batch_steps,
        obs_size=config.ppo.obs_size,
    )
    run_data = SimpleNamespace(
        step=0, ep=0, next_log=config.log.schedule,
        is_task_solved=False,
        loss_stats=defaultdict(list),
        wandb_run=start_wandb_run(config),
    )

    obs, _ = env.reset(seed=config.seed)
    batch.put(-1, obs=obs, term=False, trunc=False)

    for elapsed_steps in range(0, max_steps + batch.size, batch.size):
        with torch.no_grad():
            sample_batch(env, ppo, batch)
        train_batch(ppo, batch, run_data.loss_stats)

        if elapsed_steps >= run_data.next_log:
            run_data.step = elapsed_steps
            log_results(run_data, config, env)
            if elapsed_steps >= min_steps and run_data.is_task_solved:
                print("########## Solved! ##########")
                break

    if run_data.wandb_run is not None:
        run_data.wandb_run.finish()

def get_seed(rng):
    return int(rng.integers(1_000_000))

seed_generator = np.random.default_rng()


def make_env(
        name, n_envs, max_steps, r_scale, stats_buffer_eps, device,
        **unused
):
    kwargs = {}
    if 'LunarLander' in name:
        kwargs['enable_wind'] = unused.pop('enable_wind', False)

    wrappers = []
    frame_stack = unused.pop('frame_stack', False)
    if frame_stack:
        wrappers.append(
            partial(FrameStackObservation, stack_size=frame_stack)
        )

    print(f'Env kwargs unused: {unused}')
    env = gym.make_vec(
        name, num_envs=n_envs, max_episode_steps=max_steps,
        vectorization_mode="sync", disable_env_checker=True,
        wrappers=wrappers,
        **kwargs
    )
    if frame_stack:
        env = FlattenObservation(env)
    env = NormalizeObservation(env)
    env = TransformReward(env, func=lambda r: r * r_scale)
    env = RecordEpisodeStatistics(env, buffer_length=n_envs * stats_buffer_eps)
    env = NumpyToTorch(env, device=device)
    print(env.spec)

    obs_shape = env.observation_space.shape
    obs_size = obs_shape[-1]
    print(f'vec obs: {obs_shape} | obs: {obs_size}')
    n_actions = env.action_space[0].n
    print(f'act: {n_actions}')

    return env, obs_size, n_actions


def set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)


if hasattr(config, 'log'):
    config.log.wandb = False
    config.run.steps_range = (10_000, 25_000)
    env, *_ = make_env(**config.env.__dict__, device=config.device)
    ppo = Ppo(**config.ppo.__dict__)
    run_experiment(config, env, ppo)

env_name = "CartPole-v1"
# env_name = "LunarLander-v3"
# env_name = "MountainCar-v0"

config = SimpleNamespace(
    device=device,
    seed=get_seed(seed_generator),

    env=SimpleNamespace(
        name=env_name, n_envs=64, max_steps=500,
        r_scale=0.01, stats_buffer_eps=4,
        solved_reward=490,
    ),

    ppo=SimpleNamespace(
        ac_type=ActorCritic,
        hidden_size=32, lr=0.001, ema_lr=0.0, betas=(0.9, 0.999),
        gamma=0.995, gae_lambda=0.95, K_epochs=4, mini_batch_size=256,
        eps_clip=0.1, v_clip=False,
        v_loss_alpha=1.0, ent_loss_alpha=0.2, min_ent_loss_ratio=1 / 40.
    ),

    run=SimpleNamespace(
        steps_range=(20_000, 500_000), n_batch_steps=16,
    ),

    log=SimpleNamespace(
        schedule=10_000,
        wandb=False,
        project='ppo-v-clip-vec'
    ),
)

config.env.solved_reward *= config.env.r_scale
config.ppo.ent_loss_alpha *= config.env.r_scale
config.run.batch_size = config.run.n_batch_steps * config.env.n_envs

set_seed(config.seed)
env, obs_size, n_actions = make_env(**config.env.__dict__, device=config.device)

config.ppo.obs_size, config.ppo.n_actions = obs_size, n_actions
ppo = Ppo(**config.ppo.__dict__)

print(f'{config.seed=} | {config.ppo.lr=}')

run_experiment(config, env, ppo)

def main():
    # compare_envs()

    pass


if __name__ == '__main__':
    main()
