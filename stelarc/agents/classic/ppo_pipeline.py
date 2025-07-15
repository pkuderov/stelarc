from collections import defaultdict
from functools import partial
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers.vector import (
    FlattenObservation, NormalizeObservation, TransformReward,
    RecordEpisodeStatistics
)
from gymnasium.wrappers.vector.numpy_to_torch import NumpyToTorch

from stelarc.agents.classic.compare_envs import compare_envs
from stelarc.agents.classic.ppo import ActorCritic, Ppo, Batch
from stelarc.config import get_seed, set_seed, ns_to_dict
from stelarc.log import to_readable_num, get_stats_wrapper


@torch.no_grad()
def sample_batch(env, agent, batch: Batch):
    n_steps, n_envs = batch.shape
    # move last items from "previous" batch to the beginning of the current one
    obs, term, trunc = batch.obs[-1], batch.term[-1], batch.trunc[-1]

    for i in range(n_steps):
        # use target policy for acting
        pi, v = agent.target_policy(obs)
        a = pi.sample()
        logprob = pi.log_prob(a)

        obs_, reward, term_, trunc_, info = env.step(a.cpu())
        batch.put(
            i, obs=obs, a=a, logprob=logprob, v=v,
            r=reward, term=term, trunc=trunc
        )
        # append_episode_stats(stats, info)
        obs, term, trunc = obs_, term_, trunc_

    # put extra
    _, v = agent.target_policy(obs)
    batch.put(-1, v=v, obs=obs, term=term, trunc=trunc)

    agent.lambda_return(batch)
    return obs


def train_batch(agent, batch, loss_stats):
    new_loss_stats = agent.update(batch)
    for k, v in new_loss_stats.items():
        loss_stats[k].append(v)


def run_experiment(config, env, agent):
    if not isinstance(config.run.steps_range, tuple):
        config.run.steps_range = (config.run.steps_range, config.run.steps_range)
    min_steps, max_steps = config.run.steps_range
    batch = Batch(
        n_envs=config.env.n_envs, n_steps=config.run.n_batch_steps,
        obs_size=config.agent.obs_size,
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
        sample_batch(env, agent, batch)
        train_batch(agent, batch, run_data.loss_stats)

        if elapsed_steps >= run_data.next_log:
            run_data.step = elapsed_steps
            log_results(run_data, config, env)
            if elapsed_steps >= min_steps and run_data.is_task_solved:
                print("########## Solved! ##########")
                break

    if run_data.wandb_run is not None:
        run_data.wandb_run.finish()


def log_results(
        run_data, config, env
):
    loss_stats = run_data.loss_stats
    env = get_stats_wrapper(env)
    run_data.next_log += config.log.schedule

    step = run_data.step
    delta_ep = env.episode_count - run_data.ep
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

    _step, _sfx = to_readable_num(step)
    _ep, _ep_sfx = to_readable_num(ep)
    print(
        f'{_step:.0f}{_sfx}  [{_ep:.0f}{_ep_sfx}] {avg_sps / 1000.0:.2f} ksps'
        f'  Len: {avg_ep_len:.1f}'
        f'  Ret: {avg_ret:.1f}',
        end=' |'
    )
    for k, v in loss_stats.items():
        print(f'  {k}: {v:.5f}', end='')
    print()

    if run_data.wandb_run is not None:
        metrics = {
            'Episode': ep,
            'Global Step': step,
            'avgEpLen': avg_ep_len,
            'avgRet': avg_ret,
            'avgSPS': avg_sps,
        } | loss_stats
        run_data.wandb_run.log(metrics)

    run_data.is_task_solved = avg_ret >= config.env.solved_reward
    return avg_ret


def start_wandb_run(config):
    if not config.log.wandb:
        return None
    config.ac = config.agent.ac_type.__name__

    import wandb
    return wandb.init(
        project=config.log.project,
        tags=config.log.tags,
        config=ns_to_dict(config),
    )


def make_classic_env(
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


def test_ppo(env_name ="CartPole-v1"):
    config = SimpleNamespace(
        device='cpu',
        seed=get_seed(),

        env=SimpleNamespace(
            name=env_name, n_envs=64, max_steps=500,
            r_scale=0.01, stats_buffer_eps=4,
            solved_reward=490,
        ),

        agent=SimpleNamespace(
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
    config.agent.ent_loss_alpha *= config.env.r_scale
    config.run.batch_size = config.run.n_batch_steps * config.env.n_envs

    set_seed(config.seed)
    env, obs_size, n_actions = make_classic_env(
        **config.env.__dict__, device=config.device
    )

    config.agent.obs_size, config.agent.n_actions = obs_size, n_actions
    agent = Ppo(**config.agent.__dict__)

    print(f'{config.seed=} | {config.agent.lr=}')

    run_experiment(config, env, agent)


if __name__ == '__main__':
    # compare_envs()
    test_ppo()
