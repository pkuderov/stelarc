from collections import defaultdict
from functools import partial
from types import SimpleNamespace

import gymnasium as gym
import torch
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers.vector import (
    FlattenObservation, NormalizeObservation, TransformReward,
    RecordEpisodeStatistics
)
from gymnasium.wrappers.vector.numpy_to_torch import NumpyToTorch

from stelarc.agents.utils.batch import Batch
from stelarc.log import start_wandb_run, log_results


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

    agent.gae(batch)
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
