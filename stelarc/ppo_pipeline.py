from collections import defaultdict
from types import SimpleNamespace

import torch

from stelarc.ppo_batch import RnnBatch
from stelarc.log import start_wandb_run, log_results


@torch.no_grad()
def sample_batch(env, agent, run_data):
    batch = run_data.batch
    n_steps, n_envs = batch.shape
    # move last items from "previous" batch to the beginning of the current one
    obs, term, trunc = batch.obs[-1], batch.term[-1], batch.trunc[-1]
    state = batch.initial_state = batch.final_state
    n_correct = 0

    for i in range(n_steps):
        # use target policy for acting
        pi, v, state = agent.target_model(obs, state)
        a = pi.sample()
        logprob = pi.log_prob(a)

        obs_, reward, term_, trunc_, info = env.step(a.cpu())
        reset_mask = info['reset_mask']

        batch.put(
            i, obs=obs, a=a, logprob=logprob, v=v,
            r=reward, term=term, trunc=trunc, reset=reset_mask
        )
        n_correct += info['n_correct']
        # append_episode_stats(stats, info)
        obs, term, trunc = obs_, term_, trunc_
        state = agent.reset_state(state, reset_mask)

    # put extra
    _, v, state = agent.target_model(obs, state)
    batch.put(-1, v=v, obs=obs, term=term, trunc=trunc)
    batch.final_state = state

    run_data.stats['n_correct'].append(n_correct)

    agent.lambda_return(
        V=batch.v, r=batch.r, term=batch.term, trunc=batch.trunc, G=batch.lambda_ret,
        t=batch.n_steps
    )
    return obs, state


def train_batch(agent, batch, loss_stats):
    new_loss_stats = agent.update(batch)
    for k, v in new_loss_stats.items():
        loss_stats[k].append(v)


def run_epoch(env, agent, run_data, config, is_train):
    batch = run_data.batch
    n_batches = config.run.n_batches

    for i in range(n_batches):
        sample_batch(env, agent, run_data)
        train_batch(agent, batch, run_data.loss_stats)

        run_data.step += batch.size
        if run_data.step >= run_data.next_log:
            log_results(run_data, config, env)


def run_experiment(config, env, agent, test_env=None):
    n_epochs = config.run.n_epochs
    eval_configs = config.run.eval_configs
    run_data = SimpleNamespace(
        step=0, ep=0, next_log=config.log.schedule,
        batch=RnnBatch(
            n_envs=config.env.num_envs, n_steps=config.run.n_batch_steps,
            obs_size=agent.obs_size, action_size=agent.action_size
        ),
        # test_batch=Batch(
        #     n_envs=config.env.num_envs, n_steps=config.run.n_batch_steps,
        #     obs_size=config.agent.obs_size,
        # ),
        loss_stats=defaultdict(list),
        stats=defaultdict(list),
        test_loss_stats=defaultdict(list),
        wandb_run=start_wandb_run(config),
    )

    obs, _ = env.reset(seed=config.seed)
    run_data.batch.put(-1, obs=obs, term=False, trunc=False)

    for epoch in range(1, n_epochs + 1):
        run_epoch(
            env=env, agent=agent, run_data=run_data, config=config, is_train=True
        )

        for eval_config in eval_configs:
            # test_full_state, res = run_epoch(
            #     test_env, agent, full_state=test_full_state,
            #     epoch=epoch, total_steps=total_steps,
            #     is_train=False, **eval_config
            # )
            # _append_results(test_results, res)
            ...

    if run_data.wandb_run is not None:
        run_data.wandb_run.finish()
