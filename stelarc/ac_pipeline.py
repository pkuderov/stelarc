from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch

from stelarc.agents.utils.lambda_return import RlFlags
from stelarc.log import start_wandb_run, log_results


class Batch:
    def __init__(self, n_envs, n_steps):
        super().__init__()
        self.shape = (n_steps, n_envs)
        self.n_steps, self.n_envs = n_steps, n_envs
        self.size = n_steps * n_envs

        self.a = []
        self.logprob = []
        self.entropy = []
        self.v = []
        self.r = []
        self.flags = []

        self.lambda_ret = []

    def put(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k].append(v)

    def clear(self):
        for _, v in self.__dict__.items():
            if isinstance(v, list):
                v.clear()

    @staticmethod
    def to_flags(term, trunc, reset):
        # bitwise OR is same as sum here
        return (
            RlFlags.TERMINATED * term
            | RlFlags.TRUNCATED * trunc
            | RlFlags.RESET * reset
        )


class State:
    def __init__(self):
        self.obs = None
        self.mem_state = None


def sample_batch(env, agent, run_data):
    batch = run_data.batch
    state = run_data.state

    n_steps, n_envs = batch.shape
    obs, mem_state = state.obs, state.mem_state
    assert len(batch.v) == 0

    n_correct = 0
    n_done = 0
    for i in range(n_steps):
        # use target policy for acting
        pi, v, mem_state = agent(obs, mem_state)
        a = pi.sample()
        logprob = pi.log_prob(a)
        entropy = pi.normalised_entropy()

        obs_tn, reward, term_tn, trunc_tn, info = env.step(a.cpu())

        reset_tn = info['reset_mask']
        flags = batch.to_flags(term=term_tn, trunc=trunc_tn, reset=reset_tn)

        batch.put(
            a=a, logprob=logprob, entropy=entropy,
            r=reward, v=v, flags=flags
        )
        n_correct += info['n_correct']
        np.add.at(run_data.act_type_stats, a[..., 0][~reset_tn.numpy()], 1)

        obs = obs_tn
        mem_state = agent.reset_state(mem_state, reset_tn)

    # save state, note memory state detach
    state.obs, state.mem_state = obs, agent.detach_state(mem_state)

    # get value-based backups for lambda returns for the last step computation
    # NB: notice no-grad and no mem state mutation! Next batch, we will call
    #   an agent with the same input, but w/ updated model and grads enabled
    v = agent.evaluate(obs, mem_state)
    batch.put(v=v)

    run_data.stats['n_correct'].append(n_correct)
    agent.lambda_return(
        V=batch.v, r=batch.r, flags=batch.flags, G=batch.lambda_ret, t=batch.n_steps
    )
    # print(f'G:', batch.lambda_ret[0].numpy(), batch.lambda_ret[1].numpy())
    # these aren't needed after returns calculation
    batch.r.clear()


def train_batch(agent, batch, loss_stats):
    new_loss_stats = agent.update(batch)
    for k, v in new_loss_stats.items():
        loss_stats[k].append(v)


def train_epoch(env, agent, run_data, config):
    batch = run_data.batch
    n_batches = config.run.n_batches

    for i in range(n_batches):
        sample_batch(env, agent, run_data)
        train_batch(agent, batch, run_data.loss_stats)
        batch.clear()

        run_data.step += batch.size
        if run_data.step >= run_data.next_log:
            log_results(run_data, config, env)


def run_experiment(config, env, agent, test_env=None):
    n_epochs = config.run.n_epochs
    eval_configs = config.run.eval_configs
    run_data = SimpleNamespace(
        step=0, ep=0, next_log=config.log.schedule,
        batch=Batch(
            n_envs=config.env.num_envs, n_steps=config.run.n_batch_steps,
        ),
        state=State(),
        # test_batch=Batch(
        #     n_envs=config.env.num_envs, n_steps=config.run.n_batch_steps,
        #     obs_size=config.agent.obs_size,
        # ),
        loss_stats=defaultdict(list),
        stats=defaultdict(list),
        act_type_stats=np.zeros(3, dtype=int),
        test_loss_stats=defaultdict(list),
        wandb_run=start_wandb_run(config),
    )

    obs_t, _, = env.reset(seed=config.seed)
    run_data.state.obs = obs_t

    for epoch in range(1, n_epochs + 1):
        train_epoch(env=env, agent=agent, run_data=run_data, config=config)

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
