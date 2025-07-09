import numpy as np

from stelarc.config import ns_to_dict


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
