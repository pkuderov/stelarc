from collections import defaultdict

import numpy as np
import torch


def _append_results(agg_results, res):
    for k in res:
        agg_results[k].append(res[k])


def _to_readable_num(x):
    suffixes = ['', 'k', 'M', 'B']
    i = 0
    while abs(x) > 1000.0 or i >= len(suffixes):
        x = x / 1000.0
        i += 1

    return x, suffixes[i]


def run_epoch(
        env, agent, *, full_state, epoch, n_steps, total_steps, is_train=True, is_greedy=False
):
    o, state = full_state
    greedy = (not is_train) and is_greedy
    ret, ep_len_sum, n_eps, v_loss, pi_loss, acc = 0., 0., 0., 0., 0., 0.

    for i in range(1, n_steps + 1):
        a_agent = agent.predict(o, state, train=is_train, greedy=greedy)
        a_env = _get_numpy_action(a_agent)
        o_next, r, terminated, truncated, info = env.step(a_env)
        reset_mask = info['reset_mask']
        if is_train:
            loss = agent.learn(o, r, a_agent, o_next, terminated, reset_mask)

        ret += r.mean()
        ep_len_sum += info['ep_len_sum']
        acc += info['n_correct']
        n_eps += info['n_done']
        if is_train:
            # noinspection PyUnboundLocalVariable
            pi_loss += loss[0]
            v_loss += loss[1]

        o = o_next
        state = _reset_state(a_agent['state'], reset_mask)

    if is_train:
        pi_loss /= n_steps
        v_loss /= n_steps

    ret /= n_steps
    ep_len_sum /= n_eps
    acc /= n_eps / 100.0

    if is_train:
        total_steps += n_steps

        tot, sfx = _to_readable_num(total_steps * env.num_envs)
        print(
            f'{epoch} [{tot:.0f}{sfx}]: {acc:.2f} {ret:.3f}  '
            f'|  {ep_len_sum:.1f}  {n_eps}  '
            f'|  {pi_loss:.4f}   {v_loss:.4f}'
        )
        return (
            total_steps, (o, state),
            dict(acc=acc, ret=ret, ep_len=ep_len_sum, pi_loss=pi_loss, v_loss=v_loss)
        )

    print(f'==>  {acc:.2f} {ret:.3f}  |  {ep_len_sum:.1f}  {n_eps}')
    return (o, state), dict(acc=acc, ret=ret, ep_len=ep_len_sum)


def run_experiment(
        env, test_env, agent, n_epochs, n_steps, eval_configs,
):
    train_results = defaultdict(list)
    test_results = defaultdict(list)

    total_steps = 0
    full_state = (env.reset()[0], None)
    test_full_state = (test_env.reset()[0], None)

    for epoch in range(1, n_epochs + 1):
        if n_steps > 0:
            total_steps, full_state, res = run_epoch(
                env, agent, full_state=full_state,
                epoch=epoch, n_steps=n_steps, total_steps=total_steps,
                is_train=True, is_greedy=False,
            )
            _append_results(train_results, res)

        for eval_config in eval_configs:
            test_full_state, res = run_epoch(
                test_env, agent, full_state=test_full_state,
                epoch=epoch, total_steps=total_steps,
                is_train=False, **eval_config
            )
            _append_results(test_results, res)


def _get_numpy_action(res):
    return np.vstack([
        res['a_zma'].numpy(),
        res['a_class'].numpy(),
        res['a_row'].numpy(), res['a_col'].numpy()
    ]).T


def _reset_state(state, reset_mask):
    # TODO: use torch.where + init_state fixed random vector
    not_done = np.logical_not(reset_mask)
    not_done = torch.from_numpy(not_done).reshape(-1, 1)
    # h, c = state
    # h = h * not_done
    # c = c * not_done
    # return h, c
    return state * not_done
