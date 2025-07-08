def benchmark_non_vec_env():
    import torch
    from gymnasium.wrappers import (
        TransformReward,
        RecordEpisodeStatistics,
        NormalizeObservation,
        FrameStackObservation,
        FlattenObservation
    )
    from gymnasium.wrappers.numpy_to_torch import NumpyToTorch
    import gymnasium as gym

    def _run_test(env_name):
        # NB: for LunarLander with wind, also enable
        # Flatten + FrameStack wrappers for fair speed
        # estimate, because they are required for solution
        env = gym.make(
            env_name,
            max_episode_steps=500, disable_env_checker=True,
            # enable_wind=True,
        )
        env = TransformReward(env, func=lambda r: r / 100.0)
        if env_name == 'LunarLander-v3':
            env = FrameStackObservation(env, stack_size=3)
            env = FlattenObservation(env)
        env = NormalizeObservation(env)
        env = RecordEpisodeStatistics(env)
        env = NumpyToTorch(env)

        # should be (obs_size, ) 1D tensor
        # print(env.observation_space.shape)
        assert len(env.observation_space.shape) == 1

        print(f'"{env_name}": ', end='')

        t, st = 0, 0
        _ = env.reset()
        while st < 40_000:
            a = torch.from_numpy(env.action_space.sample().reshape(-1, 1)).squeeze()
            obs, rew, term, trunc, info = env.step(a)
            if not (term or trunc):
                continue
            ep_stats = info['episode']
            st += ep_stats['l'].sum()
            t += ep_stats['t'].sum()
            _ = env.reset()
        print(f'{st / t / 1_000:.2f} kfps')

    print('Benchmarking non-vectorised environments ==>')
    for name in ["CartPole-v1", "LunarLander-v3", "MountainCar-v0"]:
        _run_test(name)


def benchmark_vec_env():
    from functools import partial

    import torch
    from gymnasium.wrappers import FrameStackObservation, TransformReward
    from gymnasium.wrappers.vector import (
        RecordEpisodeStatistics,
        NormalizeObservation,
        FlattenObservation,
        NumpyToTorch
    )
    import gymnasium as gym

    def _run_test(env_name):
        # NB: for LunarLander with wind, also enable
        # Flatten + FrameStack wrappers for fair speed
        # estimate, because they are required for solution
        wrappers = [
            partial(TransformReward, func=lambda r: r / 100.0),
        ]
        if env_name == 'LunarLander-v3':
            wrappers.append(
                partial(FrameStackObservation, stack_size=3)
            )

        env = gym.make_vec(
            env_name, num_envs=16, vectorization_mode="sync",
            max_episode_steps=500, disable_env_checker=True,
            # enable_wind=True,
            wrappers=wrappers
        )

        if env_name == 'LunarLander-v3':
            env = FlattenObservation(env)
        env = NormalizeObservation(env)
        env = RecordEpisodeStatistics(env)
        env = NumpyToTorch(env)

        # should be (batch_size, obs_size) 2D tensor
        # print(env.observation_space.shape)
        assert len(env.observation_space.shape) == 2

        print(f'"{env_name}": ', end='')

        t, st = 0, 0
        _ = env.reset()
        while st < 40_000:
            a = torch.from_numpy(env.action_space.sample())
            obs, rew, term, trunc, info = env.step(a)
            mask = info.get('_episode', None)
            if mask is None:
                continue
            ep_stats = info['episode']
            st += ep_stats['l'][mask].sum()
            t += ep_stats['t'][mask].sum()

        st, t = st, t / env.num_envs
        print(f'{st / t / 1_000:.2f} kfps')

    print('Benchmarking vectorised environments ==>')
    for name in ["CartPole-v1", "LunarLander-v3", "MountainCar-v0"]:
        _run_test(name)


def compare_envs():
    benchmark_non_vec_env()
    benchmark_vec_env()
