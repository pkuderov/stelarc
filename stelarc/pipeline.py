from gymnasium.wrappers.vector.numpy_to_torch import NumpyToTorch
from oculr.dataset import Dataset
from oculr.env import ImageEnvironment

from stelarc.agents.utils.torch import get_device
from stelarc.config import ns_to_dict


def make_env(config):
    ds_config = ns_to_dict(config.ds)
    env_config = ns_to_dict(config.env)
    seed = config.seed
    device = get_device(config.device)

    stats_buffer_eps = env_config.pop('stats_buffer_eps', None)
    ds = Dataset(seed=seed, **ds_config)

    env = ImageEnvironment(ds=ds, seed=seed, **env_config)
    test_env = ImageEnvironment(ds=ds, seed=seed, is_eval=True, **env_config)

    if stats_buffer_eps is not None:
        from gymnasium.wrappers.vector import RecordEpisodeStatistics
        num_envs = config.env.num_envs
        env = RecordEpisodeStatistics(env, buffer_length=num_envs * stats_buffer_eps)

    env = NumpyToTorch(env, device=device)
    # print(env.spec)

    num_envs, obs_size = env.observation_space.shape
    print(f'obs: {obs_size}    act: {env.single_action_space}    num_envs: {num_envs}')

    return env, test_env, obs_size, env.single_action_space.nvec
