from gymnasium.wrappers.vector import FlattenObservation
from gymnasium.wrappers.vector.numpy_to_torch import NumpyToTorch
from oculr.dataset import Dataset
from oculr.env import ImageEnvironment

from stelarc.agents.utils.torch import get_device
from stelarc.config import ns_to_dict


def make_env(config):
    ds_config = ns_to_dict(config.ds)
    env_config = ns_to_dict(config.env)
    seed = config.seed
    # device = get_device(config.device)
    device = get_device('cpu')

    stats_buffer_eps = env_config.pop('stats_buffer_eps', None)
    ds = Dataset(seed=seed, **ds_config)

    env = ImageEnvironment(ds=ds, seed=seed, **env_config)
    test_env = ImageEnvironment(ds=ds, seed=seed, is_eval=True, **env_config)

    def _wrap_env(_env):
        _env = FlattenObservation(_env)

        if stats_buffer_eps is not None:
            from gymnasium.wrappers.vector import RecordEpisodeStatistics
            n_envs = config.env.num_envs
            _env = RecordEpisodeStatistics(_env, buffer_length=n_envs * stats_buffer_eps)

        _env = NumpyToTorch(_env, device=device)
        return _env

    env = _wrap_env(env)
    test_env = _wrap_env(test_env)

    num_envs, obs_size = env.observation_space.shape
    policy_heads_description = env.metadata['action_space_description']
    action_types_description = env.metadata['action_types_description']
    single_action_space = env.single_action_space
    print(f'obs: {obs_size}    act: {single_action_space}    num_envs: {num_envs}')

    return env, test_env, obs_size, policy_heads_description, action_types_description
