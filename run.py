def run_mnist():
    from functools import partial

    from oculr.dataset import Dataset
    from oculr.env import ImageEnvironment
    from oculr.image.buffer import PrefetchedImageBuffer

    from stelarc.agents.ac import RlClassifier
    from stelarc.pipeline import run_experiment

    seed = None
    ds = Dataset('mnist', grayscale=True, center="pixel", lp_norm=2, seed=seed)

    num_envs = 128
    env_config = dict(
        ds=ds, num_envs=num_envs, obs_hw_shape=5, max_time_steps=20, seed=None,
        answer_reward=(1.0, 0.0), step_reward=0.0,
        img_buffer_fn=partial(PrefetchedImageBuffer, prefetch_size=num_envs * 8),
        # termination_policy='first_guess',
        termination_policy='until_correct',
    )
    env = ImageEnvironment(**env_config)
    test_env = ImageEnvironment(is_eval=True, **env_config)

    agent = RlClassifier(
        obs_size=env.total_obs_size,
        encoder=[], hid_size=32, skip_connection=True, body=[32],
        pi_heads=(3, env.n_classes, *env.pos_range[1]),
        learning_rate=0.01,
        gamma=0.99,
        batch_size=num_envs * 2,
        val_loss_scale=0.25, ent_loss_scale=0.04,
        ent_scales=(1.0, 0.1, 0.4, 0.4),
        seed=seed, min_lr_scale=10.0
    )

    run_experiment(
        env, test_env, agent, n_epochs=10, n_steps=2_000,
        eval_configs=[
            dict(n_steps=400, is_greedy=True),
            dict(n_steps=400, is_greedy=False),
        ],
    )
    # run_experiment(
    #     env, test_env, agent, n_epochs=10, n_steps=10_000,
    #     eval_configs=[
    #         dict(n_steps=1_000, is_greedy=True),
    #         dict(n_steps=1_000, is_greedy=False),
    #     ],
    # )


def run_cifar_grayscale():
    from functools import partial

    from oculr.dataset import Dataset
    from oculr.env import ImageEnvironment
    from oculr.image.buffer import PrefetchedImageBuffer

    from stelarc.agents.ac import RlClassifier
    from stelarc.pipeline import run_experiment

    seed = None
    ds = Dataset('cifar', grayscale=True, center="pixel", seed=seed)

    num_envs = 128
    env_config = dict(
        ds=ds, num_envs=num_envs, obs_hw_shape=7, max_time_steps=20, seed=None,
        answer_reward=(1.0, 0.0), step_reward=0.0,
        img_buffer_fn=partial(PrefetchedImageBuffer, prefetch_size=num_envs * 8),
        termination_policy='first_guess',
        # termination_policy='until_correct',
    )
    env = ImageEnvironment(**env_config)
    test_env = ImageEnvironment(is_eval=True, **env_config)

    agent = RlClassifier(
        obs_size=env.total_obs_size,
        encoder=[], hid_size=32, skip_connection=True, body=[32],
        pi_heads=(3, env.n_classes, *env.pos_range[1]),
        learning_rate=0.01,
        gamma=0.99,
        batch_size=num_envs * 2,
        val_loss_scale=0.25, ent_loss_scale=0.2,
        ent_scales=(1.0, 0.1, 0.4, 0.4),
        seed=seed, min_lr_scale=10.0
    )

    run_experiment(
        env, test_env, agent, n_epochs=10, n_steps=2_000,
        eval_configs=[
            dict(n_steps=400, is_greedy=True),
            dict(n_steps=400, is_greedy=False),
        ],
    )
    # run_experiment(
    #     env, test_env, agent, n_epochs=10, n_steps=10_000,
    #     eval_configs=[
    #         dict(n_steps=1_000, is_greedy=True),
    #         dict(n_steps=1_000, is_greedy=False),
    #     ],
    # )


def run_cifar_rgb():
    from functools import partial

    from oculr.dataset import Dataset
    from oculr.env import ImageEnvironment
    from oculr.image.buffer import PrefetchedImageBuffer

    from stelarc.agents.ac import RlClassifier
    from stelarc.pipeline import run_experiment

    seed = None
    ds = Dataset('cifar', grayscale=False, center="pixel", seed=seed)

    num_envs = 128
    env_config = dict(
        ds=ds, num_envs=num_envs, obs_hw_shape=5, max_time_steps=20, seed=None,
        answer_reward=(1.0, 0.0), step_reward=0.0,
        img_buffer_fn=partial(PrefetchedImageBuffer, prefetch_size=num_envs * 8),
        termination_policy='first_guess',
        # termination_policy='until_correct',
    )
    env = ImageEnvironment(**env_config)
    test_env = ImageEnvironment(is_eval=True, **env_config)

    agent = RlClassifier(
        obs_size=env.total_obs_size,
        encoder=[], hid_size=32, skip_connection=True, body=[32],
        pi_heads=(3, env.n_classes, *env.pos_range[1]),
        learning_rate=0.01,
        gamma=0.99,
        batch_size=num_envs * 4,
        val_loss_scale=0.25, ent_loss_scale=0.2,
        ent_scales=(1.0, 0.1, 0.4, 0.4),
        seed=seed, min_lr_scale=10.0
    )

    run_experiment(
        env, test_env, agent, n_epochs=10, n_steps=2_000,
        eval_configs=[
            dict(n_steps=400, is_greedy=True),
            dict(n_steps=400, is_greedy=False),
        ],
    )
    # run_experiment(
    #     env, test_env, agent, n_epochs=10, n_steps=10_000,
    #     eval_configs=[
    #         dict(n_steps=1_000, is_greedy=True),
    #         dict(n_steps=1_000, is_greedy=False),
    #     ],
    # )


def main():
    # run_mnist()
    # run_cifar_grayscale()
    run_cifar_rgb()


if __name__ == '__main__':
    main()
