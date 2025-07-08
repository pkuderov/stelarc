
def main():
    from functools import partial

    from oculr.dataset import Dataset
    from oculr.env import ImageEnvironment
    from oculr.image.buffer import PrefetchedImageBuffer

    from stelarc.agents.ac import RlClassifier
    from stelarc.pipeline import run_experiment

    seed = 8041990
    ds = Dataset(seed, 'mnist', grayscale=True, center="pixel", lp_norm=1)

    # 40000: 88.47 0.313  |  2.7  |  -0.2380   0.0401
    env_config = dict(
        ds=ds, num_envs=64, obs_hw_shape=9, max_time_steps=20, seed=None,
        answer_reward=(1.0, 0.0), step_reward=0.0,
        img_buffer_fn=partial(PrefetchedImageBuffer, prefetch_size=256),
    )
    env = ImageEnvironment(**env_config)
    test_env = ImageEnvironment(is_eval=True, **env_config)

    agent = RlClassifier(
        obs_size=env.total_obs_size,
        # encoder=[32], hid_size=32, skip_connection=True, body=[32],
        encoder=[32], hid_size=32, skip_connection=True, body=[],
        pi_heads=(3, env.n_classes, *env.pos_range[1]),
        learning_rate=0.004,
        gamma=0.99,
        batch_size=256,
        # batch_size=64,
        # val_loss_scale=0.5, ent_loss_scale=0.1,
        val_loss_scale=0.25, ent_loss_scale=0.04,
        ent_scales=(1.0, 0.1, 0.4, 0.4),
        seed=seed, min_lr_scale=10.0
    )

    run_experiment(
        env, test_env, agent, n_epochs=10, n_steps=10_000,
        eval_configs=[
            dict(n_steps=1_000, is_greedy=True),
            dict(n_steps=1_000, is_greedy=False),
        ]
    )


if __name__ == '__main__':
    main()
