def run_mnist():
    from functools import partial
    from types import SimpleNamespace

    from oculr.image.buffer import PrefetchedImageBuffer

    from stelarc.agents.ppo import Agent
    from stelarc.agents.ac import RlClassifier
    from stelarc.config import get_seed, ns_to_dict
    from stelarc.pipeline import make_env
    from stelarc.pipeline_ac import run_experiment
    from stelarc.agents.utils.torch import get_device

    seed = None

    num_envs = 128
    config = SimpleNamespace(
        device='cpu',
        seed=get_seed(),

        ds=SimpleNamespace(
            ds='mnist', grayscale=True, center="pixel",
        ),
        env=SimpleNamespace(
            num_envs=num_envs, obs_hw_shape=5, max_time_steps=20,
            answer_reward=(1.0, 0.0), step_reward=0.0,
            img_buffer_fn=partial(PrefetchedImageBuffer, prefetch_size=num_envs * 8),
            # termination_policy='first_guess',
            termination_policy='until_correct',

            stats_buffer_eps=4,
        ),

        agent=SimpleNamespace(
            obs_size=None,
            layer_topology=SimpleNamespace(
                obs_encoder=[16], mem_hidden_size=16, mem_skip_connection=True,
                body=[16],
            ),

            learning_rate=0.001, lr_decay=None, lr_max_decay=10.0,
            adamw_betas=(0.9, 0.999),
            mini_batch_size=256, batch_epochs=4,

            discount_gamma=0.99, gae_lambda=0.95,
            ppo_pi_clip=0.1, ppo_v_clip_enabled=False,
            val_loss_scale=1.0,

            ent_loss_scale=0.04, ent_loss_heads_scale=(1.0, 0.1, 0.4, 0.4),
            ent_loss_scale_decay=0.997,
            ent_loss_scale_max_decay=40.,

            weights_ema_lr=0.0,
        ),

        run=SimpleNamespace(
            steps_range=(20_000, 500_000), n_batch_steps=16,
        ),

        log=SimpleNamespace(
            schedule=10_000,
            wandb=False,
            project='ppo-v-clip-vec'
        ),
    )
    config.run.batch_size = config.run.n_batch_steps * config.env.num_envs

    env, test_env, obs_size, policy_heads_box = make_env(config)

    config.agent.obs_size = obs_size
    config.agent.policy_heads = env.metadata['action_space_description']

    agent = Agent(
        seed=config.seed, device=get_device(config.device),
        **ns_to_dict(config.agent)
    )

    run_experiment(
        env, test_env, agent, n_epochs=10, n_steps=1_000,
        eval_configs=[
            dict(n_steps=200, is_greedy=True),
            dict(n_steps=200, is_greedy=False),
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
    run_mnist()


if __name__ == '__main__':
    main()
