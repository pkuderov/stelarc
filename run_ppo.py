def run_mnist():
    from functools import partial
    from types import SimpleNamespace

    from oculr.image.buffer import PrefetchedImageBuffer

    from stelarc.agents.ppo import Agent
    from stelarc.config import get_seed, ns_to_dict
    from stelarc.ppo_pipeline import run_experiment
    from stelarc.make_env import make_env
    from stelarc.agents.utils.torch import get_device

    seed = None

    num_envs = 64
    config = SimpleNamespace(
        device='cpu',
        seed=get_seed(),

        ds=SimpleNamespace(
            ds='mnist', grayscale=True, center="pixel",
        ),
        env=SimpleNamespace(
            num_envs=num_envs, obs_hw_shape=5, max_time_steps=20,
            answer_reward=(1.0, 0.0), step_reward=0.0,
            img_buffer_fn=partial(PrefetchedImageBuffer, prefetch_size=num_envs * 4),
            termination_policy='first_guess',
            # termination_policy='until_correct',

            stats_buffer_eps=20,
        ),

        model=SimpleNamespace(
            obs_size=None, obs_encoder=[],
            mem_hidden_size=32, mem_skip_connection=True,
            body_shared=[], body_separate=[32],
            policy_heads=None,
        ),

        agent=SimpleNamespace(
            learning_rate=0.003, lr_decay=None, lr_max_decay=10.0,
            adamw_betas=(0.9, 0.999),
            mini_batch_size=512, batch_epochs=2,

            discount_gamma=0.99, gae_lambda=0.95,
            ppo_pi_clip=0.1, ppo_v_clip_enabled=False,
            val_loss_scale=1.0,

            ent_loss_scale=0.5, ent_loss_heads_scale=(1.0, 0.1, 0.4, 0.4),
            ent_loss_scale_decay=0.999,
            ent_loss_scale_max_decay=400.,

            weights_ema_lr=0.0,
        ),

        run=SimpleNamespace(
            n_epochs=10, n_batches=5_000, n_batch_steps=8,
            eval_configs=[
                dict(n_batches=10, is_greedy=True),
                dict(n_batches=10, is_greedy=False),
            ],
        ),

        log=SimpleNamespace(
            schedule=100_000,
            wandb=False,
            project='rlcam-ppo'
        ),
    )
    config.run.batch_size = config.run.n_batch_steps * config.env.num_envs

    env, test_env, obs_size, policy_heads_description = make_env(config)

    config.model.obs_size = obs_size
    config.model.policy_heads = policy_heads_description

    from stelarc.model import Model
    model = Model(**ns_to_dict(config.model),)

    agent = Agent(
        model=model,
        seed=config.seed, device=get_device(config.device),
        **ns_to_dict(config.agent)
    )

    run_experiment(config, env, agent, test_env=test_env)


def main():
    run_mnist()


if __name__ == '__main__':
    main()
