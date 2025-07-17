from functools import partial
from types import SimpleNamespace

from oculr.image.buffer import PrefetchedImageBuffer

from stelarc.ac_pipeline import run_experiment
from stelarc.agents.ac import Agent
from stelarc.config import get_seed, ns_to_dict
from stelarc.make_env import make_env
from stelarc.agents.utils.torch import get_device


def run_mnist():

    seed = None

    num_envs = 128
    config = SimpleNamespace(
        device='cpu',
        seed=get_seed(),

        ds=SimpleNamespace(
            ds='mnist', grayscale=True,
            # center="pixel", lp_norm=2,
        ),
        env=SimpleNamespace(
            num_envs=num_envs, obs_hw_shape=4, max_time_steps=20,
            answer_reward=(1.0, 0.0), step_reward=0.0,
            img_buffer_fn=partial(PrefetchedImageBuffer, prefetch_size=num_envs * 4),
            termination_policy='first_guess',
            # termination_policy='until_correct',
            reset_as_step=True,

            stats_buffer_eps=20,
        ),

        model=SimpleNamespace(
            obs_size=None, obs_encoder=[16],
            mem_hidden_size=32, mem_skip_connection=False,
            body_shared=[16], body_separate=[16], body_skip_connection=True,
            policy_heads=None,
        ),

        agent=SimpleNamespace(
            learning_rate=0.008,
            lr_min=20.0, lr_decay_slope=1.0, lr_decay_speed=1.0,

            discount_gamma=0.99, gae_lambda=None,
            val_loss_scale=1.0,

            ent_loss_scale=0.1, ent_loss_heads_scale=(1.0, 0.7, 1.0, 1.0),
            ent_loss_scale_decay=0.9998,
            ent_loss_scale_max_decay=10.,
        ),

        run=SimpleNamespace(
            n_epochs=1, n_batches=2_000, n_batch_steps=8,
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

    env, test_env, obs_size, policy_heads_description, action_types_description = make_env(config)

    config.model.obs_size = obs_size
    config.model.policy_heads = policy_heads_description
    config.model.action_types = action_types_description

    from stelarc.model import Model
    model = Model(**ns_to_dict(config.model),)

    agent = Agent(
        model=model,
        seed=config.seed, device=get_device(config.device),
        **ns_to_dict(config.agent)
    )

    run_experiment(config, env, agent, test_env=test_env)


def run_cifar_grayscale():
    seed = None

    num_envs = 128
    config = SimpleNamespace(
        device='cpu',
        seed=get_seed(),

        ds=SimpleNamespace(
            ds='cifar', grayscale=True, center="pixel",
            # lp_norm=2,
        ),
        env=SimpleNamespace(
            num_envs=num_envs, obs_hw_shape=4, max_time_steps=25,
            answer_reward=(1.0, 0.0), step_reward=0.0,
            img_buffer_fn=partial(PrefetchedImageBuffer, prefetch_size=num_envs * 4),
            termination_policy='first_guess',
            # termination_policy='until_correct',
            reset_as_step=True,

            stats_buffer_eps=20,
        ),

        model=SimpleNamespace(
            obs_size=None, obs_encoder=[32],
            mem_hidden_size=64, mem_skip_connection=False,
            body_shared=[32], body_separate=[32], body_skip_connection=True,
            policy_heads=None,
        ),

        agent=SimpleNamespace(
            learning_rate=0.008,
            lr_min=20.0, lr_decay_slope=1.0, lr_decay_speed=1.0,

            discount_gamma=0.99, gae_lambda=None,
            val_loss_scale=0.25,

            ent_loss_scale=0.03, ent_loss_heads_scale=(1.0, 0.7, 1.0, 1.0),
            ent_loss_scale_decay=0.9999,
            ent_loss_scale_max_decay=10.,
        ),

        run=SimpleNamespace(
            n_epochs=10, n_batches=2_000, n_batch_steps=8,
            eval_configs=[
                dict(n_batches=10, is_greedy=True),
                dict(n_batches=10, is_greedy=False),
            ],
        ),

        log=SimpleNamespace(
            schedule=200_000,
            wandb=False,
            project='rlcam-ppo'
        ),
    )
    config.run.batch_size = config.run.n_batch_steps * config.env.num_envs

    env, test_env, obs_size, policy_heads_description, action_types_description = make_env(config)

    config.model.obs_size = obs_size
    config.model.policy_heads = policy_heads_description
    config.model.action_types = action_types_description

    from stelarc.model import Model
    model = Model(**ns_to_dict(config.model),)

    agent = Agent(
        model=model,
        seed=config.seed, device=get_device(config.device),
        **ns_to_dict(config.agent)
    )

    run_experiment(config, env, agent, test_env=test_env)


def run_cifar_rgb():
    seed = None

    num_envs = 128
    config = SimpleNamespace(
        device='cpu',
        seed=get_seed(),

        ds=SimpleNamespace(
            ds='cifar', grayscale=False, center="pixel",
            # lp_norm=2,
        ),
        env=SimpleNamespace(
            num_envs=num_envs, obs_hw_shape=3, max_time_steps=25,
            answer_reward=(1.0, 0.0),
            # step_reward=-0.00001, zoom_reward=-0.003, move_reward=0.005,
            img_buffer_fn=partial(PrefetchedImageBuffer, prefetch_size=num_envs * 4),
            termination_policy='first_guess',
            # termination_policy='until_correct',
            reset_as_step=False,

            stats_buffer_eps=50,
        ),

        model=SimpleNamespace(
            obs_size=None, obs_encoder=[32],
            mem_hidden_size=64, mem_skip_connection=False, is_lstm=False,
            body_shared=[64], body_separate=[32], body_skip_connection=False,
            policy_heads=None,
        ),

        agent=SimpleNamespace(
            learning_rate=0.008,
            lr_min=4.0, lr_decay_slope=1.0, lr_decay_speed=0.4,

            discount_gamma=0.99, gae_lambda=None,
            val_loss_scale=0.25,

            ent_loss_scale=0.03, ent_loss_heads_scale=(1.0, 0.7, 0.7, 0.7),
            ent_loss_scale_decay=0.9999,
            ent_loss_scale_max_decay=10.,
        ),

        run=SimpleNamespace(
            n_epochs=10, n_batches=4_000, n_batch_steps=4,
            eval_configs=[
                dict(n_batches=10, is_greedy=True),
                dict(n_batches=10, is_greedy=False),
            ],
        ),

        log=SimpleNamespace(
            schedule=200_000,
            wandb=False,
            project='rlcam-ppo'
        ),
    )
    config.run.batch_size = config.run.n_batch_steps * config.env.num_envs

    env, test_env, obs_size, policy_heads_description, action_types_description = make_env(config)

    config.model.obs_size = obs_size
    config.model.policy_heads = policy_heads_description
    config.model.action_types = action_types_description

    from stelarc.model import Model
    model = Model(**ns_to_dict(config.model),)

    agent = Agent(
        model=model,
        seed=config.seed, device=get_device(config.device),
        **ns_to_dict(config.agent)
    )

    run_experiment(config, env, agent, test_env=test_env)


def main():
    # run_mnist()
    # run_cifar_grayscale()
    run_cifar_rgb()


if __name__ == '__main__':
    main()
