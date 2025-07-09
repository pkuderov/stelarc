from types import SimpleNamespace

from stelarc.agents.ppo_classic import device, ActorCritic, PpoClassic
from stelarc.pipeline_ppo import run_experiment, make_classic_env
from stelarc.config import get_seed, set_seed


def test_ppo(env_name ="CartPole-v1"):
    config = SimpleNamespace()

    if hasattr(config, 'log'):
        config.log.wandb = False
        config.run.steps_range = (10_000, 25_000)
        env, *_ = make_classic_env(**config.env.__dict__, device=config.device)
        agent = PpoClassic(**config.ppo.__dict__)
        run_experiment(config, env, agent)

    config = SimpleNamespace(
        device=device,
        seed=get_seed(),

        env=SimpleNamespace(
            name=env_name, n_envs=64, max_steps=500,
            r_scale=0.01, stats_buffer_eps=4,
            solved_reward=490,
        ),

        agent=SimpleNamespace(
            ac_type=ActorCritic,
            hidden_size=32, lr=0.001, ema_lr=0.0, betas=(0.9, 0.999),
            gamma=0.995, gae_lambda=0.95, K_epochs=4, mini_batch_size=256,
            eps_clip=0.1, v_clip=False,
            v_loss_alpha=1.0, ent_loss_alpha=0.2, min_ent_loss_ratio=1 / 40.
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

    config.env.solved_reward *= config.env.r_scale
    config.agent.ent_loss_alpha *= config.env.r_scale
    config.run.batch_size = config.run.n_batch_steps * config.env.n_envs

    set_seed(config.seed)
    env, obs_size, n_actions = make_classic_env(
        **config.env.__dict__, device=config.device
    )

    config.agent.obs_size, config.agent.n_actions = obs_size, n_actions
    agent = PpoClassic(**config.agent.__dict__)

    print(f'{config.seed=} | {config.agent.lr=}')

    run_experiment(config, env, agent)


def main():
    # compare_envs()
    test_ppo()
    pass


if __name__ == '__main__':
    main()
