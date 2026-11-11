"""Train the Reacher task with stable-baselines3 PPO through Eden's gymnasium registration.

---
tier: recipe
capabilities: [rl.sb3, task.reaching, robot.yam]
hardware: cpu
optional_deps: [sb3]
run: uv run python examples/recipes/rl_libraries/sb3_ppo.py -c
---

``Sb3VecEnvWrapper.from_config`` builds the task and hands SB3 its ``policy`` observation group as
numpy. SB3 copies every step to the host, so it trains well below the throughput of ``eden train``
(rsl_rl) on the same task.

Usage:
    uv run python examples/recipes/rl_libraries/sb3_ppo.py              # GPU sim, 256 envs
    uv run python examples/recipes/rl_libraries/sb3_ppo.py -c -b 64 -s 20
"""

import argparse
import os

import eden as en
import genesis as gs
import torch

#: Steps each env collects per PPO iteration (rsl_rl's ``num_steps_per_env`` for the task).
ROLLOUT_STEPS = 24


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--cpu", action="store_true", help="Simulate and train on the CPU.")
    parser.add_argument("-b", "--num-envs", type=int, default=256, help="Number of parallel envs.")
    parser.add_argument("-s", "--steps", type=int, default=150, help="PPO iterations.")
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, log_root_path="logs/reacher_sb3", performance_mode=True)

    from stable_baselines3 import PPO

    from eden.envs.wrappers.sb3_env import Sb3VecEnvWrapper
    from eden.tasks import TASK_REGISTRY

    num_envs = 4 if en.is_smoke() else args.num_envs
    iterations = 1 if en.is_smoke() else args.steps
    config = TASK_REGISTRY.build("reacher")
    config.env_options.num_envs = num_envs
    # SB3 needs bounded actions; the reacher's action term scales them by 0.05 rad, so ±1 is ample.
    # Staggered episode starts, as `eden train` uses: without them every env times out on the same step.
    env = Sb3VecEnvWrapper.from_config(config, clip_actions=1.0, random_episode_starts=True, seed=0)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=ROLLOUT_STEPS,
        batch_size=num_envs * ROLLOUT_STEPS // 4,
        n_epochs=5,
        learning_rate=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        max_grad_norm=1.0,
        policy_kwargs={"net_arch": [256, 128], "activation_fn": torch.nn.ELU},
        tensorboard_log=en.log_dir,
        device="cpu" if args.cpu else "auto",
        seed=0,
        verbose=1,
    )
    model.learn(total_timesteps=iterations * ROLLOUT_STEPS * num_envs)
    model.save(os.path.join(en.log_dir, "sb3_ppo_reacher"))
    env.close()


if __name__ == "__main__":
    main()
