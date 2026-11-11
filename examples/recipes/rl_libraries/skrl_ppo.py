"""Train the Reacher task with skrl PPO, the critic reading its own observation group as skrl's state.

---
tier: recipe
capabilities: [rl.skrl, task.reaching, robot.yam]
hardware: cpu
optional_deps: [skrl]
run: uv run python examples/recipes/rl_libraries/skrl_ppo.py -c
---

``SkrlVecEnvWrapper`` keeps every tensor on the env's device. The policy observes the task's
``policy`` group (noise-corrupted); the value model reads the clean ``critic`` group as skrl's
``STATES``. The agent is assembled by skrl's ``Runner`` from the config dict below, the same layout
as an skrl YAML file.

Usage:
    uv run python examples/recipes/rl_libraries/skrl_ppo.py              # GPU, 4096 envs
    uv run python examples/recipes/rl_libraries/skrl_ppo.py -c -b 64 -s 20
"""

import argparse

import eden as en
import genesis as gs

#: Steps each env collects per PPO iteration.
ROLLOUT_STEPS = 24


def runner_cfg(iterations: int, log_dir: str) -> dict:
    """The skrl ``Runner`` config: the reacher's rsl_rl PPO hyperparameters, in skrl's terms."""

    def mlp(source: str) -> list[dict]:
        return [{"name": "net", "input": source, "layers": [256, 128], "activations": "elu"}]

    return {
        "seed": 0,
        "models": {
            "separate": True,
            "policy": {
                "class": "GaussianMixin",
                "initial_log_std": 0.0,
                "network": mlp("OBSERVATIONS"),
                "output": "ACTIONS",
            },
            "value": {"class": "DeterministicMixin", "network": mlp("STATES"), "output": "ONE"},
        },
        "memory": {"class": "RandomMemory", "memory_size": -1},
        "agent": {
            "class": "PPO",
            "rollouts": ROLLOUT_STEPS,
            "learning_epochs": 5,
            "mini_batches": 4,
            "discount_factor": 0.99,
            "gae_lambda": 0.95,
            "learning_rate": 1e-3,
            "learning_rate_scheduler": "KLAdaptiveLR",
            "learning_rate_scheduler_kwargs": {"kl_threshold": 0.01},
            "grad_norm_clip": 1.0,
            "entropy_loss_scale": 0.005,
            "value_loss_scale": 1.0,
            # skrl's PPO needs its inputs and value targets normalized to learn this task: without these
            # three it stays at a do-nothing policy's reward for the whole run.
            "observation_preprocessor": "RunningStandardScaler",
            "state_preprocessor": "RunningStandardScaler",
            "value_preprocessor": "RunningStandardScaler",
            "experiment": {"directory": log_dir, "experiment_name": "skrl_ppo", "write_interval": "auto"},
        },
        # "log" records Eden's per-term episode statistics (Episode_Reward/<term>, ...) in skrl's TensorBoard.
        "trainer": {"class": "SequentialTrainer", "timesteps": iterations * ROLLOUT_STEPS, "environment_info": "log"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--cpu", action="store_true", help="Simulate and train on the CPU.")
    parser.add_argument("-b", "--num-envs", type=int, default=4096, help="Number of parallel envs.")
    parser.add_argument("-s", "--steps", type=int, default=150, help="PPO iterations.")
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, log_root_path="logs/reacher_skrl", performance_mode=True)

    from skrl.utils.runner.torch import Runner

    from eden.envs.wrappers.skrl_env import SkrlVecEnvWrapper
    from eden.tasks import TASK_REGISTRY

    config = TASK_REGISTRY.build("reacher")
    config.env_options.num_envs = 4 if en.is_smoke() else args.num_envs
    # Staggered episode starts, as `eden train` uses: without them every env times out on the same step.
    env = SkrlVecEnvWrapper.from_config(
        config, state_group="critic", clip_actions=1.0, random_episode_starts=True, seed=0
    )

    runner = Runner(env, runner_cfg(1 if en.is_smoke() else args.steps, en.log_dir))
    runner.run("train")


if __name__ == "__main__":
    main()
