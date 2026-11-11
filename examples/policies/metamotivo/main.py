"""Run a pretrained MetaMotivo (FBcpr) policy on Eden's SMPL humanoid.

---
tier: policy
capabilities: [task.mimic, robot.smpl_humanoid]
hardware: cpu
optional_deps: [metamotivo]
run: uv run python examples/policies/metamotivo/main.py --vis
---

Usage
-----
::

    # Eden mode (default)
    python examples/policies/metamotivo/main.py --vis

    # Side-by-side comparison with HumEnv reference
    python examples/policies/metamotivo/main.py --compare
"""

from argparse import ArgumentParser

import torch
import numpy as np
from tqdm import tqdm
import genesis as gs

import eden as en
from eden.envs.base import RLEnvBase
from eden.options import (
    ActionManagerOptions,
    EnvOptions,
    ObservationManagerOptions,
)
from eden.extensions.policies.metamotivo.policy import MetaMotivoPolicy


def main():
    parser = ArgumentParser(description="MetaMotivo on Eden SMPL Humanoid")
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/metamotivo-S-1",
        help="MetaMotivo checkpoint to pull from the Hugging Face hub",
        choices=["facebook/metamotivo-S-1", "facebook/metamotivo-M-1"],
    )
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend")
    parser.add_argument("-v", "--vis", action="store_true", help="Show viewer")
    parser.add_argument("-s", "--steps", type=int, default=300, help="Number of steps to run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for z context")
    parser.add_argument("--goal", action="store_true", help="Use goal inference (crouching pose)")
    parser.add_argument("-b", "--num-envs", type=int, default=1, help="Number of parallel environments")
    args = parser.parse_args()

    """Run the MetaMotivo policy on Eden."""

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    env = RLEnvBase(
        env_options=EnvOptions(
            num_envs=args.num_envs,
            **MetaMotivoPolicy.env_options(),
        ),
        scene_options=en.scenes.Empty(robot=MetaMotivoPolicy.robot_options()),
        action_options=ActionManagerOptions(**MetaMotivoPolicy.action_options()),
        observation_options=ObservationManagerOptions(**MetaMotivoPolicy.observation_options()),
        show_viewer=args.vis,
        eval_mode=True,
    )
    env.build()

    policy = MetaMotivoPolicy(
        env=env,
        options=MetaMotivoPolicy.configure(model_name=args.model_name),
    )

    if args.goal:
        # fmt: off
        goal_qpos = np.array([
            0.13769039,-0.20029453,0.42305034,0.21707786,0.94573617,0.23868944,
            0.03856998,-1.05566834,-0.12680767,0.11718296,1.89464102,-0.01371153,
            -0.07981451,-0.70497424,-0.0478,-0.05700732,-0.05363342,-0.0657329,
            0.08163511,-1.06263979,0.09788937,-0.22008936,1.85898192,0.08773695,
            0.06200327,-0.3802791,0.07829525,0.06707749,0.14137152,0.08834448,
            -0.07649805,0.78328658,0.12580912,-0.01076061,-0.35937259,-0.13176489,
            0.07497022,-0.2331914,-0.11682692,0.04782308,-0.13571422,0.22827948,
            -0.23456622,-0.12406075,-0.04466465,0.2311667,-0.12232673,-0.25614032,
            -0.36237662,0.11197906,-0.08259534,-0.634934,-0.30822742,-0.93798716,
            0.08848668,0.4083417,-0.30910404,0.40950143,0.30815359,0.03266103,
            1.03959336,-0.19865537,0.25149713,0.3277561,0.16943092,0.69125975,
            0.21721349,-0.30871948,0.88890484,-0.08884043,0.38474549,0.30884107,
            -0.40933304,0.30889523,-0.29562966,-0.6271498,
        ])
        # fmt: on

        env.unwrapped.entities["robot"].set_qpos(goal_qpos)
        goal_obs = env.unwrapped.observation_manager.compute()["proprio"]
        policy.compute_goal_context(goal_obs)
    else:
        if args.seed is not None:
            torch.manual_seed(args.seed)
        policy.sample_random_context()

    obs, _ = env.reset()

    print(f"Running {args.steps} steps...")
    for _ in tqdm(range(args.steps)):
        raw_action = policy.step(obs)
        obs, _, _, _, _ = env.step(raw_action)


if __name__ == "__main__":
    main()
