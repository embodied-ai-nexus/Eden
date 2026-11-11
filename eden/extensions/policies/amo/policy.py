"""AMO whole-body control policy."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
import torch

from eden.extensions.policies.base import PolicyBase, PolicyOptions, POLICY_REGISTRY
from eden.options.managers.observations import ObservationGroupOptions
from eden.options.robots.unitree_g1 import G1_UCSD_NO_WRIST_DOFS_NAME, G1Dofs29NoWristDof
from eden.terms import ObsTerm
from eden import actions, observations
from eden.extensions.policies.amo.custom_terms.commands import AMOCommand
from eden.extensions.policies.amo.custom_terms.observations import (
    amo_adapter_output,
    amo_gait_obs,
    amo_demo_obs,
    amo_dyaw_obs,
    amo_dofs_vel,
    amo_last_action,
)
from eden.constants import ReferenceSource

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.commands import CommandTermOptions


# AMO training default pose / PD gains for the 23-DOF G1 (from the AMO release).
AMO_DEFAULT_DOFS_POS: dict[str, float] = {
    "left_hip_pitch_joint": -0.1,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.1,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.5,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.2,
    "left_elbow_joint": 0.3,
    "right_shoulder_pitch_joint": 0.5,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": -0.2,
    "right_elbow_joint": 0.3,
}
AMO_DEFAULT_DOFS_KP: dict[str, float] = {
    ".*_hip_pitch.*": 150.0,
    ".*_hip_roll.*": 150.0,
    ".*_hip_yaw.*": 150.0,
    ".*_knee.*": 300.0,
    ".*_ankle_pitch.*": 80.0,
    ".*_ankle_roll.*": 20.0,
    "waist.*": 400.0,
    "left_shoulder_pitch.*": 80.0,
    "left_shoulder_roll.*": 80.0,
    "left_shoulder_yaw.*": 40.0,
    "left_elbow.*": 60.0,
    "right_shoulder_pitch.*": 80.0,
    "right_shoulder_roll.*": 80.0,
    "right_shoulder_yaw.*": 40.0,
    "right_elbow.*": 60.0,
}
AMO_DEFAULT_DOFS_KD: dict[str, float] = {
    ".*_hip_pitch.*": 2.0,
    ".*_hip_roll.*": 2.0,
    ".*_hip_yaw.*": 2.0,
    ".*_knee.*": 4.0,
    ".*_ankle_pitch.*": 2.0,
    ".*_ankle_roll.*": 1.0,
    "waist.*": 15.0,
    "left_shoulder_pitch.*": 2.0,
    "left_shoulder_roll.*": 2.0,
    "left_shoulder_yaw.*": 1.0,
    "left_elbow.*": 1.0,
    "right_shoulder_pitch.*": 2.0,
    "right_shoulder_roll.*": 2.0,
    "right_shoulder_yaw.*": 1.0,
    "right_elbow.*": 1.0,
}


def amo_g1_robot(**overrides: Any) -> G1Dofs29NoWristDof:
    """Build a ``G1Dofs29NoWristDof`` (AMO's no-wrist robot) with AMO's training default pose and PD gains."""
    return G1Dofs29NoWristDof(
        default_dofs_pos=AMO_DEFAULT_DOFS_POS,
        default_dofs_kp=AMO_DEFAULT_DOFS_KP,
        default_dofs_kd=AMO_DEFAULT_DOFS_KD,
        **overrides,
    )


@POLICY_REGISTRY.register()
class AMOPolicy(PolicyBase):
    """AMO: Adaptive Motion Optimization for Hyper-Dexterous Humanoid Whole-Body Control."""

    # step() reads the observation dict the env publishes, so this policy owns the deploy env.
    reads_env_observations = True

    checkpoint_path: str = "amo_jit.pt"
    # ``str | None`` like every sibling: pointing ``checkpoint_path`` at local weights
    # means clearing the pinned hash, or ``_verify_md5`` rejects them.
    md5_hash: str | None = "3efbda329f247f41a7459f50f88f4bc5"

    def __init__(self, env: EnvBase, options: PolicyOptions):
        super().__init__(env, options)

        self._policy = torch.jit.load(self.checkpoint_path, map_location=env.device)
        self._policy.eval()

    @classmethod
    def robot_options(cls) -> G1Dofs29NoWristDof:
        """Return a ``G1Dofs29NoWristDof`` with AMO's training default pose and PD gains."""
        return amo_g1_robot()

    @classmethod
    def env_options(cls) -> dict[str, Any]:
        return dict(
            sim_dt=0.002,
            decimation=10,
        )

    @classmethod
    def action_options(cls, entity_name: str = "robot") -> dict[str, ActionTermOptions]:
        return dict(
            dofs_pos_controller=actions.ImplicitPDController.configure(
                reference_source=ReferenceSource.DEFAULT,
                scale=0.25,
                entity_name=entity_name,
                dofs_name=list(G1_UCSD_NO_WRIST_DOFS_NAME),
            ),
        )

    @classmethod
    def command_options(cls, **kwargs) -> dict[str, CommandTermOptions]:
        resampling_time_range = kwargs.pop("resampling_time_range", (4.0, 4.0))
        return dict(amo_command=AMOCommand.configure(resampling_time_range=resampling_time_range, **kwargs))

    @classmethod
    def observation_options(cls, entity_name: str = "robot") -> dict[str, ObservationGroupOptions]:
        return dict(
            obs_prop=ObservationGroupOptions(
                concatenate_terms=True,
                history_length=26,  # Current + 25 past frames
                flatten_history_dim=True,
                backfill=False,  # Zero pad history
                # Terms order matches play_amo.py: ang_vel, rpy, dyaw, dofs_pos, dofs_vel, last_action, gait, adapter
                ang_vel=ObsTerm.configure(
                    func=observations.base_ang_vel,
                    scale=0.25,
                    params={"entity_name": entity_name, "frame": "body"},
                ),
                rpy=ObsTerm.configure(
                    func=observations.base_rpy,
                    params={
                        "entity_name": entity_name,
                        "terms_idx": [0, 1],
                    },  # roll, pitch
                ),
                dyaw=ObsTerm.configure(
                    func=amo_dyaw_obs,
                    params={"entity_name": entity_name, "command_name": "amo_command"},
                ),
                dofs_pos=observations.DofsPos.configure(scale=1.0, entity_name=entity_name, offset_from_default=True),
                dofs_vel=ObsTerm.configure(
                    func=amo_dofs_vel,
                    scale=0.05,
                    params={"entity_name": entity_name},
                ),
                last_action=ObsTerm.configure(
                    func=amo_last_action,
                    scale=1.0,
                    params={
                        "entity_name": entity_name,
                        "action_name": "dofs_pos_controller",
                        "command_name": "amo_command",
                    },
                ),
                gait=ObsTerm.configure(
                    func=amo_gait_obs,
                    params={"entity_name": entity_name, "command_name": "amo_command"},
                ),
                adapter=ObsTerm.configure(
                    func=amo_adapter_output,
                    params={"entity_name": entity_name, "command_name": "amo_command"},
                ),
                terms_order=[
                    "ang_vel",
                    "rpy",
                    "dyaw",
                    "dofs_pos",
                    "dofs_vel",
                    "last_action",
                    "gait",
                    "adapter",
                ],
            ),
            demo_obs=ObservationGroupOptions(
                concatenate_terms=True,
                history_length=1,
                term=ObsTerm.configure(
                    func=amo_demo_obs,
                    params={"entity_name": entity_name, "command_name": "amo_command"},
                ),
            ),
        )

    def step(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            # obs_prop frame is 93 wide (3+2+2+23+23+23+2+15); history is oldest first: [t-25, ..., t-1, t].
            dim_prop = 93
            full_hist = obs["obs_prop"]  # (N, 26 * 93)
            current_prop = full_hist[..., -dim_prop:]
            obs_hist = full_hist[..., -dim_prop * 11 : -dim_prop]  # 10 frames
            extra_hist = full_hist[..., -dim_prop * 26 : -dim_prop]  # 25 frames

            obs_demo = obs["demo_obs"]
            obs_priv = torch.zeros(self.env.num_envs, 3, device=self.env.device)

            # Policy input: [prop, demo, priv, obs_hist]
            policy_in = torch.cat([current_prop, obs_demo, obs_priv, obs_hist], dim=1)

            raw_action = self._policy(policy_in, extra_hist)
            raw_action = torch.clamp(raw_action, -40.0, 40.0)

            # Combine with arm actions
            cmd_term = self.env.command_manager.get_term("amo_command")
            arm_target = cmd_term.get_arm_target()  # (N, 8) absolute

            default_arm = cmd_term.default_arm_pos
            arm_delta = (arm_target - default_arm) / 0.25

            action = torch.cat([raw_action, arm_delta], dim=1)

        return dict(dofs_pos_controller=action)
