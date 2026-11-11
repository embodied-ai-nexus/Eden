"""GMT whole-body motion-tracking policy."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

import torch

from eden.extensions.policies.base import PolicyBase, PolicyOptions, POLICY_REGISTRY
from eden.options.managers.observations import ObservationGroupOptions
from eden.options.robots.unitree_g1 import G1_UCSD_NO_WRIST_DOFS_NAME, G1Dofs29NoWristDof
from eden.terms import ObsTerm
from eden import actions, observations
from eden.extensions.policies.gmt.custom_terms.commands import GMTMotionLibCommand
from eden.extensions.policies.gmt.custom_terms.observations import dofs_vel_no_ankle_gmt
from eden.constants import ReferenceSource

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.commands import CommandTermOptions


# GMT training default pose / PD gains for the 23-DOF G1 (from the GMT release).
GMT_DEFAULT_DOFS_POS: dict[str, float] = {
    "*_hip_pitch_joint": -0.2,
    "*_hip_roll_joint": 0.0,
    "*_hip_yaw_joint": 0.0,
    "*_knee_joint": 0.4,
    "*_ankle_pitch_joint": -0.2,
    "*_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "*_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.4,
    "*_shoulder_yaw_joint": 0.0,
    "*_elbow_joint": 1.2,
    "right_shoulder_roll_joint": -0.4,
}
GMT_DEFAULT_DOFS_KP: dict[str, float] = {
    "*_hip_pitch_joint": 100.0,
    "*_hip_roll_joint": 100.0,
    "*_hip_yaw_joint": 100.0,
    "*_knee_joint": 150.0,
    "*_ankle_pitch_joint": 40.0,
    "*_ankle_roll_joint": 40.0,
    "waist_yaw_joint": 150.0,
    "waist_roll_joint": 150.0,
    "waist_pitch_joint": 150.0,
    "*_shoulder_pitch_joint": 40.0,
    "*_shoulder_roll_joint": 40.0,
    "*_shoulder_yaw_joint": 40.0,
    "*_elbow_joint": 40.0,
}
GMT_DEFAULT_DOFS_KD: dict[str, float] = {
    "*_hip_pitch_joint": 2.0,
    "*_hip_roll_joint": 2.0,
    "*_hip_yaw_joint": 2.0,
    "*_knee_joint": 4.0,
    "*_ankle_pitch_joint": 2.0,
    "*_ankle_roll_joint": 2.0,
    "waist_yaw_joint": 4.0,
    "waist_roll_joint": 4.0,
    "waist_pitch_joint": 4.0,
    "*_shoulder_pitch_joint": 5.0,
    "*_shoulder_roll_joint": 5.0,
    "*_shoulder_yaw_joint": 5.0,
    "*_elbow_joint": 5.0,
}
# Per-joint armature + damping from GMT's vendored MJCF. waist_pitch has no armature
# in the MJCF (MuJoCo=0); matched to its waist siblings rather than Eden's 0.1 default.
GMT_DEFAULT_DOFS_ARMATURE: dict[str, float] = {
    "*_hip_pitch_joint": 0.0103,
    "*_hip_roll_joint": 0.0251,
    "*_hip_yaw_joint": 0.0103,
    "*_knee_joint": 0.0251,
    "*_ankle_pitch_joint": 0.003597,
    "*_ankle_roll_joint": 0.003597,
    "waist_yaw_joint": 0.0103,
    "waist_roll_joint": 0.0103,
    "waist_pitch_joint": 0.0103,
    "*_shoulder_pitch_joint": 0.003597,
    "*_shoulder_roll_joint": 0.003597,
    "*_shoulder_yaw_joint": 0.003597,
    "*_elbow_joint": 0.003597,
}
GMT_DEFAULT_DOFS_DAMPING: dict[str, float] = {"*_joint": 2.0}


def gmt_g1_robot(**overrides: Any) -> G1Dofs29NoWristDof:
    """Build a ``G1Dofs29NoWristDof`` (GMT's 3-DOF-waist / no-wrist robot) matching GMT's training setup.

    Adds GMT's pose/gain/armature/damping overrides. Not reproduced (need GMT's
    vendored XML): ``frictionloss=0.2`` and a ~1.036 kg ``head_link``. For exact
    fidelity pass ``file=<GMT g1.xml>``.
    """
    return G1Dofs29NoWristDof(
        default_dofs_pos=GMT_DEFAULT_DOFS_POS,
        default_dofs_kp=GMT_DEFAULT_DOFS_KP,
        default_dofs_kd=GMT_DEFAULT_DOFS_KD,
        default_dofs_armature=GMT_DEFAULT_DOFS_ARMATURE,
        default_dofs_damping=GMT_DEFAULT_DOFS_DAMPING,
        **overrides,
    )


@POLICY_REGISTRY.register()
class GMTPolicy(PolicyBase):
    """
    GMT: General Motion Tracking for Humanoid Whole-Body Control.

    NOTE
    ----
    This policy only supports 23 dofs G1 robot.
    """

    # step() reads the observation dict the env publishes, so this policy owns the deploy env.
    reads_env_observations = True

    checkpoint_path: str = "gmt_g1_23dofs.pt"
    # ``str | None`` like every sibling: pointing ``checkpoint_path`` at local weights
    # means clearing the pinned hash, or ``_verify_md5`` rejects them.
    md5_hash: str | None = "eb4cfef3de37d27c3b7eae0ac7baa8bf"

    def __init__(self, env: EnvBase, options: PolicyOptions):
        super().__init__(env, options)

        self._policy = torch.jit.load(self.checkpoint_path).to(env.device)
        self._policy.eval()

    @classmethod
    def robot_options(cls) -> G1Dofs29NoWristDof:
        """Return a ``G1Dofs29NoWristDof`` matching GMT's training pose / gains / armature / damping."""
        return gmt_g1_robot()

    @classmethod
    def env_options(cls) -> dict[str, Any]:
        return dict(
            sim_dt=0.001,
            decimation=20,
        )

    @classmethod
    def action_options(cls, entity_name: str = "robot") -> dict[str, ActionTermOptions]:
        return dict(
            dofs_pos_controller=actions.ImplicitPDController.configure(
                reference_source=ReferenceSource.DEFAULT,
                scale=0.5,
                entity_name=entity_name,
                # NOTE: only 23 dofs are used for the GMT
                dofs_name=list(G1_UCSD_NO_WRIST_DOFS_NAME),
            ),
        )

    @classmethod
    def command_options(cls, motion_file: str, debug_vis: bool = False) -> dict[str, CommandTermOptions]:
        return dict(
            motion=GMTMotionLibCommand.configure(
                resampling_time_range=(60.0, 60.0),
                debug_vis=debug_vis,
                motion_file=motion_file,
            )
        )

    @classmethod
    def observation_options(cls, entity_name: str = "robot") -> dict[str, ObservationGroupOptions]:
        return dict(
            mimic_obs=ObservationGroupOptions(
                command=ObsTerm.configure(
                    func=observations.generated_commands,
                    params={"command_name": "motion"},
                ),
            ),
            obs_prop=ObservationGroupOptions(
                concatenate_terms=True,
                history_length=21,
                flatten_history_dim=True,
                backfill=False,
                ang_vel=ObsTerm.configure(
                    func=observations.base_ang_vel,
                    scale=0.25,
                    params={"entity_name": entity_name, "frame": "body"},
                ),
                rpy=ObsTerm.configure(
                    func=observations.base_rpy,
                    # NOTE: only roll and pitch are used for the GMT
                    params={"entity_name": entity_name, "terms_idx": [0, 1]},
                ),
                dofs_pos=observations.DofsPos.configure(scale=1.0, entity_name=entity_name, offset_from_default=True),
                dofs_vel=ObsTerm.configure(
                    func=dofs_vel_no_ankle_gmt,
                    scale=0.05,
                    params={"entity_name": entity_name},
                ),
                last_action=ObsTerm.configure(
                    func=observations.last_action,
                    scale=1.0,
                    params={"action_name": "dofs_pos_controller"},
                ),
                terms_order=["ang_vel", "rpy", "dofs_pos", "dofs_vel", "last_action"],
            ),
        )

    def step(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Get action from policy."""
        with torch.no_grad():
            obs_prop = obs["obs_prop"][..., -74:]
            obs_hist = obs["obs_prop"][..., :-74]
            raw_action = self._policy(torch.cat([obs["mimic_obs"], obs_prop, obs_hist], dim=1))
            raw_action = torch.clamp(raw_action, -10, 10)
        return dict(dofs_pos_controller=raw_action)
