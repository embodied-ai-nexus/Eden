"""OmniXtreme whole-body motion-tracking policy."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, ClassVar, Dict

import torch

import eden as en
from eden.terms import ObsTerm
from eden import actions, observations
from eden.options.actuators import Unitree7520_22, ActuatorSpecOptions
from eden.extensions.policies.base import PolicyBase, PolicyOptions, POLICY_REGISTRY
from eden.managers.modifiers.actions.actuators import EnvelopeClip
from eden.extensions.policies.omnixtreme.custom_terms.commands import OmniXtremeRefCommand
from eden.extensions.policies.omnixtreme.custom_terms.observations import (
    omnixtreme_prev_base_action,
    omnixtreme_anchor_ori_6d,
)
from eden.extensions.policies.omnixtreme.onnx_wrapper import (
    OnnxBasePolicyWrapper,
    OnnxResidualPolicyWrapper,
    _make_fk_onnx_session,
)
from eden.options.managers.observations import ObservationGroupOptions
from eden.constants import ReferenceSource

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.commands import CommandTermOptions

NUM_DOFS = 29

# Per-joint action scales (URDF order). OmniXtreme overrides hip pitch to use
# the 7520-22.5 motor (vs the 7520-14.3 used in stock G1Dofs29), so the scale
# for that joint is recomputed from the 7520-22 actuator spec.
ACTION_SCALE = {
    **en.robots.G1_ACTION_SCALE,
    "*_hip_pitch_joint": Unitree7520_22.RAIBERT_HEURISTIC_ACTION_SCALE(),
}


# Stock no-hands G1 defaults, built once and reused to override only the hip pitch below.
_G1_NO_HANDS = en.robots.G1Dofs29NoHands()


class OmniXtreme_G1(en.robots.G1Dofs29NoHands):
    """G1Dofs29NoHands configured to match OmniXtreme's training setup.

    OmniXtreme uses 7520-22.5 motors for the hip pitch actuators (vs the
    7520-14.3 in the stock G1, and the original BeyondMimic paper); the default
    pose, ``dofs_spec``, PD gains, and armature for the hip pitch are derived
    from that actuator spec.

    Kept as a subclass (not a ``*_robot()`` factory like the other policies)
    because ``dofs_spec`` is a ``ClassVar`` and can't be set via the constructor.
    """

    default_dofs_pos: dict[str, float] = {
        "left_hip_pitch_joint": -0.3120,
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.6690,
        "left_ankle_pitch_joint": -0.3630,
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.3120,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.6690,
        "right_ankle_pitch_joint": -0.3630,
        "right_ankle_roll_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,
        "left_shoulder_pitch_joint": 0.2,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.6,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.6,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }
    dofs_spec: ClassVar[Dict[str, ActuatorSpecOptions]] = {
        **en.robots.G1Dofs29NoHands.dofs_spec,
        "*_hip_pitch_joint": Unitree7520_22,
    }
    default_dofs_kp: dict[str, float] = {
        **_G1_NO_HANDS.default_dofs_kp,
        "*_hip_pitch_joint": Unitree7520_22.RAIBERT_HEURISTIC_KP(),
    }
    default_dofs_kd: dict[str, float] = {
        **_G1_NO_HANDS.default_dofs_kd,
        "*_hip_pitch_joint": Unitree7520_22.RAIBERT_HEURISTIC_KD(),
    }
    default_dofs_armature: dict[str, float] = {
        **_G1_NO_HANDS.default_dofs_armature,
        "*_hip_pitch_joint": Unitree7520_22.ARMATURE(),
    }
    default_dofs_force_limits: dict[str, float] = {
        **_G1_NO_HANDS.default_dofs_force_limits,
        "*_hip_pitch_joint": Unitree7520_22.EFFORT_LIMIT,
    }


@POLICY_REGISTRY.register()
class OmniXtremePolicy(PolicyBase):
    """OmniXtreme: Dual-policy humanoid motion tracking (flow-matching base + residual correction)."""

    # step() reads the observation dict the env publishes, so this policy owns the deploy env.
    reads_env_observations = True

    entity_name: str = "robot"

    checkpoint_path: str = "omnixtreme/base_policy.onnx"
    # All three hashes are ``str | None`` like every sibling: pointing a checkpoint at
    # local weights means clearing its pinned hash, or ``_verify_md5`` rejects them.
    md5_hash: str | None = "67b75756640bbb52720ee5920bbf0910"

    residual_checkpoint_path: str = "omnixtreme/residual_policy.onnx"
    residual_md5_hash: str | None = "7ad3515080d32e51241f959984c0b568"

    fk_checkpoint_path: str = "omnixtreme/fk.onnx"
    fk_md5_hash: str | None = "94c9896adda6939b0ddfa508ddc0d0be"

    def __init__(self, env: EnvBase, options: PolicyOptions):
        super().__init__(env, options)

        self._residual_path = self.fetch_asset(self.residual_checkpoint_path)
        self._fk_path = self.fetch_asset(self.fk_checkpoint_path)

        self._verify_md5(self._residual_path, self.residual_md5_hash)
        self._verify_md5(self._fk_path, self.fk_md5_hash)

        device_str = str(env.device)
        self._base_policy = OnnxBasePolicyWrapper(self.checkpoint_path, device=device_str)
        self._residual_policy = OnnxResidualPolicyWrapper(self._residual_path, device=device_str)

        # The anchor observation runs FK through the command term (see omnixtreme_anchor_ori_6d).
        cmd_term = env.command_manager.get_term("omnixtreme_motion")
        cmd_term.fk_session = _make_fk_onnx_session(self._fk_path)
        cmd_term._fk_onnx_path = self._fk_path

        # Residual gain, overridable with the RESIDUAL_GAIN environment variable.
        self._residual_gain = float(os.environ.get("RESIDUAL_GAIN", "1.0"))

        self.action_term = env.action_manager.get_term("dofs_pos_controller")

        # The residual policy reads and writes joints in BeyondMimic (Isaac Lab) order, not the
        # URDF order of the env's DOFs: ``x[:, _inv_perm]`` maps URDF -> BeyondMimic, ``x[:, _perm]`` back.
        # fmt: off
        perm_list = [
            0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28,
        ]
        inv_perm_list = [
            0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28,
        ]
        # fmt: on
        self._perm = torch.tensor(perm_list, dtype=torch.long, device=env.device)
        self._inv_perm = torch.tensor(inv_perm_list, dtype=torch.long, device=env.device)

    @classmethod
    def robot_options(cls) -> OmniXtreme_G1:
        """Return the OmniXtreme G1 (7520-22.5 hip-pitch motors, training default pose)."""
        return OmniXtreme_G1()

    @classmethod
    def env_options(cls) -> dict[str, Any]:
        return dict(sim_dt=0.004, decimation=5)

    @classmethod
    def action_options(cls, entity_name: str = "robot") -> dict[str, ActionTermOptions]:
        return dict(
            dofs_pos_controller=actions.ImplicitPDController.configure(
                reference_source=ReferenceSource.DEFAULT,
                scale=ACTION_SCALE,
                entity_name=entity_name,
                dofs_name=en.robots.G1Dofs29NoHands().dofs_name,
                modifier=EnvelopeClip.configure(),
            ),
        )

    @classmethod
    def command_options(cls, motion_file: str, debug_vis: bool = False) -> dict[str, CommandTermOptions]:
        return dict(
            omnixtreme_motion=OmniXtremeRefCommand.configure(
                resampling_time_range=(60.0, 60.0),
                motion_file=motion_file,
                debug_vis=debug_vis,
            )
        )

    @classmethod
    def observation_options(cls, entity_name: str = "robot") -> dict[str, ObservationGroupOptions]:
        return dict(
            obs_prop=ObservationGroupOptions(
                concatenate_terms=True,
                history_length=15,
                flatten_history_dim=True,
                backfill=False,
                dofs_pos=observations.DofsPos.configure(scale=1.0, entity_name=entity_name, offset_from_default=True),
                dofs_vel=observations.DofsVel.configure(scale=1.0, entity_name=entity_name),
                ang_vel=ObsTerm.configure(
                    func=observations.base_ang_vel,
                    scale=1.0,
                    params={"entity_name": entity_name, "frame": "body"},
                ),
                prev_base_action=ObsTerm.configure(
                    func=omnixtreme_prev_base_action,
                    scale=1.0,
                    params={"command_name": "omnixtreme_motion"},
                ),
                terms_order=["dofs_pos", "dofs_vel", "ang_vel", "prev_base_action"],
            ),
            command_obs=ObservationGroupOptions(
                concatenate_terms=True,
                motion=ObsTerm.configure(
                    func=observations.generated_commands,
                    params={"command_name": "omnixtreme_motion"},
                ),
                anchor=ObsTerm.configure(
                    func=omnixtreme_anchor_ori_6d,
                    params={"entity_name": entity_name, "command_name": "omnixtreme_motion"},
                ),
                terms_order=["motion", "anchor"],
            ),
        )

    def step(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            # --- Extract observations ---
            obs_prop_flat = obs["obs_prop"]  # (B, 1350) = 15 * 90
            command_obs = obs["command_obs"]  # (B, 64)
            B = obs_prop_flat.shape[0]
            dim_per_frame = NUM_DOFS * 3 + 3  # dof_pos + dof_vel + ang_vel + prev_action = 90

            real_obs = obs_prop_flat[..., -dim_per_frame:]
            real_hist = obs_prop_flat.reshape(B, 15, dim_per_frame)

            # --- Run base policy ---
            obs_dict = {
                "real_obs": real_obs,
                "command_obs": command_obs,
                "real_historical_obs_raw": real_hist,
                "self_obs": torch.zeros(B, 1, device=self.env.device),
                "initial_noise": torch.randn(B, NUM_DOFS, device=self.env.device, dtype=torch.float32),
            }
            a_base = self._base_policy(obs_dict)

            # Store base action for next-step observation
            cmd = self.env.command_manager.get_term("omnixtreme_motion")
            cmd.prev_base_action[:] = a_base

            # --- Build residual observation (183 dims) ---
            dof_pos_rel = real_obs[..., :NUM_DOFS]
            dof_vel = real_obs[..., NUM_DOFS : NUM_DOFS * 2]
            ang_vel = real_obs[..., NUM_DOFS * 2 : NUM_DOFS * 2 + 3]

            term = self.action_term
            unscaled = (term.processed_action - term.effective_offset) / term.effective_scale
            prev_action_unscaled = unscaled[:, self._inv_perm]

            # Reorder to proto (BeyondMimic) order for residual policy
            real_reordered = torch.cat(
                [
                    ang_vel,
                    dof_pos_rel[:, self._inv_perm],
                    dof_vel[:, self._inv_perm],
                    prev_action_unscaled,
                ],
                dim=-1,
            )

            ref_pos = command_obs[..., :NUM_DOFS]
            ref_vel = command_obs[..., NUM_DOFS : NUM_DOFS * 2]
            anchor6d = command_obs[..., NUM_DOFS * 2 :]
            cmd_reordered = torch.cat(
                [
                    ref_pos[:, self._inv_perm],
                    ref_vel[:, self._inv_perm],
                    anchor6d,
                ],
                dim=-1,
            )

            residual_obs = torch.cat([cmd_reordered, real_reordered, a_base[:, self._inv_perm]], dim=-1)

            # --- Run residual policy and combine ---
            a_res = self._residual_policy(residual_obs)[:, self._perm]  # reindex to URDF
            raw_actions = torch.clamp(a_base + self._residual_gain * a_res, -40.0, 40.0)

        return dict(dofs_pos_controller=raw_actions)
