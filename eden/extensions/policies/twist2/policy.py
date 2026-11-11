"""TWIST2 motion-tracking policy port (Unitree G1 29-DOF).

Reproduces the deploy-time inference pipeline of
`twist2_mjlab <https://github.com/lzyang2000/twist2_mjlab>`_
(``deploy/policy/twist2_policy.py``) inside an Eden Genesis environment so the
released ONNX actor can be driven for sim evaluation.

The kinematic reference is read through Eden's standard beyondmimic
:class:`~eden.managers.terms.commands.mimic.MotionCommand`
with ``anchor_link_name="pelvis"`` — the same path the SONIC port uses — so
TWIST2 shares the one motion loader / command term rather than carrying its own
PKL library. Convert a TWIST2 enriched PKL to the Eden ``.npz`` format with
``examples/policies/twist2/scripts/pkl_to_npz.py`` first.

I/O contract (frozen by the exported ``pretrained.onnx``): the env's ``policy``
observation group is the 11-frame history of one 127-dim frame, and the actor
reads ``current ++ history``:

    actor obs (1524) = current(127) ++ history(11 × 127, oldest first, newest = current)
        current(127) = mimic(35) ++ proprio(92)
        mimic(35)    root_vel_xy_b(2) root_z(1) root_roll_pitch(2)
                     root_yaw_ang_vel_b(1) joint_pos(29)   ← from MotionCommand
        proprio(92)  base_ang_vel(3, ×0.25) imu_roll_pitch(2)
                     joint_pos_rel(29) joint_vel_scaled(29) last_action(29)
    actor out → action (29, MJCF order)

Action transform (matches ``twist2_policy.py``):

    target[i] = DEFAULT_POS[i] + JOINT_SCALES[i] * action[i]

applied by :class:`~eden.actions.ImplicitPDController` (per-joint ``scale``,
``reference_source=DEFAULT``), so the env's ``last_action`` is the actor's raw
output. The robot's stock ``G1Dofs29`` PD gains equal TWIST2's mjlab training
gains; only the default pose is overridden to mjlab's KNEES_BENT keyframe.

TWIST2's joint order is identical to Eden's ``G1Dofs29`` MJCF order, so there is
no IsaacLab permutation. The mimic observation is heading/position-invariant
(body-frame velocities, height, roll/pitch, yaw-rate), so no heading alignment
is needed. ``num_envs > 1`` is supported (the obs/history are batched torch);
the released checkpoint's deploy contract was validated at ``num_envs=1``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from eden import actions, observations
from eden.constants import ReferenceSource
from eden.extensions.policies.base import POLICY_REGISTRY, PolicyBase, PolicyOptions
from eden.extensions.policies.twist2.constants import (
    ACTOR_CURRENT_DIM,
    ACTOR_HISTORY_LENGTH,
    BASE_ANG_VEL_SCALE,
    DEFAULT_POS,
    JOINT_SCALES,
    JOINT_VEL_SCALES,
    KNEES_BENT_DOFS_POS,
    POLICY_JOINT_NAMES,
)
from eden.extensions.policies.twist2.custom_terms.observations import (
    twist2_imu_roll_pitch,
    twist2_motion_dofs_pos,
    twist2_motion_root,
)
from eden.extensions.policies.twist2.onnx_wrapper import Twist2Actor
from eden.managers.terms.commands.mimic import MotionCommand
from eden.options.managers.observations import ObservationGroupOptions
from eden.options.robots.unitree_g1 import G1Dofs29
from eden.terms import ObsTerm

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.commands import CommandTermOptions

#: The env observation group the actor reads.
OBS_GROUP = "policy"
_ANCHOR_BODY: str = "pelvis"
# MotionCommand only needs the anchor (pelvis); the mimic reads root state from
# body index 0 and joint pos from the motion's dof channel directly.
_MOTION_BODY_NAMES: list[str] = [_ANCHOR_BODY]


@POLICY_REGISTRY.register()
class Twist2Policy(PolicyBase):
    """TWIST2 whole-body motion-tracking controller (G1 29-DOF, ONNX backend)."""

    # step() reads the observation dict the env publishes, so this policy owns the deploy env.
    reads_env_observations = True

    entity_name: str = "robot"

    # Resolved by PolicyBase (``fetch_asset``) — downloaded from the
    # ``Kashu7100/eden_policies`` HF dataset on first use. Hash pinned against
    # the uploaded actor.
    checkpoint_path: str = "twist2/pretrained.onnx"
    md5_hash: str | None = "a045b72bd49ce0993aa145f8958a5564"

    def __init__(self, env: EnvBase, options: PolicyOptions) -> None:
        super().__init__(env, options)

        self.device = env.device
        robot = self.require_entity(self.entity_name)
        # The joint observations and the action term read the robot's DOFs in its own order.
        if list(robot.dofs_name) != list(POLICY_JOINT_NAMES):
            raise ValueError(
                f"Twist2Policy drives the G1's 29 DOFs in {list(POLICY_JOINT_NAMES)} order; this robot's "
                f"dofs_name is {robot.dofs_name}. Build the robot from Twist2Policy.robot_options()."
            )

        # The built robot must carry the knees-bent keyframe in every env: the joint-position
        # observation and the action term are both relative to its default pose.
        expected_default = torch.tensor(DEFAULT_POS, device=env.device)
        built_default = robot.get_default_dofs_pos(POLICY_JOINT_NAMES).to(env.device)  # (num_envs, NUM_JOINTS)
        # ``allclose``'s tolerance, per env: the first env whose row is off is the one named.
        mismatched = ~torch.isclose(built_default, expected_default, atol=1e-6).all(dim=1)
        if mismatched.any().item():
            env_idx = int(mismatched.nonzero()[0].item())
            raise ValueError(
                f"Robot default_dofs_pos does not match TWIST2's knees-bent keyframe in env {env_idx}: "
                f"built={built_default[env_idx].tolist()} expected={expected_default.tolist()}. "
                "Build the robot from Twist2Policy.robot_options()."
            )

        # The env has to be the one this policy declares: a task env publishes its own
        # ``policy`` group, which the actor would read as garbage.
        expected_dim = ACTOR_HISTORY_LENGTH * ACTOR_CURRENT_DIM
        group_dims = env.observation_manager.group_obs_dim
        if group_dims.get(OBS_GROUP) != (expected_dim,):
            raise ValueError(
                f"Twist2Policy reads a {expected_dim}-dim '{OBS_GROUP}' observation group, but this env "
                f"publishes {group_dims}. Build the env from Twist2Policy.observation_options() (a task-less "
                f"`eden inference --policy twist2` does), not from another task's config."
            )

        self._actor = Twist2Actor(self.checkpoint_path, device=str(env.device))

    # ------------------------------------------------------------------
    # Env / action / command / observation specs
    # ------------------------------------------------------------------

    @classmethod
    def robot_options(cls) -> G1Dofs29:
        """``G1Dofs29`` with the knees-bent default pose TWIST2 trains against.

        The stock gains / action scale already match mjlab; only the default
        joint pose differs (Eden ships a home-ish pose), so we override it.
        """
        return G1Dofs29(default_dofs_pos=dict(KNEES_BENT_DOFS_POS), default_root_pos=(0.0, 0.0, 0.76))

    @classmethod
    def env_options(cls) -> dict[str, Any]:
        # 50 Hz control (decimation 2 × sim_dt 0.01) to match the deploy stack.
        return dict(
            sim_dt=0.01,
            sim_substeps=2,
            decimation=2,
            iterations=20,
            ls_iterations=20,
            max_collision_pairs=20,
            batch_dofs_info=False,
            use_gjk_collision=False,
            enable_self_collision=False,
            enable_multi_contact=False,
        )

    @classmethod
    def action_options(cls, entity_name: str = "robot") -> dict[str, ActionTermOptions]:
        """The actor's raw (normalized) action, scaled per joint onto the knees-bent default pose."""
        return dict(
            dofs_pos_controller=actions.ImplicitPDController.configure(
                entity_name=entity_name,
                dofs_name=list(POLICY_JOINT_NAMES),
                scale=dict(zip(POLICY_JOINT_NAMES, JOINT_SCALES)),
                reference_source=ReferenceSource.DEFAULT,
            ),
        )

    @classmethod
    def command_options(
        cls,
        motion_file: str = "motion.npz",
        debug_vis: bool = False,
        entity_name: str = "robot",
        ghost_entity_name: str = "",
        sampling_mode: str = "start",
    ) -> dict[str, CommandTermOptions]:
        return dict(
            motion=MotionCommand.configure(
                resampling_time_range=(1.0e9, 1.0e9),
                debug_vis=debug_vis,
                entity_name=entity_name,
                ghost_entity_name=ghost_entity_name,
                motion_file=motion_file,
                anchor_link_name=_ANCHOR_BODY,
                links_name=_MOTION_BODY_NAMES,
                pose_range={},
                velocity_range={},
                dofs_position_range=(0.0, 0.0),
                sampling_mode=sampling_mode,
            ),
        )

    @classmethod
    def observation_options(cls, entity_name: str = "robot") -> dict[str, ObservationGroupOptions]:
        """One 127-dim frame, mimic then proprio, stacked into an 11-frame history (backfilled at reset)."""
        entity = {"entity_name": entity_name}
        return {
            OBS_GROUP: ObservationGroupOptions(
                enable_corruption=False,
                history_length=ACTOR_HISTORY_LENGTH,
                flatten_history_dim=True,
                # mimic (35)
                motion_root=ObsTerm.configure(func=twist2_motion_root),
                motion_dofs_pos=ObsTerm.configure(func=twist2_motion_dofs_pos),
                # proprio (92)
                base_ang_vel=ObsTerm.configure(
                    func=observations.base_ang_vel, params={**entity, "frame": "body"}, scale=BASE_ANG_VEL_SCALE
                ),
                imu_roll_pitch=ObsTerm.configure(func=twist2_imu_roll_pitch, params=entity),
                dofs_pos=observations.DofsPos.configure(entity_name=entity_name, offset_from_default=True),
                dofs_vel=observations.DofsVel.configure(entity_name=entity_name, scale=list(JOINT_VEL_SCALES)),
                last_action=ObsTerm.configure(func=observations.last_action),
            )
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def step(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        history = obs[OBS_GROUP]  # (N, 11 * 127), oldest first, so the newest frame is the current one
        actor_obs = torch.cat([history[:, -ACTOR_CURRENT_DIM:], history], dim=-1)
        return {"dofs_pos_controller": self._actor(actor_obs)}
