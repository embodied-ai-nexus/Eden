"""HOMIE lower-body locomotion policy port (Unitree G1 29-DOF).

Wraps the frozen OpenHomie walk/stand tracker (:mod:`eden.extensions.policies.
homie.modules`) as an Eden :class:`~eden.extensions.policies.base.PolicyBase`
so it can be driven inside a Genesis environment.

HOMIE is command-driven: each step it consumes a 4-D command ``[vel_x, vel_y,
ang_vel_z, height_delta]`` (set via :meth:`HomiePolicy.set_command`, defaulting
to a zero / standing command) plus proprioception read directly from the
simulator, and emits 15 lower-body joint actions (12 leg + 3 waist). The 14 arm
DOFs are not controlled by HOMIE; during pure-locomotion inference they are
held at the HOMIE default pose (all zeros).

I/O contract
------------
Observation (built internally, 86 dim/step × 6 history = 516):

    7   command   [vel_x*2, vel_y*2, ang_vel_z*0.5, height, 0, 0, 0]
    3   base angular velocity (body frame) × 0.5
    3   projected gravity (body frame)
    29  dof_pos − HOMIE_DEFAULT_DOF_POS
    29  dof_vel × 0.05
    15  previous HOMIE actions

Action transform (per :meth:`HomieTracker.actions_to_targets`):

    lower body (0..14): target = HOMIE_DEFAULT[:15] + action * 0.25
    arms      (15..28): target = HOMIE_DEFAULT[15:29]  (held, = 0)

The 29-D absolute joint-position target is handed to Genesis's built-in PD via
an :class:`~eden.managers.terms.actions.joint_actions.ImplicitPDController`
configured with ``reference_source=ZERO`` and ``scale=1.0`` (no extra offset).

Checkpoints
-----------
The released ``model_walk.pt`` / ``model_stand.pt`` checkpoints are mirrored on
the ``Kashu7100/eden_policies`` hub and auto-download on first use (md5-pinned).
Override with local paths before instantiating::

    HomiePolicy.walk_model_path = "/path/to/model_walk.pt"
    HomiePolicy.stand_model_path = "/path/to/model_stand.pt"
    HomiePolicy.walk_md5_hash = HomiePolicy.stand_md5_hash = None  # custom weights

At least one of the two must be set; if both are set, the tracker switches
between them based on the velocity-command norm.

To train a HOMIE controller instead, use the ``homie`` task
(:mod:`eden.tasks.homie`, ``eden train --task homie``). Its policy drives the 12 legs only
over an 80-d frame, so its checkpoints are not drop-in replacements for the 15-action
release checkpoints this class loads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from genesis.utils.geom import inv_transform_by_quat

from eden import actions, observations
from eden.constants import ReferenceSource
from eden.extensions.policies.base import POLICY_REGISTRY, PolicyBase, PolicyOptions
from eden.extensions.policies.homie.modules import (
    HOMIE_ALL_PD_GAINS,
    HOMIE_DEFAULT_DOF_POS,
    HOMIE_NUM_ACTIONS,
    HomieConfig,
    HomieTracker,
)
from eden.options.managers.observations import ObservationGroupOptions
from eden.options.robots.unitree_g1 import G1_DOFS29_NAME, G1Dofs29

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.entities import RobotOptions
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.observations import ObservationGroupOptions as _OGO


_NUM_DOFS: int = 29


def homie_g1_robot(**overrides: Any) -> RobotOptions:
    """Build a ``G1Dofs29`` configured to match HOMIE's training setup.

    Overrides the default joint pose and PD gains with OpenHomie's values so
    the robot spawns in the HOMIE default pose (arms straight, elbows at 0) and
    the built-in PD reproduces the gains the tracker was trained against.

    Parameters
    ----------
    **overrides : Any
        Extra ``G1Dofs29`` fields to override (e.g. ``default_root_pos``).

    Returns
    -------
    RobotOptions
        A configured ``G1Dofs29`` instance.
    """
    default_dofs_pos = {name: HOMIE_DEFAULT_DOF_POS[i] for i, name in enumerate(G1_DOFS29_NAME)}
    default_dofs_kp = {name: HOMIE_ALL_PD_GAINS[i][0] for i, name in enumerate(G1_DOFS29_NAME)}
    default_dofs_kd = {name: HOMIE_ALL_PD_GAINS[i][1] for i, name in enumerate(G1_DOFS29_NAME)}
    return G1Dofs29(
        default_dofs_pos=default_dofs_pos,
        default_dofs_kp=default_dofs_kp,
        default_dofs_kd=default_dofs_kd,
        **overrides,
    )


@POLICY_REGISTRY.register()
class HomiePolicy(PolicyBase):
    """OpenHomie frozen walk/stand tracker for G1 lower-body locomotion.

    Reads proprioception directly from the env each step and drives the frozen
    tracker with the current 4-D command (see :meth:`set_command`). Only the
    Unitree G1 29-DOF robot is supported; ``G1Dofs29.dofs_name`` order matches
    HOMIE's joint order, so no DOF remapping is needed.
    """

    entity_name: str = "robot"

    # HOMIE uses two checkpoints (walk + stand). The base single-checkpoint
    # download is disabled (checkpoint_path stays None); the two paths below are
    # resolved through the same Hugging Face asset resolver in __init__. Override
    # with a local path or set to "" to disable a model (the tracker needs at least one).
    checkpoint_path: str | None = None
    walk_model_path: str = "homie/model_walk.pt"
    stand_model_path: str = "homie/model_stand.pt"
    # Pinned against the released bundle mirrored to Kashu7100/eden_policies.
    # Set to None when overriding with a custom (e.g. locally trained) checkpoint.
    walk_md5_hash: str | None = "556dda04493332fdaf83385c93218169"
    stand_md5_hash: str | None = "31edd14f666173d583218b5fac24e671"

    # Tracker behaviour (mirrors HomieConfig defaults).
    command_scale: float = 1.0
    walk_stand_threshold: float = 0.1
    clip_height_lower: float = 0.24
    clip_height_upper: float = 0.74

    def __init__(self, env: EnvBase, options: PolicyOptions) -> None:
        """Build the tracker, cache DOF indices, and allocate command buffers."""
        super().__init__(env, options)

        device = env.device
        self.device = device

        # Resolve the walk/stand checkpoints via the Hugging Face asset resolver
        # (downloads from Kashu7100/eden_policies on first use; passes through
        # existing local paths). Empty string disables that model.
        walk_path = self.fetch_asset(self.walk_model_path) if self.walk_model_path else ""
        stand_path = self.fetch_asset(self.stand_model_path) if self.stand_model_path else ""
        self._verify_md5(walk_path, self.walk_md5_hash) if walk_path else None
        self._verify_md5(stand_path, self.stand_md5_hash) if stand_path else None

        cfg = HomieConfig(
            walk_model_path=walk_path,
            stand_model_path=stand_path,
            command_scale=self.command_scale,
            walk_stand_threshold=self.walk_stand_threshold,
            clip_height_lower=self.clip_height_lower,
            clip_height_upper=self.clip_height_upper,
        )
        self._tracker = HomieTracker(config=cfg, num_envs=env.num_envs, device=device)

        robot = self.require_entity(self.entity_name)
        # All 29 DOFs in HOMIE / G1Dofs29 declaration order.
        _, dofs_idx = robot.find_named_dofs_idx_local(G1_DOFS29_NAME, preserve_order=True)
        self._dofs_idx = torch.as_tensor(dofs_idx, dtype=torch.long, device=device)

        # Arm joint targets are held at the HOMIE default pose (DOFs 15..28).
        self._arm_targets = (
            torch.tensor(HOMIE_DEFAULT_DOF_POS[HOMIE_NUM_ACTIONS:], dtype=torch.float32, device=device)
            .unsqueeze(0)
            .expand(env.num_envs, -1)
        )

        # Current 4-D command [vel_x, vel_y, ang_vel_z, height_delta]; settable
        # via set_command(). Defaults to zero → standing.
        self._commands = torch.zeros(env.num_envs, 4, device=device, dtype=torch.float32)

        # World-frame "down" vector, expanded to the env batch once.
        self._gravity_world = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32).expand(env.num_envs, 3)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def set_command(self, command: torch.Tensor) -> None:
        """Set the 4-D locomotion command for all (or a slice of) environments.

        Parameters
        ----------
        command : torch.Tensor
            Either ``(4,)`` (broadcast to all envs) or ``(num_envs, 4)``. Layout
            is ``[vel_x, vel_y, ang_vel_z, height_delta]`` where ``height_delta``
            is an offset about ``HOMIE_DEFAULT_HEIGHT`` (0 → 0.74 m).
        """
        command = torch.as_tensor(command, dtype=torch.float32, device=self.device)
        if command.ndim == 1:
            command = command.unsqueeze(0).expand(self._commands.shape[0], -1)
        self._commands[:] = command

    # ------------------------------------------------------------------
    # Env / action / observation specs
    # ------------------------------------------------------------------

    @classmethod
    def robot_options(cls) -> RobotOptions:
        """Return a ``G1Dofs29`` with HOMIE's default pose and PD gains baked in."""
        return homie_g1_robot()

    @classmethod
    def env_options(cls) -> dict[str, Any]:
        """Return env options for 50 Hz control (sim_dt=0.002, decimation=10)."""
        return dict(
            sim_dt=0.002,
            decimation=10,
        )

    @classmethod
    def action_options(cls, entity_name: str = "robot") -> dict[str, ActionTermOptions]:
        """Configure a 29-DOF implicit PD controller consuming absolute targets.

        :meth:`step` already produces absolute joint-position targets
        (default + scale * action for the lower body, default for the arms), so
        the controller uses ``reference_source=ZERO`` and ``scale=1.0``.
        """
        return dict(
            dofs_pos_controller=actions.ImplicitPDController.configure(
                entity_name=entity_name,
                dofs_name=G1_DOFS29_NAME,
                scale=1.0,
                reference_source=ReferenceSource.ZERO,
            ),
        )

    @classmethod
    def observation_options(cls, entity_name: str = "robot") -> dict[str, _OGO]:
        """Return a minimal placeholder observation group.

        HOMIE reads raw simulator state directly in :meth:`step`; Eden still
        requires at least one observation term to build the env.
        """
        return dict(
            policy=ObservationGroupOptions(
                concatenate_terms=True,
                enable_corruption=False,
                dofs_pos=observations.DofsPos.configure(entity_name=entity_name),
            ),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        """Reset the tracker history/prev-actions for the given environments.

        Parameters
        ----------
        envs_idx : slice | torch.Tensor | None, optional
            Environments to reset; ``None`` resets all.
        """
        self._tracker.reset(envs_idx)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def step(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Read proprioception, run the tracker, and return 29-DOF targets.

        Parameters
        ----------
        obs : dict
            Ignored — raw simulator state is read directly from the env.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{"dofs_pos_controller": (N, 29)}`` absolute joint-position targets.
        """
        # Read off the entity, not an observation group: this policy rides along as an extra deploy
        # slot on another policy's or task's env, whose groups are not its own (see the
        # "Declare the observation" paragraph of docs/guides/policy-porting.md).
        del obs
        robot = self.env.entities[self.entity_name]

        # Proprioception in HOMIE joint order (== G1Dofs29 order).
        dof_pos = robot.get_dofs_pos(dofs_idx_local=self._dofs_idx)
        dof_vel = robot.get_dofs_vel(dofs_idx_local=self._dofs_idx)

        # Body-frame base angular velocity and projected gravity (root frame).
        base_ang_vel = robot.get_ang(frame="body")
        base_quat = robot.get_quat()
        projected_gravity = inv_transform_by_quat(self._gravity_world, base_quat)

        homie_actions = self._tracker.step(
            raw_commands=self._commands,
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
        )  # (N, 15)

        lower_targets = self._tracker.actions_to_targets(homie_actions)  # (N, 15)
        targets = torch.cat([lower_targets, self._arm_targets], dim=-1)  # (N, 29)
        return {"dofs_pos_controller": targets}
