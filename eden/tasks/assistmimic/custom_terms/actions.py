"""PHC's SMPL position action, re-expressed for Genesis' hinge joints."""

from __future__ import annotations

import math

import torch

from eden.managers.action_manager import ACTION_TERM_REGISTRY, ActionTerm
from eden.utils.geom import intrinsic_xyz_from_quat, intrinsic_xyz_nearest, quat_from_axis_angle
from eden.utils.string import term_subject


@ACTION_TERM_REGISTRY.register()
class PhcSphericalPDAction(ActionTerm):
    """PHC ``isaac_pd`` action on bodies of three stacked x/y/z hinges.

    PHC's PD target is ``pd_action_scale * a`` (``a`` clamped to ``[-1, 1]``; scale ``pi``, 5 for the knees' y),
    the exponential-map coordinate of Isaac Gym's spherical joint. Genesis keeps the MJCF's three hinges
    (intrinsic X-Y-Z Euler angles), so each body's target rotation vector is converted to Euler angles — of the
    two decompositions of the rotation and every whole-turn shift of them, the one nearest the current joint
    angles (:func:`~eden.utils.geom.intrinsic_xyz_nearest`). Otherwise a knee bending past 90 deg about y,
    which PHC's knee scale reaches at an action of 0.31, would see its x and z targets jump by ``pi``.

    A target may lie past a hinge's +-180 deg limit (a knee asked for 4 rad), as in PHC: the limit stops the joint
    and the PD keeps its drive, ``kp * (target - q)``, which is the authority PHC's knee scale of 5 buys. Clamping
    the target to the limit would cut that drive everywhere short of it.

    Around 90 deg itself the hinges are near gimbal lock: a knee held there would turn an x/z action of 0.006
    (0.02 rad) into x and z targets ``+-1.5`` rad apart — the same rotation, but hinges fighting each other at
    full gain. Within ``lock_band`` of the lock that ill-conditioned combination is held near the current
    angles instead, at a bounded cost in target rotation (see ``lock_band``).

    Parameters
    ----------
    entity_name : str
        The humanoid.
    dofs_name : list[str]
        Its DOFs, grouped per body as ``<body>_x, <body>_y, <body>_z``.
    scale : float
        PHC's ``pd_action_scale``.
    knee_y_scale : float
        PHC's bumped scale for ``L_Knee_y`` / ``R_Knee_y``.
    clip : float
        Raw-action clamp (rl_games' ``clip_actions``).
    lock_band : float
        ``|cos b|`` below which the target's ill-conditioned x/z combination is pulled towards the current
        angles (:func:`~eden.utils.geom.intrinsic_xyz_nearest`); the default 0.3 is within 17 deg of 90 deg and
        costs under 0.14 rad of target rotation there. 0: the exact decomposition.
    """

    entity_name: str = "robot"
    dofs_name: str | list[str] = []
    scale: float = math.pi
    knee_y_scale: float = 5.0
    clip: float = 1.0
    lock_band: float = 0.3

    @property
    def action_dim(self) -> int:
        return self._n_dofs

    def build(self) -> None:
        super().build()
        if not 0.0 <= self.lock_band < 1.0:
            raise ValueError(
                f"{term_subject(self)}: lock_band={self.lock_band} bounds |cos b|, so it must "
                "be in [0, 1); 0 keeps the exact hinge angles, the default 0.3 damps within 17 deg of gimbal lock."
            )
        names = list(self.dofs_name)
        grouped = self._n_dofs % 3 == 0 and all(
            names[i : i + 3] == [f"{names[i].rsplit('_', 1)[0]}_{axis}" for axis in "xyz"]
            for i in range(0, self._n_dofs, 3)
        )
        if not grouped:
            raise ValueError(
                f"{term_subject(self)}: dofs_name must list each body's x, y, z hinges in that order, "
                f"got {names[:6]}..."
            )
        self._scale_vec = torch.tensor(
            [self.knee_y_scale if n in ("L_Knee_y", "R_Knee_y") else self.scale for n in names], device=self.device
        )
        self._target = torch.zeros(self.num_envs, self._n_dofs, device=self.device)

    def compute(self, actions: torch.Tensor) -> None:
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions
        rotvec = (actions.clamp(-self.clip, self.clip) * self._scale_vec).view(self.num_envs, -1, 3)
        current = self.entity.get_dofs_pos(dofs_idx_local=self.dofs_idx_local).view(self.num_envs, -1, 3)
        euler = intrinsic_xyz_nearest(intrinsic_xyz_from_quat(quat_from_axis_angle(rotvec)), current, self.lock_band)
        self._target.copy_(euler.view(self.num_envs, -1))
        self._processed_action = self._target

    def apply_actions(self) -> None:
        self.entity.control_dofs_pos(self._target, dofs_idx_local=self.dofs_idx_local)

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        envs_idx = slice(None) if envs_idx is None else envs_idx
        self._prev_action[envs_idx] = 0.0
        self._raw_action[envs_idx] = 0.0
