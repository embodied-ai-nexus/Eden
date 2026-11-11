"""Eden ``RetargetingBase`` wrapper around the dex-retargeting port.

Combines one of three SLSQP optimizers (position / vector / dexpilot) with
the upstream ``SeqRetargeting`` semantics (mean-of-limits warm start, output
low-pass filter) into a single ``DexRetargeter`` entity-extension.

Typical usage:

.. code-block:: python

    retargeter = DexRetargeter(
        env=env,
        options=DexRetargeter.configure(
            entity_name="robot",
            type="vector",
            target_origin_link_names=["wrist", "wrist", "wrist", "wrist"],
            target_task_link_names=["index_tip", "middle_tip", "ring_tip", "pinky_tip"],
            target_link_human_indices=np.array([[0, 0, 0, 0], [8, 12, 16, 20]]),
            scaling_factor=1.6,
        ),
    )
    qpos = retargeter.retarget(ref_value)              # one frame
    qpos_seq = retargeter.retarget_motion(ref_seq)     # batch
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import torch

from eden.extensions.retargeting import RetargetingBase
from eden.extensions.retargeting.dex_retargeter.constants import OPERATOR2MANO, HandType
from eden.extensions.retargeting.dex_retargeter.optimizer import (
    DexPilotOptimizer,
    Optimizer,
    PositionOptimizer,
    VectorOptimizer,
)
from eden.extensions.retargeting.dex_retargeter.utils import LPFilter
from eden.options.extensions.retargeting import RetargeterOptions

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


class DexRetargeter(RetargetingBase):
    """Port of ``dex-retargeting`` backed by Genesis kinematics.

    Parameters
    ----------
    type : {"position", "vector", "dexpilot"}
        Retargeting algorithm to instantiate.
    target_joint_names : list[str] or None
        Subset of ``entity.dofs_name`` to optimize. ``None`` = all DOFs.
    target_link_human_indices : np.ndarray or None
        Indices into the human hand keypoint array.
        - ``position``: shape ``(n,)``  — one keypoint per target link.
        - ``vector`` : shape ``(2, n)`` — origin indices on row 0, task on row 1.
        - ``dexpilot``: leave ``None``; defaults are derived from the fingertip count.
    target_link_names : list[str] or None
        Position-retargeting only. Robot links whose world positions match
        the human keypoints selected by ``target_link_human_indices``.
    target_origin_link_names, target_task_link_names : list[str] or None
        Vector-retargeting only. Each vector goes from origin to task.
    finger_tip_link_names : list[str] or None
        DexPilot only. Five (or fewer) fingertip links.
    wrist_link_name : str or None
        DexPilot only. The wrist / palm reference link.
    scaling_factor : float
        Human-to-robot scale (applies to ``vector`` and ``dexpilot``).
    huber_delta, normal_delta : float
        Huber loss beta and quadratic regularization weight on ``(x - last_qpos)``.
    project_dist, escape_dist : float
        DexPilot fingertip projection thresholds.
    has_joint_limits : bool
        Pass entity DOF limits to ``nlopt`` as bound constraints.
    low_pass_alpha : float
        Output smoothing in ``[0, 1]``. 1 = passthrough, 0 = freeze; a value outside ``[0, 1]`` disables it.
    """

    # --- algorithm selection ---
    type: Literal["position", "vector", "dexpilot"] = "vector"
    target_joint_names: Optional[list[str]] = None
    target_link_human_indices: Optional[np.ndarray] = None

    # --- link-name fields (which ones apply depends on `type`) ---
    target_link_names: Optional[list[str]] = None
    target_origin_link_names: Optional[list[str]] = None
    target_task_link_names: Optional[list[str]] = None
    finger_tip_link_names: Optional[list[str]] = None
    wrist_link_name: Optional[str] = None

    # --- optimizer hyper-parameters ---
    scaling_factor: float = 1.0
    huber_delta: float = 2e-2
    normal_delta: float = 4e-3
    project_dist: float = 0.03
    escape_dist: float = 0.05

    # --- runtime knobs ---
    has_joint_limits: bool = True
    low_pass_alpha: float = 0.1

    def __init__(self, env: "EnvBase", options: RetargeterOptions) -> None:
        super().__init__(env, options)

        self.optimizer: Optimizer = self._build_optimizer()

        # Joint-limit bounds for the target DOFs. Either pull them from the
        # entity (and forward to nlopt) or use a wide band that effectively
        # disables the constraint.
        if self.has_joint_limits:
            # entity.get_dofs_limit() returns (lower, upper), each (n_envs, n_dofs).
            # Take env 0 and stack to (n_dofs, 2).
            entity_limits = torch.stack(self.entity.get_dofs_limit(), dim=-1)[0].cpu().numpy()
            self.joint_limits = entity_limits[self.optimizer.idx_target]
            self.optimizer.set_joint_limit(self.joint_limits)
        else:
            self.joint_limits = np.tile([-1e4, 1e4], (self.optimizer.opt_dof, 1)).astype(np.float64)

        self.last_qpos = self.joint_limits.mean(axis=1).astype(np.float32)

        self.filter: Optional[LPFilter] = LPFilter(self.low_pass_alpha) if 0 <= self.low_pass_alpha <= 1 else None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_optimizer(self) -> Optimizer:
        joint_names = (
            list(self.target_joint_names) if self.target_joint_names is not None else list(self.entity.dofs_name)
        )
        builders = {
            "position": self._build_position_optimizer,
            "vector": self._build_vector_optimizer,
            "dexpilot": self._build_dexpilot_optimizer,
        }
        if self.type not in builders:
            raise ValueError(f"Unknown retargeting type: {self.type!r}. Choose from {list(builders)}.")
        return builders[self.type](joint_names)

    def _build_position_optimizer(self, joint_names: list[str]) -> PositionOptimizer:
        if self.target_link_names is None or self.target_link_human_indices is None:
            raise ValueError("Position retargeting requires target_link_names and target_link_human_indices.")
        human_idx = np.asarray(self.target_link_human_indices).squeeze()
        if human_idx.shape != (len(self.target_link_names),):
            raise ValueError(
                f"target_link_human_indices shape {human_idx.shape} does not match "
                f"{len(self.target_link_names)} target links."
            )
        return PositionOptimizer(
            self.entity,
            joint_names,
            target_link_names=list(self.target_link_names),
            target_link_human_indices=human_idx,
            huber_delta=self.huber_delta,
            norm_delta=self.normal_delta,
        )

    def _build_vector_optimizer(self, joint_names: list[str]) -> VectorOptimizer:
        if (
            self.target_origin_link_names is None
            or self.target_task_link_names is None
            or self.target_link_human_indices is None
        ):
            raise ValueError(
                "Vector retargeting requires target_origin_link_names, "
                "target_task_link_names, and target_link_human_indices."
            )
        if len(self.target_origin_link_names) != len(self.target_task_link_names):
            raise ValueError("Vector origin/task link counts differ.")
        human_idx = np.asarray(self.target_link_human_indices)
        if human_idx.shape != (2, len(self.target_origin_link_names)):
            raise ValueError(
                f"target_link_human_indices shape {human_idx.shape} should be "
                f"(2, {len(self.target_origin_link_names)})."
            )
        return VectorOptimizer(
            self.entity,
            joint_names,
            target_origin_link_names=list(self.target_origin_link_names),
            target_task_link_names=list(self.target_task_link_names),
            target_link_human_indices=human_idx,
            huber_delta=self.huber_delta,
            norm_delta=self.normal_delta,
            scaling=self.scaling_factor,
        )

    def _build_dexpilot_optimizer(self, joint_names: list[str]) -> DexPilotOptimizer:
        if self.finger_tip_link_names is None or self.wrist_link_name is None:
            raise ValueError("DexPilot retargeting requires finger_tip_link_names and wrist_link_name.")
        human_idx = np.asarray(self.target_link_human_indices) if self.target_link_human_indices is not None else None
        return DexPilotOptimizer(
            self.entity,
            joint_names,
            finger_tip_link_names=list(self.finger_tip_link_names),
            wrist_link_name=self.wrist_link_name,
            target_link_human_indices=human_idx,
            huber_delta=self.huber_delta,
            norm_delta=self.normal_delta,
            project_dist=self.project_dist,
            escape_dist=self.escape_dist,
            scaling=self.scaling_factor,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def joint_names(self) -> list[str]:
        """Full DOF name list of the underlying entity (``dofs_name`` order)."""
        return list(self.entity.dofs_name)

    @property
    def retargeting_type(self) -> str:
        return self.optimizer.retargeting_type

    def retarget(
        self,
        ref_value: np.ndarray,
        fixed_qpos: np.ndarray | None = None,
    ) -> np.ndarray:
        """Retarget one frame and return a full-DOF qpos (in ``dofs_name`` order).

        ``ref_value`` is the optimizer-specific Cartesian reference: positions
        ``(n, 3)`` for ``position``; vectors ``(n, 3)`` for ``vector`` and
        ``dexpilot``. ``fixed_qpos`` holds the non-target joints (``optimizer.idx_fixed`` order;
        zeros when ``None``).
        """
        if fixed_qpos is None:
            fixed_qpos = np.zeros(len(self.optimizer.idx_fixed), dtype=np.float32)

        qpos = self.optimizer.retarget(
            ref_value=np.asarray(ref_value, dtype=np.float32),
            fixed_qpos=np.asarray(fixed_qpos, dtype=np.float32),
            last_qpos=np.clip(self.last_qpos, self.joint_limits[:, 0], self.joint_limits[:, 1]),
        )
        self.last_qpos = qpos

        robot_qpos = np.zeros(self.optimizer.num_joints, dtype=np.float32)
        robot_qpos[self.optimizer.idx_fixed] = fixed_qpos
        robot_qpos[self.optimizer.idx_target] = qpos

        if self.filter is not None:
            robot_qpos = self.filter.next(robot_qpos)
        return robot_qpos

    def retarget_motion(
        self,
        ref_sequence: np.ndarray,
        fixed_qpos_sequence: np.ndarray | None = None,
    ) -> np.ndarray:
        """Retarget a whole trajectory.

        Parameters
        ----------
        ref_sequence : np.ndarray
            ``(T, ...)`` — per-frame ``ref_value`` arrays.
        fixed_qpos_sequence : np.ndarray, optional
            ``(T, len(idx_fixed))`` if any joints are held fixed; otherwise
            ``None``.

        Returns
        -------
        qpos : np.ndarray, shape ``(T, n_dofs)``
            Full-DOF qpos per frame in ``entity.dofs_name`` order.
        """
        return np.stack(
            [
                self.retarget(
                    ref_sequence[t],
                    fixed_qpos=fixed_qpos_sequence[t] if fixed_qpos_sequence is not None else None,
                )
                for t in range(len(ref_sequence))
            ]
        )

    def reset(self) -> None:
        """Forget previous-frame state (``last_qpos``, filter)."""
        self.last_qpos = self.joint_limits.mean(axis=1).astype(np.float32)
        if self.filter is not None:
            self.filter.reset()

    def set_qpos(self, robot_qpos: np.ndarray) -> None:
        """Manually seed ``last_qpos`` (e.g. from the simulator's current state)."""
        self.last_qpos = np.asarray(robot_qpos, dtype=np.float32)[self.optimizer.idx_target]

    def get_qpos(self, fixed_qpos: np.ndarray | None = None) -> np.ndarray:
        robot_qpos = np.zeros(self.optimizer.num_joints, dtype=np.float32)
        robot_qpos[self.optimizer.idx_target] = self.last_qpos
        if fixed_qpos is not None:
            robot_qpos[self.optimizer.idx_fixed] = fixed_qpos
        return robot_qpos

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @staticmethod
    def extract_ref_value(
        joint_pos: np.ndarray,
        target_link_human_indices: np.ndarray,
        retargeting_type: str,
    ) -> np.ndarray:
        """Build the optimizer's ``ref_value`` from a 21-keypoint MANO array.

        Same logic as upstream's ``detect_from_video.py`` snippet so existing
        MediaPipe pipelines drop in cleanly.
        """
        if retargeting_type == "POSITION":
            return joint_pos[target_link_human_indices, :]
        origin = target_link_human_indices[0, :]
        task = target_link_human_indices[1, :]
        return joint_pos[task, :] - joint_pos[origin, :]


def operator2mano(hand_type: HandType) -> np.ndarray:
    """Return the rotation that maps the operator frame to MANO."""
    return OPERATOR2MANO[hand_type]
