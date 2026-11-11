"""SLSQP-based retargeting optimizers, Genesis-backed.

Ported from ``dex_retargeting.optimizer`` (AnyTeleop / DexPilot). The
algorithm — Huber loss over keypoint positions or relative vectors, optimized
with ``nlopt``'s SLSQP and analytic gradients built from positional
Jacobians — is unchanged. Only the kinematics layer differs:

- Forward kinematics is performed by calling ``entity.set_qpos(qpos)``.
- Jacobians come from ``entity.get_jacobian(link)`` and are already in world
  frame (Genesis returns the spatial Jacobian), so the upstream
  ``link_rot @ body_jacobian`` step is dropped.
- Column ordering follows ``entity.dofs_name`` (Eden reorders internally).

Free-joint detection and the ``MimicJointKinematicAdaptor`` from upstream
are intentionally omitted; we expect hands with no base DOFs (fixed-base or
welded onto an arm) with mimic constraints already baked into the URDF.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, List, Optional

import nlopt
import numpy as np
import torch

import eden as en

if TYPE_CHECKING:
    from eden.entities.rigid import RigidEntity


# DexPilot's secondary projection gate: when two non-thumb fingertips are
# already paired with the thumb, also clip *their* pairwise distance to this
# value. Independent of ``project_dist`` (the thumb-side threshold) — taken
# verbatim from the upstream reference implementation.
_DEXPILOT_SECONDARY_PROJECT_DIST = 0.03


class Optimizer:
    """Base class. Subclasses build the per-frame objective + gradient.

    Operates directly on an Eden :class:`~eden.entities.rigid.RigidEntity`.
    All DOF arrays use ``entity.dofs_name`` order (the same order
    ``set_qpos`` and ``get_jacobian``'s columns use).
    """

    retargeting_type = "BASE"

    def __init__(
        self,
        entity: "RigidEntity",
        target_joint_names: List[str],
        target_link_human_indices: Optional[np.ndarray],
    ) -> None:
        # The requirement is that the optimised qpos carries no base DOFs — true for a fixed
        # base and equally for a hand welded onto an arm (``attach`` removes its root joint),
        # so welded dexterous hands are supported here.
        if not entity.has_immobile_base:
            raise ValueError(
                "DexRetargeter optimizers require an entity with no base DOFs (fixed-base, or "
                "welded onto another entity). Floating-hand retargeting (upstream's "
                "`add_dummy_free_joint`) is not yet supported."
            )
        self.entity = entity
        self.device = entity.device
        self.dofs_name: list[str] = list(entity.dofs_name)
        self.num_joints: int = len(self.dofs_name)

        idx_target: list[int] = []
        for name in target_joint_names:
            if name not in self.dofs_name:
                raise ValueError(f"Joint {name!r} not in entity.dofs_name. Available: {self.dofs_name}")
            idx_target.append(self.dofs_name.index(name))
        target_set = set(idx_target)
        self.target_joint_names = target_joint_names
        self.idx_target = np.array(idx_target, dtype=int)
        self.idx_fixed = np.array([i for i in range(self.num_joints) if i not in target_set], dtype=int)

        self.opt = nlopt.opt(nlopt.LD_SLSQP, len(idx_target))
        self.opt_dof = len(idx_target)

        self.target_link_human_indices = target_link_human_indices

    # ------------------------------------------------------------------
    # Eden kinematics helpers
    # ------------------------------------------------------------------
    def _cache_links(self, links_name: List[str]) -> list:
        """Resolve link names once and return cached ``RigidLink`` handles.

        Raises early if any name is missing so misconfigured presets fail at
        construction instead of mid-solve.
        """
        available = {link.name for link in self.entity.links}
        handles = []
        for name in links_name:
            if name not in available:
                raise ValueError(f"Link {name!r} not in entity. Available: {sorted(available)}")
            handles.append(self.entity.get_link(name))
        return handles

    def _setup_link_pairs(self, origin_names: List[str], task_names: List[str]) -> None:
        """Cache the union of origin+task links plus index tensors selecting each side.

        Like every other tensor in the objectives these are built on ``self.device`` (the entity's), explicitly:
        there is no torch default device — see the note in
        :meth:`DexPilotOptimizer.get_objective_function` about keeping that uniform.
        """
        self.computed_link_names: list[str] = list(set(origin_names) | set(task_names))
        self._computed_links = self._cache_links(self.computed_link_names)
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(n) for n in origin_names], device=self.device
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(n) for n in task_names], device=self.device
        )

    def _set_qpos(self, qpos: np.ndarray) -> None:
        """Apply ``qpos`` to the entity; triggers a forward-kinematics pass."""
        qpos_t = torch.as_tensor(qpos, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.entity.set_qpos(qpos_t, zero_velocity=True)

    @staticmethod
    def _link_positions(links) -> np.ndarray:
        """Stack the world-frame XYZ of each cached link handle into ``(n, 3)``."""
        # relative=False: world frame (Genesis >=1.1.1 link getters default to the user frame).
        return np.stack([link.get_pos(relative=False).squeeze(0).cpu().numpy() for link in links], axis=0)

    def _link_world_jacobian(self, link) -> np.ndarray:
        """World-frame 3xN positional Jacobian for ``link``, in ``dofs_name`` order.

        Eden's ``RigidEntity.get_jacobian`` returns ``(n_envs, 6, n_dofs)`` for a
        built entity; we take env 0 and slice off the angular rows.
        """
        return self.entity.get_jacobian(link)[0, :3, :].cpu().numpy()

    def _write_grad(
        self, links, torch_body_pos, huber_distance, x: np.ndarray, last_qpos: np.ndarray, grad: np.ndarray
    ) -> None:
        """Compute and write the SLSQP gradient via the cached link Jacobians.

        Shared by all three optimizers: they only differ in the loss they pass
        in via ``huber_distance`` (which we backprop through) and the link set.
        """
        jacobians = np.stack([self._link_world_jacobian(link) for link in links], axis=0)
        huber_distance.backward()
        grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]
        jacobians = jacobians[..., self.idx_target]
        grad_qpos = np.matmul(grad_pos, jacobians).mean(1).sum(0)
        grad_qpos += 2.0 * self.norm_delta * (x - last_qpos)
        grad[:] = grad_qpos

    # ------------------------------------------------------------------
    # nlopt plumbing
    # ------------------------------------------------------------------
    def set_joint_limit(self, joint_limits: np.ndarray, epsilon: float = 1e-3) -> None:
        if joint_limits.shape != (self.opt_dof, 2):
            raise ValueError(f"Expected joint_limits shape {(self.opt_dof, 2)}, got {joint_limits.shape}")
        self.opt.set_lower_bounds((joint_limits[:, 0] - epsilon).tolist())
        self.opt.set_upper_bounds((joint_limits[:, 1] + epsilon).tolist())

    def retarget(
        self,
        ref_value: np.ndarray,
        fixed_qpos: np.ndarray,
        last_qpos: np.ndarray,
    ) -> np.ndarray:
        """Run one SLSQP solve and return the optimized target-joint qpos.

        Parameters
        ----------
        ref_value
            Optimizer-specific Cartesian reference (positions or vectors).
        fixed_qpos
            Values for joints NOT in ``target_joint_names``, ordered to match
            ``self.idx_fixed`` (i.e. ``entity.dofs_name`` filtered by that index).
        last_qpos
            Initial guess and regularization target for this frame.
        """
        if len(fixed_qpos) != len(self.idx_fixed):
            raise ValueError(f"Optimizer expects {len(self.idx_fixed)} fixed joints, got {len(fixed_qpos)}.")
        objective_fn = self.get_objective_function(
            ref_value,
            fixed_qpos,
            np.asarray(last_qpos, dtype=np.float32),
        )
        self.opt.set_min_objective(objective_fn)
        try:
            qpos = self.opt.optimize(last_qpos)
            return np.asarray(qpos, dtype=np.float32)
        except RuntimeError as e:  # nlopt raises this on solver failure
            en.logger.warning(f"nlopt solver failure: {e}")
            return np.asarray(last_qpos, dtype=np.float32)

    @abstractmethod
    def get_objective_function(
        self,
        ref_value: np.ndarray,
        fixed_qpos: np.ndarray,
        last_qpos: np.ndarray,
    ): ...


class PositionOptimizer(Optimizer):
    """Match absolute world-frame positions of robot links to keypoints."""

    retargeting_type = "POSITION"

    def __init__(
        self,
        entity: "RigidEntity",
        target_joint_names: List[str],
        target_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta: float = 0.02,
        norm_delta: float = 4e-3,
    ) -> None:
        super().__init__(entity, target_joint_names, target_link_human_indices)
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta)
        self.norm_delta = norm_delta
        self._target_links = self._cache_links(target_link_names)
        self.opt.set_ftol_abs(1e-5)

    def get_objective_function(self, target_pos, fixed_qpos, last_qpos):
        expected = (len(self._target_links), 3)
        if target_pos.shape != expected:
            raise ValueError(
                f"PositionOptimizer: target_pos shape {target_pos.shape} does not match "
                f"{expected} (one (x, y, z) per target link). Did you pass the full "
                f"21-keypoint MANO array? Slice by target_link_human_indices first, "
                f"e.g. via DexRetargeter.extract_ref_value."
            )
        qpos = np.zeros(self.num_joints, dtype=np.float32)
        qpos[self.idx_fixed] = fixed_qpos
        torch_target_pos = torch.as_tensor(target_pos, dtype=torch.float32, device=self.device)
        torch_target_pos.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_target] = x
            self._set_qpos(qpos)
            body_pos = self._link_positions(self._target_links)

            torch_body_pos = torch.as_tensor(body_pos, dtype=torch.float32, device=self.device)
            torch_body_pos.requires_grad_()
            huber_distance = self.huber_loss(torch_body_pos, torch_target_pos)
            result = float(huber_distance.detach().item())

            if grad.size > 0:
                self._write_grad(self._target_links, torch_body_pos, huber_distance, x, last_qpos, grad)
            return result

        return objective


class VectorOptimizer(Optimizer):
    """Match relative vectors between robot link pairs to keypoint pairs."""

    retargeting_type = "VECTOR"

    def __init__(
        self,
        entity: "RigidEntity",
        target_joint_names: List[str],
        target_origin_link_names: List[str],
        target_task_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta: float = 0.02,
        norm_delta: float = 4e-3,
        scaling: float = 1.0,
    ) -> None:
        super().__init__(entity, target_joint_names, target_link_human_indices)
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="mean")
        self.norm_delta = norm_delta
        self.scaling = scaling
        self._setup_link_pairs(target_origin_link_names, target_task_link_names)
        self.opt.set_ftol_abs(1e-6)

    def get_objective_function(self, target_vector, fixed_qpos, last_qpos):
        expected = (len(self.origin_link_indices), 3)
        if target_vector.shape != expected:
            raise ValueError(
                f"VectorOptimizer: target_vector shape {target_vector.shape} does not "
                f"match {expected} (one (dx, dy, dz) per origin/task link pair). Did "
                f"you pass the full 21-keypoint MANO array? Compute "
                f"`kp[task_idx] - kp[origin_idx]` first, e.g. via "
                f"DexRetargeter.extract_ref_value."
            )
        qpos = np.zeros(self.num_joints, dtype=np.float32)
        qpos[self.idx_fixed] = fixed_qpos
        torch_target_vec = torch.as_tensor(target_vector, dtype=torch.float32, device=self.device) * self.scaling
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_target] = x
            self._set_qpos(qpos)
            body_pos = self._link_positions(self._computed_links)

            torch_body_pos = torch.as_tensor(body_pos, dtype=torch.float32, device=self.device)
            torch_body_pos.requires_grad_()
            robot_vec = torch_body_pos[self.task_link_indices, :] - torch_body_pos[self.origin_link_indices, :]
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1)
            huber_distance = self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
            result = float(huber_distance.detach().item())

            if grad.size > 0:
                self._write_grad(self._computed_links, torch_body_pos, huber_distance, x, last_qpos, grad)
            return result

        return objective


class DexPilotOptimizer(Optimizer):
    """DexPilot vector retargeting with thumb-to-fingertip projection.

    A direct port of ``dex_retargeting.optimizer.DexPilotOptimizer``. Generates
    its own link pairs (each fingertip-to-wrist + every fingertip-pair) and
    snaps thumb-to-fingertip distances inside ``project_dist`` to encourage
    pinch grasps. See https://arxiv.org/abs/1910.03135.
    """

    retargeting_type = "DEXPILOT"

    def __init__(
        self,
        entity: "RigidEntity",
        target_joint_names: List[str],
        finger_tip_link_names: List[str],
        wrist_link_name: str,
        target_link_human_indices: Optional[np.ndarray] = None,
        huber_delta: float = 0.03,
        norm_delta: float = 4e-3,
        project_dist: float = 0.03,
        escape_dist: float = 0.05,
        eta1: float = 1e-4,
        eta2: float = 3e-2,
        scaling: float = 1.0,
    ) -> None:
        if not (2 <= len(finger_tip_link_names) <= 5):
            raise ValueError(f"DexPilot supports 2..5 fingertips; got {len(finger_tip_link_names)}.")
        self.num_fingers = len(finger_tip_link_names)

        origin_link_index, task_link_index = self._generate_link_indices(self.num_fingers)
        if target_link_human_indices is None:
            target_link_human_indices = (np.stack([origin_link_index, task_link_index], axis=0) * 4).astype(int)

        links_name = [wrist_link_name] + list(finger_tip_link_names)
        origin_names = [links_name[i] for i in origin_link_index]
        task_names = [links_name[i] for i in task_link_index]

        super().__init__(entity, target_joint_names, target_link_human_indices)
        self.scaling = scaling
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="none")
        self.norm_delta = norm_delta
        self.project_dist = project_dist
        self.escape_dist = escape_dist
        self.eta1 = eta1
        self.eta2 = eta2

        self._setup_link_pairs(origin_names, task_names)
        self.opt.set_ftol_abs(1e-6)

        (
            self.projected,
            self.s2_project_index_origin,
            self.s2_project_index_task,
            self.projected_dist,
        ) = self._set_dexpilot_cache(self.num_fingers, eta1, eta2)

    @staticmethod
    def _generate_link_indices(num_fingers: int) -> tuple[list[int], list[int]]:
        origin_link_index: list[int] = []
        task_link_index: list[int] = []
        for i in range(1, num_fingers):
            for j in range(i + 1, num_fingers + 1):
                origin_link_index.append(j)
                task_link_index.append(i)
        for i in range(1, num_fingers + 1):
            origin_link_index.append(0)
            task_link_index.append(i)
        return origin_link_index, task_link_index

    @staticmethod
    def _set_dexpilot_cache(num_fingers: int, eta1: float, eta2: float):
        projected = np.zeros(num_fingers * (num_fingers - 1) // 2, dtype=bool)
        s2_project_index_origin: list[int] = []
        s2_project_index_task: list[int] = []
        for i in range(0, num_fingers - 2):
            for j in range(i + 1, num_fingers - 1):
                s2_project_index_origin.append(j)
                s2_project_index_task.append(i)
        projected_dist = np.array([eta1] * (num_fingers - 1) + [eta2] * ((num_fingers - 1) * (num_fingers - 2) // 2))
        return projected, s2_project_index_origin, s2_project_index_task, projected_dist

    def get_objective_function(self, target_vector, fixed_qpos, last_qpos):
        qpos = np.zeros(self.num_joints, dtype=np.float32)
        qpos[self.idx_fixed] = fixed_qpos

        len_proj = len(self.projected)
        len_s2 = len(self.s2_project_index_task)
        len_s1 = len_proj - len_s2

        target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1)
        self.projected[:len_s1][target_vec_dist[0:len_s1] < self.project_dist] = True
        self.projected[:len_s1][target_vec_dist[0:len_s1] > self.escape_dist] = False
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[:len_s1][self.s2_project_index_origin],
            self.projected[:len_s1][self.s2_project_index_task],
        )
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[len_s1:len_proj], target_vec_dist[len_s1:len_proj] <= _DEXPILOT_SECONDARY_PROJECT_DIST
        )

        normal_weight = np.ones(len_proj, dtype=np.float32)
        high_weight = np.array([200] * len_s1 + [400] * len_s2, dtype=np.float32)
        weight = np.where(self.projected, high_weight, normal_weight)
        # `as_tensor(..., device=self.device)`, not `from_numpy`: every tensor in this objective is built on
        # the entity's device, explicitly, since there is no torch default device (Genesis stopped setting one in
        # https://github.com/Genesis-Embodied-AI/Genesis/pull/3431). `from_numpy` aliases the numpy buffer, so it is
        # always CPU: on a GPU backend, multiplying the CUDA huber term by it raises inside nlopt, and `retarget`'s
        # `except RuntimeError` turns that into a per-frame warning while handing back `last_qpos` -- a robot hand
        # that silently never moves.
        weight = torch.as_tensor(
            np.concatenate([weight, np.ones(self.num_fingers, dtype=np.float32) * len_proj + self.num_fingers]),
            dtype=torch.float32,
            device=self.device,
        )

        normal_vec = target_vector * self.scaling
        dir_vec = target_vector[:len_proj] / (target_vec_dist[:, None] + 1e-6)
        projected_vec = dir_vec * self.projected_dist[:, None]
        reference_vec = np.where(self.projected[:, None], projected_vec, normal_vec[:len_proj])
        reference_vec = np.concatenate([reference_vec, normal_vec[len_proj:]], axis=0)
        torch_target_vec = torch.as_tensor(reference_vec, dtype=torch.float32, device=self.device)
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_target] = x
            self._set_qpos(qpos)
            body_pos = self._link_positions(self._computed_links)

            torch_body_pos = torch.as_tensor(body_pos, dtype=torch.float32, device=self.device)
            torch_body_pos.requires_grad_()
            robot_vec = torch_body_pos[self.task_link_indices, :] - torch_body_pos[self.origin_link_indices, :]
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1)
            huber_distance = (self.huber_loss(vec_dist, torch.zeros_like(vec_dist)) * weight / robot_vec.shape[0]).sum()
            result = float(huber_distance.detach().item())

            if grad.size > 0:
                self._write_grad(self._computed_links, torch_body_pos, huber_distance, x, last_qpos, grad)
            return result

        return objective
