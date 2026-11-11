"""Task-space action terms: differential IK, absolute pose IK, and operational-space control."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import genesis as gs
import torch
from genesis.utils.geom import inv_quat, transform_by_quat

import eden as en
from eden.managers.action_manager import ACTION_TERM_REGISTRY, ActionTerm
from eden.managers.terms.utils import resolve_link_idx
from eden.utils.geom import combine_frame_transforms, quat_box_minus, quat_from_axis_angle, quat_mul
from eden.utils.string import term_subject
from eden.utils.torch import resolve_scalar_or_vector

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions


#: Metres and radians per normalized action unit when a config names no scale: robosuite's OSC
#: action limits.
_DEFAULT_POSITION_SCALE = 0.05
_DEFAULT_ORIENTATION_SCALE = 0.5


def _resolve_axis_parameter(
    value: float | list[float] | tuple[float, ...],
    axis_count: int,
    *,
    device: torch.device | str,
    minimum: float | None = None,
) -> torch.Tensor:
    """Resolve and validate a scalar or axis-wise parameter as a fixed-length tensor."""
    resolved = resolve_scalar_or_vector(
        value,
        axis_count,
        device,
        what="Axis parameter",
        dtype=gs.tc_float,
    )
    is_finite = math.isfinite(resolved) if isinstance(resolved, float) else torch.isfinite(resolved).all().item()
    if not is_finite:
        raise ValueError("Axis parameter must contain only finite values")
    if minimum is not None:
        is_below_minimum = resolved < minimum if isinstance(resolved, float) else (resolved < minimum).any().item()
        if is_below_minimum:
            raise ValueError(f"Axis parameter entries must be >= {minimum}")
    if isinstance(resolved, float):
        return torch.full((axis_count,), resolved, device=device, dtype=gs.tc_float)
    return resolved


@ACTION_TERM_REGISTRY.register()
class DifferentialIKController(ActionTerm):
    """Action term that maps task-space commands to joint targets via differential IK.

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    dofs_name: list[str]
        The names of the DOFs to control.
    ee_link_name: str
        The name of the end-effector link.
    ee_local_pos: tuple[float, float, float]
        Control-point position in the end-effector link frame, in metres: the point the
        position rows move, and the point a rotation command turns the link about. The
        default is the link origin.
    position_scale: float | list[float]
        Metres per normalized action unit for the position axes. A scalar applies
        to x, y, and z; a length-3 list configures the axes independently.
    orientation_scale: float | list[float]
        Radians per normalized action unit for the orientation axes. A scalar
        applies to rx, ry, and rz; a length-3 list configures the axes independently.
    damping: float
        Levenberg-Marquardt damping ``lambda`` for the differential-IK least
        squares ``Jt (J Jt + lambda I)^-1``. Larger values trade tracking
        accuracy for stability near singularities, where an under-damped solve
        produces large ``delta_q`` spikes (visible as end-effector jitter).
        mjlab's ``DifferentialIKActionCfg`` uses 0.1 (the default here); an
        under-damped value like 1e-4 is effectively undamped.
    max_dq: float | None
        Optional per-step clamp (rad) on the IK joint delta, mirroring mjlab's
        ``max_dq``. Caps how far any joint may move in a single control step,
        which bounds end-effector speed and smooths motion. ``None`` disables it.
    position_weight: float
        Task-space weight on the position (x, y, z) rows of the IK solve.
    orientation_weight: float
        Task-space weight on the orientation (rx, ry, rz) rows of the IK solve,
        mirroring mjlab's ``DifferentialIKActionCfg.orientation_weight``. The
        damped least squares weights both the Jacobian rows and the task error
        per axis, so ``orientation_weight=0.0`` makes it **position-only** — the
        wrist orientation is left uncontrolled (free in the nullspace) rather
        than driven by the policy, which removes wrist jitter without pinning
        the wrist. Defaults to 1.0 (full 6-DOF). Relies on
        ``damping > 0`` to keep zero-weighted rows of ``J Jt`` invertible.
    """

    ee_link_name: str = ""
    ee_local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_scale: float | list[float] = _DEFAULT_POSITION_SCALE
    orientation_scale: float | list[float] = _DEFAULT_ORIENTATION_SCALE
    damping: float = 0.1
    max_dq: float | None = None
    position_weight: float = 1.0
    orientation_weight: float = 1.0

    def __init__(
        self,
        env: EnvBase,
        options: ActionTermOptions,
    ):
        super().__init__(env=env, options=options)
        self._ee_link = None
        self._ee_local_pos = None
        self._diag = None
        self._scale = None
        self._task_weight = None

    def build(self) -> None:
        super().build()

        self._ee_link = self.entity.get_link(self.ee_link_name)
        if len(self.ee_local_pos) != 3 or not all(math.isfinite(value) for value in self.ee_local_pos):
            raise ValueError(
                f"{term_subject(self)}: ee_local_pos must contain three finite values, got {self.ee_local_pos}"
            )
        self._ee_local_pos = torch.tensor(self.ee_local_pos, dtype=gs.tc_float, device=self.device)

        # Jacobian columns are in dofs_name order; build indices to select
        # this action's DOF subset from the reordered Jacobian. A welded child (a
        # composite's gripper / dex hand) has no base columns either — Genesis ``attach``
        # removed its root joint — so gate on ``has_immobile_base``, matching ``_jac_perm``.
        base_offset = 0 if self.entity.has_immobile_base else 6
        self._jac_dofs_idx = torch.as_tensor(
            [base_offset + self.entity.dofs_idx_map[idx] for idx in self.dofs_idx_local.tolist()],
            dtype=gs.tc_int,
            device=self.device,
        )

        self._scale = torch.cat(
            (
                _resolve_axis_parameter(self.position_scale, 3, device=self.device),
                _resolve_axis_parameter(self.orientation_scale, 3, device=self.device),
            )
        )

        # Per-axis task weights; None means uniform 1.0 and skips the weighting.
        if self.position_weight != 1.0 or self.orientation_weight != 1.0:
            self._task_weight = torch.tensor(
                [self.position_weight] * 3 + [self.orientation_weight] * 3,
                device=self.device,
                dtype=gs.tc_float,
            ).unsqueeze(0)  # (1, 6)

    @property
    def action_dim(self):
        return 6  # (x, y, z, rx, ry, rz)

    @property
    def effective_scale(self) -> torch.Tensor:
        """Return the resolved task-space scale in ``(x, y, z, rx, ry, rz)`` order."""
        return self._scale

    @property
    def diag(self):
        if self._diag is None:
            self._diag = self.damping * torch.eye(6, device=self.device, dtype=gs.tc_float)
        return self._diag

    def compute(self, actions: torch.Tensor) -> None:
        assert actions.shape == (
            self.num_envs,
            self.action_dim,
        ), f"Expected actions shape: (num_envs, {self.action_dim}), got {actions.shape=}"
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions

        delta_target = self._scale * torch.tanh(actions)  # (n_envs, 6)
        # (n_envs, 6, n_dofs), the linear rows at the control point.
        J = self.entity.get_jacobian(link=self._ee_link, local_point=self._ee_local_pos)[:, :, self._jac_dofs_idx]
        if self._task_weight is not None:
            # Weighted damped least squares: weight the Jacobian rows and task
            # error per axis. orientation_weight=0 zeros the orientation rows
            # (damping keeps J Jt invertible) -> position-only IK, wrist free.
            J = J * self._task_weight.unsqueeze(-1)  # (n_envs, 6, n_dofs)
            delta_target = delta_target * self._task_weight  # (n_envs, 6)
        J_T = J.transpose(1, 2)  # (n_envs, n_dofs, 6)
        # `solve_ex`, not `solve`: `solve` reads the LU info back to raise on a singular system, a GPU->CPU
        # sync every control step. `J Jt + damping * I` is positive definite, so there is nothing to catch.
        delta_q = J_T @ torch.linalg.solve_ex(J @ J_T + self.diag, delta_target)[0].unsqueeze(-1)
        delta_q = delta_q.squeeze(-1)  # (n_envs, n_dofs)

        # Per-step joint-delta clamp (mjlab's max_dq): bounds joint speed.
        if self.max_dq is not None:
            delta_q = delta_q.clamp(-self.max_dq, self.max_dq)

        self._processed_action = delta_q + self.entity.get_dofs_pos(dofs_idx_local=self.genesis_dofs_idx)

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._prev_action[envs_idx] = 0.0
        self._raw_action[envs_idx] = 0.0

    def apply_actions(self) -> None:
        self.entity.control_dofs_pos(self._processed_action, dofs_idx_local=self.genesis_dofs_idx)


def _damped_pseudoinverse(
    matrix: torch.Tensor,
    identity: torch.Tensor,
    relative_damping: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a scaled Tikhonov pseudoinverse and an invalid-batch mask."""
    is_finite = torch.isfinite(matrix).all(dim=(-2, -1))
    safe_matrix = torch.where(is_finite[..., None, None], matrix, 0.0)
    scale = safe_matrix.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(torch.finfo(matrix.dtype).eps)
    scaled_matrix = safe_matrix / scale
    ridge_root = math.sqrt(relative_damping)
    expanded_identity = identity.expand(matrix.shape[:-2] + identity.shape[-2:])
    scaled_identity = ridge_root * expanded_identity
    block_system = torch.cat(
        (
            torch.cat((scaled_identity, scaled_matrix), dim=-1),
            torch.cat((scaled_matrix.transpose(-2, -1), -scaled_identity), dim=-1),
        ),
        dim=-2,
    )
    block_rhs = torch.cat((expanded_identity, torch.zeros_like(expanded_identity)), dim=-2)
    # ``solve_ex`` keeps the CUDA hot path asynchronous while exposing per-batch factorization failures on device.
    solve_result = torch.linalg.solve_ex(block_system, block_rhs, check_errors=False)
    pseudoinverse = solve_result.result[..., matrix.shape[-1] :, :] / scale
    is_invalid = ~is_finite | (solve_result.info != 0) | ~torch.isfinite(pseudoinverse).all(dim=(-2, -1))
    pseudoinverse.masked_fill_(is_invalid[..., None, None], 0.0)
    return pseudoinverse, is_invalid


def _solve_mass_matrix_jacobian_transpose(
    mass_matrix: torch.Tensor,
    jacobian: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve ``M X = J^T`` and return ``X`` with an invalid-batch mask."""
    # ``solve_ex`` keeps the CUDA hot path asynchronous while exposing per-batch failures on device.
    solve_result = torch.linalg.solve_ex(
        mass_matrix,
        jacobian.transpose(-2, -1),
        check_errors=False,
    )
    mass_solve_jacobian_t = solve_result.result
    is_invalid = (
        ~torch.isfinite(mass_matrix).all(dim=(-2, -1))
        | ~torch.isfinite(jacobian).all(dim=(-2, -1))
        | (solve_result.info != 0)
        | ~torch.isfinite(mass_solve_jacobian_t).all(dim=(-2, -1))
    )
    mass_solve_jacobian_t.masked_fill_(is_invalid[..., None, None], 0.0)
    return mass_solve_jacobian_t, is_invalid


@ACTION_TERM_REGISTRY.register()
class OperationalSpaceController(ActionTerm):
    """Operational-space control to a persistent relative position/pose target.

    Policy actions are clipped, scaled, and composed with the measured world-frame control-point pose in
    :meth:`compute`. The resulting absolute target stays fixed while :meth:`apply_actions` recomputes feedback,
    Jacobian, inertia, nullspace, and DOF force on every physics substep. Gravity compensation is configured on the
    rigid material of the kinematic-tree root, including the root parent rather than only the controlled child in a
    welded tree. This controller does not add Coriolis or centrifugal bias forces. The entity's configured force range
    saturates the controller command, not the total effective effort including material gravity compensation.

    The controller supports fixed entities and entities welded through a static parent chain. ``dofs_name`` defines one
    ordered controller space shared by the Jacobian, mass matrix, DOF state, nullspace objective, and output force.
    When dynamics are enabled, build rejects a selected DOF set that omits an ancestor DOF of the control-point link.
    The selected mass submatrix further assumes remaining excluded branch DOFs are fixed or held so their acceleration
    can be neglected, or are dynamically decoupled; assigning them another controller does not by itself remove dynamic
    coupling. Operational inertia uses a relative Tikhonov approximation to the Moore-Penrose pseudoinverse, so a
    rank-deficient Jacobian has a finite damped minimum-norm response.

    Planned extensions
    ------------------
    TODO(generic-osc-extensions): Add absolute pose targets, configurable task/root/world command frames, per-axis
    motion selection, absolute wrench and hybrid motion-force control, variable impedance, and command interpolation
    as concrete Eden callers require them. Extend the dynamics contract to floating bases and passive coupled
    coordinates only with an explicit generalized-coordinate index space. Keep multi-end-effector, mobile-manipulator,
    and whole-body coordination in a composite layer above this single-control-point action term.

    Parameters
    ----------
    ee_link_name : str
        Link carrying the controlled point.
    ee_local_pos : tuple[float, float, float]
        Control-point position in the end-effector link frame, in metres.
    ee_local_quat : tuple[float, float, float, float]
        Control-point orientation in the end-effector link frame as a normalized wxyz quaternion.
    control_mode : {"position", "pose"}
        Whether actions command three world-frame translation deltas or translation plus world-frame axis-angle
        rotation deltas.
    position_scale : float | tuple[float, float, float]
        Metres per translation action unit, scalar or axis-wise.
    orientation_scale : float | tuple[float, float, float]
        Radians per axis-angle rotation action unit, scalar or axis-wise. Ignored in position mode.
    action_clip : float | tuple[float, ...]
        Symmetric pre-scaling action limit for each controlled task axis.
    kp : float | tuple[float, ...]
        Nonnegative task-space proportional gains for the controlled axes.
    damping_ratio : float | tuple[float, ...]
        Nonnegative damping ratios. Derivative gains are 2 * damping_ratio * sqrt(kp).
    inertial_decoupling : {"none", "full", "position_orientation"}
        none maps the feedback wrench directly, full multiplies it by the coupled operational inertia, and
        position_orientation uses separate translational and rotational operational inertias.
    operational_inertia_damping : float
        Positive dimensionless Tikhonov ridge coefficient relative to the squared largest absolute entry of each
        task-space inverse-inertia matrix.
    nullspace_kp : float | None
        Positive DOF-space posture gain, or None to disable the secondary posture objective. Enabling it requires
        more selected DOFs than controlled task axes.
    nullspace_damping_ratio : float
        Nonnegative damping ratio for the nullspace posture objective.
    nullspace_reference : {"default", "reset_pose"}
        Track the entity default DOF positions or the measured DOF positions at the most recent reset.
    """

    ee_link_name: str = ""
    ee_local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ee_local_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    control_mode: Literal["position", "pose"] = "pose"
    position_scale: float | tuple[float, float, float] = _DEFAULT_POSITION_SCALE
    orientation_scale: float | tuple[float, float, float] = _DEFAULT_ORIENTATION_SCALE
    action_clip: float | tuple[float, ...] = 1.0
    kp: float | tuple[float, ...] = 100.0
    damping_ratio: float | tuple[float, ...] = 1.0
    inertial_decoupling: Literal["none", "full", "position_orientation"] = "none"
    operational_inertia_damping: float = 1.0e-6
    nullspace_kp: float | None = None
    nullspace_damping_ratio: float = 1.0
    nullspace_reference: Literal["default", "reset_pose"] = "default"

    def __init__(self, env: EnvBase, options: ActionTermOptions) -> None:
        super().__init__(env=env, options=options)
        self._ee_link_idx: int | None = None
        self._ee_local_pos: torch.Tensor | None = None
        self._ee_local_quat: torch.Tensor | None = None
        self._dofs_idx_name_order: torch.Tensor | None = None
        self._position_scale: torch.Tensor | None = None
        self._orientation_scale: torch.Tensor | None = None
        self._action_clip: torch.Tensor | None = None
        self._scaled_action: torch.Tensor | None = None
        self._kp: torch.Tensor | None = None
        self._kd: torch.Tensor | None = None
        self._task_identity: torch.Tensor | None = None
        self._dofs_identity: torch.Tensor | None = None
        self._nullspace_acceleration: torch.Tensor | None = None
        self._nullspace_reference_pos: torch.Tensor | None = None
        self._goal_pos: torch.Tensor | None = None
        self._goal_quat: torch.Tensor | None = None
        self._nullspace_kd = 0.0
        self._needs_mass_matrix = False

    @property
    def action_dim(self) -> int:
        return 3 if self.control_mode == "position" else 6

    def build(self) -> None:
        self._resolve_controller_parameters()
        super().build()
        self._resolve_control_topology()

        tensor_options = {"dtype": gs.tc_float, "device": self.device}
        self._scaled_action = torch.zeros((self.num_envs, self.action_dim), **tensor_options)
        if self._needs_mass_matrix:
            self._task_identity = torch.eye(self.action_dim, **tensor_options).unsqueeze(0)
        if self.nullspace_kp is not None:
            self._dofs_identity = torch.eye(self.num_dofs, **tensor_options).unsqueeze(0)
            self._nullspace_acceleration = torch.zeros((self.num_envs, self.num_dofs), **tensor_options)
            self._nullspace_reference_pos = torch.zeros((self.num_envs, self.num_dofs), **tensor_options)
            self._nullspace_kd = 2.0 * self.nullspace_damping_ratio * self.nullspace_kp**0.5
        self._goal_pos = torch.zeros((self.num_envs, 3), **tensor_options)
        self._goal_quat = torch.zeros((self.num_envs, 4), **tensor_options)

        self._set_goal_to_current(slice(None))
        if self._nullspace_reference_pos is not None:
            self._refresh_nullspace_reference(slice(None))

    def _resolve_controller_parameters(self) -> None:
        if self.control_mode not in ("position", "pose"):
            raise ValueError(
                f"{term_subject(self)}: control_mode must be 'position' or 'pose', got {self.control_mode!r}"
            )
        if self.inertial_decoupling not in ("none", "full", "position_orientation"):
            raise ValueError(
                f"{term_subject(self)}: inertial_decoupling must be 'none', 'full', or "
                f"'position_orientation', got {self.inertial_decoupling!r}"
            )
        if self.nullspace_reference not in ("default", "reset_pose"):
            raise ValueError(
                f"{term_subject(self)}: nullspace_reference must be 'default' or 'reset_pose', "
                f"got {self.nullspace_reference!r}"
            )
        if not math.isfinite(self.operational_inertia_damping) or self.operational_inertia_damping <= 0.0:
            raise ValueError(
                f"{term_subject(self)}: operational_inertia_damping must be positive and finite, "
                f"got {self.operational_inertia_damping}"
            )
        self._position_scale = _resolve_axis_parameter(self.position_scale, 3, device=self.device).unsqueeze(0)
        if self.control_mode == "pose":
            self._orientation_scale = _resolve_axis_parameter(self.orientation_scale, 3, device=self.device).unsqueeze(
                0
            )
        self._action_clip = _resolve_axis_parameter(
            self.action_clip, self.action_dim, device=self.device, minimum=0.0
        ).unsqueeze(0)
        self._kp = _resolve_axis_parameter(self.kp, self.action_dim, device=self.device, minimum=0.0).unsqueeze(0)
        self._kd = (
            2.0
            * _resolve_axis_parameter(
                self.damping_ratio,
                self.action_dim,
                device=self.device,
                minimum=0.0,
            ).unsqueeze(0)
            * torch.sqrt(self._kp)
        )
        if self.nullspace_kp is not None and (not math.isfinite(self.nullspace_kp) or self.nullspace_kp <= 0.0):
            raise ValueError(
                f"{term_subject(self)}: nullspace_kp must be positive and finite or None, got {self.nullspace_kp}"
            )
        if not math.isfinite(self.nullspace_damping_ratio) or self.nullspace_damping_ratio < 0.0:
            raise ValueError(
                f"{term_subject(self)}: nullspace_damping_ratio must be nonnegative and finite, "
                f"got {self.nullspace_damping_ratio}"
            )
        if len(self.ee_local_pos) != 3 or not all(math.isfinite(value) for value in self.ee_local_pos):
            raise ValueError(
                f"{term_subject(self)}: ee_local_pos must contain three finite values, got {self.ee_local_pos}"
            )
        if len(self.ee_local_quat) != 4 or not all(math.isfinite(value) for value in self.ee_local_quat):
            raise ValueError(
                f"{term_subject(self)}: ee_local_quat must contain four finite values, got {self.ee_local_quat}"
            )
        quaternion_norm = math.sqrt(sum(component * component for component in self.ee_local_quat))
        if not math.isclose(quaternion_norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"{term_subject(self)}: ee_local_quat must be normalized, got norm={quaternion_norm}")
        self._ee_local_pos = torch.tensor(self.ee_local_pos, dtype=gs.tc_float, device=self.device)
        self._ee_local_quat = torch.tensor(self.ee_local_quat, dtype=gs.tc_float, device=self.device).unsqueeze(0)

    def _resolve_control_topology(self) -> None:
        entity = self.entity
        if self.nullspace_kp is not None and self.num_dofs <= self.action_dim:
            raise ValueError(
                f"{term_subject(self)}: nullspace control requires more selected DOFs than controlled task axes, "
                f"got num_dofs={self.num_dofs} and task_dim={self.action_dim}"
            )
        self._needs_mass_matrix = self.inertial_decoupling != "none" or self.nullspace_kp is not None
        root_dofs = int(entity.n_dofs) - int(entity.num_dofs)
        if root_dofs != 0:
            raise ValueError(
                f"{term_subject(self)} currently requires an entity with no root DOFs, "
                f"got total_dofs={entity.n_dofs}, named_dofs={entity.num_dofs}"
            )
        attachment_parent = entity.attachment_parent if entity.is_attached else None
        while attachment_parent is not None:
            if int(attachment_parent.n_dofs) != 0:
                raise ValueError(
                    f"{term_subject(self)} cannot control an entity welded to a moving parent because the "
                    f"parent DOFs are absent from its Jacobian and mass matrix"
                )
            attachment_parent = attachment_parent.attachment_parent if attachment_parent.is_attached else None

        ee_link_idx = resolve_link_idx(entity, [self.ee_link_name], self.device)
        if len(ee_link_idx) != 1:
            raise ValueError(
                f"{term_subject(self)}: ee_link_name must resolve exactly one link, "
                f"got {len(ee_link_idx)} for {self.ee_link_name!r}"
            )
        self._ee_link_idx = int(ee_link_idx[0])
        self._dofs_idx_name_order = torch.as_tensor(
            [entity.dofs_idx_map[index] for index in self.dofs_idx_local.tolist()],
            dtype=gs.tc_int,
            device=self.device,
        )
        full_jacobian = entity.get_jacobian(link=entity.links[self._ee_link_idx], local_point=self._ee_local_pos)
        if full_jacobian.ndim != 3 or tuple(full_jacobian.shape[-2:]) != (6, entity.num_dofs):
            raise ValueError(
                f"{term_subject(self)}: expected a Jacobian with trailing shape "
                f"(6, {entity.num_dofs}), got {tuple(full_jacobian.shape)}"
            )
        if self._needs_mass_matrix:
            ancestor_dofs_idx_local = set()
            link = entity.links[self._ee_link_idx]
            while True:
                for joint in link.joints:
                    ancestor_dofs_idx_local.update(joint.dofs_idx_local)
                if not entity.link_start <= link.parent_idx < entity.link_end:
                    break
                link = entity.links[link.parent_idx - entity.link_start]
            selected_dofs_idx_local = set(self.dofs_idx_local.tolist())
            missing_dofs_idx_local = sorted(ancestor_dofs_idx_local - selected_dofs_idx_local)
            if missing_dofs_idx_local:
                missing_dofs_name = [entity.dofs_name[entity.dofs_idx_map[index]] for index in missing_dofs_idx_local]
                raise ValueError(
                    f"{term_subject(self)}: dynamics require every ancestor DOF of the control-point link; "
                    f"dofs_name omits {missing_dofs_name}"
                )
            mass_matrix = entity.get_mass_mat()
            if mass_matrix.ndim != 3 or tuple(mass_matrix.shape[-2:]) != (entity.num_dofs, entity.num_dofs):
                raise ValueError(
                    f"{term_subject(self)}: expected a mass matrix with trailing shape "
                    f"({entity.num_dofs}, {entity.num_dofs}), got {tuple(mass_matrix.shape)}"
                )

    def _control_point_pose(self, entity=None) -> tuple[torch.Tensor, torch.Tensor]:
        if entity is None:
            entity = self.resolve_entity(self.entity_name)
        link_pos = entity.get_links_pos(ls_idx_local=self._ee_link_idx).squeeze(1)
        link_quat = entity.get_links_quat(ls_idx_local=self._ee_link_idx).squeeze(1)
        control_pos = link_pos + transform_by_quat(self._ee_local_pos.unsqueeze(0).expand_as(link_pos), link_quat)
        control_quat = quat_mul(link_quat, self._ee_local_quat.expand_as(link_quat))
        return control_pos, control_quat

    def _set_goal_to_current(self, envs_idx: slice | torch.Tensor) -> None:
        current_pos, current_quat = self._control_point_pose()
        self._goal_pos[envs_idx] = current_pos[envs_idx]
        self._goal_quat[envs_idx] = current_quat[envs_idx]

    def _update_goal_from_action(
        self,
        current_pos: torch.Tensor,
        current_quat: torch.Tensor,
        scaled_action: torch.Tensor,
    ) -> None:
        """Update the persistent goal from one policy action; benchmark subclasses may specialize this contract."""
        torch.add(current_pos, scaled_action[:, :3], out=self._goal_pos)
        if self.control_mode == "pose":
            self._goal_quat[:] = quat_mul(quat_from_axis_angle(scaled_action[:, 3:]), current_quat)
        else:
            self._goal_quat[:] = current_quat

    def _reset_goal(self, envs_idx: slice | torch.Tensor) -> None:
        """Reset goal state; benchmark subclasses may preserve a construction-time orientation instead."""
        self._set_goal_to_current(envs_idx)

    def _orientation_error(self, goal_quat: torch.Tensor, current_quat: torch.Tensor) -> torch.Tensor:
        """Return axis-angle orientation error; benchmark subclasses may override the error representation."""
        return quat_box_minus(goal_quat, current_quat)

    def _refresh_nullspace_reference(self, envs_idx: slice | torch.Tensor) -> None:
        entity = self.resolve_entity(self.entity_name)
        if self.nullspace_reference == "reset_pose":
            reference = entity.get_dofs_pos(dofs_idx_local=self.genesis_dofs_idx)
        else:
            reference = entity.default_dofs_pos.index_select(1, self._dofs_idx_name_order)
            if reference.shape[0] == 1 and self.num_envs != 1:
                reference = reference.expand(self.num_envs, -1)
        self._nullspace_reference_pos[envs_idx] = reference[envs_idx]

    @property
    def effective_scale(self) -> torch.Tensor:
        """Return the resolved scale in position-then-orientation axis order."""
        if self.control_mode == "position":
            return self._position_scale[0]
        return torch.cat((self._position_scale[0], self._orientation_scale[0]))

    def compute(self, actions: torch.Tensor) -> None:
        expected_shape = (self.num_envs, self.action_dim)
        if tuple(actions.shape) != expected_shape:
            raise ValueError(
                f"{term_subject(self)}: actions must have shape {expected_shape}, got {tuple(actions.shape)}"
            )
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions
        self._scaled_action.copy_(actions)
        torch.clamp(self._scaled_action, min=-self._action_clip, max=self._action_clip, out=self._scaled_action)
        self._scaled_action[:, :3].mul_(self._position_scale)
        if self.control_mode == "pose":
            self._scaled_action[:, 3:].mul_(self._orientation_scale)

        entity = self.resolve_entity(self.entity_name)
        current_pos, current_quat = self._control_point_pose(entity)
        self._update_goal_from_action(current_pos, current_quat, self._scaled_action)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._prev_action[envs_idx] = 0.0
        self._raw_action[envs_idx] = 0.0
        self._processed_action[envs_idx] = 0.0
        self._reset_goal(envs_idx)
        if self._nullspace_reference_pos is not None:
            self._refresh_nullspace_reference(envs_idx)

    def apply_actions(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        current_pos, current_quat = self._control_point_pose(entity)
        full_jacobian = entity.get_jacobian(link=entity.links[self._ee_link_idx], local_point=self._ee_local_pos)[
            :, : self.action_dim
        ]
        jacobian = full_jacobian.index_select(-1, self._dofs_idx_name_order)
        full_dofs_vel = entity.get_dofs_vel()
        task_velocity = (full_jacobian @ full_dofs_vel.unsqueeze(-1)).squeeze(-1)
        if self.control_mode == "position":
            task_error = self._goal_pos - current_pos
        else:
            task_error = torch.cat(
                [self._goal_pos - current_pos, self._orientation_error(self._goal_quat, current_quat)],
                dim=-1,
            )
        desired_acceleration = self._kp * task_error - self._kd * task_velocity
        invalid_envs = (
            ~torch.isfinite(full_jacobian).all(dim=(-2, -1))
            | ~torch.isfinite(full_dofs_vel).all(dim=-1)
            | ~torch.isfinite(desired_acceleration).all(dim=-1)
        )

        if self._needs_mass_matrix:
            mass_matrix = entity.get_mass_mat()
            mass_matrix = mass_matrix.index_select(1, self.dofs_idx_local).index_select(2, self.dofs_idx_local)
            mass_solve_jacobian_t, invalid_mass_solve = _solve_mass_matrix_jacobian_transpose(
                mass_matrix,
                jacobian,
            )
            invalid_envs |= invalid_mass_solve
            mass_solve_jacobian_t.masked_fill_(invalid_envs[:, None, None], 0.0)
            if self.inertial_decoupling == "full" or self.control_mode == "position" or self.nullspace_kp is not None:
                task_inertia, invalid_task_inertia = _damped_pseudoinverse(
                    jacobian @ mass_solve_jacobian_t,
                    self._task_identity,
                    self.operational_inertia_damping,
                )
                invalid_envs |= invalid_task_inertia
            if self.nullspace_kp is not None:
                jacobian_bar = mass_solve_jacobian_t @ task_inertia
                nullspace = self._dofs_identity - jacobian_bar @ jacobian
        if self.inertial_decoupling == "none":
            desired_wrench = desired_acceleration
        elif self.inertial_decoupling == "full" or self.control_mode == "position":
            desired_wrench = (task_inertia @ desired_acceleration.unsqueeze(-1)).squeeze(-1)
        else:
            block_identity = self._task_identity[:, :3, :3]
            position_inertia, invalid_position_inertia = _damped_pseudoinverse(
                jacobian[:, :3] @ mass_solve_jacobian_t[:, :, :3],
                block_identity,
                self.operational_inertia_damping,
            )
            orientation_inertia, invalid_orientation_inertia = _damped_pseudoinverse(
                jacobian[:, 3:] @ mass_solve_jacobian_t[:, :, 3:],
                block_identity,
                self.operational_inertia_damping,
            )
            invalid_envs |= invalid_position_inertia | invalid_orientation_inertia
            desired_wrench = torch.cat(
                [
                    (position_inertia @ desired_acceleration[:, :3].unsqueeze(-1)).squeeze(-1),
                    (orientation_inertia @ desired_acceleration[:, 3:].unsqueeze(-1)).squeeze(-1),
                ],
                dim=-1,
            )
        dofs_force = (jacobian.transpose(-2, -1) @ desired_wrench.unsqueeze(-1)).squeeze(-1)

        if self.nullspace_kp is not None:
            dofs_pos = entity.get_dofs_pos(dofs_idx_local=self.genesis_dofs_idx)
            dofs_vel = entity.get_dofs_vel(dofs_idx_local=self.genesis_dofs_idx)
            self._nullspace_acceleration[:] = (
                self.nullspace_kp * (self._nullspace_reference_pos - dofs_pos) - self._nullspace_kd * dofs_vel
            )
            nullspace_force = mass_matrix @ self._nullspace_acceleration.unsqueeze(-1)
            dofs_force += (nullspace.transpose(-2, -1) @ nullspace_force).squeeze(-1)

        invalid_envs |= ~torch.isfinite(dofs_force).all(dim=-1)
        dofs_force.masked_fill_(invalid_envs[:, None], 0.0)
        self._processed_action.copy_(dofs_force)
        self.verify_command_shape(self._processed_action)
        entity.control_dofs_force(self._processed_action, dofs_idx_local=self.genesis_dofs_idx)


@ACTION_TERM_REGISTRY.register()
class IKPoseController(ActionTerm):
    """Absolute task-space pose controller: solve IK for a base-frame target pose each step.

    The action is a base-frame target pose ``[x, y, z, qw, qx, qy, qz]`` (wxyz) for the
    end-effector link, optionally expressed at a virtual tool frame offset from that link.
    Each control step the target is transformed to the world frame, the tool offset is
    removed, and Genesis' IK is solved from the *measured* joint configuration so the
    resulting joint targets stay on the current elbow branch. Policies that emit
    accumulated Cartesian targets (RobotArena-style delta accumulation, DROID Cartesian
    velocity effectors) drive this term; a policy that emits per-step deltas relative to
    the measured pose should use :class:`DifferentialIKController` instead.

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    dofs_name: list[str]
        The arm DOFs the IK solution is applied to.
    ee_link_name: str
        The end-effector link the target pose refers to.
    ee_link_offset_pos: tuple[float, float, float]
        Position of the virtual tool frame in the end-effector link frame. The link is
        driven to the pose that places this offset frame at the target.
    ee_link_offset_quat: tuple[float, float, float, float]
        Orientation (wxyz) of the virtual tool frame in the end-effector link frame.
    max_joint_delta: float | None
        Per-step clamp (rad) on every joint's change from its measured position; also
        turns an IK branch discontinuity into a bounded step. ``None`` disables it.
    joint_limit_margin: float
        Keep commanded joints at least this far (rad) inside their limits.
    max_ik_position_error: float | None
        Reject an IK solution whose position residual (m) exceeds this and hold the
        measured arm pose instead. ``None`` disables the check.
    max_ik_rotation_error: float | None
        Same for the rotation residual (rad).
    """

    ee_link_name: str = "gripper_link"
    ee_link_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ee_link_offset_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    # Optional safety limits for callers that execute velocity-policy targets; disabled by default.
    max_joint_delta: float | None = None
    joint_limit_margin: float = 0.0
    max_ik_position_error: float | None = None
    max_ik_rotation_error: float | None = None

    def __init__(self, env: EnvBase, options: ActionTermOptions):
        super().__init__(env=env, options=options)
        self._ee_link = None
        self._ee_link_offset_pos_inv = None
        self._ee_link_offset_quat_inv = None

    def build(self) -> None:
        super().build()
        if self.entity.n_qs != self.entity.n_dofs:
            # The IK seed and solution are full Genesis qpos vectors; indexing them by DOF index
            # is only valid when qpos and DOF indices coincide (no free/ball joints).
            raise ValueError(
                f"{term_subject(self)} requires a fixed-base entity with one qpos per DOF; "
                f"it has n_qs={self.entity.n_qs}, n_dofs={self.entity.n_dofs}."
            )
        self._all_qs_idx = torch.arange(self.entity.n_qs, device=self.device)
        self._ee_link = self.entity.get_link(self.ee_link_name)
        offset_pos = torch.as_tensor(self.ee_link_offset_pos, dtype=torch.float32, device=self.device).unsqueeze(0)
        offset_quat = torch.as_tensor(self.ee_link_offset_quat, dtype=torch.float32, device=self.device).unsqueeze(0)
        offset_quat = offset_quat / offset_quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self._ee_link_offset_quat_inv = inv_quat(offset_quat)
        self._ee_link_offset_pos_inv = transform_by_quat(-offset_pos, self._ee_link_offset_quat_inv)

    @property
    def action_dim(self) -> int:
        return 7

    def compute(self, actions: torch.Tensor) -> None:
        assert actions.shape == (
            self.num_envs,
            self.action_dim,
        ), f"Expected actions shape: (num_envs, {self.action_dim}), got {actions.shape=}"
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions

        target_pos_b = actions[:, :3]
        target_quat_b = actions[:, 3:7]
        target_quat_b = target_quat_b / target_quat_b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        # The target refers to the virtual tool frame; drive the EE link to the
        # pose that places that offset frame at the requested target.
        target_pos_w, target_quat_w = combine_frame_transforms(
            self.entity.get_pos(),
            self.entity.get_quat(),
            target_pos_b,
            target_quat_b,
        )
        target_pos_w, target_quat_w = combine_frame_transforms(
            target_pos_w,
            target_quat_w,
            self._ee_link_offset_pos_inv,
            self._ee_link_offset_quat_inv,
        )

        # Genesis-local indices: read through dofs_idx_local, never index a dofs_name-ordered array.
        current_arm = self.entity.get_dofs_pos(dofs_idx_local=self.genesis_dofs_idx)
        # Full qpos in Genesis-native order, the layout inverse_kinematics expects for init_qpos.
        current_qpos = self.entity.get_qpos(qs_idx_local=self._all_qs_idx)
        ik_result = self.entity.inverse_kinematics(
            self.ee_link_name,
            poss=target_pos_w,
            quats=target_quat_w,
            # Explicitly seed from the measured configuration. This prevents
            # random-restart IK from selecting a distant Franka elbow branch.
            init_qpos=current_qpos,
            dofs_idx_local=self.genesis_dofs_idx,
            return_error=True,
        )
        target_q, ik_error = ik_result
        if not isinstance(target_q, torch.Tensor):
            target_q = torch.as_tensor(target_q, dtype=torch.float32, device=self.device)
        target_q = target_q.to(device=self.device, dtype=torch.float32)
        if target_q.ndim == 1:
            target_q = target_q.unsqueeze(0)

        # inverse_kinematics returns the full Genesis-native qpos (n_qs == n_dofs, checked in build()).
        processed = target_q[:, self.dofs_idx_local]
        if not isinstance(ik_error, torch.Tensor):
            ik_error = torch.as_tensor(ik_error, dtype=torch.float32, device=self.device)
        ik_error = ik_error.to(device=self.device, dtype=torch.float32)
        if ik_error.ndim == 1:
            ik_error = ik_error.unsqueeze(0)

        finite = torch.isfinite(processed).all(dim=-1) & torch.isfinite(ik_error).all(dim=-1)
        acceptable = finite.clone()
        if self.max_ik_position_error is not None:
            acceptable &= ik_error[..., :3].norm(dim=-1) <= float(self.max_ik_position_error)
        if self.max_ik_rotation_error is not None:
            acceptable &= ik_error[..., 3:6].norm(dim=-1) <= float(self.max_ik_rotation_error)
        if not acceptable.all():
            # Never pass a failed IK result into the rigid solver: one NaN
            # position target permanently poisons the simulation state. Hold
            # the affected arm at its measured joints and continue instead.
            processed = torch.where(acceptable.unsqueeze(-1), processed, current_arm)
            rejected = ~acceptable
            position_residual = ik_error[..., :3].norm(dim=-1)
            rotation_residual = ik_error[..., 3:6].norm(dim=-1)
            rejected_position_residual = position_residual[rejected].detach().cpu().tolist()
            rejected_rotation_residual = rotation_residual[rejected].detach().cpu().tolist()
            rejected_target_pos_b = target_pos_b[rejected].detach().cpu().tolist()
            rejected_target_quat_b = target_quat_b[rejected].detach().cpu().tolist()
            en.logger.warning(
                f"{term_subject(self)} rejected an invalid or excessive-residual IK target; "
                f"holding the current arm pose for {int(rejected.sum().item())} environment(s). "
                f"position_residual_m={rejected_position_residual} "
                f"(limit={self.max_ik_position_error}), "
                f"rotation_residual_rad={rejected_rotation_residual} "
                f"(limit={self.max_ik_rotation_error}), "
                f"target_pos_base_m={rejected_target_pos_b}, "
                f"target_quat_base_wxyz={rejected_target_quat_b}."
            )

        # DROID's RobotIKSolver limits every Franka joint to 0.2 rad per
        # Cartesian command. Applying the same bound here also makes an IK
        # branch discontinuity harmless instead of a full-arm jump.
        if self.max_joint_delta is not None:
            max_delta = float(self.max_joint_delta)
            processed = current_arm + torch.clamp(processed - current_arm, -max_delta, max_delta)

        # DROID's dm_robotics Cartesian effector keeps 0.3 rad away from joint
        # limits. Clamp the commanded target to the same interior band. If the
        # measured state is already outside it, max_joint_delta above moves it
        # back gradually rather than teleporting it.
        if self.joint_limit_margin > 0.0:
            lower, upper = self.entity.get_dofs_limit(dofs_idx_local=self.genesis_dofs_idx)
            lower = torch.as_tensor(lower, dtype=torch.float32, device=self.device)
            upper = torch.as_tensor(upper, dtype=torch.float32, device=self.device)
            safe_lower = lower + float(self.joint_limit_margin)
            safe_upper = upper - float(self.joint_limit_margin)
            processed = torch.maximum(safe_lower, torch.minimum(processed, safe_upper))
            if self.max_joint_delta is not None:
                max_delta = float(self.max_joint_delta)
                processed = current_arm + torch.clamp(processed - current_arm, -max_delta, max_delta)
        self._processed_action = processed

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._prev_action[envs_idx] = 0.0
        self._raw_action[envs_idx] = 0.0

    def apply_actions(self) -> None:
        self.entity.control_dofs_pos(self._processed_action, dofs_idx_local=self.genesis_dofs_idx)
