"""OmniRetarget optimization-based retargeter."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
from genesis.utils.geom import inv_transform_by_trans_quat, rotvec_to_quat, transform_quat_by_quat

import eden as en
from eden.extensions.retargeting import RetargetingBase
from eden.options.extensions.retargeting import RetargeterOptions
from eden.utils.geom import quat_to_R, R_to_quat
from eden.utils.pc import get_entity_surface_points

from eden.extensions.retargeting.omniretarget.utils import (
    compute_delaunay_simplices,
    get_adjacency_list,
    calculate_laplacian_matrix,
)


class OmniRetargeter(RetargetingBase):
    """Interaction-mesh retargeter backed by Eden/Genesis.

    Works on a floating-base entity (qpos ``[pos, quat_wxyz, joints]``) and on a fixed-base or
    welded one (qpos ``joints`` only). Every name below is checked at construction: a link, DOF,
    demo joint or ``foot_links`` key the entity or config does not have raises, naming it.

    Parameters
    ----------
    foot_links : dict[str, str]
        ``{virtual_name: entity_link_name}`` for foot contact points.
        A plain list ``[link_name, ...]`` is also accepted (virtual name = link name).
    demo_joints : list[str]
        Joint names of the demo motion's joint axis (e.g. SMPL-H).
    laplacian_match_links : dict[str, str]
        ``{demo_joint_name: entity_link_name}`` for the Laplacian mesh.
    link_offsets : dict[str, np.ndarray]
        ``{name: local_offset_xyz}`` for virtual offset points.
    manual_lb, manual_ub : dict[str, float]
        Manual joint-limit overrides keyed by DOF name.
    manual_cost : dict[str, float]
        Per-joint cost weights keyed by DOF name.
    nominal_tracking_dofs : list[str]
        DOF names for nominal-pose tracking.
    foot_sticking_map : dict[str, list[str]]
        Maps sticking flag keys (e.g. ``"L_Toe"``) to lists of
        ``foot_links`` keys to constrain when that flag is active.
    laplacian_weights : float
        Weight of every interaction-mesh vertex's Laplacian-deformation term.
    smooth_weight : float
        Weight pulling each frame's solution toward the previous frame's.
    step_size : float
        Trust-region radius for SQP.
    foot_sticking_tolerance : float
    foot_z_sticking_tolerance : float
        Tolerance for optional Z sticking during active foot contact.
    foot_ground_z : float
        World-frame ground plane height used by optional foot-floor constraint.
    foot_ground_margin : float
        Extra non-penetration margin above ``foot_ground_z``.
    collision_detection_threshold : float
    penetration_tolerance : float
    w_nominal_tracking_init : float
    nominal_tracking_tau : float
        Exponential decay constant for nominal tracking weight.
    activate_foot_sticking : bool
    activate_foot_z_sticking : bool
        Keep a sticking foot's z close to the previous frame's, to reduce vertical jitter during stance.
    activate_foot_ground_constraint : bool
        Bound every ``foot_links`` point above ``foot_ground_z + foot_ground_margin`` on every frame.
    activate_obj_non_penetration : bool
    activate_joint_limits : bool
    w_root : float
        Weight of the root-position tracking term (floating base only; the first
        ``laplacian_match_links`` entry is the root).
    n_iter_init, n_iter : int
        Maximum SQP iterations on the first frame and on every later one (both >= 1).
    """

    # -- configurable: task description ------------------------------------
    foot_links: dict[str, str] = {}
    demo_joints: list = []
    laplacian_match_links: dict = {}
    link_offsets: dict[str, np.ndarray] = {}
    manual_lb: dict[str, float] = {}
    manual_ub: dict[str, float] = {}
    manual_cost: dict[str, float] = {}
    nominal_tracking_dofs: list = []
    foot_sticking_map: dict[str, list[str]] = {}

    # -- configurable: solver hyper-parameters -----------------------------
    laplacian_weights: float = 10.0
    smooth_weight: float = 0.2
    step_size: float = 0.2
    foot_sticking_tolerance: float = 1e-3
    foot_z_sticking_tolerance: float = 5e-3
    foot_ground_z: float = 0.0
    foot_ground_margin: float = 0.0
    collision_detection_threshold: float = 0.1
    penetration_tolerance: float = 1e-3
    w_nominal_tracking_init: float = 5.0
    nominal_tracking_tau: float = 10.0
    activate_foot_sticking: bool = True
    activate_foot_z_sticking: bool = False
    activate_foot_ground_constraint: bool = False
    activate_obj_non_penetration: bool = True
    activate_joint_limits: bool = True
    w_root: float = 100.0
    n_iter_init: int = 50
    n_iter: int = 10

    def __init__(self, env, options: RetargeterOptions) -> None:
        super().__init__(env, options)

        # -- normalize mutable fields ------------------------------------------
        if isinstance(self.foot_links, list):
            self.foot_links = {k: k for k in self.foot_links}
        for name in ("n_iter_init", "n_iter"):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"OmniRetargeter: {name}={getattr(self, name)!r} runs no SQP iteration, so no frame would "
                    f"be solved. Pass {name} >= 1 (the defaults are n_iter_init=50, n_iter=10)."
                )
        self._refuse_unknown(
            "laplacian_match_links keys",
            self.laplacian_match_links,
            self.demo_joints,
            "demo_joints (the demo's joint axis)",
        )
        self.smplh_mapped_joint_indices: list[int] = [
            self.demo_joints.index(name) for name in self.laplacian_match_links
        ]

        # -- validate link names against entity ---------------------------------
        available_links = [link.name for link in self.entity.links]
        self._refuse_unknown(
            "laplacian_match_links values",
            self.laplacian_match_links.values(),
            available_links,
            f"the links of entity '{self.entity_name}'",
        )
        self._refuse_unknown(
            "foot_links values", self.foot_links.values(), available_links, f"the links of entity '{self.entity_name}'"
        )
        self._refuse_unknown(
            "foot_sticking_map entries",
            [key for keys in self.foot_sticking_map.values() for key in keys],
            self.foot_links,
            "the foot_links keys",
        )

        # -- DOF layout --------------------------------------------------------
        #
        # Eden floating-base DOF layout:
        #   qpos:  [base_pos(3), base_quat(4), joints(N)]  → total 7 + N
        #   DOF:   [base_lin(3), base_ang(3),  joints(N)]  → total 6 + N
        #
        # The optimisation variable ``dqa`` lives in DOF (tangent) space.
        # ---------------------------------------------------------------
        self.dofs_name: list[str] = list(self.entity.dofs_name)
        self.n_joints: int = len(self.entity.dofs_name)
        self.nq_opt: int = self.entity.n_dofs
        # 6 base tangent DOFs only when a free root exists. ``has_immobile_base`` (not
        # ``is_fixed_base``) is the question: a welded child's root joint is removed by
        # ``attach``, so its qpos and DOF vectors have no base prefix either.
        self._joint_dof_start: int = 0 if self.entity.has_immobile_base else 6
        # The same split in qpos: ``[pos(3), quat_wxyz(4), joints]``, or just ``joints``.
        self._joint_q_start: int = 0 if self.entity.has_immobile_base else 7
        self.nq: int = self._joint_q_start + self.n_joints
        # ``_integrate_dqa`` adds a step to each joint q, which only holds when every joint has
        # one q and one DOF; a ball joint (4 q / 3 DOFs) or a ``dofs_name`` that leaves joints
        # out would put every slice past it in the wrong place.
        expected_n_dofs = self._joint_dof_start + self.n_joints
        if self.nq != self.entity.n_qs or expected_n_dofs != self.nq_opt:
            multi_dof = sorted({name for name in self.dofs_name if self.dofs_name.count(name) > 1})
            raise ValueError(
                f"OmniRetargeter: entity '{self.entity_name}' has n_qs={self.entity.n_qs} and "
                f"n_dofs={self.nq_opt}, but OmniRetarget handles only "
                f"{'a fixed base' if self.entity.has_immobile_base else 'a free root (7 q / 6 DOFs)'} plus "
                f"one-DOF hinge or slide joints, all listed in dofs_name ({self.n_joints} entries here), "
                f"which would give n_qs={self.nq} and n_dofs={expected_n_dofs}. "
                f"Joints with more than one DOF: {multi_dof or 'none (so dofs_name leaves joints out)'}. "
                f"Retarget an entity built from one-DOF joints, e.g. a ball joint split into three hinges."
            )

        for field in ("manual_lb", "manual_ub", "manual_cost", "nominal_tracking_dofs"):
            self._refuse_unknown(
                field, getattr(self, field), self.dofs_name, f"the dofs_name of entity '{self.entity_name}'"
            )

        # -- joint limits & cost weights (DOF space) ---------------------------
        self._init_limits_and_costs()

        # -- nominal tracking indices (joint DOFs only) ------------------------
        self.nominal_tracking_joint_indices = np.array(
            [self.dofs_name.index(name) for name in self.nominal_tracking_dofs], dtype=int
        )
        self.nominal_tracking_dof_indices = self._joint_dof_start + self.nominal_tracking_joint_indices

    @staticmethod
    def _refuse_unknown(field: str, names, available, where: str) -> None:
        """Raise naming every entry of ``names`` missing from ``available``, rather than let it be skipped.

        Parameters
        ----------
        field : str
            The configured field the names came from, as the reader wrote it.
        names : Iterable[str]
            The names to check.
        available : Iterable[str]
            The names they must come from.
        where : str
            What ``available`` is, for the message.
        """
        available = list(available)
        unknown = [name for name in dict.fromkeys(names) if name not in available]
        if unknown:
            raise ValueError(
                f"OmniRetargeter: {field} {unknown} are not in {where}. Available: {available}. "
                f"Fix the spelling or drop the entry: an unknown name is an error, not skipped."
            )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_limits_and_costs(self) -> None:
        """Set up DOF-space joint limits and per-joint cost weights."""
        # ``get_dofs_limit`` covers the joint DOFs only (not the 6 base DOFs), in ``dofs_name``
        # order, which is the joint order of qpos and of the DOF vector; env 0's row is the limit.
        lower, upper = (bound[0].cpu().numpy() for bound in self.entity.get_dofs_limit())
        self.dqa_lb = np.full(self.nq_opt, -1e6)
        self.dqa_ub = np.full(self.nq_opt, 1e6)
        self.dqa_lb[self._joint_dof_start :] = lower
        self.dqa_ub[self._joint_dof_start :] = upper

        # Manual overrides (keys are DOF names, checked in ``__init__``)
        for name, v in self.manual_lb.items():
            self.dqa_lb[self._joint_dof_start + self.dofs_name.index(name)] = v
        for name, v in self.manual_ub.items():
            self.dqa_ub[self._joint_dof_start + self.dofs_name.index(name)] = v

        # Per-joint cost weights
        self.Q_diag = np.zeros(self.nq_opt)
        for name, v in self.manual_cost.items():
            self.Q_diag[self._joint_dof_start + self.dofs_name.index(name)] = v

    # ------------------------------------------------------------------
    # Eden kinematics helpers
    # ------------------------------------------------------------------

    def _set_entity_qpos(self, q: np.ndarray) -> None:
        """Set the entity configuration and run forward kinematics.

        Parameters
        ----------
        q : np.ndarray
            The ``(nq,)`` qpos: ``[pos, quat_wxyz, joints]`` on a floating base, ``joints`` on a fixed one.
        """
        qpos_t = torch.as_tensor(q, dtype=torch.float32, device=self.entity.device).unsqueeze(0)
        self.entity.set_qpos(qpos_t)

    def _get_link_jacobian_and_pos(
        self, link_name: str, offset: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return positional Jacobian and world position for a link.

        Parameters
        ----------
        link_name : str
            Entity link name.
        offset : (3,) array, optional
            Local-frame offset from the link origin.  When provided, the
            ``local_point`` parameter of ``get_jacobian`` handles the
            velocity transfer internally.

        Returns
        -------
        J : (3, nq_opt)
            Positional Jacobian in DOF space, sliced to optimised DOFs.
        pos : (3,)
            World-frame position (of offset point when *offset* is given).
        """
        link = self.entity.get_link(link_name)

        has_offset = offset is not None and np.any(offset != 0)

        # get_jacobian → (n_envs, 6, n_dofs);  take env 0, convert to numpy
        # for CVXPY.  When local_point is given, Genesis computes the
        # Jacobian at the offset point directly (correct moment arm).
        if has_offset:
            offset_t = torch.as_tensor(offset, dtype=torch.float32, device=self.entity.device)
            J_full_6 = self.entity.get_jacobian(link, local_point=offset_t)[0].cpu().numpy()
        else:
            J_full_6 = self.entity.get_jacobian(link)[0].cpu().numpy()

        # Slice to the DOFs we're optimising (columns are already in
        # dofs_name order thanks to RigidEntity.get_jacobian reordering).
        J_6 = J_full_6[:, : self.nq_opt]

        return J_6[:3, :], self._link_point(link, offset).cpu().numpy()

    @staticmethod
    def _link_point(link, offset: np.ndarray | None) -> torch.Tensor:
        """World position of the point at local ``offset`` on ``link`` (its origin for no or a zero offset).

        Parameters
        ----------
        link : RigidLink
            The link the point rides on.
        offset : (3,) array or None
            The point in the link's local frame.
        """
        # relative=False keeps the world frame (Genesis >=1.1.1 link getters default to the
        # user frame, alignment stripped).
        pos = link.get_pos(relative=False).squeeze(0)
        if offset is None or not np.any(offset != 0):
            return pos
        R = quat_to_R(link.get_quat(relative=False).squeeze(0))
        return pos + R @ torch.as_tensor(offset, dtype=pos.dtype, device=pos.device)

    def _get_manipulator_jacobians(
        self,
        links: dict[str, str],
        obj_frame: bool = False,
        obj_pos: np.ndarray | None = None,
        obj_quat: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Compute positional Jacobians & positions for a set of links.

        Parameters
        ----------
        links : dict
            ``{demo_joint_name: robot_link_name}``.
        obj_frame : bool
            If True, rotate Jacobians/positions into the object's local frame.
        obj_pos, obj_quat : array-like
            Object world pose (required when *obj_frame* is True).
            Quaternion is ``(w, x, y, z)``.

        Returns
        -------
        J_dict : {name: (3, nq_opt)}
        p_dict : {name: (3,)}
        """
        if obj_frame:
            assert obj_quat is not None, "obj_quat is required when obj_frame is True"
            assert obj_pos is not None, "obj_pos is required when obj_frame is True"
            R_inv = quat_to_R(obj_quat).T

        J_dict: dict[str, np.ndarray] = {}
        p_dict: dict[str, np.ndarray] = {}

        for name, link_name in links.items():
            J, pos = self._get_link_jacobian_and_pos(link_name, offset=self.link_offsets.get(name))
            if obj_frame:
                J = R_inv @ J
                pos = R_inv @ (pos - obj_pos)
            J_dict[name] = J
            p_dict[name] = pos

        return J_dict, p_dict

    def _get_link_positions(
        self,
        q: np.ndarray,
        links: dict[str, str],
    ) -> dict[str, np.ndarray]:
        """Set qpos and return world-frame positions for *links*."""
        self._set_entity_qpos(q)
        return {
            name: self._link_point(self.entity.get_link(link_name), self.link_offsets.get(name)).cpu().numpy()
            for name, link_name in links.items()
        }

    # ------------------------------------------------------------------
    # Integration: dqa (DOF tangent) → updated qpos
    # ------------------------------------------------------------------

    def _integrate_dqa(self, q: np.ndarray, dqa: np.ndarray) -> np.ndarray:
        """Apply a tangent-space step *dqa* to configuration *q*.

        Parameters
        ----------
        q : (nq,)
            Current qpos: ``[pos(3), quat_wxyz(4), joints(N)]``, or ``joints(N)`` on a fixed base.
        dqa : (nq_opt,)
            Tangent-space update.

        Returns
        -------
        q_new : same shape as *q*.
        """
        q_new = np.copy(q)
        q_new[self._joint_q_start :] += dqa[self._joint_dof_start :]
        if not self.entity.has_immobile_base:
            q_new[0:3] += dqa[0:3]
            # Base rotation via exponential map (rotation vector → quaternion)
            # transform_quat_by_quat(v, u) = u * v, so this left-multiplies
            # step_quat onto the current orientation (world-frame angular vel).
            step_quat = rotvec_to_quat(dqa[3:6])
            q_new[3:7] = transform_quat_by_quat(q_new[3:7], step_quat)
            q_new[3:7] /= np.linalg.norm(q_new[3:7]) + 1e-12
        return q_new

    # ------------------------------------------------------------------
    # Helper: qpos → DOF-space values (for costs / tracking)
    # ------------------------------------------------------------------

    def _qpos_to_dof_values(self, q: np.ndarray) -> np.ndarray:
        """Extract DOF-aligned position values from *q* (qpos).

        Base rotation (DOF 3-5) is set to zero because there is no unique
        scalar representation; only base position and joint angles are
        meaningful for position-level costs.
        """
        vals = np.zeros(self.nq_opt)
        vals[self._joint_dof_start :] = q[self._joint_q_start :]
        if not self.entity.has_immobile_base:
            vals[0:3] = q[0:3]
        return vals

    # ------------------------------------------------------------------
    # Core retargeting
    # ------------------------------------------------------------------

    def _compute_q_init(
        self,
        human_joint_motions: np.ndarray,
        object_name: str | None = None,
        object_poses: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute a default initial qpos from the first frame.

        Sets the base position to the human root, orients the robot to
        face the object (when applicable), and zeros all joint angles.

        Parameters
        ----------
        human_joint_motions : np.ndarray
            Human joint positions, shape ``(T, n_demo_joints, 3)``.
        object_name : str or None
            Name of the object to face, or None to skip orientation.
        object_poses : np.ndarray or None
            Object poses, shape ``(T, 7)`` as ``[pos(3), quat_wxyz(4)]``.

        Returns
        -------
        q_init : np.ndarray
            The initial qpos, shape ``(nq,)``: ``7 + n_joints`` on a floating base, ``n_joints`` on a fixed one.
        """
        root_pos = human_joint_motions[0, 0, :3]

        if object_name is not None and object_poses is not None:
            # Orient robot to face the object
            obj_pos_0 = object_poses[0, :3]
            d = obj_pos_0[:2] - root_pos[:2]
            d = d / (np.linalg.norm(d) + 1e-12)
            x_axis = np.array([d[0], d[1], 0.0])
            z_axis = np.array([0.0, 0.0, 1.0])
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis) + 1e-12
            rot_mat = np.column_stack([x_axis, y_axis, z_axis])
            quat = R_to_quat(rot_mat)
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0])

        if not self.entity.has_immobile_base:
            return np.concatenate([root_pos, quat, np.zeros(self.n_joints)])
        else:
            return np.zeros(self.n_joints)

    def retarget_motion(
        self,
        human_joint_motions: np.ndarray,
        foot_sticking_sequences: list[dict[str, bool]],
        object_name: str | None = None,
        object_poses: np.ndarray | None = None,
        scale_factor: float = 1.0,
        q_a_init: np.ndarray | None = None,
        q_nominal_list: np.ndarray | None = None,
        n_surface_points: int = 100,
        constant_nominal_weight: bool = True,
    ) -> np.ndarray:
        """Run the full retargeting loop over all frames.

        Parameters
        ----------
        human_joint_motions : np.ndarray
            Human joint positions, shape ``(T, n_demo_joints, 3)``.
        foot_sticking_sequences : list of dicts
            ``{toe_name: bool}`` per frame.
        object_name : str or None
            Entity name in the environment (e.g. ``"largebox"``), or
            ``None`` for retargeting without an interaction object.
        object_poses : (T, 7) or None
            ``[pos(3), quat_wxyz(4)]``.  Required when *object_name* is
            not ``None``.
        scale_factor : float
            Ratio ``robot_height / human_height`` used to scale object
            surface points to the demo (human) frame.
        q_a_init : np.ndarray or None
            Optional initial qpos, shape ``(nq,)``: ``7 + n_joints`` on a floating base,
            ``n_joints`` on a fixed one. Defaults to the first nominal frame when
            ``q_nominal_list`` is given, else to :meth:`_compute_q_init`.
        q_nominal_list : np.ndarray or None
            Optional nominal trajectory, shape ``(T, nq)``.
        n_surface_points : int
            Number of surface points to sample from the object entity.
        constant_nominal_weight : bool
            If True, use a constant nominal tracking weight; if False,
            apply exponential decay over frames.

        Returns
        -------
        retargeted : (T, nq)
        """
        num_frames = human_joint_motions.shape[0]
        self._check_motion_inputs(
            human_joint_motions, foot_sticking_sequences, object_name, object_poses, q_a_init, q_nominal_list
        )

        # -- Obtain object surface points from entity --------------------------
        if object_name is not None:
            obj_entity = self.env.entities[object_name]
            points_list = get_entity_surface_points(obj_entity, n_points=n_surface_points)
            object_points_local = np.concatenate([p.cpu().numpy() for p in points_list if p is not None], axis=0)
            object_points_local_demo = object_points_local * scale_factor
        else:
            object_points_local = np.zeros((0, 3))
            object_points_local_demo = np.zeros((0, 3))

        if q_a_init is None:
            if q_nominal_list is not None:
                q_a_init = q_nominal_list[0]
            else:
                q_a_init = self._compute_q_init(human_joint_motions, object_name, object_poses)
        # The running solution: each frame starts from the previous frame's result, which is also
        # what its smoothness and foot-sticking terms are measured against.
        q = np.array(q_a_init, dtype=float)
        retargeted_motions = []

        en.logger.info(f"Starting motion retargeting for {num_frames} frames...")

        for i in tqdm(range(num_frames)):
            if object_poses is not None:
                obj_trans = object_poses[i, :3]
                obj_quat = object_poses[i, 3:]
            else:
                obj_trans = np.zeros(3)
                obj_quat = np.array([1.0, 0.0, 0.0, 0.0])

            human_mapped = human_joint_motions[i, self.smplh_mapped_joint_indices]

            # Transform to object frame when applicable
            if object_name is None:
                human_in_obj = human_mapped
            else:
                human_in_obj = inv_transform_by_trans_quat(human_mapped, obj_trans, obj_quat)

            # Interaction mesh
            all_pts = np.vstack([human_in_obj, object_points_local_demo])
            simplices = compute_delaunay_simplices(all_pts)
            adj_list = get_adjacency_list(simplices, len(all_pts))
            target_lap = calculate_laplacian_matrix(all_pts, adj_list) @ all_pts

            # Nominal weight schedule
            if constant_nominal_weight:
                w_nom = self.w_nominal_tracking_init
            else:
                w_nom = self.w_nominal_tracking_init * np.exp(-i / self.nominal_tracking_tau)

            q_nominal_joints = None
            if q_nominal_list is not None:
                q_nominal_joints = np.asarray(q_nominal_list)[i, self._joint_q_start :]

            q, _cost = self.iterate(
                q_n=q,
                q_t_last=q,
                target_laplacian=target_lap,
                adj_list=adj_list,
                obj_pts_local=object_points_local,
                foot_sticking=foot_sticking_sequences[i],
                w_nominal_tracking=w_nom,
                q_nominal_joints=q_nominal_joints,
                init_t=(i == 0),
                n_iter=self.n_iter_init if i == 0 else self.n_iter,
                obj_pos=obj_trans,
                obj_quat=obj_quat,
                target_root_pos=(human_in_obj[0] if len(human_in_obj) > 0 else None),
                object_name=object_name,
            )

            retargeted_motions.append(q)

        return np.array(retargeted_motions)

    def _check_motion_inputs(
        self,
        human_joint_motions: np.ndarray,
        foot_sticking_sequences: list[dict[str, bool]],
        object_name: str | None,
        object_poses: np.ndarray | None,
        q_a_init: np.ndarray | None,
        q_nominal_list: np.ndarray | None,
    ) -> None:
        """Refuse inputs whose shapes disagree, before any frame is solved.

        Parameters
        ----------
        human_joint_motions : np.ndarray
            ``(T, len(demo_joints), 3)``.
        foot_sticking_sequences : list of dicts
            One flag dict per frame.
        object_name : str or None
            The interaction object's entity name.
        object_poses : np.ndarray or None
            ``(T, 7)`` when ``object_name`` is given.
        q_a_init : np.ndarray or None
            ``(nq,)``.
        q_nominal_list : np.ndarray or None
            ``(T, nq)``.
        """
        where = "OmniRetargeter.retarget_motion"
        shape = np.shape(human_joint_motions)
        if len(shape) != 3 or shape[1:] != (len(self.demo_joints), 3):
            raise ValueError(
                f"{where}: human_joint_motions has shape {shape}, but must be "
                f"(T, {len(self.demo_joints)}, 3): one xyz per demo_joints entry, in demo_joints order."
            )
        num_frames = shape[0]
        if len(foot_sticking_sequences) != num_frames:
            raise ValueError(
                f"{where}: foot_sticking_sequences has {len(foot_sticking_sequences)} "
                f"entries for {num_frames} frames of human_joint_motions; pass one flag dict per frame."
            )
        if self.activate_foot_sticking:
            # A flag is read only through ``foot_sticking_map``, so one the map does not key would
            # leave its foot never sticking, silently.
            unmapped = sorted({key for flags in foot_sticking_sequences for key in flags} - set(self.foot_sticking_map))
            if unmapped:
                raise ValueError(
                    f"{where}: foot_sticking_sequences flags {unmapped} are not "
                    f"foot_sticking_map keys {sorted(self.foot_sticking_map)}, so those feet would never stick. "
                    f"Rename the flags to the map's keys, key foot_sticking_map by the toe names the flags were "
                    f"computed for (TOE_NAMES_BY_FORMAT[<format>] in eden.extensions.retargeting.constants), "
                    f"or pass activate_foot_sticking=False to retarget without foot sticking."
                )
        if object_name is not None:
            if object_poses is None:
                raise ValueError("object_poses is required when object_name is provided.")
            if object_name not in self.env.entities:
                raise ValueError(
                    f"{where}: object_name {object_name!r} is not an entity of this env. "
                    f"Entities: {sorted(self.env.entities)}."
                )
        if object_poses is not None and np.shape(object_poses) != (num_frames, 7):
            raise ValueError(
                f"{where}: object_poses has shape {np.shape(object_poses)}, but must be "
                f"({num_frames}, 7): [pos(3), quat_wxyz(4)] per frame."
            )
        if q_a_init is not None and np.shape(q_a_init) != (self.nq,):
            raise ValueError(
                f"{where}: q_a_init has shape {np.shape(q_a_init)}, but entity "
                f"'{self.entity_name}' has qpos size {self._qpos_layout()}."
            )
        if q_nominal_list is not None and np.shape(q_nominal_list) != (num_frames, self.nq):
            raise ValueError(
                f"{where}: q_nominal_list has shape {np.shape(q_nominal_list)}, but must "
                f"be ({num_frames}, {self.nq}): one qpos per frame, of size {self._qpos_layout()}."
            )

    def _qpos_layout(self) -> str:
        """This entity's qpos size and layout, for an error message."""
        if self.entity.has_immobile_base:
            return f"{self.nq} = n_joints"
        return f"{self.nq} = 7 + n_joints: [pos, quat_wxyz, joints]"

    # ------------------------------------------------------------------
    # SQP outer loop
    # ------------------------------------------------------------------

    def iterate(
        self,
        q_n: np.ndarray,
        q_t_last: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[set],
        obj_pts_local: np.ndarray,
        foot_sticking: dict[str, bool],
        w_nominal_tracking: float,
        q_nominal_joints: np.ndarray | None,
        init_t: bool,
        n_iter: int,
        obj_pos: np.ndarray,
        obj_quat: np.ndarray,
        target_root_pos: np.ndarray | None = None,
        object_name: str | None = None,
    ) -> tuple[np.ndarray, float]:
        # Pre-compute foot positions at q_t_last (for sticking constraint)
        p_foot_t_last = self._get_link_positions(q_t_last, self.foot_links)

        last_cost = np.inf
        for _ in range(n_iter):
            q_n, cost = self.solve_single_iteration(
                q_n=q_n,
                q_t_last=q_t_last,
                target_laplacian=target_laplacian,
                adj_list=adj_list,
                obj_pts_local=obj_pts_local,
                foot_sticking=foot_sticking,
                w_nominal_tracking=w_nominal_tracking,
                q_nominal_joints=q_nominal_joints,
                init_t=init_t,
                obj_pos=obj_pos,
                obj_quat=obj_quat,
                target_root_pos=target_root_pos,
                p_foot_t_last=p_foot_t_last,
                object_name=object_name,
            )
            if np.abs(cost - last_cost) < 1e-4:
                break
            last_cost = cost

        return q_n, cost

    # ------------------------------------------------------------------
    # Single SQP iteration (QP construction & solve)
    # ------------------------------------------------------------------

    def solve_single_iteration(
        self,
        q_n: np.ndarray,
        q_t_last: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[set],
        obj_pts_local: np.ndarray,
        foot_sticking: dict[str, bool],
        w_nominal_tracking: float,
        q_nominal_joints: np.ndarray | None,
        init_t: bool,
        obj_pos: np.ndarray,
        obj_quat: np.ndarray,
        target_root_pos: np.ndarray | None = None,
        p_foot_t_last: dict[str, np.ndarray] | None = None,
        object_name: str | None = None,
    ) -> tuple[np.ndarray, float]:
        """Build & solve a single convex sub-problem.

        Returns ``(q_new, cost)``.
        """
        # ---- 1. Set current configuration ------------------------------------
        q = q_n
        self._set_entity_qpos(q)

        # ---- 2. Jacobians & positions for Laplacian links --------------------
        in_obj_frame = object_name is not None
        J_dict, p_dict = self._get_manipulator_jacobians(
            self.laplacian_match_links,
            obj_frame=in_obj_frame,
            obj_pos=obj_pos,
            obj_quat=obj_quat,
        )

        robot_link_keys = list(self.laplacian_match_links.keys())
        V_r = len(robot_link_keys)
        V_o = len(obj_pts_local)
        V = V_r + V_o

        # Stack Jacobians: robot links have non-zero J, object pts have zero J
        J_V = np.zeros((3 * V, self.nq_opt))
        for idx, key in enumerate(robot_link_keys):
            J_V[3 * idx : 3 * (idx + 1), :] = J_dict[key]
        # Object points: rows stay zero (object doesn't move with robot DOFs)

        robot_pts = np.array([p_dict[k] for k in robot_link_keys])  # (V_r, 3)
        vertices = np.vstack([robot_pts, obj_pts_local])  # (V, 3)

        # ---- 3. Laplacian constraint -----------------------------------------
        L = calculate_laplacian_matrix(vertices, adj_list)
        Kron = sp.kron(L, sp.eye(3, format="csr"), format="csr")
        J_L = Kron @ J_V  # (3V, nq_opt)

        lap0 = L @ vertices
        lap0_vec = lap0.reshape(-1)
        target_lap_vec = target_laplacian.reshape(-1)

        w_v = self.laplacian_weights * np.ones(V)
        sqrt_w3 = np.sqrt(np.repeat(w_v, 3))

        # ---- 4. CVXPY variables & constraints --------------------------------
        dqa = cp.Variable(self.nq_opt, name="dqa")
        lap_var = cp.Variable(3 * V, name="laplacian")

        constraints = [
            cp.Constant(J_L) @ dqa - lap_var == -lap0_vec,
        ]

        # ---- 4a. Joint limits ------------------------------------------------
        if self.activate_joint_limits:
            j_start = self._joint_dof_start
            j_end = j_start + self.n_joints
            joints_dqa = dqa[j_start:j_end]
            joints_curr = q[self._joint_q_start :]
            joints_lb = self.dqa_lb[j_start:j_end]
            joints_ub = self.dqa_ub[j_start:j_end]
            constraints.append(joints_dqa >= joints_lb - joints_curr)
            constraints.append(joints_dqa <= joints_ub - joints_curr)

        # ---- 4b. Trust region (SOC) ------------------------------------------
        soc_constraint = cp.SOC(self.step_size, dqa)
        constraints.append(soc_constraint)

        # ---- 4c. Foot sticking -----------------------------------------------
        if self.activate_foot_sticking and self.foot_sticking_map:
            for sticking_key, foot_link_keys in self.foot_sticking_map.items():
                if not foot_sticking.get(sticking_key, False):
                    continue
                for key in foot_link_keys:
                    link_name = self.foot_links[key]
                    J_foot, p_foot = self._get_link_jacobian_and_pos(link_name, offset=self.link_offsets.get(key))
                    if p_foot_t_last is not None and key in p_foot_t_last:
                        p_prev = p_foot_t_last[key]
                    else:
                        p_prev = p_foot

                    # Constrain XY displacement: foot should stay put
                    # J_xy @ dqa ≈ (p_prev - p_curr)_xy ± tol
                    diff_xy = p_prev[:2] - p_foot[:2]
                    J_xy = J_foot[:2, :]
                    tol = self.foot_sticking_tolerance
                    constraints.append(J_xy @ dqa >= diff_xy - tol)
                    constraints.append(J_xy @ dqa <= diff_xy + tol)

                    # Optional Z sticking during contact.
                    if self.activate_foot_z_sticking:
                        J_z = J_foot[2, :]
                        z_ref = p_prev[2]
                        if self.activate_foot_ground_constraint:
                            z_ref = max(z_ref, self.foot_ground_z + self.foot_ground_margin)
                        diff_z = z_ref - p_foot[2]
                        z_tol = self.foot_z_sticking_tolerance
                        constraints.append(J_z @ dqa >= diff_z - z_tol)
                        constraints.append(J_z @ dqa <= diff_z + z_tol)

        # ---- 4d. Foot-ground non-penetration (world-frame) -------------------
        if self.activate_foot_ground_constraint and self.foot_links:
            z_lb = self.foot_ground_z + self.foot_ground_margin
            for key, link_name in self.foot_links.items():
                J_foot, p_foot = self._get_link_jacobian_and_pos(link_name, offset=self.link_offsets.get(key))
                J_z = J_foot[2, :]
                constraints.append(J_z @ dqa >= z_lb - p_foot[2])

        # ---- 4e. Object non-penetration (point-based) ------------------------
        if self.activate_obj_non_penetration and V_o > 0 and V_r > 0:
            # robot_pts and obj_pts_local are BOTH in the object frame
            # (when object_name is not None) or both in world frame
            # (when object_name is None), so distances are consistent.
            diff = robot_pts[:, None, :] - obj_pts_local[None, :, :]  # (Vr, Vo, 3)
            dists = np.linalg.norm(diff, axis=-1)  # (Vr, Vo)

            close_mask = dists < self.collision_detection_threshold
            close_indices = np.argwhere(close_mask)

            for r_idx, o_idx in close_indices:
                dist = dists[r_idx, o_idx]
                if dist < 1e-6:
                    continue
                normal = diff[r_idx, o_idx] / dist
                J_r = J_V[3 * r_idx : 3 * (r_idx + 1), :]
                constraints.append(normal @ (J_r @ dqa) >= self.penetration_tolerance - dist)

        # ---- 5. Objective ----------------------------------------------------
        obj_terms = []

        # Laplacian deformation
        obj_terms.append(cp.sum_squares(cp.multiply(sqrt_w3, lap_var - target_lap_vec)))

        # Root position tracking
        # p_dict and J_dict are already in the correct frame (object frame
        # when object_name is not None, world frame otherwise).
        # target_root_pos is also in the same frame (human_in_obj[0] when
        # object, raw world position when ground).
        if target_root_pos is not None and V_r > 0 and not self.entity.has_immobile_base:
            root_pos = p_dict[robot_link_keys[0]]
            root_J = J_dict[robot_link_keys[0]]

            pos_err = root_pos - target_root_pos
            obj_terms.append(self.w_root * cp.sum_squares(root_J @ dqa + pos_err))

        # Nominal joint tracking
        if w_nominal_tracking > 0 and q_nominal_joints is not None and len(self.nominal_tracking_joint_indices) > 0:
            idx_j = self.nominal_tracking_joint_indices  # into joint array
            idx_dqa = self.nominal_tracking_dof_indices  # into dqa
            q_curr_joints = q[self._joint_q_start :]
            z = dqa[idx_dqa] - (q_nominal_joints[idx_j] - q_curr_joints[idx_j])
            obj_terms.append(w_nominal_tracking * cp.sum_squares(z))

        # Per-joint regularisation
        Q_sqrt = np.sqrt(np.asarray(self.Q_diag, dtype=float))
        q_dof = self._qpos_to_dof_values(q)
        obj_terms.append(cp.sum_squares(cp.multiply(Q_sqrt, dqa + q_dof)))

        # Temporal smoothness: pull q + dqa toward the previous frame's solution
        dqa_smooth = self._qpos_to_dof_values(q_t_last) - q_dof
        obj_terms.append(self.smooth_weight * cp.sum_squares(dqa - dqa_smooth))

        # ---- 6. Solve --------------------------------------------------------
        problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)

        try:
            problem.solve(solver=cp.CLARABEL)
        except cp.error.SolverError:
            problem.solve(solver=cp.SCS)

        # Fallback: drop SOC trust region on first frame if infeasible
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and init_t:
            constraints = [c for c in constraints if c is not soc_constraint]
            problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
            try:
                problem.solve(solver=cp.CLARABEL)
            except cp.error.SolverError:
                problem.solve(solver=cp.SCS)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            en.logger.warning(f"QP solve failed: {problem.status}")
            return q, float("inf")

        dqa_val = dqa.value
        q_new = self._integrate_dqa(q, dqa_val)

        return q_new, problem.value
