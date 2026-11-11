"""OTRetarget: joint robot and multi-object motion retargeting via optimal transport (arXiv 2609.36602).

Per frame ``t`` the retargeter solves (Eq. 1)::

    min_{q_t, {T^o_t}}  E_int + E_sty_r + E_sty_o + |q_t (-) q_{t-1}|^2      s.t.  C(q_t, {T^o_t})

* ``E_int`` (Eq. 5-6) carries the demonstration's surface interactions. Every probe ``p`` (dense surface
  samples on the robot links and on the objects) reads a proximity triple ``(d, w, n)`` from every
  channel ``c`` (the ground and each object): signed distance, witness (closest surface point) and the
  witness-to-probe direction. Its target triple comes from the human probe it is transported from
  (``src(p)``, one entropic OT plan per pre-assigned part/link group, solved in the T-pose) or, for an
  object probe, from the same point of the demonstrated object. Residuals penalise the depth gap,
  the geodesic witness displacement and the wrong side of the demonstrated contact plane, weighted by
  proximity kernels and normalised per link.
* ``E_sty_r`` / ``E_sty_o`` (Eq. 7) keep the motion style: tracked links follow the human joint
  orientations through T-pose alignment offsets, and positions derived from those orientations by
  forward kinematics over the robot's own link lengths (native scale — nothing is rescaled); each
  object is centred on its demonstrated trajectory.
* ``C`` holds joint position and velocity limits, signed-distance non-penetration of every
  probe-channel pair, capsule self-collision, and per-object displacement/corridor boxes.

The robot pose and every free object pose are solved jointly by a trust-region SQP (Eq. 8): Gauss-Newton
Hessian, linearised constraints, one dense QP per iterate (ProxQP, :mod:`.solver`), 50 iterations on the
cold-started first frame and a few warm-started ones after.

This module is the method alone; it knows nothing of feet or hands. What one kind of embodiment or data needs on top
is an :class:`Addon` (``OTRetargeter.addons``): :class:`~.stance.FloorStance` (legged robots on a floor) and
:class:`~.grasp.HandGrasp` (dexterous hands). ``examples/recipes/retargeting/otretarget/README.md`` tabulates every
departure from the paper, with its measurements.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import eden as en
from eden.extensions.retargeting import RetargetingBase
from eden.extensions.retargeting.otretarget.capsules import build_link_capsules
from eden.extensions.retargeting.otretarget.channels import Channel, MeshChannel, PlaneChannel
from eden.extensions.retargeting.otretarget.kinematics import RigKinematics
from eden.extensions.retargeting.otretarget.probes import (
    sample_human_probes,
    sample_object_probes,
    sample_rig_probes,
    sample_surface,
)
from eden.extensions.retargeting.otretarget.solver import (
    GaussNewton,
    Local,
    Rows,
    pad,
    single_threaded_blas,
    solve_qp,
)
from eden.extensions.retargeting.otretarget.transport import group_correspondences, transport_correspondence
from eden.options.extensions.retargeting import RetargeterOptions
from eden.options.options import DynamicConfigurableOptions
from eden.utils.common import ConfigurableMixin
from eden.utils.geom import (
    R_to_quat,
    axis_angle_from_quat,
    quat_from_axis_angle,
    quat_to_R,
    segment_closest_points,
    skew,
)
from eden.utils.mesh import get_link_mesh

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.extensions.retargeting.otretarget.human import HumanSurfaceMotion


@dataclass
class InteractionObject:
    """A manipulated (or static) object of the demonstration.

    Attributes
    ----------
    name : str
        Channel name (used for the output poses).
    mesh : trimesh.Trimesh
        The object geometry in the retargeted scene, in the object frame.
    poses : np.ndarray
        ``(T, 7)`` demonstrated poses ``[pos, quat_wxyz]`` of the object frame.
    demo_mesh : trimesh.Trimesh, optional
        The geometry the human actually interacted with, when ``mesh`` is a substitute (Sec. III-E).
        Demonstrated witnesses are mapped onto ``mesh`` by an object-to-object transport plan; distances
        keep their metric value.
    fixed : bool
        Hold the object on its demonstrated trajectory instead of making its pose a decision variable.
    weight : float
        The object's ``f_l`` (default 1): multiplies the interaction weights of its probes, as ``link_weights`` does
        for a robot link. It belongs to the object, so a retargeter reused over clips needs no per-clip option.
    """

    name: str
    mesh: trimesh.Trimesh
    poses: np.ndarray
    demo_mesh: trimesh.Trimesh | None = None
    fixed: bool = False
    weight: float = 1.0

    def __post_init__(self) -> None:
        # float64 whatever the source (an npz track is often float32): the rotation converters keep the input dtype.
        self.poses = np.asarray(self.poses, dtype=np.float64)
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError(f"InteractionObject {self.name!r}: weight={self.weight!r} must be finite and >= 0.")


@dataclass
class RetargetResult:
    """Retargeted trajectories.

    Attributes
    ----------
    qpos : np.ndarray
        ``(T, n_q)`` rig configuration (root qpos ``[pos, quat_wxyz, joints]``, then each welded entity's joints).
    entity_qpos : dict[str, np.ndarray]
        The same split per entity, ready for ``entity.set_qpos``.
    object_poses : dict[str, np.ndarray]
        ``(T, 7)`` solved ``[pos, quat_wxyz]`` per object.
    fps : float
        Frame rate.
    info : dict[str, np.ndarray]
        Per-frame solver diagnostics (``iterations``, ``cost`` and its terms ``cost/<term>`` at the last
        linearisation, ``active_pairs``, ``solve_ms``, the number of iterates whose QP needed the
        elastic relaxation, ``qp_relaxed``, or found no step at all, ``qp_failed``, the trust-region steps
        rejected for deepening a penetration, ``steps_rejected``, and the worst probe penetration at the
        solution, ``penetration_mm``), plus what the add-ons report.
    stats : dict[str, float]
        Whole-run figures: ``setup_s`` (probing, transport plans, channel grids), ``solve_s`` (the frame
        loop), ``total_s``, the problem size (``robot_probes``, ``human_probes``, ``object_probes``,
        ``channels``, ``capsule_pairs``, ``n_dof``, ``n_decision``) and ``frames``.
    """

    qpos: np.ndarray
    entity_qpos: dict[str, np.ndarray]
    object_poses: dict[str, np.ndarray]
    fps: float
    info: dict[str, np.ndarray] = field(default_factory=dict)
    stats: dict[str, float] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Write everything to one ``.npz``."""
        arrays = {"qpos": self.qpos, "fps": np.float64(self.fps)}
        arrays |= {f"entity_qpos/{k}": v for k, v in self.entity_qpos.items()}
        arrays |= {f"object_poses/{k}": v for k, v in self.object_poses.items()}
        arrays |= {f"info/{k}": v for k, v in self.info.items()}
        arrays |= {f"stats/{k}": np.float64(v) for k, v in self.stats.items()}
        np.savez_compressed(path, **arrays)


@dataclass
class Targets:
    """Demonstrated triples for one probe set against one channel (channel frame)."""

    distance: np.ndarray  # (N,) demo signed distance, +inf where unknown (beyond d_bar or no source)
    witness: np.ndarray  # (N, 3)
    normal: np.ndarray  # (N, 3) outward surface normal at the witness: sign(d) * n
    # Geodesic-table weights at the witnesses, filled row by row as residuals need them (see ``_geodesic_hat``).
    geodesic_hat: list | None = field(default=None, repr=False)
    # (N,) probes an add-on marks for this frame (``HandGrasp``: the probes that close on the object; None: none)
    closed: np.ndarray | None = field(default=None, repr=False)


@dataclass
class Frame:
    """Targets and bounds of one frame's solve. ``q_prev`` and the previous object poses are what it is smoothed
    toward (``q_prev`` None: not smoothed); ``free_cols`` (velocity columns) are the only ones that move when set, and
    ``stage`` names the solve (``"main"``, or an add-on's stage). ``extra`` holds what add-ons put in for their terms.
    """

    t: int
    q_prev: np.ndarray | None
    obj_prev_pos: np.ndarray
    obj_prev_rot: np.ndarray
    demo_pos: np.ndarray
    demo_rot: np.ndarray
    demo_step: np.ndarray | None
    demo_delta: np.ndarray | None
    root_delta: np.ndarray | None
    robot_t: list[Targets]
    object_t: list[Targets]
    human_t: list[Targets]  # every human probe's triple, per channel
    style_pos: np.ndarray
    style_rot: np.ndarray
    dt: float
    free_cols: np.ndarray | None = None
    stage: str = "main"
    extra: dict = field(default_factory=dict)


@dataclass
class ProbeSet:
    """One probe set (robot or object probes) against one channel at a linearisation point."""

    kind: str  # "robot" | "object"
    ci: int  # channel index
    x: np.ndarray  # (N, 3) channel-frame probe positions
    d: np.ndarray  # signed distances (+inf off the grid)
    grad: np.ndarray  # distance gradients (channel frame)
    wit: np.ndarray  # witnesses (channel frame)
    wjac: np.ndarray  # witness Jacobians
    targets: Targets
    active: np.ndarray
    collide: np.ndarray
    world: np.ndarray  # (N, 3) world probe positions
    R: np.ndarray  # channel rotation
    tau: np.ndarray  # channel origin


@dataclass
class Linearization:
    """What an add-on's terms see while :meth:`OTRetargeter._linearize` builds one QP."""

    frame: Frame
    gn: GaussNewton
    obj_pos: np.ndarray
    obj_rot: np.ndarray
    link_pos: np.ndarray
    link_rot: np.ndarray
    link_jac: np.ndarray
    r_world: np.ndarray  # (N, 3) robot probes, world
    support_C: list  # extra inequality rows (``C dx >= l``), dense
    support_l: list


@dataclass
class Run:
    """One ``retarget_motion`` call: the demonstration and the outputs written so far."""

    human: HumanSurfaceMotion
    objects: list[InteractionObject]
    out_q: np.ndarray
    out_obj: np.ndarray
    info: dict[str, np.ndarray]


class Addon(ConfigurableMixin[DynamicConfigurableOptions]):
    """A stage or cost term added to :class:`OTRetargeter` for one kind of embodiment or data (see ``addons``).

    The retargeter calls each hook at a fixed point of its loop; the defaults do nothing. Options are the add-on's
    class attributes, as for any configurable; an unknown key is an error at construction.
    """

    #: Cost terms this add-on adds (reported as ``RetargetResult.info["cost/<term>"]``).
    _terms: ClassVar[tuple[str, ...]] = ()
    #: Per-frame diagnostics this add-on writes into ``RetargetResult.info``.
    _info: ClassVar[tuple[str, ...]] = ()

    def __init__(self, rt: OTRetargeter, options) -> None:
        super().__init__(options)
        self.rt = rt

    def setup(self, human: HumanSurfaceMotion) -> None:
        """Per demonstration, after the transport plans and the object channels; resets per-clip state."""

    def cold_starts(self, human: HumanSurfaceMotion, obj_pos: np.ndarray, obj_rot: np.ndarray) -> list[np.ndarray]:
        """Further cold-start configurations of frame 0 (each is solved; ``rank_start`` helps pick one)."""
        return []

    def rank_start(self, human, frame: Frame, q, obj_pos, obj_rot, n_starts: int) -> float:
        """Rank of one frame-0 solve among ``n_starts`` (lower wins; ties go to fewer relaxed QPs, then cost)."""
        return 0.0

    def frame(self, human: HumanSurfaceMotion, frame: Frame, forward: bool) -> None:
        """Adjust a frame's targets, or add to ``frame.extra``. ``forward`` is True once per frame, in time order."""

    def set_terms(self, lin: Linearization, s: ProbeSet) -> None:
        """Residuals or rows on one probe set (called after its depth residual)."""

    def terms(self, lin: Linearization) -> None:
        """Residuals or rows on the whole rig (called after the style terms)."""

    def after_frame(self, run: Run, frame: Frame, q, obj_pos, obj_rot, penetration: float):
        """After frame ``frame.t`` is solved: returns ``(q to continue from, q written out, penetration)``."""
        return q, q, penetration


def _addon_classes() -> dict[str, type[Addon]]:
    from eden.extensions.retargeting.otretarget.grasp import HandGrasp
    from eden.extensions.retargeting.otretarget.stance import FloorStance

    return {"floor_stance": FloorStance, "hand_grasp": HandGrasp}


class OTRetargeter(RetargetingBase):
    """Optimal-transport interaction retargeter for a floating-base rig, with objects.

    Rationale and measurements for the options that depart from the paper:
    ``examples/recipes/retargeting/otretarget/README.md``.

    Parameters
    ----------
    hand_entity_names : list[str]
        Entities welded below ``entity_name`` (e.g. ``["left_hand", "right_hand"]``) that form one rig with it.
    correspondences : list
        ``[[human parts...], [robot links...]]`` groups; one transport plan per group (Sec. III-B).
    style_links : dict[str, str]
        ``{robot link: human joint}`` tracked by the style cost; the first entry is the rig root.
    human_root_joint : str
        The human joint the rig root's position target sits on (the pelvis; ``Hips`` on the SOMA rig).
    tpose : dict[str, float]
        Joint values that put the robot in the human rest T-pose (unlisted joints are zero).
    robot_probe_density, human_probe_density : dict[str, float]
        ``{name regex: probes per m^2}`` for robot links / human parts.
    object_probe_density : float
        Probes per m^2 on objects (1000 in the paper).
    min_probes_per_link : int
        Floor on probes per link or human part.
    probe_exclude_links : list[str]
        Link regexes that carry no probes.
    use_visual_mesh : bool
        Sample robot probes and fit capsules on visual meshes.
    object_solid_voxel : float
        Voxel size (m) at which an object mesh whose winding-number sign is not reliable is closed into a solid
        before its distance field is baked (:func:`~.channels.watertight_solid`); 0 keeps every scan.
    sdf_resolution : float
        Voxel size of the channel grids (1 cm).
    exact_band : float
        Probes closer than this to a mesh channel read the exact mesh distance, witness and derivatives instead of the
        grid's (0 keeps the grid everywhere).
    geodesic_samples : int
        Surface samples of each object's geodesic table (all pairs: 4000 samples take 64 MB); the density is
        ``geodesic_samples / area``, never below ``object_probe_density``.
    max_distance : float
        ``d_bar``: probe-channel pairs closer than this (in the demonstration or now) are active (15 cm).
    kernel_width : float
        ``sigma`` of the quadratic proximity kernels (13 cm).
    kernel_cap : float
        ``beta_max``, the cap of the depth kernel under penetration (5).
    ot_epsilon : float
        Entropic regularisation of the transport plans (0.1).
    ot_normal_weight : float
        Weight of the normal-agreement term of the transport cost (0: the paper's position-only cost).
    interaction_weights : list[float]
        ``(lambda_d, lambda_x, lambda_n)`` for robot probes ((256, 625, 2500)).
    object_interaction_scale : float
        Multiplier of those weights for object probes (4).
    link_weights : dict[str, float]
        Per-link ``f_l`` (default 1): ``{regex: weight}`` over the rig's link names, first match wins; each regex must
        match a link. An object's ``f_l`` is its :attr:`InteractionObject.weight`.
    style_pos_weight, style_rot_weight : float
        ``lambda_pos`` and ``lambda_rot`` (the paper does not publish them).
    style_link_weights : dict[str, float]
        Per tracked link multiplier of both style weights (default 1): ``{regex: weight}`` over ``style_links`` keys.
    object_pos_weight, object_rot_weight : float
        ``lambda_pos_o``, ``lambda_rot_o``: how strongly each free object is centred on its demonstrated trajectory.
    object_smooth_weight : float
        Weight of each free object's frame-to-frame displacement held to the demonstrated one (not in Eq. 1).
    smooth_weight : float
        Weight of ``|q_t (-) q_{t-1}|^2``.
    collision_margin : float
        ``eps_col``: allowed penetration (0.3 mm).
    collision_activation : float
        Probe-channel pairs closer than this get a non-penetration row.
    collision_spacing : float
        Non-penetration rows sit on one robot probe per voxel of this size per link (a link smaller than eight voxels
        gets an eighth of its extent instead).
    self_collision : bool
        Add capsule self-collision rows.
    self_collision_activation : float
        Capsule pairs with clearance below this get a row.
    self_collision_exclude : list
        ``[regex_a, regex_b]`` link-name pairs left out of self-collision (both orders).
    capsule_radius_quantile : float
        Radius quantile of the capsule fits.
    joint_velocity_limit : float
        Default joint velocity limit (rad/s or m/s).
    joint_velocity_limits : dict[str, float]
        Per-joint overrides.
    joint_limits : dict[str, list[float]]
        ``{joint-name regex: [lower, upper]}``: the range the retargeted motion may use, intersected with the joint's
        own limits (never wider), first match wins.
    root_speed_slack : float
        (m/s) Per axis, the root's displacement stays within the demonstrator's root-joint displacement plus
        ``root_speed_slack * dt`` (0 disables).
    object_corridor : list[float]
        ``b_cor`` (m, rad): per-axis bound on an object's deviation from its demonstrated pose.
    object_speed_slack : list[float]
        (m/s, rad/s) added to the demonstrated per-frame displacement to form ``b_env``.
    trust_region : list[float]
        Per-DoF step boxes ``(Delta translation m, Delta rotation/joint rad)`` (5 cm, 0.1 rad).
    penetration_slack : float
        A step that drives a probe deeper than it was by more than this (m) is re-solved (see ``_solve_frame``).
    witness_lateral_cap : float
        Cap on the lateral curvature factor rho / |c| of the witness residual's Gauss-Newton model.
    reweight_every : int
        Accepted steps between re-evaluations of the reweighted interaction terms within a frame.
    min_trust_scale : float
        Smallest fraction of ``trust_region`` the step halving goes to; there the step is taken regardless.
    first_frame_iterations, iterations : int
        SQP budgets for the cold first frame and the warm-started rest (50, 6).
    step_tolerance : float
        Stop iterating once the step's infinity norm falls below this.
    ground : bool
        Add the floor channel; the cold start stands the rig on it.
    ground_height : float
        Floor height (m).
    addons : dict[str, dict]
        ``{name: options}`` of the add-ons to run, in order: ``floor_stance``
        (:class:`~.stance.FloorStance`), ``hand_grasp`` (:class:`~.grasp.HandGrasp`).
    """

    hand_entity_names: list[str] = []
    correspondences: list = []
    style_links: dict[str, str] = {}
    human_root_joint: str = "pelvis"
    tpose: dict[str, float] = {}

    robot_probe_density: dict[str, float] = {".*": 2000.0}
    human_probe_density: dict[str, float] = {".*": 2000.0}
    object_probe_density: float = 1000.0
    min_probes_per_link: int = 24
    probe_exclude_links: list[str] = []
    use_visual_mesh: bool = True

    object_solid_voxel: float = 0.004
    sdf_resolution: float = 0.01
    exact_band: float = 0.01
    geodesic_samples: int = 4000
    max_distance: float = 0.15
    kernel_width: float = 0.13
    kernel_cap: float = 5.0
    ot_epsilon: float = 0.1
    ot_normal_weight: float = 0.5

    interaction_weights: list[float] = [256.0, 625.0, 2500.0]
    object_interaction_scale: float = 4.0
    link_weights: dict[str, float] = {}
    style_pos_weight: float = 0.03
    style_link_weights: dict[str, float] = {}
    style_rot_weight: float = 3.0
    object_pos_weight: float = 10.0
    object_rot_weight: float = 1.0
    smooth_weight: float = 0.5
    object_smooth_weight: float = 100.0

    collision_margin: float = 3e-4
    collision_activation: float = 0.15
    collision_spacing: float = 0.01
    self_collision: bool = True
    self_collision_activation: float = 0.03
    self_collision_exclude: list = []
    capsule_radius_quantile: float = 0.8
    joint_velocity_limit: float = 20.0
    joint_velocity_limits: dict[str, float] = {}
    joint_limits: dict[str, list[float]] = {}
    root_speed_slack: float = 1.0
    object_corridor: list[float] = [0.5, 1.0]
    object_speed_slack: list[float] = [1.0, 3.0]
    trust_region: list[float] = [0.05, 0.10]
    penetration_slack: float = 1e-3
    witness_lateral_cap: float = 4.0
    reweight_every: int = 10
    min_trust_scale: float = 1.0 / 64.0

    first_frame_iterations: int = 50
    iterations: int = 6
    step_tolerance: float = 1e-4

    ground: bool = True
    ground_height: float = 0.0

    addons: dict[str, dict] = {}

    def __init__(self, env: EnvBase, options: RetargeterOptions) -> None:
        super().__init__(env, options)
        self._validate_scalars()
        self.kin = RigKinematics(env, [self.entity_name, *self.hand_entity_names])
        kin = self.kin
        if not kin.floating:
            raise ValueError(
                f"OTRetargeter: entity {self.entity_name!r} has a fixed base, but the method retargets the root pose "
                f"with the rest of the body. Retarget a floating-base rig."
            )
        # -- names -----------------------------------------------------------
        joints = kin.var_names[kin.n_base_v :]
        self._refuse_unknown("tpose", self.tpose, joints, "the rig's joints")
        self._refuse_unknown("joint_velocity_limits", self.joint_velocity_limits, joints, "the rig's joints")
        for pattern, bounds in self.joint_limits.items():
            if not any(re.fullmatch(pattern, n) for n in joints) or len(bounds) != 2 or not bounds[0] < bounds[1]:
                raise ValueError(
                    f"OTRetargeter: joint_limits entry {pattern!r}: {bounds!r} must fully match at least one joint of the "
                    f"rig ({joints}) and give [lower, upper] with lower < upper."
                )
        for i, name in enumerate(joints):
            bounds = next((b for p, b in self.joint_limits.items() if re.fullmatch(p, name)), None)
            if bounds is not None:
                kin.joint_lower[i] = max(kin.joint_lower[i], float(bounds[0]))
                kin.joint_upper[i] = min(kin.joint_upper[i], float(bounds[1]))
        for pattern in self.link_weights:
            # Robot links only (an object carries its own InteractionObject.weight): a typo here read as weight 1.
            if not any(re.fullmatch(pattern, n) for n in kin.link_names):
                raise ValueError(
                    f"OTRetargeter: link_weights entry {pattern!r} fully matches none of the rig's links "
                    f"({list(kin.link_names)}); fix the regex or drop the entry (weight an object with "
                    f"InteractionObject(weight=...))."
                )
        for pattern in self.probe_exclude_links:
            try:
                matched = any(re.fullmatch(pattern, n) for n in kin.link_names)
            except re.error as exc:
                raise ValueError(
                    f"OTRetargeter: probe_exclude_links entry {pattern!r} is not a regex ({exc})."
                ) from None
            if not matched:
                raise ValueError(
                    f"OTRetargeter: probe_exclude_links entry {pattern!r} fully matches none of the rig's links "
                    f"({list(kin.link_names)}); fix the regex or drop the entry."
                )
        if not self.correspondences:
            raise ValueError(
                "OTRetargeter: correspondences is empty; give at least one [[human parts], [robot links]] group."
            )
        group_links = [link for group in self.correspondences for link in self._group(group)[1]]
        self._refuse_unknown("correspondences robot links", group_links, kin.link_names, "the rig's links")
        if not self.style_links:
            raise ValueError("OTRetargeter: style_links is empty; give at least the root link -> 'pelvis'.")
        self._refuse_unknown("style_links keys", self.style_links, kin.link_names, "the rig's links")
        for pattern, value in self.style_link_weights.items():
            if not any(re.fullmatch(pattern, link) for link in self.style_links) or value < 0:
                raise ValueError(
                    f"OTRetargeter: style_link_weights entry {pattern!r}: {value!r} must be >= 0 and fully match at "
                    f"least one style_links key ({list(self.style_links)}); fix the regex or drop the entry."
                )
        root_link = kin.link_names[int(np.flatnonzero(kin.link_parent < 0)[0])]
        first = next(iter(self.style_links))
        if first != root_link:
            raise ValueError(
                f"OTRetargeter: the first style_links entry is {first!r}, but it must be the rig root {root_link!r}: "
                f"the style targets are forward kinematics from the root aligned to the human pelvis."
            )

        # -- reference (T-) pose ---------------------------------------------
        self._q_tpose = np.zeros(kin.n_q)
        self._q_tpose[3] = 1.0
        for name, value in self.tpose.items():
            self._q_tpose[self._q_index(name)] = value
        kin.set_configuration(self._q_tpose)
        self._tpose_link_pos, self._tpose_link_rot = kin.link_poses()

        # -- robot probes ------------------------------------------------------
        self.robot_probes = sample_rig_probes(
            kin, self.robot_probe_density, self.min_probes_per_link, self.probe_exclude_links, self.use_visual_mesh
        )
        self._probe_labels = [kin.link_names[c] for c in self.robot_probes.carrier]
        # Non-penetration rows on one probe per voxel, the voxel shrunk to an eighth of a small link's extent:
        # at 1 cm a 2-4 cm phalanx keeps a handful of rows and a thin pole slips in between them.
        local, carrier = self.robot_probes.local, self.robot_probes.carrier
        spacing = np.full(len(kin.link_names), float(self.collision_spacing))
        for li in np.unique(carrier):
            pts = local[carrier == li]
            spacing[li] = min(spacing[li], 0.125 * float((pts.max(axis=0) - pts.min(axis=0)).max()))
        self._collision_probe = _voxel_subset(local, carrier, np.maximum(spacing, 1e-3)[carrier])

        # -- self-collision capsules ------------------------------------------
        self.capsules = None
        if self.self_collision:
            links = {link.name: link for ent in kin.entities for link in ent.links}
            pts = {}
            for li, name in enumerate(kin.link_names):
                geoms = links[name].vgeoms if self.use_visual_mesh else links[name].geoms
                if len(geoms) == 0:
                    continue
                mesh = get_link_mesh(links[name], use_visual_mesh=self.use_visual_mesh)
                if mesh.area > 1e-8:
                    pts[li] = sample_surface(mesh, 20000.0, 64, seed=li)[0]
            q_zero = np.zeros(kin.n_q)
            q_zero[3] = 1.0
            kin.set_configuration(q_zero)
            zero_pose = kin.link_poses()
            self.capsules = build_link_capsules(
                pts,
                kin.link_parent,
                [zero_pose, (self._tpose_link_pos, self._tpose_link_rot)],
                radius_quantile=self.capsule_radius_quantile,
                link_names=list(kin.link_names),
                exclude=[tuple(pair) for pair in self.self_collision_exclude],
            )

        # -- limits ------------------------------------------------------------
        self._vel_limit = np.full(kin.n_v - kin.n_base_v, float(self.joint_velocity_limit))
        for name, value in self.joint_velocity_limits.items():
            self._vel_limit[kin.var_names.index(name) - kin.n_base_v] = value
        self._link_weight = np.array([_match_weight(n, self.link_weights) for n in kin.link_names])
        self._limb_of_link, self._joint_q, self._joint_limb = self._limbs()

        # -- add-ons -----------------------------------------------------------
        classes = _addon_classes()
        unknown = [n for n in self.addons if n not in classes]
        if unknown:
            raise ValueError(
                f"OTRetargeter: addons {unknown} are not add-ons. Available: {list(classes)}. Fix the name or drop the "
                f"entry."
            )
        self._addons: list[Addon] = [
            classes[name](self, classes[name].configure(**(cfg or {}))) for name, cfg in self.addons.items()
        ]
        self._terms = _TERMS + tuple(t for a in self._addons for t in a._terms)

    def addon(self, name: str) -> Addon | None:
        """The add-on ``name`` of this retargeter (see ``addons``), or None when it is not configured."""
        return next((a for n, a in zip(self.addons, self._addons) if n == name), None)

    # ------------------------------------------------------------------
    # validation helpers
    # ------------------------------------------------------------------

    def _validate_scalars(self) -> None:
        for name, length in (
            ("interaction_weights", 3),
            ("object_corridor", 2),
            ("object_speed_slack", 2),
            ("trust_region", 2),
        ):
            if len(getattr(self, name)) != length:
                raise ValueError(f"OTRetargeter: {name}={getattr(self, name)!r} must have {length} entries.")
        # {option: (in range, the rule)}. Each of these otherwise fails minutes into ``_setup`` or never: a zero
        # ``reweight_every`` divides by zero, a non-positive trust region or velocity limit gives NaN or crossed step
        # bounds, a negative weight makes the QP non-convex (every iterate "failed", nothing raised), and a kernel
        # wider than ``max_distance`` turns on witness residuals for probes whose demonstrated target is unknown.
        smallest_box = self.min_trust_scale * min(self.trust_region)
        checks = {
            "max_distance": (self.max_distance > 0, "> 0"),
            "kernel_width": (0 < self.kernel_width <= self.max_distance, f"in (0, max_distance={self.max_distance}]"),
            "kernel_cap": (self.kernel_cap >= 1, ">= 1"),
            "ot_epsilon": (self.ot_epsilon > 0, "> 0"),
            "ot_normal_weight": (self.ot_normal_weight >= 0, ">= 0"),
            "sdf_resolution": (self.sdf_resolution > 0, "> 0"),
            "object_solid_voxel": (self.object_solid_voxel >= 0, ">= 0"),
            "geodesic_samples": (self.geodesic_samples >= 16, ">= 16"),
            "exact_band": (self.exact_band >= 0, ">= 0"),
            "first_frame_iterations": (self.first_frame_iterations >= 1, ">= 1"),
            "iterations": (self.iterations >= 1, ">= 1"),
            "reweight_every": (self.reweight_every >= 1, ">= 1"),
            "collision_margin": (self.collision_margin >= 0, ">= 0"),
            "collision_spacing": (self.collision_spacing > 0, "> 0"),
            "penetration_slack": (self.penetration_slack >= 0, ">= 0"),
            "root_speed_slack": (self.root_speed_slack >= 0, ">= 0"),
            "witness_lateral_cap": (self.witness_lateral_cap >= 1, ">= 1"),
            "joint_velocity_limit": (self.joint_velocity_limit > 0, "> 0"),
            "trust_region": (min(self.trust_region) > 0, "> 0 in both entries"),
            "min_trust_scale": (0 < self.min_trust_scale <= 1, "in (0, 1]"),
            # A step this small is then always inside the step box, which is what makes it convergence.
            "step_tolerance": (
                0 < self.step_tolerance < smallest_box,
                f"in (0, min_trust_scale * min(trust_region) = {smallest_box:g})",
            ),
        }
        for name in (
            "interaction_weights",
            "object_interaction_scale",
            "style_pos_weight",
            "style_rot_weight",
            "object_pos_weight",
            "object_rot_weight",
            "smooth_weight",
            "object_smooth_weight",
        ):
            checks[name] = (min(np.atleast_1d(getattr(self, name))) >= 0, ">= 0 (a negative weight is non-convex)")
        bad = [f"{k}={getattr(self, k)!r} must be {rule}" for k, (ok, rule) in checks.items() if not ok]
        bad += [
            f"joint_velocity_limits[{k!r}]={v!r} must be > 0" for k, v in self.joint_velocity_limits.items() if v <= 0
        ]
        for pattern, value in self.link_weights.items():
            if value < 0:
                bad.append(f"link_weights[{pattern!r}]={value!r} must be >= 0")
            try:
                re.compile(pattern)
            except re.error as exc:
                bad.append(f"link_weights key {pattern!r} is not a regex ({exc})")
        if bad:
            raise ValueError(f"OTRetargeter: {'; '.join(bad)}.")

    @staticmethod
    def _refuse_unknown(field_name: str, names, available, where: str) -> None:
        available = list(available)
        unknown = [n for n in dict.fromkeys(names) if n not in available]
        if unknown:
            raise ValueError(
                f"OTRetargeter: {field_name} {unknown} are not in {where}. Available: {available}. "
                f"Fix the spelling or drop the entry: an unknown name is an error, not skipped."
            )

    @staticmethod
    def _group(group) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if len(group) != 2:
            raise ValueError(f"OTRetargeter: correspondence {group!r} must be [[human parts...], [robot links...]].")
        return tuple(group[0]), tuple(group[1])

    def _q_index(self, joint: str) -> int:
        return self.kin.n_base_q + self.kin.var_names.index(joint) - self.kin.n_base_v

    # ------------------------------------------------------------------
    # setup per demonstration
    # ------------------------------------------------------------------

    def _setup(self, human: HumanSurfaceMotion, objects: list[InteractionObject]) -> None:
        """Everything that depends on the demonstration: probes, transport plans, style, channels, the add-ons."""
        self._human_setup(human)
        self._object_setup(objects, human.num_frames)
        for addon in self._addons:
            addon.setup(human)

    def _human_setup(self, human: HumanSurfaceMotion) -> None:
        """Probe the human, solve the transport plans and the style alignment for this body."""
        kin = self.kin
        human_parts = [p for group in self.correspondences for p in self._group(group)[0]]
        self._refuse_unknown("correspondences human parts", human_parts, human.joint_names, "the body model's joints")
        self._refuse_unknown(
            "style_links values", self.style_links.values(), human.joint_names, "the body model's joints"
        )
        if self.human_root_joint not in human.joint_names:
            raise ValueError(
                f"OTRetargeter: human_root_joint={self.human_root_joint!r} is not a joint of this body model "
                f"({human.joint_names}); set it to the pelvis joint ('pelvis' for SMPL-X, 'Hips' for SOMA)."
            )

        self.human_probes = sample_human_probes(human, self.human_probe_density, self.min_probes_per_link)
        h_rest = self.human_probes.positions(human.rest_vertices)
        h_labels = [human.joint_names[j] for j in self.human_probes.part]
        r_tpose = self.robot_probes.world(self._tpose_link_pos, self._tpose_link_rot)
        r_normals = np.einsum("nab,nb->na", self._tpose_link_rot[self.robot_probes.carrier], self.robot_probes.normals)
        groups = [self._group(g) for g in self.correspondences]
        self.src, self.probe_group = group_correspondences(
            h_rest,
            h_labels,
            r_tpose,
            self._probe_labels,
            groups,
            self.ot_epsilon,
            self.human_probes.normals(human.rest_vertices),
            r_normals,
            self.ot_normal_weight,
            # Marginals by the area each sample stands for: groups mixing densities stay area-balanced.
            self.human_probes.area,
            self.robot_probes.area,
            device=self.env.device,
        )

        # Style: alignment offsets O_l = R_h,rest^T R_l,T and the tracked-link tree with T-pose offsets.
        self._style_link_idx = np.array([kin.link_index(n) for n in self.style_links], dtype=np.int64)
        self._style_weight = np.array([_match_weight(n, self.style_link_weights) for n in self.style_links])
        self._style_joint_idx = np.array([human.joint_index(j) for j in self.style_links.values()], dtype=np.int64)
        R_h_rest = human.rest_joint_rot[self._style_joint_idx]
        R_l_t = self._tpose_link_rot[self._style_link_idx]
        self._style_offset = np.einsum("lba,lbc->lac", R_h_rest, R_l_t)
        tracked = {int(li): k for k, li in enumerate(self._style_link_idx)}
        parent = np.full(len(self._style_link_idx), -1, dtype=np.int64)
        for k, li in enumerate(self._style_link_idx):
            cur = kin.link_parent[li]
            while cur >= 0 and int(cur) not in tracked:
                cur = kin.link_parent[cur]
            parent[k] = tracked[int(cur)] if cur >= 0 else -1
        self._style_parent = parent
        # Evaluation order for the FK of target orientations: every tracked link after its tracked parent.
        depth = np.zeros(len(parent), dtype=np.int64)
        for k in range(len(parent)):
            cur = parent[k]
            while cur >= 0:
                depth[k] += 1
                cur = parent[cur]
        self._style_order = np.argsort(depth, kind="stable")
        p_t = self._tpose_link_pos[self._style_link_idx]
        self._style_bone = np.zeros((len(parent), 3))
        for k in range(len(parent)):
            if parent[k] >= 0:
                self._style_bone[k] = R_l_t[parent[k]].T @ (p_t[k] - p_t[parent[k]])
        self._pelvis_joint = human.joint_index(self.human_root_joint)

    def _style_targets(self, human: HumanSurfaceMotion, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Target orientations ``R_h(l) O_l`` and positions by FK of those orientations from the pelvis."""
        rot = np.einsum("lab,lbc->lac", human.joint_rot[t, self._style_joint_idx], self._style_offset)
        pos = np.zeros((len(rot), 3))
        for k in self._style_order:
            p = self._style_parent[k]
            pos[k] = human.joint_pos[t, self._pelvis_joint] if p < 0 else pos[p] + rot[p] @ self._style_bone[k]
        return pos, rot

    def _initial_configuration(
        self,
        human: HumanSurfaceMotion,
        obj_pos: np.ndarray,
        obj_rot: np.ndarray,
        iterations: int = 100,
        pulls: list | None = None,
        pull_weight: float = 0.0,
    ) -> np.ndarray:
        """Cold start of the first frame: the demonstrated posture, standing on the floor.

        The joints are fitted to the first frame's style orientations alone (root held on the pelvis, joint limits as
        bounds), the rig is moved vertically until its lowest probe is as high as the demonstrator's lowest point, and
        limbs that start inside an object are drawn back (``_untangle``). Started from zero joints at the pelvis
        instead, a robot in a demonstrated squat began with its straight legs through the floor and ended the first
        frame lying or kneeling. ``pulls`` (``[(link, point in the link frame, world target)]``) are also pulled to
        their targets at ``pull_weight``; the root may then move off the pelvis to reach them.
        """
        kin = self.kin
        pos0, rot0 = self._style_targets(human, 0)
        q = np.zeros(kin.n_q)
        q[0:3] = pos0[0]
        q[3:7] = R_to_quat(rot0[0])
        q[kin.n_base_q :] = np.clip(0.0, kin.joint_lower, kin.joint_upper)
        sl = self._style_link_idx
        w = np.repeat(self._style_weight, 3)
        damping = 1e-3 * np.eye(kin.n_v)
        pulls = pulls if pull_weight > 0 else []
        for _ in range(iterations):
            kin.set_configuration(q)
            link_pos, link_rot = kin.link_poses()
            jac = kin.link_jacobians(link_pos, link_rot)
            e = axis_angle_from_quat(R_to_quat(np.einsum("lab,lcb->lac", link_rot[sl], rot0))).reshape(-1)
            J = jac[sl, 3:6].reshape(-1, kin.n_v)
            H = J.T @ (w[:, None] * J) + damping
            g = J.T @ (w * e)
            lo = np.full(kin.n_v, -self.trust_region[1])
            hi = -lo
            lo[0:3] = hi[0:3] = 0.0
            if pulls:
                lo[0:3], hi[0:3] = -self.trust_region[0], self.trust_region[0]
                for li, local, target in pulls:
                    p = link_pos[li] + link_rot[li] @ local
                    Jp = jac[li, 0:3] - skew(p - link_pos[li]) @ jac[li, 3:6]
                    H += pull_weight * Jp.T @ Jp
                    g += pull_weight * Jp.T @ (p - target)
                H[0:3, 0:3] += 10.0 * np.eye(3)
                g[0:3] += 10.0 * (q[0:3] - pos0[0])
            lo, hi = self._joint_bounds(q, lo, hi)
            step, _ = solve_qp(H, g, np.zeros((0, kin.n_v)), np.zeros(0), lo, hi)
            q = kin.integrate(q, step)
            if np.abs(step).max() < self.step_tolerance:
                break
        q = self._on_floor(q, human)
        return self._on_floor(self._untangle(q, obj_pos, obj_rot)[0], human)

    def _joint_bounds(self, q: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Step bounds ``lo``, ``hi`` (velocity columns) tightened so that no joint leaves its limits."""
        kin = self.kin
        joints = q[kin.n_base_q :]
        lo[kin.n_base_v :] = np.maximum(lo[kin.n_base_v :], kin.joint_lower - joints)
        hi[kin.n_base_v :] = np.minimum(hi[kin.n_base_v :], kin.joint_upper - joints)
        return lo, hi

    def _on_floor(self, q: np.ndarray, human: HumanSurfaceMotion) -> np.ndarray:
        """``q`` moved vertically so that the rig's lowest probe is at the demonstrator's lowest point in frame 0 (never
        below ``ground_height``); unchanged without ``ground``.
        """
        if not self.ground:
            return q
        self.kin.set_configuration(q)
        lowest = self.robot_probes.world(*self.kin.link_poses())[:, 2].min()
        q = q.copy()
        q[2] += max(float(human.vertices[0][:, 2].min()), self.ground_height) - lowest
        return q

    def _free_distance(self, points: np.ndarray, obj_pos: np.ndarray, obj_rot: np.ndarray):
        """Exact ``(signed distance, witness)`` of world ``points`` (``(N, 3)``, or one ``(3,)``) against each free
        object at ``obj_pos``, ``obj_rot``: yields ``(channel, object, d, w)``.
        """
        for ci, oi in enumerate(self._channel_object):
            if oi in self._free:
                x = (points - obj_pos[oi]) @ obj_rot[oi]
                d, w = self.channels[ci].exact_distance(x[None] if x.ndim == 1 else x)
                yield ci, oi, d, w

    def _untangle(
        self, q: np.ndarray, obj_pos: np.ndarray, obj_rot: np.ndarray, depth: float = 0.002, step: float = 0.1
    ) -> tuple[np.ndarray, bool]:
        """The cold start with every limb that starts inside a free object drawn back toward the T-pose.

        A limb is a chain of joints between two branchings of the kinematic tree (a leg, an arm up to the palm, a
        finger). Inside a thin part the rows on its two faces point in opposite directions and no local step gets a
        link out, so each limb with a probe deeper than ``depth`` in a free object is blended back toward its T-pose
        values by ``step`` until it is clear; the cold iterations then bring it to the demonstrated contact from
        outside, on the side it came from. A limb still inside when fully drawn back hands the retraction on to the
        limb it hangs from (a thumb to its arm). The object is never moved: that lost which side of it each hand
        belonged on. Returns the configuration and whether any limb moved.
        """
        kin = self.kin
        if not self._free:
            return q, False
        limb_of_link, joint_q, joint_limb = self._limb_of_link, self._joint_q, self._joint_limb
        parent = kin.link_parent
        # The T-pose: arms out to the sides at shoulder height, hands open, legs straight. The zero pose is not clear of
        # what the hands hold: a G1 at zero holds its forearms straight forward at waist height, through a chair back.
        neutral = np.clip(self._q_tpose[kin.n_base_q :], kin.joint_lower, kin.joint_upper)[joint_q - kin.n_base_q]
        target = q[joint_q].copy()
        alpha = np.ones(int(joint_limb.max()) + 1)
        out = q.copy()
        for _ in range(4 * int(np.ceil(1.0 / step))):
            out[joint_q] = neutral + alpha[joint_limb] * (target - neutral)
            kin.set_configuration(out)
            world = self.robot_probes.world(*kin.link_poses())
            inside = np.zeros(len(world), dtype=bool)
            for _, _, d, _ in self._free_distance(world, obj_pos, obj_rot):
                inside |= d < -depth
            limbs = set(np.unique(limb_of_link[self.robot_probes.carrier[inside]]).tolist()) - {-1}
            while any(alpha[k] <= 0.0 for k in limbs):
                limbs = {k for k in limbs if alpha[k] > 0.0} | {
                    int(limb_of_link[parent[k]]) for k in limbs if alpha[k] <= 0.0 and parent[k] >= 0
                }
                limbs.discard(-1)
            if not limbs:
                break
            limbs = np.array(sorted(limbs))
            alpha[limbs] = np.maximum(alpha[limbs] - step, 0.0)
        out[joint_q] = neutral + alpha[joint_limb] * (target - neutral)
        return out, bool((alpha < 1.0).any())

    def _limbs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Limbs of the rig: chains of joints between branchings of the link tree.

        Returns the limb of every link (-1 for links no joint moves), and the configuration index and limb of every
        joint. A limb is named by its top link, the child of a link with two or more children (or of the root).
        """
        kin = self.kin
        parent = kin.link_parent
        children = np.bincount(parent[parent >= 0], minlength=len(parent))
        top = np.arange(len(parent))
        for li in range(len(parent)):
            cur = li
            while parent[cur] >= 0 and children[parent[cur]] < 2 and parent[parent[cur]] >= 0:
                cur = parent[cur]
            top[li] = cur
        jointed = np.full(len(parent), -1)
        jointed[kin._joint_link] = kin._joint_link
        limb_of_link = np.full(len(parent), -1)
        for li in range(len(parent)):
            cur = li
            while cur >= 0 and jointed[cur] < 0:
                cur = parent[cur]
            limb_of_link[li] = -1 if cur < 0 else top[cur]
        joint_q = kin.n_base_q + kin._joint_var - kin.n_base_v
        return limb_of_link, joint_q, top[kin._joint_link]

    def _object_setup(self, objects: list[InteractionObject], n_frames: int) -> None:
        names = [o.name for o in objects]
        if len(set(names)) != len(names) or "ground" in names:
            raise ValueError(f"OTRetargeter: object names {names} must be unique and not 'ground'.")
        for o in objects:
            if np.shape(o.poses) != (n_frames, 7):
                raise ValueError(
                    f"OTRetargeter: object {o.name!r} poses have shape {np.shape(o.poses)}; expected ({n_frames}, 7) "
                    f"[pos, quat_wxyz] per human frame."
                )
        self._object_weight = np.array([o.weight for o in objects], dtype=np.float64)
        self.objects = objects
        poses = np.stack([o.poses for o in objects], axis=1) if objects else np.zeros((n_frames, 0, 7))
        self._demo_pos = poses[..., :3].astype(np.float64)
        self._demo_rot = quat_to_R(poses[..., 3:])
        self.channels: list[Channel] = []
        self._channel_object: list[int] = []  # object index per channel, -1 for the ground
        if self.ground:
            self.channels.append(PlaneChannel("ground", self.max_distance))
            self._channel_object.append(-1)
        self._demo_channels: dict[int, MeshChannel] = {}
        self._witness_map: dict[int, tuple[cKDTree, np.ndarray, np.ndarray]] = {}
        for oi, o in enumerate(objects):
            ch = self._mesh_channel(o.name, o.mesh)
            self.channels.append(ch)
            self._channel_object.append(oi)
            if o.demo_mesh is not None:
                demo = self._mesh_channel(o.name, o.demo_mesh)
                self._demo_channels[oi] = demo
                # Witness transfer (Sec. III-E): demo-surface samples -> substitute-surface samples by OT,
                # each in its object frame normalised to unit radius; metric targets are kept as is.
                d_pts = demo.geodesics.points
                s_pts = ch.geodesics.points
                src = transport_correspondence(s_pts, d_pts, self.ot_epsilon, device=self.env.device)
                self._witness_map[oi] = (cKDTree(d_pts), s_pts[src], ch.geodesics.normals[src])
        # Object probes on the scene geometry; their demonstrated counterparts on the demo geometry.
        self.object_probes = sample_object_probes(
            [o.mesh for o in objects], self.object_probe_density, self.min_probes_per_link
        )
        demo_local = self.object_probes.local.copy()
        for oi, o in enumerate(objects):
            if o.demo_mesh is None:
                continue
            mask = self.object_probes.carrier == oi
            demo_pts = sample_object_probes([o.demo_mesh], self.object_probe_density, self.min_probes_per_link).local
            demo_local[mask] = demo_pts[
                transport_correspondence(
                    demo_pts, self.object_probes.local[mask], self.ot_epsilon, device=self.env.device
                )
            ]
        self._object_probe_demo_local = demo_local
        self._free = [oi for oi, o in enumerate(objects) if not o.fixed]
        self._free_slot = {oi: self.kin.n_v + 6 * k for k, oi in enumerate(self._free)}
        self.n_x = self.kin.n_v + 6 * len(self._free)

    def _mesh_channel(self, name: str, mesh: trimesh.Trimesh) -> MeshChannel:
        # Geodesic-table density: ``geodesic_samples`` over the area, at least the probe density.
        density = max(self.object_probe_density, self.geodesic_samples / max(float(mesh.area), 1e-9))
        return MeshChannel(
            name, mesh, self.max_distance, self.sdf_resolution, density, solid_voxel=self.object_solid_voxel
        )

    # ------------------------------------------------------------------
    # demonstration triples
    # ------------------------------------------------------------------

    def _channel_pose(self, ci: int, obj_pos: np.ndarray, obj_rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        oi = self._channel_object[ci]
        if oi < 0:
            return np.eye(3), np.array([0.0, 0.0, self.ground_height])
        return obj_rot[oi], obj_pos[oi]

    def _read_triples(self, channel: Channel, x_local: np.ndarray, oi: int) -> Targets:
        """Demonstrated ``(d, w, sign(d) n)`` of channel-frame points (unknown beyond the grid -> +inf)."""
        q = channel.query(x_local)
        if isinstance(channel, MeshChannel):
            # Near the surface the demonstrated triple is read exactly, as the robot's are (``exact_band``).
            near = q.valid & (q.distance < self.exact_band)
            if near.any():
                ex = channel.exact_query(x_local[near])
                for name in ("distance", "witness", "distance_grad"):
                    getattr(q, name)[near] = getattr(ex, name)
        d = np.where(q.valid, q.distance, np.inf)
        w = q.witness
        delta = x_local - w
        length = np.linalg.norm(delta, axis=1)
        grad_dir = q.distance_grad / np.maximum(np.linalg.norm(q.distance_grad, axis=1), 1e-12)[:, None]
        sign = np.where(d < 0, -1.0, 1.0)
        normal = np.where(
            (length > 1e-6)[:, None], sign[:, None] * delta / np.maximum(length, 1e-12)[:, None], grad_dir
        )
        if oi in self._witness_map:
            tree, s_pts, s_nrm = self._witness_map[oi]
            idx = tree.query(w)[1]
            w = s_pts[idx]
            normal = s_nrm[idx]
        return Targets(d, w, normal)

    def _demo_targets(self, human: HumanSurfaceMotion, t: int) -> tuple[list[Targets], list[Targets], list[Targets]]:
        """Per channel: targets of every robot probe, of every object probe, and every human probe's own triple."""
        h_world = self.human_probes.positions(human.vertices[t].astype(np.float64))
        demo_pos, demo_rot = self._demo_pos[t], self._demo_rot[t]
        o_world = (
            demo_pos[self.object_probes.carrier]
            + np.einsum("nab,nb->na", demo_rot[self.object_probes.carrier], self._object_probe_demo_local)
            if len(self.object_probes)
            else np.zeros((0, 3))
        )
        robot_t, object_t, human_t = [], [], []
        has_src = self.src >= 0
        src = np.maximum(self.src, 0)
        for ci, channel in enumerate(self.channels):
            oi = self._channel_object[ci]
            R, tau = self._channel_pose(ci, demo_pos, demo_rot)
            demo_channel = self._demo_channels.get(oi, channel)
            h = self._read_triples(demo_channel, (h_world - tau) @ R, oi)
            human_t.append(h)
            robot_t.append(Targets(np.where(has_src, h.distance[src], np.inf), h.witness[src], h.normal[src]))
            o = self._read_triples(demo_channel, (o_world - tau) @ R, oi)
            # An object probe does not interact with its own object.
            o.distance = np.where(self.object_probes.carrier == oi, np.inf, o.distance)
            object_t.append(o)
        return robot_t, object_t, human_t

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def retarget_motion(
        self,
        human: HumanSurfaceMotion,
        objects: list[InteractionObject] | None = None,
        q_init: np.ndarray | None = None,
        progress: bool = True,
    ) -> RetargetResult:
        """Retarget a demonstration frame by frame.

        Parameters
        ----------
        human : HumanSurfaceMotion
            The demonstrator's posed surface and skeleton (z-up, metres, native scale).
        objects : list[InteractionObject], optional
            Objects of the scene with their demonstrated poses (same frames as ``human``).
        q_init : np.ndarray, optional
            ``(n_q,)`` rig configuration to start the first frame from; defaults to the cold start
            (``_initial_configuration`` and the add-ons' ``cold_starts``).
        progress : bool
            Show a progress bar.

        Returns
        -------
        RetargetResult
        """
        kin = self.kin
        objects = list(objects or [])
        n_frames = human.num_frames
        if n_frames < 1 or not human.fps > 0:
            raise ValueError(
                f"OTRetargeter.retarget_motion: the demonstration has {n_frames} frames at fps={human.fps!r}; it needs "
                f"at least one frame and fps > 0 (check the slice that made it, e.g. HumanSurfaceMotion.slice past the "
                f"end)."
            )
        t0 = time.time()
        self._setup(human, objects)
        setup_s = time.time() - t0
        en.logger.info(
            f"OTRetarget setup {setup_s:.1f}s: {len(self.robot_probes)} robot probes, "
            f"{len(self.human_probes)} human probes, {len(self.object_probes)} object probes, "
            f"{len(self.channels)} channels, {0 if self.capsules is None else len(self.capsules.pairs)} capsule pairs."
        )

        obj_pos = self._demo_pos[0].copy()
        obj_rot = self._demo_rot[0].copy()
        if q_init is None:
            # One BLAS thread here too: with a pool, sums are ordered differently from run to run, and a 1e-16
            # difference in the cold start grew into 0.4 rad by frame 40 of a chair flip (runs were not reproducible).
            with single_threaded_blas():
                starts = [self._initial_configuration(human, obj_pos, obj_rot)]
                for addon in self._addons:
                    starts += addon.cold_starts(human, obj_pos, obj_rot)
        else:
            q = np.asarray(q_init, dtype=np.float64).copy()
            if q.shape != (kin.n_q,):
                raise ValueError(
                    f"OTRetargeter: q_init has shape {q.shape}; the rig's configuration has {kin.n_q} entries."
                )
            starts = [q]

        keys = [*_INFO, *(k for a in self._addons for k in a._info), *(f"cost/{k}" for k in self._terms)]
        run = Run(human, objects, np.zeros((n_frames, kin.n_q)), np.zeros((len(objects), n_frames, 7)), {})
        run.info = {k: np.zeros(n_frames) for k in keys}
        iterator = range(n_frames)
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="OTRetarget")
        # Every product in an iterate is small (tens of columns): a multithreaded BLAS spends far longer
        # waking its pool than computing (6 ms vs 0.03 ms for a 100x81 Gram matrix on a 64-core node).
        t1 = time.time()
        with single_threaded_blas():
            self._run_frames(iterator, run, starts, obj_pos, obj_rot)
        solve_s = time.time() - t1
        stats = {
            "setup_s": setup_s,
            "solve_s": solve_s,
            "total_s": setup_s + solve_s,
            "frames": n_frames,
            "robot_probes": len(self.robot_probes),
            "human_probes": len(self.human_probes),
            "object_probes": len(self.object_probes),
            "channels": len(self.channels),
            "capsule_pairs": 0 if self.capsules is None else len(self.capsules.pairs),
            "n_dof": kin.n_v,
            "n_decision": self.n_x,
        }
        return RetargetResult(
            qpos=run.out_q,
            entity_qpos=dict(zip(kin.entity_names, kin.entity_qpos(run.out_q))),
            object_poses={o.name: run.out_obj[oi] for oi, o in enumerate(objects)},
            fps=human.fps,
            info=run.info,
            stats=stats,
        )

    def _run_frames(self, iterator, run: Run, starts, obj_pos, obj_rot) -> None:
        """The per-frame SQP loop, writing into ``run``. Frame 0 is solved from each configuration of ``starts`` and
        keeps the best solve (``_first_frame``).
        """
        info = run.info
        q = starts[0]
        q_prev, obj_prev_pos, obj_prev_rot = None, obj_pos.copy(), obj_rot.copy()
        for t in iterator:
            tf = time.time()
            frame = self._frame_inputs(run.human, t, q_prev, obj_prev_pos, obj_prev_rot, forward=True)
            for oi, o in enumerate(run.objects):
                if o.fixed:
                    obj_pos[oi], obj_rot[oi] = frame.demo_pos[oi], frame.demo_rot[oi]
            if t == 0:
                q, obj_pos, obj_rot, local, it = self._first_frame(run.human, starts, obj_pos, obj_rot, frame, info)
            else:
                q, obj_pos, obj_rot, local, it = self._solve_frame(q, obj_pos, obj_rot, frame, self.iterations, info, t)
            q_out, penetration = q, local.penetration
            for addon in self._addons:
                q, q_out, penetration = addon.after_frame(run, frame, q, obj_pos, obj_rot, penetration)
            run.out_q[t] = q_out
            info["penetration_mm"][t] = 1e3 * penetration
            for oi in range(len(run.objects)):
                run.out_obj[oi, t, :3] = obj_pos[oi]
                quat = R_to_quat(obj_rot[oi])
                # Continuous in sign: the previous output frame's hemisphere (the demonstrated pose's at frame 0).
                prev = run.out_obj[oi, t - 1, 3:] if t > 0 else run.objects[oi].poses[0, 3:]
                run.out_obj[oi, t, 3:] = quat if quat @ prev >= 0.0 else -quat
            info["iterations"][t] = it + 1
            info["cost"][t] = local.cost
            for k, v in local.terms.items():
                info[f"cost/{k}"][t] = v
            info["active_pairs"][t] = local.n_active
            info["solve_ms"][t] = 1e3 * (time.time() - tf)
            q_prev, obj_prev_pos, obj_prev_rot = q.copy(), obj_pos.copy(), obj_rot.copy()

    def _frame_inputs(self, human, t, q_prev, obj_prev_pos, obj_prev_rot, forward=False) -> Frame:
        """The targets and bounds of frame ``t`` (see :class:`Frame`). ``forward`` is True once per frame of the forward
        pass, in time order (add-ons advance their state on it).
        """
        robot_t, object_t, human_t = self._demo_targets(human, t)
        style_pos, style_rot = self._style_targets(human, t)
        demo_pos, demo_rot = self._demo_pos[t], self._demo_rot[t]
        demo_step = demo_delta = root_delta = None
        if t > 0:
            prev_rot = self._demo_rot[t - 1]
            demo_delta = np.concatenate(
                [
                    demo_pos - self._demo_pos[t - 1],
                    axis_angle_from_quat(R_to_quat(demo_rot @ np.transpose(prev_rot, (0, 2, 1)))),
                ],
                axis=1,
            )
            demo_step = np.abs(demo_delta)
            root_delta = human.joint_pos[t, self._pelvis_joint] - human.joint_pos[t - 1, self._pelvis_joint]
        frame = Frame(
            t,
            q_prev,
            obj_prev_pos,
            obj_prev_rot,
            demo_pos,
            demo_rot,
            demo_step,
            demo_delta,
            root_delta,
            robot_t,
            object_t,
            human_t,
            style_pos,
            style_rot,
            1.0 / human.fps,
        )
        for addon in self._addons:
            addon.frame(human, frame, forward)
        return frame

    def _first_frame(self, human, starts, obj_pos, obj_rot, frame, info):
        """The cold solve of frame 0 from each configuration of ``starts``; keeps the best one and adds its QP counts
        to ``info``. Returns what ``_solve_frame`` does. The add-ons rank the solves first (``Addon.rank_start``), then
        one without relaxed QPs wins, then the lower cost.
        """
        best = None
        for q in starts:
            local_info = _scratch_info()
            out = self._solve_frame(
                q, obj_pos.copy(), obj_rot.copy(), frame, self.first_frame_iterations, local_info, 0
            )
            for _ in range(3):
                # A limb can also end the cold solve inside an object (a thumb caught in a perforated chair back):
                # draw it back and solve the first frame again, before every later frame inherits it.
                untangled, moved = self._untangle(out[0], out[1], out[2])
                if not moved:
                    break
                out = self._solve_frame(untangled, out[1], out[2], frame, self.first_frame_iterations, local_info, 0)
            relaxed = local_info["qp_relaxed"][0] + local_info["qp_failed"][0] > 0
            ranks = tuple(a.rank_start(human, frame, out[0], out[1], out[2], len(starts)) for a in self._addons)
            score = (*ranks, relaxed, out[3].cost)
            if best is None or score < best[0]:
                best = (score, out, local_info)
        for k, v in best[2].items():
            info[k][0] += v[0]
        return best[1]

    def solve(self, q, obj_pos, obj_rot, frame: Frame, n_iter: int) -> tuple:
        """One solve of ``frame`` from ``q`` for an add-on's stage: ``(q, obj_pos, obj_rot, local, qp_failed)``.

        The object poses are copied; QP counts are not reported.
        """
        info = _scratch_info()
        q, obj_pos, obj_rot, local, _ = self._solve_frame(q, obj_pos.copy(), obj_rot.copy(), frame, n_iter, info, 0)
        return q, obj_pos, obj_rot, local, info["qp_failed"][0] > 0

    # ------------------------------------------------------------------
    # one SQP iterate
    # ------------------------------------------------------------------

    def _solve_frame(self, q, obj_pos, obj_rot, frame, n_iter, info, t):
        """Trust-region SQP on one frame (Eq. 8).

        Each iterate solves the Gauss-Newton QP inside the per-DoF step box. A step is taken if it drives no probe
        deeper into a surface than it was (beyond ``penetration_slack``). Otherwise it is solved again with a cutting
        plane per probe it drove deeper, or in a box half the size of the rejected step. If no step of the frame is
        taken and the frame did not converge, the least deep rejected one is: whole steps cycled (jitter), and
        rejected frames froze the robot. The cost is not compared: a step that drives no probe deeper is taken even
        if the Gauss-Newton model overpredicted its decrease.

        The reweighted interaction terms are evaluated afresh at the start of the frame and kept for its
        iterates (refreshed every ``reweight_every`` accepted steps), so the iterates of one frame descend
        one weighted least-squares problem. Iteration stops on a tiny step or the frame's budget.
        """
        local = self._linearize(q, obj_pos, obj_rot, frame)
        full = np.ones(self.n_x)
        scale = full.copy()
        accepted = 0
        it = 0
        cuts: dict = {}
        fallback = None  # the least deep rejected step, taken if the frame would otherwise not move
        converged = False
        for it in range(n_iter):
            step, status = self._solve_local(local, scale)
            info["qp_relaxed"][t] += status == "relaxed"
            info["qp_failed"][t] += status == "failed"
            if status == "failed":
                # No step is not convergence: retry in a smaller box (fewer rows reach their bounds).
                if scale.max() <= self.min_trust_scale:
                    break
                scale = np.maximum(0.5 * scale, self.min_trust_scale)
                continue
            if np.abs(step).max() < self.step_tolerance:
                # Interior to the box (its smallest half-width, min_trust_scale * trust_region, is above the
                # tolerance; checked at construction), so the current point solves the QP, cuts included.
                converged = True
                break
            q_new, pos_new, rot_new = self._apply_step(q, obj_pos, obj_rot, step)
            candidate = self._linearize(q_new, pos_new, rot_new, frame, frozen=local.frozen, cuts=cuts)
            deeper = self._deepened(local.contacts, candidate.contacts)
            if deeper and scale.max() > self.min_trust_scale:
                info["steps_rejected"][t] += 1
                if fallback is None or candidate.penetration < fallback[0]:
                    fallback = (candidate.penetration, q_new, pos_new, rot_new, candidate)
                if self._add_cuts(cuts, candidate.contacts, deeper):
                    # A probe at an edge or a thin wall slid over it (its linearised row allows that): the tangent
                    # plane of the surface it was driven into keeps it out while the rest of the body moves on.
                    local = self._linearize(q, obj_pos, obj_rot, frame, frozen=local.frozen, cuts=cuts)
                    continue
                # Shrink the box to half the rejected step, DoF by DoF: a box shrunk as a whole leaves a step well
                # inside it unchanged, and every retry was rejected alike.
                scale = np.maximum(0.5 * np.abs(step) / local.trust, self.min_trust_scale)
                continue
            q, obj_pos, obj_rot, local = q_new, pos_new, rot_new, candidate
            scale = full.copy()
            accepted += 1
            if cuts:
                # The planes steer the step that needed them and are dropped once it is taken: kept for the rest of
                # the frame they held fingers off the surfaces the demonstration touches.
                cuts = {}
                local = self._linearize(q, obj_pos, obj_rot, frame, frozen=local.frozen)
            if accepted % self.reweight_every == 0:
                local = self._linearize(q, obj_pos, obj_rot, frame)
        if accepted == 0 and fallback is not None and not converged:
            # Every step of the frame was rejected: a frame that does not move hands the same state to the next,
            # which rejects alike. The least deep rejected step is taken. Not after a converged solve: the cutting
            # planes showed that staying put is optimal, and the rejected step drove a probe deeper.
            _, q, obj_pos, obj_rot, local = fallback
            if cuts:
                local = self._linearize(q, obj_pos, obj_rot, frame, frozen=local.frozen)
        return q, obj_pos, obj_rot, local, it

    def _deepened(self, before: list, after: list) -> dict:
        """Probes that ``after`` puts deeper than ``before`` did, beyond ``penetration_slack``: ``{(kind, ch): idx}``.

        Probe by probe: one global maximum let a step drive probe A deeper while another probe (or a persistent
        capsule overlap) was deeper still, and blocked every step while one probe was stuck deep.
        """
        floor = -self.collision_margin
        was = {(kind, ci): dict(zip(idx.tolist(), d.tolist(), strict=True)) for kind, ci, idx, _, d, _ in before}
        out = {}
        for kind, ci, idx, _, d, _ in after:
            prev = was.get((kind, ci), {})
            limit = np.array([min(prev.get(i, floor), floor) for i in idx.tolist()]) - self.penetration_slack
            deep = idx[d < limit]
            if len(deep):
                out[(kind, ci)] = deep
        return out

    def _apply_step(self, q, obj_pos, obj_rot, step):
        """Configuration, free-object positions and rotations after ``step``."""
        q_new = self.kin.integrate(q, step[: self.kin.n_v])
        pos_new, rot_new = obj_pos.copy(), obj_rot.copy()
        for oi in self._free:
            s = self._free_slot[oi]
            pos_new[oi] = obj_pos[oi] + step[s : s + 3]
            rot_new[oi] = quat_to_R(quat_from_axis_angle(step[s + 3 : s + 6])) @ obj_rot[oi]
        return q_new, pos_new, rot_new

    @staticmethod
    def _add_cuts(cuts: dict, contacts: list, deepened: dict) -> int:
        """Add a cutting plane for every probe in ``deepened`` (from :meth:`_deepened`) that has none yet.

        ``cuts`` maps ``(kind, channel)`` to ``(probe indices, channel-frame points, distances, unit normals)``: the
        tangent plane of the surface each probe was driven into, at the rejected step. Returns the number added.
        """
        added = 0
        for kind, ci, idx, x, d, grad in contacts:
            if (kind, ci) not in deepened:
                continue
            empty = (np.zeros(0, dtype=np.int64), np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3)))
            k_old, x_old, d_old, n_old = cuts.get((kind, ci), empty)
            new = np.isin(idx, deepened[(kind, ci)]) & ~np.isin(idx, k_old)
            if not new.any():
                continue
            n = grad[new] / np.maximum(np.linalg.norm(grad[new], axis=1), 1e-12)[:, None]
            cuts[(kind, ci)] = (np.r_[k_old, idx[new]], np.r_[x_old, x[new]], np.r_[d_old, d[new]], np.r_[n_old, n])
            added += int(new.sum())
        return added

    def _solve_local(self, local: Local, trust_scale) -> tuple[np.ndarray, str]:
        """Solve the QP of a linearisation inside the step box scaled by ``trust_scale``."""
        trust = local.trust * trust_scale
        lo = np.maximum(local.hard_lo, -trust)
        hi = np.minimum(local.hard_hi, trust)
        bad = lo > hi
        if bad.any():
            # A hard bound unreachable within one trust step: move as far toward it as allowed.
            h_lo, h_hi, tr = local.hard_lo[bad], local.hard_hi[bad], trust[bad]
            # The midpoint is selected only where neither hard bound lies beyond the box, so both are finite there.
            toward = np.where(h_lo > tr, tr, np.where(h_hi < -tr, -tr, np.clip(0.5 * (h_lo + h_hi), -tr, tr)))
            lo[bad] = hi[bad] = toward
        l_in = local.l_in
        unreachable = False
        if len(local.C):
            # A row no step inside the box can satisfy is relaxed to the most the box allows, with a margin below the
            # capacity whatever its sign (0.95 * capacity is above a negative capacity).
            capacity = np.maximum(local.C * hi[None], local.C * lo[None]).sum(axis=1)
            unreachable = bool((l_in > capacity).any())
            l_in = np.minimum(l_in, capacity - 0.05 * np.abs(capacity))
        step, status = solve_qp(local.H, local.g, local.C, l_in, lo, hi)
        # A row lowered here is relaxed as much as one the elastic shift lowers: report it, or ``qp_relaxed`` and the
        # frame-0 ranking (fewer relaxed QPs wins) cannot see it.
        return step, "relaxed" if unreachable and status == "solved" else status

    def _linearize(
        self, q, obj_pos, obj_rot, frame: Frame, frozen: list | None = None, cuts: dict | None = None
    ) -> Local:
        """Linearise Eq. 8 at a configuration: the Gauss-Newton model, constraint rows and hard bounds.

        The interaction weights are iteratively reweighted (Sec. III-C): the kernels ``beta`` depend on the
        current distance, the per-link normalisation on the current active set. With ``frozen`` None they are
        evaluated here and recorded in the result; passing that record back evaluates the *same* weighted
        least-squares problem at another configuration, so the iterates of a frame descend one problem (a
        Gauss-Newton model built with fixed weights does not predict a cost whose weights move). Step acceptance
        reads penetration only (see :meth:`_solve_frame`); ``Local.cost`` is reported and ranks frame-0 starts.
        """
        kin = self.kin
        n_x, n_v = self.n_x, kin.n_v
        q_prev = frame.q_prev
        kin.set_configuration(q)
        link_pos, link_rot = kin.link_poses()
        link_jac = kin.link_jacobians(link_pos, link_rot)
        gn = GaussNewton(n_x, n_v, link_jac, self._terms)
        r_world = self.robot_probes.world(link_pos, link_rot)
        o_world = (
            obj_pos[self.object_probes.carrier]
            + np.einsum("nab,nb->na", obj_rot[self.object_probes.carrier], self.object_probes.local)
            if len(self.object_probes)
            else np.zeros((0, 3))
        )

        # ---- pass 1: proximity of every probe set to every channel ----------------
        sets: list[ProbeSet] = []
        carrier_count = np.zeros(len(kin.link_names) + len(self.objects))
        sid = -1
        for ci, channel in enumerate(self.channels):
            R, tau = self._channel_pose(ci, obj_pos, obj_rot)
            for kind, world, targets in (
                ("robot", r_world, frame.robot_t[ci]),
                ("object", o_world, frame.object_t[ci]),
            ):
                if len(world) == 0:
                    continue
                sid += 1
                x = (world - tau) @ R
                qr = channel.query(x)
                d = np.where(qr.valid, qr.distance, np.inf)
                demo_close = targets.distance < self.max_distance
                if kind == "object":
                    own = self.object_probes.carrier == self._channel_object[ci]
                    demo_close &= ~own
                    d = np.where(own, np.inf, d)
                # Off-grid probes the demonstration wants in contact (or that a frozen weighting still
                # holds a residual for), and every probe within ``exact_band`` of a mesh: read the exact
                # surface instead. Inside a feature a few grid cells thick the interpolated distance reads
                # a probe 4 mm deep as touching.
                wanted = demo_close.copy()
                if frozen is not None:
                    wanted[frozen[sid][0]] = True
                    wanted[frozen[sid][2]] = True
                far = wanted & ~qr.valid
                if isinstance(channel, MeshChannel):
                    far |= qr.valid & (d < self.exact_band)
                grad, wit, wjac = qr.distance_grad, qr.witness, qr.witness_jac
                if far.any():
                    ex = channel.exact_query(x[far])
                    d = d.copy()
                    d[far] = ex.distance
                    grad, wit, wjac = grad.copy(), wit.copy(), wjac.copy()
                    grad[far] = ex.distance_grad
                    wit[far] = ex.witness
                    wjac[far] = ex.witness_jac
                active = demo_close | (d < self.max_distance)
                if kind == "robot":
                    active &= self.src >= 0
                    carriers = self.robot_probes.carrier
                    collide = (d < self.collision_activation) & self._collision_probe
                else:
                    carriers = len(kin.link_names) + self.object_probes.carrier
                    collide = d < self.collision_activation
                    if self._channel_object[ci] not in self._free_slot:
                        # A fixed object's probe against the ground or a fixed object: no step moves either, so its
                        # row is all zeros, and a demonstrated table resting 3 mm into the floor read as 3 mm of
                        # penetration on every frame.
                        collide &= np.isin(self.object_probes.carrier, self._free)
                np.add.at(carrier_count, carriers[active], 1.0)
                sets.append(ProbeSet(kind, ci, x, d, grad, wit, wjac, targets, active, collide, world, R, tau))

        # ---- pass 2: residuals and non-penetration rows ------------------------------
        rows_C, rows_l = [], []
        lin = Linearization(frame, gn, obj_pos, obj_rot, link_pos, link_rot, link_jac, r_world, [], [])
        contacts = []
        n_active = 0
        record = []
        for sid, s in enumerate(sets):
            kind, ci, x, d, grad, wit, wjac, targets = s.kind, s.ci, s.x, s.d, s.grad, s.wit, s.wjac, s.targets
            world, R, tau = s.world, s.R, s.tau
            # Non-penetration: d + grad^T dx_c >= -eps.
            c_idx = np.flatnonzero(s.collide)
            if len(c_idx):
                rows = self._rows(kind, ci, c_idx, world[c_idx], R, tau, obj_pos, link_pos, grad[c_idx])
                rows_C.append(gn.dense(rows))
                rows_l.append(-self.collision_margin - d[c_idx])
                contacts.append((kind, ci, c_idx, x[c_idx], d[c_idx], grad[c_idx]))
            if cuts and (kind, ci) in cuts:
                # Cutting planes from rejected steps (see ``_solve_frame``): probe p stays on the outer side of the
                # tangent plane of the surface it was driven into, n . (x_p - x_cut) + d_cut >= -eps.
                k_idx, x_cut, d_cut, n_cut = cuts[(kind, ci)]
                rows = self._rows(kind, ci, k_idx, world[k_idx], R, tau, obj_pos, link_pos, n_cut)
                lin.support_C.append(gn.dense(rows))
                lin.support_l.append(-self.collision_margin - d_cut - ((x[k_idx] - x_cut) * n_cut).sum(axis=1))
            if frozen is None:
                weights = self._interaction_weights(kind, s.active, d, targets, carrier_count)
            else:
                weights = frozen[sid]
            record.append(weights)
            kk, w_d, kt, w_x, w_n = weights
            n_active += len(np.union1d(kk, kt))
            if len(kk):
                # r^d = d - (d_hat)_+
                d_hat_pos = np.maximum(np.minimum(targets.distance[kk], self.max_distance), 0.0)
                rows = self._rows(kind, ci, kk, world[kk], R, tau, obj_pos, link_pos, grad[kk])
                gn.add(f"{kind}_depth", rows, d[kk] - d_hat_pos, w_d)
            for addon in self._addons:
                addon.set_terms(lin, s)
            if len(kt):
                # r^x = rho_c(w; w_hat), the geodesic witness displacement, carried as the vector
                # (rho / |c|) c along the chord c = w - w_hat: the same cost rho^2, but smooth where the witness
                # reaches its target (a Gauss-Newton step on the cone rho overshoots and flips back every iterate).
                # Outward normals at both ends (the distance gradient at the current witness) keep each end on
                # its own face of a thin part when the geodesic table is interpolated.
                n_now = grad[kt] / np.maximum(np.linalg.norm(grad[kt], axis=1), 1e-12)[:, None]
                channel = self.channels[ci]
                if isinstance(channel, MeshChannel):
                    hat = self._geodesic_hat(channel, targets, kt)
                    rho, g_rho = channel.geodesic(wit[kt], targets.witness[kt], n_now, targets.normal[kt], hat=hat)
                else:
                    rho, g_rho = channel.geodesic(wit[kt], targets.witness[kt], n_now, targets.normal[kt])
                chord_vec = wit[kt] - targets.witness[kt]
                if isinstance(channel, PlaneChannel):
                    chord_vec[:, 2] = 0.0
                chord = np.linalg.norm(chord_vec, axis=1)
                tiny = chord < 1e-9
                safe = np.where(tiny, 1.0, chord)
                u = np.where(tiny[:, None], 0.0, chord_vec / safe[:, None])
                # Jacobian of r = rho u w.r.t. the witness: u grad(rho)^T + (rho / |c|) (I - u u^T). Its gradient
                # J^T r = rho grad(rho) is exact whatever factor multiplies the lateral part, so that factor only
                # shapes the curvature model. rho / |c| grows without bound when the witness sits across a thin part
                # from its target (geodesic round the edge, chord through it) and made the QP ill-conditioned: capped.
                lateral = np.where(tiny, 1.0, np.minimum(rho / safe, self.witness_lateral_cap))
                for axis in range(3):
                    d_w = u[:, axis, None] * g_rho - lateral[:, None] * u[:, axis, None] * u
                    d_w[:, axis] += lateral
                    gamma_x = np.einsum("ka,kab->kb", d_w, wjac[kt])
                    rows = self._rows(kind, ci, kt, world[kt], R, tau, obj_pos, link_pos, gamma_x)
                    gn.add(f"{kind}_witness", rows, rho * u[:, axis], w_x)
                # r^n = (sign(d_hat) n_hat . (x - w_hat))_-
                m_hat = targets.normal[kt]
                side = (m_hat * (x[kt] - targets.witness[kt])).sum(1)
                wrong = side < 0
                if wrong.any():
                    kw = kt[wrong]
                    rows = self._rows(kind, ci, kw, world[kw], R, tau, obj_pos, link_pos, m_hat[wrong])
                    gn.add(f"{kind}_normal", rows, side[wrong], w_n[wrong])

        # ---- self-collision ------------------------------------------------------------
        if self.capsules is not None and len(self.capsules.pairs):
            caps = self.capsules
            wa, wb = caps.world(link_pos, link_rot)
            i, j = caps.pairs[:, 0], caps.pairs[:, 1]
            pa, pb = segment_closest_points(wa[i], wb[i], wa[j], wb[j])
            dist = np.linalg.norm(pa - pb, axis=1)
            clearance = dist - caps.radius[i] - caps.radius[j]
            close = np.flatnonzero(clearance < self.self_collision_activation)
            if len(close):
                n = (pa[close] - pb[close]) / np.maximum(dist[close], 1e-9)[:, None]
                la, lb = caps.link[i[close]], caps.link[j[close]]
                ua = np.concatenate([n, np.cross(pa[close] - link_pos[la], n)], axis=1)
                ub = np.concatenate([n, np.cross(pb[close] - link_pos[lb], n)], axis=1)
                G = np.zeros((len(close), n_x))
                G[:, :n_v] = np.einsum("ki,kix->kx", ua, link_jac[la]) - np.einsum("ki,kix->kx", ub, link_jac[lb])
                rows_C.append(G)
                rows_l.append(-self.collision_margin - clearance[close])

        # ---- style --------------------------------------------------------------------
        sl = self._style_link_idx
        e_rot = axis_angle_from_quat(R_to_quat(np.einsum("lab,lcb->lac", link_rot[sl], frame.style_rot)))
        root_f = np.sqrt(self._style_weight)[:, None]  # per-link style weights, as row scales
        gn.add_dense(
            "style_rot",
            (root_f * e_rot).reshape(-1),
            pad((root_f[:, :, None] * link_jac[sl, 3:6]).reshape(-1, n_v), n_x),
            self.style_rot_weight,
        )
        e_pos = link_pos[sl] - frame.style_pos
        gn.add_dense(
            "style_pos",
            (root_f * e_pos).reshape(-1),
            pad((root_f[:, :, None] * link_jac[sl, 0:3]).reshape(-1, n_v), n_x),
            self.style_pos_weight,
        )
        for addon in self._addons:
            addon.terms(lin)
        demo_pos, demo_rot = frame.demo_pos, frame.demo_rot
        for oi in self._free:
            s = self._free_slot[oi]
            J = np.zeros((3, n_x))
            J[:, s : s + 3] = np.eye(3)
            gn.add_dense("object_pos", obj_pos[oi] - demo_pos[oi], J, self.object_pos_weight)
            J = np.zeros((3, n_x))
            J[:, s + 3 : s + 6] = np.eye(3)
            gn.add_dense(
                "object_rot", axis_angle_from_quat(R_to_quat(obj_rot[oi] @ demo_rot[oi].T)), J, self.object_rot_weight
            )
        if q_prev is not None:
            gn.add_dense("smooth", kin.difference(q, q_prev), pad(np.eye(n_v), n_x), self.smooth_weight)
        moved = {
            oi: np.concatenate(
                [
                    obj_pos[oi] - frame.obj_prev_pos[oi],
                    axis_angle_from_quat(R_to_quat(obj_rot[oi] @ frame.obj_prev_rot[oi].T)),
                ]
            )
            for oi in self._free
        }
        if frame.demo_delta is not None and self.object_smooth_weight > 0:
            # Each free object moves between frames as the demonstrated one did: the residual is its
            # displacement minus the demonstrated displacement, so jitter is damped, not the motion.
            for oi in self._free:
                s = self._free_slot[oi]
                J = np.zeros((6, n_x))
                J[:, s : s + 6] = np.eye(6)
                gn.add_dense("object_smooth", moved[oi] - frame.demo_delta[oi], J, self.object_smooth_weight)

        # ---- box constraints: limits, velocity, object boxes, trust region ----------------
        hard_lo = np.full(n_x, -np.inf)
        hard_hi = np.full(n_x, np.inf)
        joints = q[kin.n_base_q :]
        hard_lo[kin.n_base_v : n_v] = kin.joint_lower - joints
        hard_hi[kin.n_base_v : n_v] = kin.joint_upper - joints
        dt = frame.dt
        if q_prev is not None:
            prev = q_prev[kin.n_base_q :]
            hard_lo[kin.n_base_v : n_v] = np.maximum(hard_lo[kin.n_base_v : n_v], prev - self._vel_limit * dt - joints)
            hard_hi[kin.n_base_v : n_v] = np.minimum(hard_hi[kin.n_base_v : n_v], prev + self._vel_limit * dt - joints)
            if kin.floating and frame.root_delta is not None and self.root_speed_slack > 0:
                # The root moves as the demonstrator's root joint did, within the slack (see ``root_speed_slack``).
                centre = q_prev[0:3] + frame.root_delta - q[0:3]
                hard_lo[0:3] = np.maximum(hard_lo[0:3], centre - self.root_speed_slack * dt)
                hard_hi[0:3] = np.minimum(hard_hi[0:3], centre + self.root_speed_slack * dt)
        b_cor = np.repeat(np.asarray(self.object_corridor, dtype=np.float64), 3)
        slack = np.repeat(np.asarray(self.object_speed_slack, dtype=np.float64), 3) * dt
        for oi in self._free:
            s = self._free_slot[oi]
            cum = np.concatenate(
                [obj_pos[oi] - demo_pos[oi], axis_angle_from_quat(R_to_quat(obj_rot[oi] @ demo_rot[oi].T))]
            )
            hard_lo[s : s + 6] = np.maximum(hard_lo[s : s + 6], -b_cor - cum)
            hard_hi[s : s + 6] = np.minimum(hard_hi[s : s + 6], b_cor - cum)
            if frame.demo_step is not None:
                b_env = frame.demo_step[oi] + slack
                hard_lo[s : s + 6] = np.maximum(hard_lo[s : s + 6], -b_env - moved[oi])
                hard_hi[s : s + 6] = np.minimum(hard_hi[s : s + 6], b_env - moved[oi])
        if frame.free_cols is not None:
            # Only these velocity columns move (an add-on's stage); the objects are held.
            held = np.ones(n_x, dtype=bool)
            held[:n_v][frame.free_cols] = False
            hard_lo[held] = 0.0
            hard_hi[held] = 0.0
        trust = np.full(n_x, self.trust_region[1])
        trust[0:3] = self.trust_region[0]
        for oi in self._free:
            trust[self._free_slot[oi] : self._free_slot[oi] + 3] = self.trust_region[0]

        C = np.concatenate(rows_C) if rows_C else np.zeros((0, n_x))
        l_in = np.concatenate(rows_l) if rows_l else np.zeros(0)
        # Each row reads d + G dx >= -eps, so its lower bound is -eps - d: the deepest penetration is max(-d).
        penetration = max(0.0, float((l_in + self.collision_margin).max())) if len(l_in) else 0.0
        if lin.support_C:
            C = np.concatenate([C, *lin.support_C])
            l_in = np.concatenate([l_in, *lin.support_l])
        # Unit-gradient rows: a row no step can move (an interpolated SDF gradient cancelling on the
        # centreline of a feature thinner than the grid) is dropped — it would also break the solver's
        # row equilibration — and the rest are normalised for conditioning.
        norms = np.linalg.norm(C, axis=1)
        keep = norms > 1e-9
        C = C[keep] / norms[keep, None]
        l_in = l_in[keep] / norms[keep]
        return Local(
            gn.H,
            gn.g,
            C,
            l_in,
            hard_lo,
            hard_hi,
            trust,
            gn.terms,
            n_active,
            penetration,
            record,
            gn.term_grads,
            contacts,
        )

    @staticmethod
    def _geodesic_hat(channel: MeshChannel, targets: Targets, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Geodesic-table weights at the target witnesses ``rows``, each computed once per frame.

        The target end of a witness residual is fixed for the frame, but a frame linearises about nine times (every
        iterate and every candidate step). The cache lives on the frame's ``targets`` and is dropped if its witnesses
        or normals are replaced.
        """
        cache = targets.geodesic_hat
        if cache is None or cache[0] is not targets.witness or cache[1] is not targets.normal:
            n = len(targets.witness)
            cache = targets.geodesic_hat = [targets.witness, targets.normal, None, None, np.zeros(n, dtype=bool)]
        todo = rows[~cache[4][rows]]
        if len(todo):
            idx, alpha = channel.geodesics.endpoint_weights(targets.witness[todo], targets.normal[todo])
            if cache[2] is None:
                cache[2] = np.zeros((len(cache[4]), idx.shape[1]), dtype=idx.dtype)
                cache[3] = np.zeros((len(cache[4]), idx.shape[1]))
            cache[2][todo], cache[3][todo] = idx, alpha
            cache[4][todo] = True
        return cache[2][rows], cache[3][rows]

    def _interaction_weights(self, kind, active, d, targets, carrier_count):
        """Pairs and weights of the interaction residuals of one probe set against one channel (Eq. 6).

        Returns ``(depth pairs, depth weights, tangential pairs, witness weights, normal weights)``: pairs
        whose kernel vanishes are left out.
        """
        lam_d, lam_x, lam_n = self.interaction_weights
        sigma, beta_max = self.kernel_width, self.kernel_cap
        k_idx = np.flatnonzero(active)
        empty = (np.zeros(0, dtype=np.int64), np.zeros(0))
        if len(k_idx) == 0:
            return (*empty, *empty, np.zeros(0))
        n_links = len(self.kin.link_names)
        if kind == "robot":
            carriers = self.robot_probes.carrier[k_idx]
            scale = 1.0
            f = self._link_weight[carriers]
        else:
            carriers = n_links + self.object_probes.carrier[k_idx]
            scale = self.object_interaction_scale
            f = self._object_weight[carriers - n_links]
        kappa = scale * f / np.maximum(carrier_count[carriers], 1.0)
        d_hat_pos = np.maximum(np.minimum(targets.distance[k_idx], self.max_distance), 0.0)
        beta = np.minimum(np.clip(1.0 - np.minimum(d_hat_pos, d[k_idx]) / sigma, 0.0, None) ** 2, beta_max)
        beta_bar = np.clip(1.0 - d_hat_pos / sigma, 0.0, None) ** 2
        w_d = kappa * lam_d * beta**2
        keep = w_d > 0
        tang = beta_bar > 0
        return (
            k_idx[keep],
            w_d[keep],
            k_idx[tang],
            kappa[tang] * lam_x * beta_bar[tang] ** 2,
            kappa[tang] * lam_n * beta_bar[tang] ** 2,
        )

    def _rows(self, kind, ci, idx, world, R, tau, obj_pos, link_pos, gamma) -> Rows:
        """Rows of scalar residuals with channel-frame gradient ``gamma`` (``d r / d x_c``) at probes ``idx``.

        With ``x_c = R^T (X - tau)``, the world gradient is ``c = R gamma``. A robot probe on link ``l``
        moves with ``J_v - [X - o_l]x J_omega``, giving the 6-vector ``[c; (X - o_l) x c]`` against the link's
        spatial Jacobian; a probe on a free object moves with its ``(dtau, dtheta)``; a free channel object
        moves ``x_c`` by ``-R^T dtau + R^T [X - tau]x dtheta``.
        """
        c = gamma @ R.T
        n_obj = 6 * len(self._free)
        u_o = np.zeros((len(idx), n_obj))
        if kind == "robot":
            link = self.robot_probes.carrier[idx]
            u_r = np.concatenate([c, np.cross(world - link_pos[link], c)], axis=1)
        else:
            link = None
            u_r = None
            carriers = self.object_probes.carrier[idx]
            for oi in np.unique(carriers):
                if oi in self._free_slot:
                    m = carriers == oi
                    s = self._free_slot[oi] - self.kin.n_v
                    u_o[m, s : s + 3] = c[m]
                    u_o[m, s + 3 : s + 6] = np.cross(world[m] - obj_pos[oi], c[m])
        oc = self._channel_object[ci]
        if oc in self._free_slot:
            s = self._free_slot[oc] - self.kin.n_v
            u_o[:, s : s + 3] -= c
            u_o[:, s + 3 : s + 6] += np.cross(c, world - tau)
        return Rows(link, u_r, u_o)


#: Cost terms of the method, reported per frame in ``RetargetResult.info`` (``cost/<term>``); add-ons add theirs.
_TERMS = (
    "robot_depth",
    "robot_witness",
    "robot_normal",
    "object_depth",
    "object_witness",
    "object_normal",
    "style_rot",
    "style_pos",
    "object_pos",
    "object_rot",
    "smooth",
    "object_smooth",
)
#: Per-frame diagnostics of the method.
_INFO = (
    "iterations",
    "cost",
    "active_pairs",
    "solve_ms",
    "qp_relaxed",
    "qp_failed",
    "steps_rejected",
    "penetration_mm",
)


def _scratch_info() -> dict[str, np.ndarray]:
    """QP counters of a solve that is not reported per frame."""
    return {k: np.zeros(1) for k in ("qp_relaxed", "qp_failed", "steps_rejected")}


def _match_weight(name: str, weights: dict[str, float]) -> float:
    """``f_l`` of a link or object: the first ``weights`` regex that fully matches ``name``, else 1."""
    for pattern, value in weights.items():
        if re.fullmatch(pattern, name):
            return float(value)
    return 1.0


def _voxel_subset(local: np.ndarray, carrier: np.ndarray, spacing) -> np.ndarray:
    """Mask keeping one point per ``spacing`` voxel (scalar, or one per point) of each carrier's frame."""
    spacing = np.broadcast_to(np.asarray(spacing, dtype=np.float64), (len(local),))[:, None]
    keys = np.concatenate([carrier[:, None], np.floor(local / spacing).astype(np.int64)], axis=1)
    _, first = np.unique(keys, axis=0, return_index=True)
    mask = np.zeros(len(local), dtype=bool)
    mask[first] = True
    return mask
