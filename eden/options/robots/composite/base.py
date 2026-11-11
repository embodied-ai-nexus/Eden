"""Base options for composite robots (an arm + an attached gripper, and kin).

A composite robot bundles several unit-robot :class:`RobotOptions` into one rig
that a scene expands into its constituent entities + welds. Unlike the unit
robots in ``eden/options/robots``, a composite is *itself* a validated Options
(a :class:`~eden.options.entities.CompositeEntityOptions`), so it round-trips
through the config system, can declare its own fields, and is added to a scene by
name like any other entity::

    from eden.options.robots import Droid

    droid = Droid(base_pos=(0.0, 0.0, 0.4))
    scene = en.scenes.Empty(is_support_enabled=True, robot=droid)   # -> robot (arm) + gripper + weld
    # ...the same object exposes the controller metadata as shortcut properties:
    DifferentialIKController.configure(entity_name=droid.arm_name,
                                       ee_link_name=droid.ee_link,
                                       dofs_name=droid.arm_dofs)

Two end-effector cases share one arm base (:class:`_ArmCompositeOptions`):
:class:`ManipulatorOptions` bolts on a two-finger **gripper** (the *manipulator*);
:class:`DexManipulatorOptions` bolts on a multi-DOF dexterous **hand**. Concrete
rigs (Franka + Robotiq, UR5e, Franka + XHand, …) subclass one of these with default
fields and register in :data:`COMPOSITE_ROBOT_REGISTRY`. A humanoid carrying an end-effector
per wrist is the third shape, :class:`HumanoidOptions` — a sibling of the arm base rather than
a subclass, because its root is a humanoid with *two* welds instead of an arm with one (see
:mod:`eden.options.robots.composite.oli_wuji`).
mjlab attaches the end-effector to each arm at a tool frame via ``with_gripper``;
Eden's equivalent is the scene ``attachments_dict``, which the composite fills in
from :meth:`expand`.
"""

from __future__ import annotations

import numpy as np
from genesis.typing import UnitVec4FType, Vec3FType
from genesis.utils import geom as gu

from eden.options.entities import (
    AttachmentSpec,
    BaseEntityOptions,
    CompositeEntityOptions,
    EntityOptions,
    RobotOptions,
)
from eden.utils.geom import align_up_and_front
from eden.utils.registry import Registry


def weld_quat(child: EntityOptions, mount_quat: UnitVec4FType) -> tuple[float, float, float, float]:
    """The ``attach_quat`` that makes ``mount_quat`` a rotation of URDF *link* frames.

    Genesis applies a weld's quaternion **before** the morph frame offset, and Eden keeps the
    child's ``up``/``front`` alignment in that offset (``Entity._create_morph_from_options``), so
    the parent link gets ``attach @ align``. Right-multiplying by ``align^-1`` cancels it, making
    ``mount_quat`` mean the child's base-link orientation in the parent link frame — the identity
    when the two URDF frames coincide.

    A no-op for a child with the Eden default axes. The Wuji hands' alignment is a 180 deg
    rotation, so leaving it in welds the hand pointing back up the forearm, silently.

    Parameters
    ----------
    child : EntityOptions
        The entity being welded, read for its ``up`` / ``front`` declaration. A primitive has no
        asset axes to align and is not a supported end-effector.
    mount_quat : array-like[float, float, float, float]
        Intended (w, x, y, z) orientation of the child's base link in the parent link frame.
        Scaling is harmless (``quat_to_R`` normalizes); zero raises, since Genesis would only
        surface a bare ``ZeroDivisionError``.

    Returns
    -------
    tuple[float, float, float, float]
        The (w, x, y, z) quaternion to pass as the weld's ``attach_quat``.
    """
    quat = np.asarray(mount_quat, dtype=np.float64)
    if np.linalg.norm(quat) < 1e-12:
        raise ValueError(f"mount_quat must be a non-zero quaternion (w, x, y, z); got {tuple(mount_quat)}.")
    align = align_up_and_front(child.up, child.front)
    return tuple(float(v) for v in gu.R_to_quat(gu.quat_to_R(quat) @ align.T))


def has_frame_alignment(child: EntityOptions) -> bool:
    """Whether ``child``'s ``up``/``front`` rotate its asset at load, so a weld must cancel them."""
    return not np.allclose(align_up_and_front(child.up, child.front), np.eye(3), atol=1e-9)


def _apply_base_pos(base_pos: tuple[float, float, float] | None, body: RobotOptions) -> None:
    """Write the composite's opt-in placement knob onto its primary ``body`` (no-op when unset).

    The equality check avoids a redundant assignment when the body already carries the requested
    placement. The two ways a rig is duplicated deliberately treat drifted state (a ``body`` pose
    mutated directly after construction, out of sync with ``base_pos``) differently:
    ``copy.deepcopy`` is a snapshot — it does not re-run this hook, so the drift survives — while
    validated load re-runs it, so the file's ``base_pos`` is re-applied over the serialized body
    pose. Pinned by ``test_base_pos_is_reapplied_on_load_but_not_on_copy``.
    """
    if base_pos is None:
        return
    base_pos = tuple(base_pos)
    if tuple(body.default_root_pos) != base_pos:
        body.default_root_pos = base_pos


def _validate_welded_child(field: str, child: EntityOptions) -> None:
    """Reject a world-pinned end-effector: a welded child is posed by its parent link.

    ``attach_to`` drops the child's root joint and carries it on the parent link, so pinning it
    to the world as well is a contradiction the sim resolves by nailing the end-effector to the
    origin while the arm (or humanoid) walks away from it. Nothing downstream catches that — the
    scene builds, the weld is emitted, and only the trajectory is wrong — so it is checked where
    the rig is written. Every in-tree rig already passes ``is_fixed_base=False``; this is for the
    rigs assembled from the shared bases with a caller's own end-effector.
    """
    if getattr(child, "is_fixed_base", False):
        raise ValueError(
            f"{field}={type(child).__name__}(is_fixed_base=True) cannot be welded: a welded child "
            f"is posed by its parent link, so it must not also be pinned to the world. Pass "
            f"{type(child).__name__}(is_fixed_base=False). Only the rig's root body is pinned."
        )


class _ArmCompositeOptions(CompositeEntityOptions):
    """Shared base for a single-arm rig: an arm + one end-effector welded to it.

    Owns the arm, the link used as the IK/attach frame, the grasp-point links, the
    optional arm base-placement knob, and the arm-DOF shortcut. Subclasses add the
    second unit robot (a gripper or a dexterous hand) under a task-friendly field name
    and implement the composite contract (:meth:`entity_parts` / :meth:`attachments`);
    the base class turns those into the scene expansion (deep-copied) and the
    primary-name check.

    Private, and **not constructible**: it declares no :meth:`entity_parts`, so
    :meth:`CompositeEntityOptions.model_post_init` raises ``NotImplementedError`` on any
    attempt to instantiate it. Use :class:`ManipulatorOptions` (arm + gripper) or
    :class:`DexManipulatorOptions` (arm + dex hand), or subclass this to add a third arm
    shape. The underscore is the signal; the raise is the enforcement.
    """

    #: The arm unit robot. Subclasses provide a concrete default.
    arm: RobotOptions
    #: Link on the arm where the end-effector attaches; also the IK frame.
    ee_link: str
    #: End-effector links whose midpoint is the grasp point (the EE used by
    #: rewards/obs): the gripper pads for a gripper, the fingertips for a dex hand.
    grasp_links: list[str] = []
    #: Scene entity name for the arm — this IS the entity name in the scene, and how
    #: downstream terms reference it (e.g. ``entity_name="robot"``). The arm is the
    #: primary body, so the scene placement key must equal it. Give two composites in
    #: one scene distinct names (e.g. ``arm_name="left"``).
    arm_name: str = "robot"
    #: Convenience knob to place the arm base (tasks that mount the arm on a table raise
    #: it along z). ``None`` leaves ``arm.default_root_pos`` untouched, so a caller can set
    #: the placement on ``arm`` directly instead.
    base_pos: tuple[float, float, float] | None = None
    #: Optional mounting offset of the end-effector base **in the ``ee_link`` frame** (named to
    #: match ``CameraOptions.attach_offset`` / ``attach_quat``). ``None`` (the default) welds the
    #: end-effector at its own ``default_root_pos`` / ``default_root_quat``; setting ``attach_offset``
    #: and/or ``attach_quat`` overrides that with an explicit transform
    #: (https://github.com/Genesis-Embodied-AI/Genesis/pull/3041), so a rig can state the
    #: tool-frame → end-effector pose here instead of baking it into the asset or a dedicated attach
    #: body. Setting only one component leaves the other at the end-effector's own
    #: ``default_root_pos`` / ``default_root_quat`` (never silently reset to zero / identity). A rig
    #: that defaults a component to its own mount refuses a caller who sets only the other one, which
    #: would keep half of that mount.
    attach_offset: Vec3FType | None = None
    attach_quat: UnitVec4FType | None = None

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        self._check_whole_mount()
        # The arm base placement is a single knob on the composite (opt-in).
        _apply_base_pos(self.base_pos, self.arm)

    def _check_whole_mount(self) -> None:
        """Refuse one mount component set over a rig's default for the other, which would mix two mounts."""
        mount = ("attach_offset", "attach_quat")
        given = [name for name in mount if name in self.model_fields_set]
        if len(given) != 1:
            return
        (other,) = (name for name in mount if name not in given)
        kept = type(self).model_fields[other].default
        if kept is None:
            return
        rig = type(self).__name__
        own_pose = {"attach_offset": "default_root_pos", "attach_quat": "default_root_quat"}[other]
        raise ValueError(
            f"{rig}({given[0]}=...) sets half of a mount: {rig}'s {other} defaults to its own mount, {kept}, and the "
            f"weld would keep it. Set both, {rig}(attach_offset=(x, y, z), attach_quat=(w, x, y, z)), or pass "
            f"{other}=None to take the end-effector's own {own_pose} for it."
        )

    @property
    def arm_dofs(self) -> list[str]:
        """Arm DOFs driven by the joint-position / IK controller (== ``arm.dofs_name``)."""
        return list(self.arm.dofs_name)

    def _attach_spec(self, ee: EntityOptions) -> AttachmentSpec:
        """Attachment spec welding the end-effector ``ee`` onto the arm's ``ee_link`` (see :attr:`attach_offset`).

        ``ee`` must be an :class:`~eden.options.entities.EntityOptions` — the weld reads its
        ``up``/``front`` axes, which a bare primitive does not declare.

        Without a mount offset this is the 2-tuple ``(arm_name, ee_link)`` — the end-effector mounts
        at its own ``default_root_pos`` / ``default_root_quat``. Setting ``attach_offset`` and/or
        ``attach_quat`` emits the 4-tuple ``(arm_name, ee_link, attach_offset, attach_quat)`` that
        overrides it. An unset component falls back to the end-effector's own ``default_root_pos`` /
        ``default_root_quat`` — the same per-component default the 2-tuple / ``RigidEntity.attach_to``
        would apply — so setting only one component never silently discards the other from the asset.

        The quaternion goes through :func:`weld_quat`, which is also why an end-effector with a
        frame alignment never takes the 2-tuple shortcut — that path would reach Genesis
        uncancelled.
        """
        if self.attach_offset is None and self.attach_quat is None and not has_frame_alignment(ee):
            return (self.arm_name, self.ee_link)
        pos, quat = self._mount_pose(ee)
        return (self.arm_name, self.ee_link, pos, weld_quat(ee, quat))

    def _mount_pose(self, ee: EntityOptions) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """The end-effector base link's ``(pos, wxyz quat)`` in the ``ee_link`` frame, per-component defaulted."""
        pos = tuple(ee.default_root_pos) if self.attach_offset is None else tuple(self.attach_offset)
        quat = tuple(ee.default_root_quat) if self.attach_quat is None else tuple(self.attach_quat)
        return pos, quat


class ManipulatorOptions(_ArmCompositeOptions):
    """An arm + an attached two-finger gripper, added to a scene as one rig.

    The gripper is welded to ``arm``'s ``ee_link`` (see :meth:`attachments`). The
    gripper's driven DOFs and open/close targets are exposed as shortcut properties over
    the sub-option, so they can't drift from the robot definition.
    """

    #: The gripper unit robot. Subclasses provide a concrete default.
    gripper: RobotOptions
    #: Scene entity name for the gripper (e.g. ``gripper_name="left_gripper"``).
    gripper_name: str = "gripper"

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        _validate_welded_child("gripper", self.gripper)

    # --- shortcut properties over the gripper sub-option (single source of truth) ---
    @property
    def gripper_dofs(self) -> list[str]:
        """Gripper DOFs the binary controller drives (== keys of ``gripper.open_dofs_pos``)."""
        return list(self.gripper.open_dofs_pos.keys())

    @property
    def gripper_open(self) -> dict[str, float]:
        return dict(self.gripper.open_dofs_pos)

    @property
    def gripper_close(self) -> dict[str, float]:
        return dict(self.gripper.close_dofs_pos)

    @property
    def stroke(self) -> float:
        """Pad-to-pad opening (m) at ``gripper_open`` (== ``gripper.stroke``)."""
        if self.gripper.stroke is None:
            raise ValueError(
                f"{type(self.gripper).__name__} declares no stroke; set the ``stroke`` ClassVar on the gripper "
                f"options (the pad-to-pad opening at open_dofs_pos) to use it in a {type(self).__name__}."
            )
        return float(self.gripper.stroke)

    @property
    def tcp_offset(self) -> tuple[float, float, float]:
        """The tool point — pad midpoint with the gripper open — in the ``ee_link`` frame (m).

        The gripper's own ``tcp_offset`` (declared in its base-link frame) carried through the
        mounting pose the weld applies, so a rig with an ``attach_offset`` / ``attach_quat`` reports
        the point where the pads actually are. The config-time twin of the grasp frame the
        robosuite terms measure on the built rig.
        """
        if self.gripper.tcp_offset is None:
            raise ValueError(
                f"{type(self.gripper).__name__} declares no tcp_offset; set the ``tcp_offset`` ClassVar on the "
                f"gripper options (pad midpoint in its base-link frame) to use it in a {type(self).__name__}."
            )
        pos, quat = self._mount_pose(self.gripper)
        tcp = gu.quat_to_R(np.asarray(quat, dtype=np.float64)) @ np.asarray(self.gripper.tcp_offset, dtype=np.float64)
        return tuple(float(v) for v in np.asarray(pos, dtype=np.float64) + tcp)

    @property
    def max_substep_dt(self) -> float | None:
        """Finest substep either half of the rig needs (``None`` when neither declares one).

        A rig is only as stiff as its stiffest constraint allows: welding a Robotiq onto an arm that
        imposes nothing still means the *rig* cannot be stepped above the Robotiq's limit. Read it
        when choosing ``EnvOptions.sim_dt`` / ``sim_substeps`` for a scene that holds the rig.
        """
        limits = [limit for limit in (self.arm.max_substep_dt, self.gripper.max_substep_dt) if limit is not None]
        return min(limits) if limits else None

    def hold_width(self, width: float, clearance: float = 0.002) -> dict[str, float]:
        """Finger DOF targets that put the pads ``width + 2 * clearance`` apart.

        Linear between ``gripper_open`` (pads :attr:`stroke` apart) and ``gripper_close`` (pads
        touching), clamped to that range — so a Franka Hand holding a 4 cm peg gets each prismatic
        finger at ~2.2 cm, and a Robotiq the matching driver angle. Meant for objects welded to the
        hand; a free object is grasped by the gripper action instead.
        """
        closed = min(max(1.0 - (width + 2.0 * clearance) / self.stroke, 0.0), 1.0)
        return {
            dof: self.gripper_open[dof] + closed * (self.gripper_close[dof] - self.gripper_open[dof])
            for dof in self.gripper_dofs
        }

    def preset_hold(self, width: float, clearance: float = 0.002) -> None:
        """Start the gripper closed on an object ``width`` m wide: writes :meth:`hold_width` into its defaults.

        Unlike :meth:`hold_width`, which clamps because it is asked for a pose and a pose always
        exists, this refuses a width the gripper cannot close on. The two differ because the
        outcomes differ: a clamped *query* is a number the caller can inspect, whereas a clamped
        *preset* silently starts the episode with the gripper wide open around an object welded into
        it — a rig that looks like it is holding something it is not, with nothing to notice.
        """
        if width <= 0.0:
            raise ValueError(
                f"{type(self).__name__}.preset_hold needs the width of an object to hold, got {width} m. "
                f"To start the gripper simply shut, set its defaults from gripper_close instead."
            )
        opening = width + 2.0 * clearance
        if opening > self.stroke:
            raise ValueError(
                f"{type(self).__name__} cannot hold a {width} m object at {clearance} m clearance: that needs a "
                f"{opening} m opening and {type(self.gripper).__name__}'s stroke is {self.stroke} m. Widths up to "
                f"{self.stroke - 2.0 * clearance} m fit at this clearance."
            )
        self.gripper.default_dofs_pos.update(self.hold_width(width, clearance))

    # --- composite contract: name -> sub-entity, plus the arm->gripper weld ---
    def entity_parts(self) -> list[tuple[str, BaseEntityOptions]]:
        return [(self.arm_name, self.arm), (self.gripper_name, self.gripper)]

    def attachments(self) -> dict[str, AttachmentSpec]:
        return {self.gripper_name: self._attach_spec(self.gripper)}


class DexManipulatorOptions(_ArmCompositeOptions):
    """An arm + an attached multi-DOF dexterous hand, added to a scene as one rig.

    The dexterous counterpart of :class:`ManipulatorOptions`: the two-finger gripper is
    replaced by a hand welded to ``arm``'s ``ee_link`` and driven in joint space over
    *every* DOF (:attr:`hand_dofs`) — a dexterous hand has no scalar open/close target,
    so there is deliberately no ``hand_open`` / ``hand_close``. The grasp point is the
    centroid of the fingertip links (the inherited :attr:`grasp_links`, aliased
    :attr:`fingertip_links` for readability at dex call sites). Concrete rigs (Franka +
    XHand, xArm + XHand, KUKA + SharPa, …) subclass it with default fields and register
    in :data:`COMPOSITE_ROBOT_REGISTRY`.
    """

    #: The dexterous hand unit robot. Subclasses provide a concrete default.
    hand: RobotOptions
    #: Scene entity name for the hand (e.g. ``hand_name="left_hand"``).
    hand_name: str = "hand"

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        _validate_welded_child("hand", self.hand)

    # --- shortcut properties over the hand sub-option (single source of truth) ---
    @property
    def hand_dofs(self) -> list[str]:
        """Hand DOFs driven in joint space by the PD controller (== ``hand.dofs_name``)."""
        return list(self.hand.dofs_name)

    @property
    def fingertip_links(self) -> list[str]:
        """Fingertip links whose centroid is the grasp point (== ``grasp_links``)."""
        return list(self.grasp_links)

    # --- composite contract: name -> sub-entity, plus the arm->hand weld ---
    def entity_parts(self) -> list[tuple[str, BaseEntityOptions]]:
        return [(self.arm_name, self.arm), (self.hand_name, self.hand)]

    def attachments(self) -> dict[str, AttachmentSpec]:
        return {self.hand_name: self._attach_spec(self.hand)}


class HumanoidOptions(CompositeEntityOptions):
    """Shared base for a bimanual humanoid rig: a hand-stripped humanoid + an end-effector per wrist.

    The bimanual counterpart of :class:`_ArmCompositeOptions`. Where an arm rig welds one
    end-effector to one arm, a humanoid rig welds a **left and a right** one — a dexterous hand
    or a two-finger gripper, whichever the embodiment carries — onto the flange frames the
    humanoid asset declares, and so expands into *three* scene entities rather than two::

        rig = HumanoidOptions(robot=OliNoHands(), left_hand=..., right_hand=...,
                              left_ee_link="left_hand_mount", right_ee_link="right_hand_mount")
        scene = en.scenes.Empty(robot=rig)
        # -> entities "robot", "left_hand", "right_hand"

    The humanoid is the composite's primary body, so the scene placement key must equal
    :attr:`robot_name`. Concrete rigs (LimX Oli + Wuji hands, …) subclass this with default
    fields and register in :data:`COMPOSITE_ROBOT_REGISTRY`; the useful default is the
    *hand-stripped* humanoid variant (``OliNoHands``, ``G1Dofs29NoHands``), so the arms carry
    the real end-effector's mass and not that plus a moulded phantom one.

    Everything a controller needs is a shortcut property over the sub-options, so the DOF lists
    and grasp links cannot drift from the robot definitions::

        JointPositionAction.configure(entity_name=rig.robot_name, dofs_name=rig.robot_dofs)
        DifferentialIKController.configure(entity_name=rig.robot_name, ee_link_name=rig.right_ee_link)

    A gripper-ended rig reads its open/close targets off the sub-option
    (``rig.left_hand.open_dofs_pos``): unlike :class:`ManipulatorOptions`, this base stays
    end-effector agnostic, since a humanoid can wear a different one per side.

    The mount transform is likewise shared by default (:attr:`mount_offset` /
    :attr:`mount_quat`) but overridable per wrist (:attr:`left_mount_quat` and friends), because
    "both flanges take the same transform" holds only where the asset frames are mirror images —
    true of Oli's ``*_hand_mount`` pair, not a property of humanoids in general.

    What the rig stands on follows the same shape: :attr:`feet_links` reads the humanoid's own
    ``feet_links_name`` unless :attr:`feet_links_name` overrides it, for a rig whose ground
    contact is not the humanoid's declared feet.
    """

    #: The humanoid, and the composite's primary body. Subclasses provide a concrete default.
    robot: RobotOptions
    #: The end-effectors — a dexterous hand or a gripper per side. Subclasses provide concrete
    #: defaults, and must make them free-base: a welded child is posed by its parent link, so it
    #: must not also be world-pinned. Enforced at construction — an ``is_fixed_base=True``
    #: end-effector would otherwise build fine and simulate wrong.
    left_hand: RobotOptions
    right_hand: RobotOptions

    #: Flange frames the end-effectors bolt to, and the natural IK frames for arm control. Name
    #: the asset's own mount frame where it has one (e.g. ``left_hand_mount``) rather than the
    #: parent wrist link — that is what keeps :attr:`mount_offset` an identity instead of a
    #: number re-derived here.
    left_ee_link: str
    right_ee_link: str

    #: End-effector links per side whose centroid is the grasp point (the EE used by rewards /
    #: obs): the fingertips of a hand, the pads of a gripper. Read them off the sub-option's
    #: metadata where it has some, so swapping the end-effector means swapping these too.
    left_grasp_links: list[str] = []
    right_grasp_links: list[str] = []

    #: The links this *rig* stands on, overriding the ones its humanoid declares.
    #:
    #: ``RobotOptions.feet_links_name`` is a property of the humanoid asset, and inheriting it is
    #: right for the usual rig — welding hands to the wrists does not change what the legs stand
    #: on, so ``OliWuji`` gets Oli's ankle rolls for free. It stops being right as soon as the
    #: assembly reaches the ground: weld foot shells, skates or a board binding onto the ankles
    #: and the contact links belong to the *welded child*, which the humanoid's own declaration
    #: cannot name.
    #:
    #: ``None`` (the default) reads the humanoid's own feet, so a rig that does not care never
    #: mentions this. Any explicit value wins, ``()`` included — an empty tuple declares "this rig
    #: has no feet in the leg-odometry sense", which is how the deploy-time base-linear-velocity
    #: estimator is made to refuse rather than return a quietly biased number (the same opt-out
    #: ``Go2W`` uses). Read the resolved value off :attr:`feet_links`, never off this field.
    feet_links_name: tuple[str, ...] | None = None

    #: Scene entity names — these ARE the entity names in the scene, and how downstream terms
    #: reference the three bodies. Give two humanoids in one scene distinct names.
    robot_name: str = "robot"
    left_hand_name: str = "left_hand"
    right_hand_name: str = "right_hand"

    #: Convenience knob to place the humanoid's pelvis. ``None`` leaves
    #: ``robot.default_root_pos`` untouched, so a caller can set the placement on ``robot``
    #: directly instead.
    base_pos: tuple[float, float, float] | None = None

    #: The end-effector base link's pose in the flange frame, applied to *both* wrists — the
    #: common case, where the flange pair is symmetric and the per-side chirality lives in the
    #: ``_L`` / ``_R`` assets (as it does for the Wuji hands on Oli's ``*_hand_mount`` frames).
    #: Identity by default, since the asset is expected to place the frame; override to model an
    #: adapter that shifts or rotates the end-effector. Annotations must stay ``Vec3FType`` /
    #: ``UnitVec4FType`` — a bare tuple drops the quaternion normalizer and the
    #: ``allow_inf_nan=False`` guard.
    mount_offset: Vec3FType = (0.0, 0.0, 0.0)
    mount_quat: UnitVec4FType = (1.0, 0.0, 0.0, 0.0)

    #: Per-side overrides of the two above, for a humanoid whose wrist frames are *not* mirror
    #: images of each other — welding one end-effector asset to both sides of such a rig needs
    #: two different transforms, which the shared knob cannot express. ``None`` (the default)
    #: means "use the shared value", so a symmetric rig never mentions these; set a side's offset
    #: and/or quat to override just that component on just that wrist.
    left_mount_offset: Vec3FType | None = None
    left_mount_quat: UnitVec4FType | None = None
    right_mount_offset: Vec3FType | None = None
    right_mount_quat: UnitVec4FType | None = None

    def model_post_init(self, context) -> None:
        # super() runs the duplicate-name check, so a colliding rig fails where it was written.
        super().model_post_init(context)
        # Both end-effectors are welded, so neither may also be pinned to the world.
        _validate_welded_child("left_hand", self.left_hand)
        _validate_welded_child("right_hand", self.right_hand)
        # Pelvis placement is a single knob on the composite (opt-in), as on _ArmCompositeOptions.
        _apply_base_pos(self.base_pos, self.robot)

    # --- shortcut properties over the sub-options (single source of truth) ---
    @property
    def robot_dofs(self) -> list[str]:
        """The humanoid's body DOFs, in the robot's canonical order (== ``robot.dofs_name``)."""
        return list(self.robot.dofs_name)

    @property
    def left_hand_dofs(self) -> list[str]:
        """Left end-effector DOFs driven in joint space by the PD controller (== ``left_hand.dofs_name``)."""
        return list(self.left_hand.dofs_name)

    @property
    def right_hand_dofs(self) -> list[str]:
        """Right end-effector DOFs driven in joint space by the PD controller (== ``right_hand.dofs_name``)."""
        return list(self.right_hand.dofs_name)

    @property
    def left_hand_actuated_dofs(self) -> list[str]:
        """The left end-effector's driven DOFs, its class's ``actuated_dofs_name``.

        The action columns of a hand whose other joints follow them (:attr:`left_hand_mimic_dofs`).
        """
        return list(type(self.left_hand).actuated_dofs_name)

    @property
    def right_hand_actuated_dofs(self) -> list[str]:
        """The right end-effector's driven DOFs; see :attr:`left_hand_actuated_dofs`."""
        return list(type(self.right_hand).actuated_dofs_name)

    @property
    def left_hand_mimic_dofs(self) -> dict[str, tuple[str, float, float]]:
        """The left end-effector's couplings, ``follower -> (driver, multiplier, offset)``.

        Read off its ``metadata.mimic_dofs`` (empty for a hand that declares none): what a joint PD action
        term's ``mimic_dofs`` takes.
        """
        return dict(getattr(type(self.left_hand).metadata, "mimic_dofs", None) or {})

    @property
    def right_hand_mimic_dofs(self) -> dict[str, tuple[str, float, float]]:
        """The right end-effector's couplings; see :attr:`left_hand_mimic_dofs`."""
        return dict(getattr(type(self.right_hand).metadata, "mimic_dofs", None) or {})

    @property
    def feet_links(self) -> tuple[str, ...]:
        """The links this rig stands on: :attr:`feet_links_name` if set, else the humanoid's own.

        The resolved answer, and the one consumers should read — :attr:`feet_links_name` is the
        *override*, and is ``None`` for every rig that simply inherits. Resolved on read rather
        than written back at construction, so swapping ``robot`` still moves the feet with it and
        a saved config records the rig's intent (inherit) instead of freezing today's answer.
        """
        if self.feet_links_name is not None:
            return tuple(self.feet_links_name)
        return tuple(self.robot.feet_links_name)

    def _mount_spec(
        self,
        ee: EntityOptions,
        ee_link: str,
        offset_override: Vec3FType | None,
        quat_override: UnitVec4FType | None,
    ) -> AttachmentSpec:
        """Weld spec seating end-effector ``ee`` on ``ee_link`` at this side's mount transform.

        ``offset_override`` / ``quat_override`` are the side's :attr:`left_mount_offset`-style
        fields: ``None`` falls back to the shared :attr:`mount_offset` / :attr:`mount_quat`, so a
        symmetric flange pair states the transform once and an asymmetric one overrides per side.

        Always the explicit 4-tuple, unlike :meth:`_ArmCompositeOptions._attach_spec`: on a
        humanoid the mount belongs to this rig, not to either end-effector's ``default_root_*``,
        which is what a 2-tuple weld would use instead.

        The quaternion goes through :func:`weld_quat`, which cancels the child's ``up`` / ``front``
        alignment so the mount quaternion means the end-effector base link's orientation *in the
        flange link frame* — an identity when the two URDF frames coincide. Leaving that
        cancellation out fails silently: a hand whose axes rotate its asset 180 deg ends up
        pointing back up the forearm.
        """
        offset = self.mount_offset if offset_override is None else offset_override
        quat = self.mount_quat if quat_override is None else quat_override
        return (self.robot_name, ee_link, tuple(offset), weld_quat(ee, quat))

    # --- composite contract: name -> sub-entity, plus the two wrist welds ---
    def entity_parts(self) -> list[tuple[str, BaseEntityOptions]]:
        return [
            (self.robot_name, self.robot),
            (self.left_hand_name, self.left_hand),
            (self.right_hand_name, self.right_hand),
        ]

    def attachments(self) -> dict[str, AttachmentSpec]:
        return {
            self.left_hand_name: self._mount_spec(
                self.left_hand, self.left_ee_link, self.left_mount_offset, self.left_mount_quat
            ),
            self.right_hand_name: self._mount_spec(
                self.right_hand, self.right_ee_link, self.right_mount_offset, self.right_mount_quat
            ),
        }


# Registry of known composite robots (name -> ManipulatorOptions / DexManipulatorOptions /
# HumanoidOptions subclass); look one up by name and add new arms, dexterous hands, or humanoids
# as new subclasses in their own module. Mirrors ROBOT_REGISTRY for unit robots.
COMPOSITE_ROBOT_REGISTRY = Registry("composite_robot")
