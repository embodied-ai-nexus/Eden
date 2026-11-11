"""The arm + two-finger gripper a scripted demonstrator drives, from whichever robot a scene holds there.

A demonstrator needs a handful of facts about the robot: which entity carries the arm DOFs and the
link the IK term drives, which entity carries the fingers, and where the tool point (the pad
midpoint) sits in that link's frame. Two shapes of scene entry supply them:

* a composite :class:`~eden.options.robots.ManipulatorOptions` (robosuite's Panda, UR5e + Robotiq,
  DROID): an arm entity and a welded gripper entity, the tool point carried through the weld;
* a single :class:`~eden.options.entities.RobotOptions` whose asset includes its own fingers
  (SIMPLER's WidowX250): the fingers are the DOFs of ``open_dofs_pos``, the arm every other actuated
  DOF, the IK link ``ee_links_name[0]``, and the tool point the robot's ``tcp_offset``.

:func:`gripper_rig` reads either into one :class:`GripperRig`, so nothing downstream asks which
shape it is. The action terms are found on the built env by what they drive
(:func:`find_rig_actions`), not by name: tasks name them ``robot_arm`` (robosuite) or
``ee_controller`` (SIMPLER, Polaris).

:class:`RigDriver` is what moves the rig: it reads the tool point's pose off the built robot and
turns a target tool pose and gripper value into the two terms' commands — the task's own action
space, so a recorded action replays as-is. Every demonstrator drives the robot through it (the
keyframe controller and the tool routines alike).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import inv_quat

from eden.extensions.scripted_demos.primitives import effector_rot, tool_quat
from eden.managers.terms.actions.binary_actions import BinaryJointController
from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.options.entities import BaseEntityOptions, CompositeEntityOptions, RobotOptions
from eden.options.robots import ManipulatorOptions
from eden.utils.geom import axis_angle_from_quat, quat_from_axis_angle, quat_mul, transform_by_quat, yaw_from_quat

if TYPE_CHECKING:
    from eden.entities import RigidEntity
    from eden.envs.base import EnvBase
    from eden.options.entities import SceneOptions

#: Largest ``|delta / scale|`` the arm command is allowed to ask for. The IK term squashes its
#: input with ``tanh``, so the recorded action is ``atanh`` of this ratio: 0.9 keeps it under 1.5.
MAX_COMMAND_RATIO = 0.9
#: Headroom [rad] the planner keeps from either stop of the arm's last joint.
WRIST_MARGIN = 0.35
#: A last joint whose axis is this far from vertical does not turn the hand's heading; its range is not planned around.
MIN_WRIST_VERTICALITY = 0.5


@dataclass(frozen=True)
class GripperRig:
    """The facts a scripted demonstrator reads off an arm + two-finger gripper.

    Attributes
    ----------
    slot : str
        Scene entry the rig was read from.
    arm_name : str
        Entity carrying the arm DOFs and :attr:`ee_link`.
    gripper_name : str
        Entity carrying the finger DOFs — :attr:`arm_name` itself for a single-body robot.
    ee_link : str
        Link the IK term drives, on :attr:`arm_name`.
    arm_dofs : tuple[str, ...]
        Arm DOFs, base to wrist; the last is the joint that turns the hand about its approach axis.
    gripper_dofs : tuple[str, ...]
        Finger DOFs.
    tcp_offset : tuple[float, float, float]
        Tool point (pad midpoint, gripper open) in the :attr:`ee_link` frame (m). Its direction is
        the hand's approach axis.
    """

    slot: str
    arm_name: str
    gripper_name: str
    ee_link: str
    arm_dofs: tuple[str, ...]
    gripper_dofs: tuple[str, ...]
    tcp_offset: tuple[float, float, float]


def gripper_rig(scene_options: SceneOptions, slot: str) -> GripperRig:
    """Read the arm + gripper at scene entry ``slot`` (a composite rig or a robot with its own fingers).

    Raises
    ------
    KeyError
        ``slot`` is not a scene entry.
    TypeError
        The entry is neither shape, or a single-body robot does not declare a fact the rig needs.
    """
    options = getattr(scene_options, slot, None)
    if options is None:
        raise KeyError(
            f"Scene has no entry '{slot}' to read an arm + gripper from; scene entities: {entity_entries(scene_options)}."
        )
    if isinstance(options, ManipulatorOptions):
        return GripperRig(
            slot=slot,
            arm_name=options.arm_name,
            gripper_name=options.gripper_name,
            ee_link=options.ee_link,
            arm_dofs=tuple(options.arm_dofs),
            gripper_dofs=tuple(options.gripper_dofs),
            tcp_offset=tuple(options.tcp_offset),
        )
    if isinstance(options, RobotOptions):
        return _single_body_rig(slot, options)
    raise TypeError(
        f"Scene entry '{slot}' is a {type(options).__name__}; a scripted demonstrator drives an arm + two-finger "
        f"gripper — a ManipulatorOptions composite, or a RobotOptions whose asset carries its own fingers."
    )


def entity_entries(scene_options: SceneOptions) -> list[str]:
    """Names of the scene's entity entries — its ``attachments_dict`` and other plain fields left out."""
    return [
        name
        # .keys(), not iteration: SceneOptions is a pydantic model, which iterates (name, value) pairs.
        for name in scene_options.keys()  # noqa: SIM118
        if isinstance(getattr(scene_options, name), (BaseEntityOptions, CompositeEntityOptions))
    ]


def _single_body_rig(slot: str, robot: RobotOptions) -> GripperRig:
    """A robot whose asset includes the fingers: the facts come from its class-level declarations."""
    name = type(robot).__name__
    missing = [
        declaration
        for declaration, present in (
            ("open_dofs_pos (the finger DOFs and their open targets)", bool(getattr(robot, "open_dofs_pos", None))),
            ("ee_links_name (the link the IK term drives, first entry)", bool(robot.ee_links_name)),
            ("tcp_offset (pad midpoint in that link's frame)", robot.tcp_offset is not None),
        )
        if not present
    ]
    if missing:
        raise TypeError(
            f"Scene entry '{slot}' holds {name}, which a scripted demonstrator can only drive as an arm with its own "
            f"fingers if the robot declares {'; '.join(missing)}. Add the missing ClassVar(s) to {name} — measure "
            f"tcp_offset off the built finger pads, don't guess it."
        )
    fingers = tuple(robot.open_dofs_pos)
    actuated = robot.actuated_dofs_name or robot.dofs_name
    return GripperRig(
        slot=slot,
        arm_name=slot,
        gripper_name=slot,
        ee_link=robot.ee_links_name[0],
        arm_dofs=tuple(dof for dof in actuated if dof not in fingers),
        gripper_dofs=fingers,
        tcp_offset=tuple(float(v) for v in robot.tcp_offset),
    )


def find_gripper_rigs(scene_options: SceneOptions) -> list[GripperRig]:
    """Every arm + gripper in the scene, in scene-entry order — one for a single-arm task, two for a two-arm one."""
    rigs = []
    for slot in entity_entries(scene_options):
        try:
            rigs.append(gripper_rig(scene_options, slot))
        except TypeError:
            continue
    return rigs


def find_gripper_rig(scene_options: SceneOptions) -> GripperRig:
    """The one arm + gripper in the scene, when the caller does not name its slot.

    Raises
    ------
    TypeError
        The scene holds none, or several — name the slot then.
    """
    rigs = find_gripper_rigs(scene_options)
    if len(rigs) != 1:
        found = [rig.slot for rig in rigs]
        raise TypeError(
            f"Expected exactly one arm + gripper in the scene, found {len(rigs)} ({found}); pass the slot explicitly "
            f"(gripper_rig(scene_options, '<slot>'))."
        )
    return rigs[0]


@dataclass(frozen=True)
class RigActions:
    """The two action terms that move a rig: an IK term on its arm and a binary term on its fingers."""

    arm: str
    gripper: str


def find_rig_actions(env: EnvBase, rig: GripperRig, *, exclusive: bool = True) -> RigActions:
    """Find the ``DifferentialIKController`` driving ``rig.ee_link`` and the ``BinaryJointController`` on its fingers.

    Matched on what each term drives — entity, link, DOFs — not on its name. With ``exclusive`` (the
    default) every action term of the env must be one of the two: a demonstrator commands both each
    step and has nothing to send a third. A multi-arm demonstrator passes ``exclusive=False`` for each
    arm and checks that the arms' terms together cover every term
    (:class:`~eden.extensions.scripted_demos.multi_arm.MultiArmDemonstrator`).
    """
    manager = env.action_manager
    terms = {name: manager.get_term(name) for name in manager.active_terms}
    arm = [
        name
        for name, term in terms.items()
        if isinstance(term, DifferentialIKController)
        and term.entity_name == rig.arm_name
        and term.ee_link_name == rig.ee_link
    ]
    gripper = [
        name
        for name, term in terms.items()
        if isinstance(term, BinaryJointController)
        and term.entity_name == rig.gripper_name
        and set(term.dofs_name) == set(rig.gripper_dofs)
    ]
    layout = {name: f"{type(term).__name__}(entity={term.entity_name})" for name, term in terms.items()}
    if len(arm) != 1 or len(gripper) != 1:
        raise TypeError(
            f"A scripted demonstrator drives rig '{rig.slot}' through one DifferentialIKController on "
            f"'{rig.arm_name}'/'{rig.ee_link}' and one BinaryJointController on '{rig.gripper_name}' "
            f"{list(rig.gripper_dofs)}; found {len(arm)} and {len(gripper)} among the action terms {layout}."
        )
    extra = sorted(set(terms) - {arm[0], gripper[0]})
    if exclusive and extra:
        raise TypeError(
            f"Action terms {extra} drive something other than rig '{rig.slot}'; a scripted demonstrator commands "
            f"only its IK term '{arm[0]}' and gripper term '{gripper[0]}', and every term needs a command each step. "
            f"Action terms: {layout}."
        )
    return RigActions(arm=arm[0], gripper=gripper[0])


class RigDriver:
    """Reads a rig's tool pose and turns target tool poses into the task's IK + gripper commands, batched over envs.

    Effector rotations are relative to the rig's straight-down pose (:attr:`down_quat`): identity
    points the approach axis at the floor with heading 0, and
    :func:`~eden.extensions.scripted_demos.primitives.effector_rot` builds one from a heading and a tilt.

    Parameters
    ----------
    env : EnvBase
        A built environment holding the rig at ``slot`` (:func:`gripper_rig`), moved by one
        ``DifferentialIKController`` and one ``BinaryJointController`` (:func:`find_rig_actions`).
    slot : str
        Scene entry holding the rig.
    workspace : tuple[tuple[float, float, float], tuple[float, float, float]]
        Low / high corners every tool target is clipped to. The low ``z`` is the lowest the
        *fingertips* may go: the tool point's floor is raised by :attr:`tip_reach`, measured on the
        built rig.
    max_step_translation, max_step_rotation : float
        Largest end-effector move asked for in one control step, per axis [m] and in angle [rad]
        (OCBench's ``ee_action_delta`` clip).
    exclusive : bool
        Whether the rig's two action terms must be all the env has (:func:`find_rig_actions`); a
        multi-arm demonstrator drives each arm with ``False``.
    """

    def __init__(
        self,
        env: EnvBase,
        slot: str,
        *,
        workspace: tuple[tuple[float, float, float], tuple[float, float, float]],
        max_step_translation: float = 0.05,
        max_step_rotation: float = 0.3,
        exclusive: bool = True,
    ):
        self.env = env
        self.device = device = env.device
        n = env.num_envs
        self.rig = rig = gripper_rig(env.scene_options, slot)
        tcp = torch.tensor(rig.tcp_offset, device=device, dtype=torch.float32)
        if float(tcp.norm()) < 1e-3:
            raise ValueError(
                f"Rig '{rig.slot}' has tcp_offset {rig.tcp_offset}: the tool point must lie off '{rig.ee_link}' — its "
                f"direction is the approach axis the demonstrator points at the floor."
            )
        self.arm: RigidEntity = env.entities[rig.arm_name]
        self.gripper: RigidEntity = env.entities[rig.gripper_name]
        self._ee_link_idx = torch.tensor([self.arm.get_link(rig.ee_link).idx_local], device=device)
        self._tcp = tcp.expand(n, 3)

        actions = find_rig_actions(env, rig, exclusive=exclusive)
        self.arm_term_name, self.gripper_term_name = actions.arm, actions.gripper
        self._scale = env.action_manager.get_term(actions.arm).effective_scale
        # The arm's last joint: its range bounds the headings a plan may ask for (dofs_name order
        # throughout — get_dofs_pos / get_dofs_limit / get_jacobian all use it).
        wrist = rig.arm_dofs[-1]
        if wrist not in self.arm.dofs_name:
            raise KeyError(
                f"Rig '{rig.slot}' lists arm DOF '{wrist}', which '{rig.arm_name}' does not have: "
                f"{list(self.arm.dofs_name)}."
            )
        self._wrist = list(self.arm.dofs_name).index(wrist)
        gripper_term = env.action_manager.get_term(actions.gripper)
        self._finger_dofs_idx = gripper_term.genesis_dofs_idx
        # Clamped to the joint range: a close target past it (a 2F-85 driven beyond its stop) would
        # leave empty fingers short of "closed", and read as holding something.
        low, high = (limit[:1] for limit in self.gripper.get_dofs_limit(dofs_idx_local=self._finger_dofs_idx))
        self._finger_open = torch.clamp(gripper_term.open_targets, low, high)
        self._finger_close = torch.clamp(gripper_term.close_targets, low, high)
        self.max_step_translation = max_step_translation
        self.max_step_rotation = max_step_rotation

        #: ``(1, 4)`` world orientation of the IK link with the hand pointing straight down, heading 0.
        self.down_quat = self._measure_down_quat(tcp)
        #: How far [m] the fingers reach past the tool point along the approach axis.
        self.tip_reach = self._measure_tip_reach(tcp)
        # The workspace floor bounds the fingertips, not the tool point between the pads: a 3 cm cube's
        # centre is lower than a WidowX's fingers reach past their pads, so a grasp aimed there drives
        # the fingers — and the cube in them — into the table.
        #: Low / high ``(3,)`` corners of the tool point's reachable box (the floor raised by :attr:`tip_reach`).
        self.workspace = (
            torch.tensor(workspace[0], device=device, dtype=torch.float32),
            torch.tensor(workspace[1], device=device, dtype=torch.float32),
        )
        self.workspace[0][2] += self.tip_reach

    def _measure_down_quat(self, tcp: torch.Tensor) -> torch.Tensor:
        """The ``ee_link`` orientation with the hand pointing straight down, measured off the home pose. ``(1, 4)``.

        The approach axis is the tool point's direction in the ``ee_link`` frame — ``+z`` on a Franka
        Hand or Robotiq mount, ``+x`` on the WidowX's ``gripper_link``. The minimal rotation that
        points it at the floor is applied to the home orientation, then the heading is removed: that
        of the ``ee_link`` axis most perpendicular to the approach, so effector heading 0 has it
        along world +x.
        """
        device = self.device
        approach_local = tcp / tcp.norm()
        basis = torch.eye(3, device=device)
        reference_local = basis[(basis @ approach_local).abs().argmin()]
        ee_quat = self.ee_quat()[:1]
        approach = transform_by_quat(approach_local.unsqueeze(0), ee_quat)
        down = torch.tensor([[0.0, 0.0, -1.0]], device=device)
        axis = torch.linalg.cross(approach, down)
        angle = torch.atan2(axis.norm(dim=-1, keepdim=True), (approach * down).sum(-1, keepdim=True))
        swing = quat_from_axis_angle(axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-9) * angle)
        pointing_down = quat_mul(swing, ee_quat)
        reference = transform_by_quat(reference_local.unsqueeze(0), pointing_down)
        heading = torch.atan2(reference[:, 1], reference[:, 0])
        return quat_mul(effector_rot(-heading), pointing_down)

    def finger_links(self) -> list:
        """The finger links with collision geometry: every such link of a separate gripper entity, or the links the
        finger DOFs move.
        """
        rig = self.rig
        if rig.gripper_name != rig.arm_name:
            links = [link for link in self.gripper.links if link.n_geoms]
        else:
            joints = {joint.name: joint for joint in self.gripper.joints}
            links = [joints[dof].link for dof in rig.gripper_dofs if dof in joints and joints[dof].link.n_geoms]
        if not links:
            raise ValueError(
                f"Rig '{rig.slot}': no finger link of '{rig.gripper_name}' has collision geometry to measure; "
                f"finger DOFs {list(rig.gripper_dofs)}."
            )
        return links

    def _measure_tip_reach(self, tcp: torch.Tensor) -> float:
        """How far [m] the fingers extend past the tool point along the approach axis, on the built rig.

        The corners of each finger link's collision box at the home pose, projected on the approach
        axis. Boxes are world-axis-aligned, so a tilted hand reads slightly long — the safe side.
        """
        ee_quat = self.ee_quat()[:1]
        tool = self.ee_pos()[:1] + transform_by_quat(tcp.unsqueeze(0), ee_quat)
        approach = transform_by_quat((tcp / tcp.norm()).unsqueeze(0), ee_quat)[0]
        corners = torch.tensor(
            [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], device=self.device, dtype=torch.float32
        )
        reach = 0.0
        for link in self.finger_links():
            aabb = link.get_AABB().reshape(-1, 2, 3)[0].to(self.device)  # (2, 3): min, max (env 0)
            points = aabb[0] + corners * (aabb[1] - aabb[0])
            reach = max(reach, float(((points - tool) @ approach).max()))
        return reach

    def contact_links(self) -> list:
        """The links whose touch is the fingers': :meth:`finger_links` less a separate gripper's root (its palm /
        mount).
        """
        if self.rig.gripper_name == self.rig.arm_name:
            return self.finger_links()
        root = self.gripper.links[0]
        links = [link for link in self.finger_links() if link.idx != root.idx]
        if not links:
            raise ValueError(
                f"Rig '{self.rig.slot}': only the root link '{root.name}' of '{self.rig.gripper_name}' has collision "
                f"geometry, so no finger contact can be told from the palm's."
            )
        return links

    @cached_property
    def _contact_links_idx(self) -> torch.Tensor:
        return torch.tensor([link.idx for link in self.contact_links()], device=self.device)

    @cached_property
    def _rig_links_idx(self) -> torch.Tensor:
        entities = {id(self.arm): self.arm, id(self.gripper): self.gripper}.values()
        return torch.tensor([link.idx for entity in entities for link in entity.links], device=self.device)

    @cached_property
    def finger_axis_yaw(self) -> float:
        """Heading [rad] of the line the fingers close along, in the effector frame, in ``[0, pi)``.

        Read off the built fingers the first time it is asked (the keyframe controller never asks):
        the contact links' box centres, projected on the plane across the approach axis — their
        principal direction is the closing line — relative to the effector's heading, so a target
        heading turns it with the hand.
        """
        ee_quat = self.ee_quat()[:1]
        approach = transform_by_quat((self._tcp[:1] / self._tcp[:1].norm()), ee_quat)[0]
        centres = torch.stack(
            [link.get_AABB().reshape(-1, 2, 3)[0].mean(0).to(self.device) for link in self.contact_links()]
        )
        if len(centres) < 2:
            raise ValueError(
                f"Rig '{self.rig.slot}': the closing direction is measured from at least two finger links with "
                f"collision geometry; found {len(centres)} ({[link.name for link in self.contact_links()]})."
            )
        across = centres - centres.mean(0)
        across = across - (across @ approach).unsqueeze(-1) * approach
        _, _, v = torch.linalg.svd(across, full_matrices=False)
        axis = v[0]
        _, rot = self.tool_pose()
        # A line has no direction: fold into [0, pi).
        return (math.atan2(float(axis[1]), float(axis[0])) - float(yaw_from_quat(rot[:1])[0])) % math.pi

    # --- state readers --------------------------------------------------------------------------

    def ee_pos(self) -> torch.Tensor:
        """``(N, 3)`` world position of the IK link."""
        return self.arm.get_links_pos(ls_idx_local=self._ee_link_idx)[:, 0]

    def ee_quat(self) -> torch.Tensor:
        """``(N, 4)`` world orientation of the IK link."""
        return self.arm.get_links_quat(ls_idx_local=self._ee_link_idx)[:, 0]

    def tool_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Tool-point position ``(N, 3)`` and effector rotation ``(N, 4)`` (identity = pointing straight down)."""
        ee_quat = self.ee_quat()
        pos = self.ee_pos() + transform_by_quat(self._tcp, ee_quat)
        return pos, quat_mul(ee_quat, inv_quat(self.down_quat).expand_as(ee_quat))

    def gripper_closure(self) -> torch.Tensor:
        """``(N,)`` how far the fingers are from open (0) toward their closed targets (1), averaged over finger DOFs.

        Fingers closed on an object stop short of 1 — the proprioceptive "am I holding something".
        """
        q = self.gripper.get_dofs_pos(dofs_idx_local=self._finger_dofs_idx)
        span = self._finger_close - self._finger_open
        moving = span.abs() > 1e-6
        fraction = torch.where(moving, (q - self._finger_open) / torch.where(moving, span, 1.0), 0.0)
        return fraction.sum(-1) / moving.sum(-1).clamp(min=1)

    def finger_touches(self, min_force: float) -> torch.Tensor:
        """``(N, L)`` bool over the scene's global link indices: which links a finger presses on with at least
        ``min_force`` [N].

        The robot's own links never count.
        """
        contacts = self.gripper.get_contacts()
        link_a, link_b, valid = contacts["link_a"], contacts["link_b"], contacts["valid_mask"]
        force = contacts["force_a"].norm(dim=-1)
        finger_a, finger_b = torch.isin(link_a, self._contact_links_idx), torch.isin(link_b, self._contact_links_idx)
        rig_a, rig_b = torch.isin(link_a, self._rig_links_idx), torch.isin(link_b, self._rig_links_idx)
        pressing = valid & (force >= min_force)
        touches = torch.zeros(len(valid), self._num_scene_links, dtype=torch.long, device=self.device)
        for other, touching in ((link_b, pressing & finger_a & ~rig_b), (link_a, pressing & finger_b & ~rig_a)):
            # Padding rows hold whatever the buffer held: clamp their index, add nothing for them.
            index = torch.where(touching, other, 0).long().clamp(0, self._num_scene_links - 1)
            touches.scatter_add_(1, index, touching.long())
        return touches > 0

    def finger_contact(self, min_force: float) -> torch.Tensor:
        """``(N,)`` bool: a finger link touches something that is not the robot, pressing at least ``min_force`` [N]."""
        return self.finger_touches(min_force).any(dim=-1)

    @cached_property
    def _num_scene_links(self) -> int:
        return 1 + max(link.idx for entity in self.env.entities.values() for link in getattr(entity, "links", []))

    def pinching(self, entity) -> torch.Tensor:
        """``(N,)`` bool: finger links on *both* sides of the closing line touch ``entity`` — it is held between them.

        Two links of one finger (a 2F-85's pad and its knuckle) touching it do not count.
        """
        contacts = self.gripper.get_contacts(with_entity=entity._entity)
        link_a, link_b, valid = contacts["link_a"], contacts["link_b"], contacts["valid_mask"]
        tool_pos, rot = self.tool_pose()
        heading = yaw_from_quat(rot) + self.finger_axis_yaw
        closing = torch.stack([torch.cos(heading), torch.sin(heading), torch.zeros_like(heading)], dim=-1)
        left = right = torch.zeros(len(tool_pos), dtype=torch.bool, device=self.device)
        for link in self.contact_links():
            touching = (valid & ((link_a == link.idx) | (link_b == link.idx))).any(dim=-1)
            centre = (
                link.get_AABB().reshape(-1, 2, 3).mean(dim=1).to(self.device)
            )  # (N, 3); a frame can sit on the joint axis
            side = ((centre - tool_pos) * closing).sum(-1)
            left = left | (touching & (side > 0))
            right = right | (touching & (side < 0))
        return left & right

    def yaw_window(self) -> torch.Tensor:
        """``(N, 2)`` effector headings reachable before the last arm joint comes within :data:`WRIST_MARGIN` of a stop.

        Linearized about the current pose: a heading change ``d psi`` turns the last joint by
        ``d psi / k``, where ``k`` is the vertical component of its world axis (about +-1 for a wrist
        pointing down). Re-measure it at every plan, so the redundant arm's drift is accounted for.
        """
        q = self.arm.get_dofs_pos()[:, self._wrist]
        low, high = (limit[:, self._wrist] for limit in self.arm.get_dofs_limit())
        k = self.arm.get_jacobian(self.arm.get_link(self.rig.ee_link))[:, 5, self._wrist]
        _, rot = self.tool_pose()
        yaw = yaw_from_quat(rot)
        a = yaw + k * (low + WRIST_MARGIN - q)
        b = yaw + k * (high - WRIST_MARGIN - q)
        window = torch.stack([torch.minimum(a, b), torch.maximum(a, b)], dim=-1)
        unbounded = (k.abs() < MIN_WRIST_VERTICALITY) | ~torch.isfinite(window).all(dim=-1)
        inf = torch.full_like(window, math.inf)
        return torch.where(unbounded.unsqueeze(-1), torch.stack([-inf[:, 0], inf[:, 1]], dim=-1), window)

    # --- commands -------------------------------------------------------------------------------

    def clip(self, tool_pos: torch.Tensor) -> torch.Tensor:
        """``tool_pos`` clipped into :attr:`workspace`."""
        low, high = self.workspace
        return torch.maximum(torch.minimum(tool_pos, high), low)

    def command(
        self, tool_pos: torch.Tensor, rot: torch.Tensor, grasp: torch.Tensor, hold: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """The action dict that moves the tool point toward ``tool_pos`` / ``rot`` with gripper ``grasp``.

        Parameters
        ----------
        tool_pos : torch.Tensor
            ``(N, 3)`` target tool-point position; clipped into :attr:`workspace`.
        rot : torch.Tensor
            ``(N, 4)`` target effector rotation.
        grasp : torch.Tensor
            ``(N,)`` gripper value, 0 open to 1 closed.
        hold : torch.Tensor, optional
            ``(N,)`` bool: these envs' arms get a zero command (hold still).

        Returns
        -------
        dict[str, torch.Tensor]
            The IK term's ``(N, 6)`` command and the gripper term's ``(N, 1)``, keyed by term name.
        """
        tool_pos = self.clip(tool_pos)
        target_quat = tool_quat(rot, self.down_quat)
        target_ee = tool_pos - transform_by_quat(self._tcp, target_quat)
        ee_quat = self.ee_quat()
        translation = (target_ee - self.ee_pos()).clamp(-self.max_step_translation, self.max_step_translation)
        rotation = axis_angle_from_quat(quat_mul(target_quat, inv_quat(ee_quat)))
        angle = rotation.norm(dim=-1, keepdim=True)
        rotation = rotation * (self.max_step_rotation / angle.clamp(min=self.max_step_rotation))
        delta = torch.cat([translation, rotation], dim=-1)
        ratio = (delta / self._scale).clamp(-MAX_COMMAND_RATIO, MAX_COMMAND_RATIO)
        arm = torch.atanh(ratio)
        if hold is not None:
            arm = torch.where(hold.unsqueeze(-1), 0.0, arm)
        gripper = (2.0 * grasp - 1.0).unsqueeze(-1)
        return {self.arm_term_name: arm, self.gripper_term_name: gripper}
