"""The grip of a parallel jaw on a Mochi soft body: how hard it squeezes, and RoboSimStudio's two latches.

:class:`JawGrip` reads each pad's squeeze along the closing axis from the per-link contact force. The latches
pin soft-body vertices to the hand and carry them there until the jaws open, each engaging as RoboSimStudio's
does:

* :class:`SettledGraspLatch` is the ``wipe_the_bar`` latch (``bar.Latch``): the jaws grip the foam by contact,
  and once that grip has settled the foam under the pads is pinned. RoboSimStudio needs it because friction alone
  let the sponge creep out of the jaws stroke after stroke, and so does Mochi: the demo's routine without it cleaned
  0.07 of the stripe before the sponge slid out of the jaws in its first stroke.
* :class:`FeatureGraspLatch` is the benchmark's deformable grasp (the agent's ``attach``, a ``ParticleDragger``
  in pin mode): closing seizes the vertices of a declared feature (a towel's edge strip) within 3 cm of the tool
  centre, and a close with none in reach seizes nothing. RoboSimStudio's fold demo pinches the towel by friction
  instead, on a VBD slab whose two faces 5 mm apart the jaws gather; a Mochi shell is one surface, which the
  jaws' tips press into the bench rather than gather (``docs/tasks/mochi.md``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import genesis as gs
import torch
from genesis.utils.geom import inv_transform_by_quat, transform_by_quat

from eden.managers.action_manager import ACTION_TERM_REGISTRY
from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.managers.terms.actions.binary_actions import BinaryJointController
from eden.tasks.mochi.common import YAM_FINGER_SPEED, link_point
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.entities.mochi import MochiSoftEntity
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions


def pad_squeeze(robot, pads: torch.Tensor, axis_link, closing_axis: torch.Tensor) -> torch.Tensor:
    """Each pad's push on what is between the jaws (N), ``(num_envs, 2)``: its contact force along the closing axis.

    ``pads`` holds the two pad links' local indices, the one on the closing axis's + side first; ``closing_axis``
    is the unit vector from it toward the other, in ``axis_link``'s frame. A pad is pushed back out of the jaw by
    what it squeezes, so a pad resting on the bench (a force across the axis) reads nothing.
    """
    force = robot.get_links_net_contact_force()[:, pads]  # (B, 2, 3)
    quat = axis_link.get_quat()
    closing = transform_by_quat(closing_axis.expand_as(quat[:, :3]), quat)
    return torch.stack([-(force[:, 0] * closing).sum(-1), (force[:, 1] * closing).sum(-1)], dim=-1)


@METRIC_TERM_REGISTRY.register()
class JawGrip(MetricTerm):
    """How hard a parallel jaw squeezes what is between its pads (N): the weaker pad's push along the closing axis.

    RoboSimStudio takes a grasp as made once each pad presses with 2 N; a hand closed on air or on one edge reads
    0. A shared quantity (configure ``contributes_to_success=False``) the policy observes and rewards gate on.

    Parameters
    ----------
    entity_name : str
        The robot.
    pad_links_name : tuple[str, str]
        The two pad links, the one on the closing axis's + side first.
    axis_link_name : str
        The link whose frame holds the closing axis.
    closing_axis : tuple[float, float, float]
        Unit vector in that frame from the first pad toward the second.
    """

    entity_name: str = "robot"
    pad_links_name: tuple[str, str] = ("lf_down", "rf_down")
    axis_link_name: str = "link_6"
    closing_axis: tuple[float, float, float] = (-1.0, 0.0, 0.0)

    def build(self) -> None:
        robot = self.resolve_entity(self.entity_name)
        _, pads = robot.find_named_links_idx_local(list(self.pad_links_name), preserve_order=True)
        self._pads = torch.as_tensor(pads, dtype=torch.long, device=self.device)
        self._axis_link = robot.get_link(self.axis_link_name)
        self._axis = torch.tensor(self.closing_axis, dtype=gs.tc_float, device=self.device)

    def compute(self) -> torch.Tensor:
        robot = self.resolve_entity(self.entity_name)
        return pad_squeeze(robot, self._pads, self._axis_link, self._axis).amin(dim=-1).clamp_min(0.0)


# A finger target within this (m) of its command has arrived: the ramp lands on it to rounding.
_TARGET_ARRIVED = 1e-6


@ACTION_TERM_REGISTRY.register()
class RateLimitedGripper(BinaryJointController):
    """A binary gripper (action > 0 closes) whose finger targets travel toward open or closed at ``speed``.

    A servo gripper moves at a set speed and then squeezes. Driven straight to the end of their travel, the
    fingers' position servos would close as fast as their stiffness pulls them (Mochi clamps no drive force), so
    the squeeze stiffness would also set the speed. Here the target ramps instead: the servo gains set only how
    hard the jaws hold.

    Parameters
    ----------
    speed : float
        How far a finger's target moves per second (joint units: m/s for the YAM's slides).
    """

    speed: float = YAM_FINGER_SPEED

    def build(self) -> None:
        super().build()
        if self.speed <= 0.0:
            raise ValueError(f"{term_subject(self)}: speed must be positive, got {self.speed}.")
        self._target = torch.zeros(self.num_envs, len(self.genesis_dofs_idx), dtype=gs.tc_float, device=self.device)
        # Envs whose target restarts from where their fingers are: every env at first, and any env reset since.
        self._restart = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def compute(self, actions: torch.Tensor) -> None:
        super().compute(actions)
        fingers = self.entity.get_dofs_pos(dofs_idx_local=self.genesis_dofs_idx)
        self._target = torch.where(self._restart[:, None], fingers, self._target)
        self._restart.zero_()
        step = self.speed * self._env.dt
        self._target = self._target + (self._processed_action - self._target).clamp(-step, step)
        self._processed_action = self._target

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        super().reset(envs_idx)
        self._restart[slice(None) if envs_idx is None else envs_idx] = True


class _PinningGripper(RateLimitedGripper):
    """A binary gripper (action > 0 closes) that carries pinned soft-body vertices on its hand until it opens.

    Subclasses choose, in :meth:`compute`, which vertices to pin and when (:meth:`_pin`); this base keeps each
    env's pinned set, moves it with the hand every physics step and releases it when the jaws are told to open or
    the env resets.
    """

    soft_entity_name: str = ""
    hand_link_name: str = "link_6"
    max_vertices: int = 128

    def __init__(self, env: EnvBase, options: ActionTermOptions):
        super().__init__(env=env, options=options)
        self._soft: MochiSoftEntity | None = None
        self._pinned: torch.Tensor | None = None
        self._pinned_local: torch.Tensor | None = None

    def build(self) -> None:
        super().build()
        from eden.entities.mochi import MochiSoftEntity

        soft = self._env.entities.get(self.soft_entity_name)
        if not isinstance(soft, MochiSoftEntity):
            raise TypeError(
                f"{term_subject(self)} holds a Mochi soft body, but soft_entity_name={self.soft_entity_name!r} is "
                f"{'not an entity' if soft is None else f'a {type(soft).__name__}'}. Mochi soft bodies in this "
                f"scene: {[n for n, e in self._env.entities.items() if isinstance(e, MochiSoftEntity)]}."
            )
        self._soft = soft
        self._hand = self.entity.get_link(self.hand_link_name)
        k = min(self.max_vertices, soft.n_vertices)
        self._pinned = torch.full((self.num_envs, k), -1, dtype=gs.tc_int, device=self.device)
        self._pinned_local = torch.zeros(self.num_envs, k, 3, dtype=gs.tc_float, device=self.device)

    @property
    def is_latched(self) -> torch.Tensor:
        """Whether each env has soft-body vertices pinned in its jaws, ``(num_envs,)`` bool."""
        return (self._pinned >= 0).any(dim=-1)

    def _closing(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.squeeze(-1) if actions.dtype == torch.bool else (actions.squeeze(-1) * self.scale) > 0

    def _release(self, envs_mask: torch.Tensor) -> None:
        rows = torch.where(envs_mask[:, None], self._pinned, -1)
        self._soft.set_vertices_fixed(rows, is_fixed=False)
        self._pinned.masked_fill_(envs_mask[:, None], -1)

    def _pin(self, pin: torch.Tensor, vertices: torch.Tensor, verts: torch.Tensor) -> None:
        """Pin ``vertices`` ``(num_envs, k)`` where ``pin`` holds, where they stand now (so nothing jumps)."""
        offset = verts.gather(1, vertices[..., None].expand(-1, -1, 3)) - self._hand.get_pos()[:, None, :]
        quat = self._hand.get_quat()[:, None, :].expand(-1, offset.shape[1], -1)
        self._pinned.copy_(torch.where(pin, vertices.to(gs.tc_int), self._pinned))
        self._pinned_local.copy_(torch.where(pin[..., None], inv_transform_by_quat(offset, quat), self._pinned_local))

    def apply_actions(self) -> None:
        super().apply_actions()
        # Carried on the hand every physics step: re-aimed once a control step, the pinned vertices would trail a
        # moving hand by that step's travel.
        pos, quat = self._hand.get_pos(), self._hand.get_quat()
        quat = quat[:, None, :].expand(-1, self._pinned.shape[1], -1)
        self._soft.set_vertices_target(self._pinned, pos[:, None, :] + transform_by_quat(self._pinned_local, quat))

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        super().reset(envs_idx)
        # Pins outlive a scene reset (they are not part of the solver state), so they are released here.
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[slice(None) if envs_idx is None else envs_idx] = True
        self._release(mask)


@ACTION_TERM_REGISTRY.register()
class SettledGraspLatch(_PinningGripper):
    """A binary gripper (action > 0 closes) that pins the soft body under its pads once a grasp has settled.

    RoboSimStudio's ``bar.Latch``. While the jaws are told to close, each control step checks for a grasp made by
    contact: the jaw opening has moved no more than ``still_tolerance`` over the last ``settle_steps`` steps, each
    pad presses with ``min_press`` or more, and the opening is ``min_squeeze`` or more under ``free_width`` (the
    part's own width across the jaws). Then the soft-body vertices within ``reach`` of either pad's contact face
    are pinned where they stand and carried on the hand until the jaws are told to open or the env resets. A hand
    closed on air or on one edge never meets the test and never latches.

    Parameters
    ----------
    soft_entity_name : str
        The Mochi soft body the jaws hold.
    hand_link_name : str
        The hand link: the closing axis is in its frame, and the pinned vertices ride on it.
    pad_links_name : tuple[str, str]
        The two pad links, the one on the closing axis's + side first.
    pad_face_center, pad_face_half : tuple[float, float, float]
        The pad's contact face as a box in its link's frame: centre and half extents (m).
    closing_axis : tuple[float, float, float]
        Unit vector in the hand's frame from the first pad toward the second.
    free_width : float
        The part's width across the jaws (m): RoboSimStudio's 50 mm sponge.
    settle_steps : int
        Control steps the opening must hold still over (RoboSimStudio's 0.2 s).
    still_tolerance : float
        How far (m) the opening may move over those steps (0.3 mm).
    min_press : float
        Normal force (N) each pad must press with (2 N).
    min_squeeze : float
        How far (m) under ``free_width`` the opening must be (2 mm).
    reach : float
        Vertices within this (m) of a pad's contact face are pinned (6 mm).
    max_vertices : int
        At most this many vertices are pinned per env (the nearest).
    """

    pad_links_name: tuple[str, str] = ("lf_down", "rf_down")
    pad_face_center: tuple[float, float, float] = (0.0, -0.0024, 0.046)
    pad_face_half: tuple[float, float, float] = (0.006, 0.002, 0.04)
    closing_axis: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    free_width: float = 0.05
    settle_steps: int = 6
    still_tolerance: float = 3e-4
    min_press: float = 2.0
    min_squeeze: float = 0.002
    reach: float = 0.006

    def build(self) -> None:
        super().build()
        if self.settle_steps < 1:
            raise ValueError(f"{term_subject(self)}: settle_steps must be at least 1, got {self.settle_steps}.")
        self._pad_links = [self.entity.get_link(name) for name in self.pad_links_name]
        _, pads = self.entity.find_named_links_idx_local(list(self.pad_links_name), preserve_order=True)
        self._pads = torch.as_tensor(pads, dtype=torch.long, device=self.device)
        self._axis = torch.tensor(self.closing_axis, dtype=gs.tc_float, device=self.device)
        self._face_center = torch.tensor(self.pad_face_center, dtype=gs.tc_float, device=self.device)
        self._face_half = torch.tensor(self.pad_face_half, dtype=gs.tc_float, device=self.device)
        # The opening is the sum of the fingers' travel toward open.
        self._open_sign = torch.sign(self._open_action - self._close_action).to(gs.tc_float)
        self._openings = torch.full((self.num_envs, self.settle_steps), torch.inf, device=self.device)

    def _face_distance(self, verts: torch.Tensor) -> torch.Tensor:
        """Distance of every vertex to the nearer pad's contact face, ``(num_envs, n_vertices)``."""
        distances = []
        for link in self._pad_links:
            quat = link.get_quat()[:, None, :].expand(-1, verts.shape[1], -1)
            local = inv_transform_by_quat(verts - link.get_pos()[:, None, :], quat) - self._face_center
            distances.append((local.abs() - self._face_half).clamp_min(0.0).norm(dim=-1))
        return torch.minimum(*distances)

    def compute(self, actions: torch.Tensor) -> None:
        super().compute(actions)
        close = self._closing(actions)
        self._release(~close)
        opening = (self.entity.get_dofs_pos(dofs_idx_local=self.genesis_dofs_idx) * self._open_sign).sum(dim=-1)
        self._openings = torch.cat([self._openings[:, 1:], opening[:, None]], dim=-1)
        self._openings.masked_fill_(~close[:, None], torch.inf)

        still = (self._openings.amax(dim=-1) - self._openings.amin(dim=-1)) <= self.still_tolerance
        press = pad_squeeze(self.entity, self._pads, self._hand, self._axis).amin(dim=-1) >= self.min_press
        squeezed = opening <= self.free_width - self.min_squeeze
        engage = close & ~self.is_latched & still & press & squeezed

        verts = self._soft.get_vertex_positions()
        distance, nearest = self._face_distance(verts).topk(self._pinned.shape[1], largest=False)
        self._pin(engage[:, None] & (distance <= self.reach), nearest, verts)

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        super().reset(envs_idx)
        self._openings[slice(None) if envs_idx is None else envs_idx] = torch.inf


@ACTION_TERM_REGISTRY.register()
class FeatureGraspLatch(_PinningGripper):
    """A binary gripper (action > 0 closes) that seizes a soft body's declared feature near its tool centre.

    RoboSimStudio's deformable grasp (``attach``: a ``ParticleDragger`` in pin mode). On the control step a close
    completes (the finger targets reach the closed command, which they travel to at ``speed``), the vertices of
    ``feature_vertices`` within ``radius`` of the tool centre (the nearest ``max_vertices`` of them) are pinned where
    they stand and carried on the hand until the jaws are told to open or the env resets. A close with none in reach
    seizes nothing until the next close. RoboSimStudio seizes on the close command itself; with jaws that take
    their travel's time to shut, that would carry the cloth on jaws still open, before the pads reached it.

    Parameters
    ----------
    soft_entity_name : str
        The Mochi soft body the jaws hold.
    hand_link_name : str
        The hand link the tool centre is fixed to, which the pinned vertices ride on.
    tcp_offset : tuple[float, float, float]
        The tool centre in the hand's frame (m).
    feature_vertices : list[int]
        The declared feature the jaws can seize (a towel's edge strip). Empty: the whole body.
    radius : float
        Vertices within this (m) of the tool centre are seized: RoboSimStudio's 3 cm.
    max_vertices : int
        At most this many (the nearest) are seized per env: RoboSimStudio's 128.
    """

    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    feature_vertices: list[int] = []
    radius: float = 0.03

    def build(self) -> None:
        super().build()
        n = self._soft.n_vertices
        bad = [v for v in self.feature_vertices if not 0 <= v < n]
        if bad:
            raise ValueError(
                f"{term_subject(self)}: feature_vertices {bad} are not vertices of "
                f"'{self.soft_entity_name}', which has "
                f"{n} (indices 0..{n - 1})."
            )
        self._outside = torch.zeros(n, dtype=torch.bool, device=self.device)
        if self.feature_vertices:
            self._outside[:] = True
            self._outside[list(self.feature_vertices)] = False
        self._tcp_offset = torch.tensor(self.tcp_offset, dtype=gs.tc_float, device=self.device)
        self._closed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def compute(self, actions: torch.Tensor) -> None:
        super().compute(actions)
        close = self._closing(actions)
        self._release(~close)
        closed = close & ((self._target - self._close_action).abs().amax(dim=-1) <= _TARGET_ARRIVED)
        seize = closed & ~self._closed
        self._closed.copy_(closed)
        verts = self._soft.get_vertex_positions()
        tcp = link_point(self._hand, self._tcp_offset)
        distance = (verts - tcp[:, None, :]).norm(dim=-1).masked_fill(self._outside, torch.inf)
        distance, nearest = distance.topk(self._pinned.shape[1], largest=False)
        self._pin(seize[:, None] & (distance <= self.radius), nearest, verts)

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        super().reset(envs_idx)
        self._closed[slice(None) if envs_idx is None else envs_idx] = False


@METRIC_TERM_REGISTRY.register()
class LatchHolds(MetricTerm):
    """1 while the named latching gripper holds soft-body vertices pinned, else 0: a shared quantity, not a gate.

    Configure ``contributes_to_success=False``; rewards that pay for holding the part gate on it.
    """

    action_name: str = "gripper"

    def build(self) -> None:
        latch = self._env.action_manager.get_term(self.action_name)
        if not isinstance(latch, _PinningGripper):
            raise TypeError(
                f"LatchHolds reads a latching gripper (SettledGraspLatch, FeatureGraspLatch), but action term "
                f"'{self.action_name}' is a {type(latch).__name__}. Set action_name to the task's gripper."
            )
        self._latch = latch

    def compute(self) -> torch.Tensor:
        return self._latch.is_latched.float()
