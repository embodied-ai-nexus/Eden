"""Grasp-frame and pad-contact helpers shared by the robosuite terms.

robosuite measures every reaching distance from the gripper's ``grip_site`` — a frame between
the fingertips — and detects a grasp as *both* fingerpad geom groups touching the object
(``ManipulationEnv._check_grasp``). Both notions are reproduced here off the rig the scene holds
(:class:`RigMixin`) so they hold for the Franka Hand and for a welded Robotiq alike:

* :class:`RigMixin` — the one robot-shaped field a term declares: the scene slot its arm + gripper
  composite sits in. Everything else about the rig is read off that composite in ``build()``.
* :class:`GraspFrameMixin` — the grasp point is the midpoint of the pad **collision geoms**,
  measured once at ``build()`` as a fixed offset in the ``ee_link`` frame and reproduced each step
  by rigid-body FK from that link. Pad *link* origins can sit centimetres proximal of the actual
  pad surfaces (the Robotiq's do), so the geoms are the right reference; the midpoint of two
  symmetric pads does not move with finger opening, so a fixed offset is exact.
* :func:`pads_touching` — per-env "both pads in contact with the object", from the contact pairs
  of the most recent physics step.
* :class:`ObjectPointMixin` — a point of the manipulated object (its root, one link, or a fixed
  offset in either frame — robosuite's sites), read in the MJCF body-origin frame robosuite uses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import inv_transform_by_quat, transform_by_quat

from eden.managers.terms.utils import resolve_link_idx
from eden.options.robots import ManipulatorOptions

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.entities.rigid import RigidEntity


def pad_geoms_by_link(gripper: RigidEntity, grasp_links: list[str]) -> list[list]:
    """The collision geoms of each link in ``grasp_links``, grouped per link; raises if a link carries none.

    Grouped rather than flat because the grasp point is the midpoint *of the two pads*, not the mean
    of however many geoms each happens to be modelled with: a gripper whose pads carry unequal geom
    counts would otherwise pull the point sideways towards the busier pad, silently. Both in-tree
    grippers are symmetric (the Franka Hand one geom per pad, the Robotiq 2F85 two), so for them the
    two definitions agree.
    """
    found: dict[str, list] = {name: [] for name in grasp_links}
    for geom in gripper.geoms:
        if geom.link.name in found:
            found[geom.link.name].append(geom)
    empty = [name for name, geoms in found.items() if not geoms]
    if empty:
        available = sorted({geom.link.name for geom in gripper.geoms})
        raise ValueError(
            f"grasp_links {empty} of entity '{gripper.name}' carry no collision geoms; the grasp point is the "
            f"midpoint of the pad geoms, so it cannot be measured. Links with geoms: {available}."
        )
    return [found[name] for name in grasp_links]


class RigMixin:
    """The one robot-shaped field a robosuite term declares: the scene slot holding its rig.

    A robosuite config places a :class:`~eden.options.robots.ManipulatorOptions` composite at each
    robot slot (``robot``, or ``robot0`` / ``robot1``), and that object already carries everything a
    term needs — the names the arm and gripper entities take, the link the gripper is welded to, the
    pad links, both DOF lists. Reading them off it in ``build()`` is what makes ``--robot`` a
    scene-level swap: no term holds a copy of the rig, so there is nothing downstream to rewrite and
    nothing that can be left pointing at the old embodiment.
    """

    rig: str

    def _rig_options(self) -> ManipulatorOptions:
        """The composite at scene slot :attr:`rig` (the env keeps the scene options it was built from)."""
        scene = self._env.scene_options
        composite = getattr(scene, self.rig, None)
        if composite is None:
            raise ValueError(
                f"{type(self).__name__}.rig names scene slot '{self.rig}', which this scene does not have. "
                f"Slots: {sorted(scene.keys())}."
            )
        if not isinstance(composite, ManipulatorOptions):
            raise TypeError(
                f"{type(self).__name__}.rig names scene slot '{self.rig}', which holds a "
                f"{type(composite).__name__}; the robosuite terms read the arm + gripper composite the config "
                f"places there (see eden.tasks.robosuite.rigs.make_rig)."
            )
        return composite


class GraspFrameMixin(RigMixin):
    """Resolve the grasp frame once in ``build()``; read it each step with :meth:`_grasp_pos_w`.

    The host term declares :attr:`~RigMixin.rig`; the arm, the gripper, the mount link and the pad
    links all come off that composite. Call :meth:`_build_grasp_frame` from ``build()``.
    """

    def _build_grasp_frame(self) -> None:
        rig = self._rig_options()
        self._arm_name = rig.arm_name
        arm: RigidEntity = self.resolve_entity(rig.arm_name)
        gripper: RigidEntity = self.resolve_entity(rig.gripper_name)
        self._ee_idx = resolve_link_idx(arm, [rig.ee_link], self.device)
        pads = pad_geoms_by_link(gripper, list(rig.grasp_links))
        # World-frame pad midpoint at build time (every env is in the same pose then), expressed in
        # the ee_link frame. Genesis runs forward kinematics at scene build, so the geom poses are
        # populated before any step. Mean *within* each pad first, then across the pads, so the
        # midpoint stays between them whatever each pad's geom count.
        per_pad = [torch.stack([geom.get_pos(relative=False) for geom in geoms]).mean(dim=0) for geoms in pads]
        pads_w = torch.stack(per_pad).mean(dim=0)  # (num_envs, 3)
        ee_pos = arm.get_links_pos(ls_idx_local=self._ee_idx)[:, 0]
        ee_quat = arm.get_links_quat(ls_idx_local=self._ee_idx)[:, 0]
        offset = inv_transform_by_quat(pads_w - ee_pos, ee_quat)[:1]  # (1, 3)
        if not torch.isfinite(offset).all() or offset.norm() > 0.5:
            raise ValueError(
                f"Grasp frame of '{rig.gripper_name}' measured {offset.squeeze(0).tolist()} m from link "
                f"'{rig.ee_link}' of '{rig.arm_name}' — not a plausible pad midpoint. Check that the rig's ee_link "
                f"is the gripper mount frame and its grasp_links are the two finger pads."
            )
        self._grasp_offset = offset.expand(self.num_envs, 3).contiguous()

    def _ee_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        arm: RigidEntity = self.resolve_entity(self._arm_name)
        ee_pos = arm.get_links_pos(ls_idx_local=self._ee_idx)[:, 0]
        ee_quat = arm.get_links_quat(ls_idx_local=self._ee_idx)[:, 0]
        return ee_pos, ee_quat

    def _grasp_pos_w(self) -> torch.Tensor:
        """World position of the grasp point (pad-geom midpoint). ``(num_envs, 3)``."""
        ee_pos, ee_quat = self._ee_pose_w()
        return ee_pos + transform_by_quat(self._grasp_offset, ee_quat)


def pad_link_global_indices(gripper: RigidEntity, grasp_links: list[str]) -> list[int]:
    """Global (scene-wide) link indices of the pad links, as ``get_contacts`` reports them."""
    return [gripper.get_link(name).idx for name in grasp_links]


def links_touching(contacts: dict[str, torch.Tensor], link_idx: int) -> torch.Tensor:
    """Per-env bool: some valid contact pair in ``contacts`` involves global link ``link_idx``."""
    hit = (contacts["link_a"] == link_idx) | (contacts["link_b"] == link_idx)
    return (hit & contacts["valid_mask"]).any(dim=-1)


def pads_touching(gripper: RigidEntity, obj: Entity, pad_idx: list[int]) -> torch.Tensor:
    """Per-env bool ``(num_envs,)``: every pad link in ``pad_idx`` touches ``obj`` (robosuite ``_check_grasp``)."""
    contacts = gripper.get_contacts(with_entity=obj)
    grasping = links_touching(contacts, pad_idx[0])
    for idx in pad_idx[1:]:
        grasping = grasping & links_touching(contacts, idx)
    return grasping


class ObjectPointMixin:
    """Resolve an object point once in ``build()``; read it each step with :meth:`_object_point_w`.

    The point is ``object_offset`` in the frame of ``object_link_name`` (a link of the object) or, when
    that is ``None``, in the object's MJCF root frame — i.e. robosuite's body / site positions. Both are
    read with ``relative=True``: Genesis re-origins a free MJCF body onto its inertial frame, and
    robosuite's numbers are body-origin numbers. Requires the host term to declare the three
    fields; call :meth:`_build_object_point` from ``build()``.
    """

    object_name: str
    object_link_name: str | None
    object_offset: tuple[float, float, float]

    def _build_object_point(self) -> None:
        obj = self.resolve_entity(self.object_name)
        self._obj_link = None
        if self.object_link_name is not None:
            self._obj_link = resolve_link_idx(obj, [self.object_link_name], self.device)
        self._has_offset = any(v != 0.0 for v in self.object_offset)
        offset = torch.tensor(self.object_offset, dtype=torch.float32, device=self.device)
        self._obj_offset = offset.expand(self.num_envs, 3).contiguous()

    def _object_point_w(self) -> torch.Tensor:
        """World position of the object point. ``(num_envs, 3)``."""
        obj = self.resolve_entity(self.object_name)
        if self._obj_link is None:
            pos = obj.get_pos(relative=True)
            if not self._has_offset:
                return pos
            return pos + transform_by_quat(self._obj_offset, obj.get_quat(relative=True))
        # ``relative=True`` strips the centre-of-mass anchor when the named link is the object's root. A
        # fixed *child* link of a free single-body object still reports a frame displaced by the root's
        # anchor, which no flag strips (docs/guides/entities.md): name a child link only on an articulated
        # object.
        pos = obj.get_links_pos(ls_idx_local=self._obj_link, relative=True)[:, 0]
        if not self._has_offset:
            return pos
        quat = obj.get_links_quat(ls_idx_local=self._obj_link, relative=True)[:, 0]
        return pos + transform_by_quat(self._obj_offset, quat)
