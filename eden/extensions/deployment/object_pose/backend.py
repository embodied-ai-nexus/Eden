"""``TrackedObjectMixin``: any deployment backend, plus scene objects posed from a camera every tick.

Mix it in ahead of a robot's backend::

    @DEPLOYMENT_REGISTRY.register()
    class FrankaWithCubeDeployment(TrackedObjectMixin, FrankaPandaDeployment):
        pass

    deployment = FrankaWithCubeDeployment(env, DeploymentOptions(tracked_objects={
        "cube": {"parent_offset_pos": (0.6, 0.0, 0.8), "parent_offset_quat": (0.0, 0.707, -0.707, 0.0)},
    }))
    deployment.attach_pose_source("cube", EstimatorPoseSource(camera, AprilTagEstimator(tags=CUBE_TAGS)))
    deployment.connect()

Each tick, after the robot is read and synced onto the scene, every tracked object's latest pose is
carried into the world frame and written into its scene entity (``set_pos`` / ``set_quat`` with
``relative=True``: the authored frame its ``default_root_pos`` uses). The chain, right to left from
what the source measures to what the scene holds::

    T_world_entity = T_world_parent · T_parent_report · T_report_obj · T_obj_entity

- ``T_report_obj``: the source's sample, the object in the source's report frame (the camera's
  optical frame, or an anchor tag's).
- ``T_parent_report`` (:attr:`TrackedObjectSpec.parent_offset_pos` / ``_quat``): where that report
  frame sits on its parent. A camera on a tripod: its extrinsic in the world. An anchor tag on the
  wrist, or a camera on the wrist: its pose in that link's frame (hand-eye calibration).
- ``T_world_parent``: identity for ``parent_link=None`` (the world, turned with a floating base's
  START yaw, as the MuJoCo composite turns its props), else the link's pose from the robot's forward
  kinematics this tick.
- ``T_obj_entity`` (:attr:`TrackedObjectSpec.object_offset_pos` / ``_quat``): the scene entity's
  authored frame in the frame the source tracks (a mesh origin that differs from the URDF root, an
  axis convention that differs from the task's).

Mapping the Wuji rig onto it (``examples/projects/manipulation/wuji/deploy``): reorient's cube is
``parent_link`` = the wrist-tag link with the source in that tag's frame; pen spin's pen is
``parent_link`` = the palm with ``parent_offset`` = ``TAG_IN_PALM[side]`` and ``object_offset`` the
shaft-axis rotation and offset. The left-hand mirroring is Wuji's alone: a subclass overrides
:meth:`TrackedObjectMixin._source_pose`.
"""

from __future__ import annotations

import math
from abc import ABC
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, field_validator

import eden as en
from eden.extensions.deployment.object_pose.sources import PoseSample, PoseSource, wait_for_pose
from eden.utils.common import ConfigurableMixin
from eden.utils.geom import transform_pos_quat_by_trans_quat

if TYPE_CHECKING:
    from eden.extensions.deployment.utils.state import RobotState


class ObjectPoseLost(RuntimeError):
    """A tracked object's pose went stale while the policy drove the robot (``on_stale="raise"``)."""


class TrackedObjectSpec(BaseModel):
    """How one tracked object's source pose reaches its scene entity. See the module docstring for the chain.

    Attributes
    ----------
    parent_link : str | None
        The robot link the source's report frame is fixed to; ``None``: fixed in the world.
    parent_entity : str | None
        The entity that owns ``parent_link``; ``None``: the deployment's own ``entity_name``.
    parent_offset_pos, parent_offset_quat : tuple
        The report frame's pose in the parent frame (wxyz): the camera extrinsic, the anchor tag's
        mount, or the hand-eye calibration.
    object_offset_pos, object_offset_quat : tuple
        The scene entity's authored frame in the tracked object frame (wxyz).
    max_age_s : float | None
        A pose older than this, while the policy drives, is stale (``on_stale`` says what then).
        ``None`` keeps using the last pose however old.
    connect_timeout_s : float
        How long ``connect`` waits for a first live pose before refusing to start.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_link: str | None = None
    parent_entity: str | None = None
    parent_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    parent_offset_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    object_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    object_offset_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    max_age_s: float | None = 0.5
    connect_timeout_s: float = 10.0

    @field_validator("parent_offset_quat", "object_offset_quat")
    @classmethod
    def _unit_quat(cls, q: tuple[float, ...]) -> tuple[float, ...]:
        norm = math.sqrt(sum(v * v for v in q))
        if not abs(norm - 1.0) < 1e-3:
            raise ValueError(f"{q} is not a unit wxyz quaternion (norm {norm:.4g}); normalise it, wxyz order.")
        return tuple(v / norm for v in q)

    @field_validator("max_age_s")
    @classmethod
    def _positive_age(cls, v: float | None) -> float | None:
        if v is not None and not v > 0:
            raise ValueError(f"max_age_s={v!r} must be positive, or None to keep the last pose.")
        return v


class TrackedObjectMixin(ConfigurableMixin, ABC):
    """Scene objects posed from :class:`PoseSource` s, for any :class:`~eden.extensions.deployment.base.DeploymentBase`.

    Mixed in *ahead of* the robot's backend (as :class:`~eden.extensions.deployment.base.SoftShutdownMixin`
    is; a :class:`ConfigurableMixin` so its two options count as the backend's), so its hooks wrap the backend's: ``_sync_state`` (pose the
    objects after the robot's FK is current), ``_before_step`` (the staleness guard), ``_handle_start``
    (no START without a live pose), ``connect`` and ``close`` (the sources' lifetimes).

    Options
    -------
    tracked_objects : dict
        ``{entity_name: TrackedObjectSpec | dict}``: the scene entities to pose and how.
    on_stale : {"stop", "raise"}
        A stale pose under the policy: ``"stop"`` does what the operator's STOP does (the robot
        holds where it is) and logs which object went stale; ``"raise"`` raises
        :class:`ObjectPoseLost`, ending the run as the backend ends any failed tick.

    The pose sources are live objects (a camera, a thread, a socket), not options: bind each with
    :meth:`attach_pose_source` before ``connect``.
    """

    tracked_objects: dict = {}  # noqa: RUF012 -- a configurable parameter; ConfigurableMixin deep-copies it per instance
    on_stale: Literal["stop", "raise"] = "stop"

    # ------------------------------------------------------------------ construction

    def _resolve_options(self) -> None:
        """Validate every tracked object against the scene, before the backend swaps its entity (see the base)."""
        super()._resolve_options()
        owner = type(self).__name__
        if self.on_stale not in ("stop", "raise"):
            raise ValueError(f"{owner}: on_stale={self.on_stale!r} is not one of ('stop', 'raise').")
        entities = self._env.entities
        self._object_specs: dict[str, TrackedObjectSpec] = {}
        #: ``{object: (parent entity name, parent link local index)}``; absent for a world-fixed parent.
        self._object_parent_links: dict[str, tuple[str, int]] = {}
        for name, raw in self.tracked_objects.items():
            if name not in entities:
                raise KeyError(
                    f"{owner}: tracked_objects names {name!r}, which is not a scene entity; entities: {sorted(entities)}"
                )
            if name == self.entity_name:
                raise ValueError(
                    f"{owner}: tracked_objects names {name!r}, the deployed robot itself; list only objects."
                )
            spec = raw if isinstance(raw, TrackedObjectSpec) else TrackedObjectSpec(**raw)
            self._object_specs[name] = spec
            if spec.parent_link is None:
                if spec.parent_entity is not None:
                    raise ValueError(
                        f"{owner}: tracked_objects[{name!r}] sets parent_entity={spec.parent_entity!r} without "
                        "parent_link; give the link the report frame is fixed to, or drop parent_entity for the world."
                    )
                continue
            parent = spec.parent_entity or self.entity_name
            if parent not in entities:
                raise KeyError(
                    f"{owner}: tracked_objects[{name!r}].parent_entity={parent!r} is not a scene entity; entities: "
                    f"{sorted(entities)}"
                )
            links = [link.name for link in entities[parent].links]
            if spec.parent_link not in links:
                raise KeyError(
                    f"{owner}: tracked_objects[{name!r}].parent_link={spec.parent_link!r} is not a link of "
                    f"{parent!r}; its links: {links}"
                )
            _, idx = entities[parent].find_named_links_idx_local([spec.parent_link])
            self._object_parent_links[name] = (parent, idx[0])
        self._pose_sources: dict[str, PoseSource] = {}
        self._object_never_seen: set[str] = set(self._object_specs)

    def attach_pose_source(self, entity_name: str, source: PoseSource) -> None:
        """Bind the source of ``entity_name``'s pose. Owned from here: :meth:`close` closes it."""
        if entity_name not in self._object_specs:
            raise KeyError(
                f"{type(self).__name__}.attach_pose_source: {entity_name!r} is not in tracked_objects "
                f"{sorted(self._object_specs)}; add it there first."
            )
        if entity_name in self._pose_sources:
            raise ValueError(
                f"{type(self).__name__}.attach_pose_source: {entity_name!r} already has {self._pose_sources[entity_name].name}."
            )
        self._pose_sources[entity_name] = source

    # ------------------------------------------------------------------ lifetime

    def connect(self) -> None:
        """Connect the robot, then start every source and wait for each object's first live pose."""
        missing = sorted(set(self._object_specs) - set(self._pose_sources))
        if missing:
            raise RuntimeError(
                f"{type(self).__name__}: no pose source attached for {missing}. Fix: call "
                "attach_pose_source(name, source) for each before connect()."
            )
        super().connect()
        for name, source in self._pose_sources.items():
            source.start()
        for name, source in self._pose_sources.items():
            spec = self._object_specs[name]
            wait_for_pose(source, spec.connect_timeout_s, spec.max_age_s or math.inf, f"the {name!r} object")
            en.logger.info(f"[object_pose] {name!r}: live from {source.name}")

    def close(self) -> None:
        try:
            super().close()
        finally:
            for source in self._pose_sources.values():
                try:
                    source.close()
                except Exception as exc:  # noqa: BLE001 -- one source failing to close must not keep the others open
                    en.logger.warning(f"[object_pose] closing {source.name} failed: {exc}")

    # ------------------------------------------------------------------ per tick

    def _stale_objects(self) -> list[str]:
        """``"name (why)"`` for every object whose pose is missing or older than its ``max_age_s``."""
        stale = []
        for name, source in self._pose_sources.items():
            limit = self._object_specs[name].max_age_s
            sample = source.latest()
            if sample is None:
                stale.append(f"{name!r} (no pose yet)")
            elif limit is not None and sample.age > limit:
                stale.append(f"{name!r} ({sample.age:.2f} s old, limit {limit:g} s)")
        return stale

    def _handle_start(self) -> None:
        """Refuse START while any tracked object's pose is stale: the policy would act on where the object was."""
        if stale := self._stale_objects():
            en.logger.error(
                f"[object_pose] START refused: stale object pose for {', '.join(stale)}. Put it in view and START again."
            )
            return
        super()._handle_start()

    def _before_step(self) -> None:
        super()._before_step()
        if not self.policy_active:
            return
        if stale := self._stale_objects():
            msg = f"object pose lost under the policy: {', '.join(stale)}"
            if self.on_stale == "raise":
                raise ObjectPoseLost(msg)
            en.logger.error(f"[object_pose] {msg} -- STOP (hold). START again once it is back in view.")
            self.request_stop()

    def _source_pose(self, name: str, sample: PoseSample) -> tuple[np.ndarray, np.ndarray]:
        """``T_report_obj`` for ``name`` from its sample. Identity here; a rig overrides it (Wuji's left-hand mirror)."""
        return sample.pos, sample.quat

    def _world_parent_pose(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """``T_world_parent``, ``(1, 3)`` / ``(1, 4)``: the parent link's FK, or the world turned by the START yaw."""
        device = self._env.device
        if name in self._object_parent_links:
            parent, idx = self._object_parent_links[name]
            entity = self._env.entities[parent]  # re-resolved each tick: the deployment swaps the robot's entity
            return entity.get_links_pos([idx])[:, 0], entity.get_links_quat([idx])[:, 0]
        # A world-fixed camera sees the world; the policy's world is turned by the yaw captured at START on a
        # floating base (``auto_yaw_align``), so the object turns with it, as MujocoCompositeDeployment's props do.
        root = self._state_entity
        yaw = 0.0 if root.has_immobile_base else root.yaw_offset
        quat = torch.tensor([[math.cos(-0.5 * yaw), 0.0, 0.0, math.sin(-0.5 * yaw)]], device=device)
        return torch.zeros((1, 3), device=device), quat

    def _sync_state(self, state: RobotState) -> None:
        """Sync the robot (its FK is what a link-parented object is placed by), then pose every tracked object."""
        super()._sync_state(state)
        device = self._env.device

        def t(x) -> torch.Tensor:
            return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)[None]

        for name, source in self._pose_sources.items():
            sample = source.latest()
            if sample is None:
                continue  # the entity stays where it was; START is refused until a pose arrives
            if name in self._object_never_seen:
                self._object_never_seen.discard(name)
                en.logger.info(f"[object_pose] {name!r}: first pose from {source.name}")
            spec = self._object_specs[name]
            pos, quat = self._source_pose(name, sample)
            # T_report_entity = T_report_obj · T_obj_entity
            pos, quat = transform_pos_quat_by_trans_quat(
                t(spec.object_offset_pos), t(spec.object_offset_quat), t(pos), t(quat)
            )
            # T_parent_entity = T_parent_report · T_report_entity
            pos, quat = transform_pos_quat_by_trans_quat(
                pos, quat, t(spec.parent_offset_pos), t(spec.parent_offset_quat)
            )
            # T_world_entity = T_world_parent · T_parent_entity
            pos, quat = transform_pos_quat_by_trans_quat(pos, quat, *self._world_parent_pose(name))
            entity = self._env.entities[name]
            entity.set_pos(pos, relative=True)
            entity.set_quat(quat, relative=True)
            # Same keys as MujocoCompositeDeployment's props, so a hardware and a sim-to-sim --debug_record line up.
            state.extra[f"{name}.pos"] = pos[0].cpu().numpy().astype(np.float64)
            state.extra[f"{name}.quat"] = quat[0].cpu().numpy().astype(np.float64)
            state.extra[f"{name}.pose_age_s"] = float(sample.age)
