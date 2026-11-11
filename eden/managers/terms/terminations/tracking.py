"""Termination terms that gate episodes on robot-vs-motion-ref tracking error.

Two families:

* **Direct world-frame gates** (function-style, registry keys
  ``motion_{base,link,dofs}_*_tracking_error_above``): hard ``> threshold``
  tests on raw distances. Symmetric with
  :mod:`eden.managers.terms.rewards.tracking`. The dexmachina ManipTrans
  curriculum kills envs whose pos / rot / per-link error exceed an ``eps``;
  these terms supply the gates (decaying the eps is a curriculum's job).
* **Anchor-yaw-warped gates** (``bad_anchor_*``, ``BadMotionLinkPosZOnly``):
  compare robot state against the yaw-warped motion target so the gate
  doesn't fire just because the robot drifted in heading.

Also exposes :func:`motion_ends`, which fires when the motion buffer for
the entity runs out of frames.

Reduction conventions for multi-link / multi-DOF terms:

* ``"max"`` (default) — strict per-element gate; any link/DOF beyond
  threshold triggers. Matches ManipTrans semantics.
* ``"mean"`` — averaged-error gate; lenient.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import torch

from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY, TerminationTerm
from eden.managers.terms.utils import gravity_vec
from genesis.utils.geom import inv_transform_by_quat

from eden.utils.geom import quat_error_magnitude
from eden.utils.string import term_subject
from eden.utils.motion_reference.lookup import (
    anchor_state_once,
    assert_motion_field_present,
    current_motion_frame,
    get_motion_dofs_idx,
    get_motion_link_idx,
    get_robot_dofs_idx,
    get_robot_link_idx,
)

if TYPE_CHECKING:
    from eden.envs.base import MimicRLEnvBase


@TERMINATION_TERM_REGISTRY.register()
def motion_base_position_tracking_error_above(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    threshold: float,
) -> torch.Tensor:
    """Terminate when world-frame base-position L2 error exceeds ``threshold``."""
    assert_motion_field_present(env, entity_name, "base_pos_w")
    entity = env.get_entity(entity_name)
    ref = env.motion_manager.data[entity_name]
    return torch.norm(entity.get_pos() - current_motion_frame(ref.base_pos_w), dim=-1) > threshold


@TERMINATION_TERM_REGISTRY.register()
def motion_base_orientation_tracking_error_above(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    threshold: float,
) -> torch.Tensor:
    """Terminate when base quat-error magnitude (wxyz) exceeds ``threshold``."""
    assert_motion_field_present(env, entity_name, "base_quat_w")
    entity = env.get_entity(entity_name)
    ref = env.motion_manager.data[entity_name]
    return quat_error_magnitude(entity.get_quat(), current_motion_frame(ref.base_quat_w)) > threshold


@TERMINATION_TERM_REGISTRY.register()
def motion_link_position_tracking_error_above(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    links_name: list[str],
    threshold: float,
    reduction: Literal["mean", "max"] = "max",
) -> torch.Tensor:
    """Terminate when link-position L2 error reduction exceeds ``threshold``."""
    assert_motion_field_present(env, entity_name, "links_pos_w")
    motion_idx = get_motion_link_idx(env, entity_name, links_name)
    robot_idx = get_robot_link_idx(env, entity_name, links_name)
    ref = env.motion_manager.data[entity_name]
    motion_pos = current_motion_frame(ref.links_pos_w, motion_idx)
    robot_pos = env.get_entity(entity_name).get_links_pos(ls_idx_local=robot_idx)
    dist = torch.norm(motion_pos - robot_pos, dim=-1)
    err = dist.max(-1).values if reduction == "max" else dist.mean(-1)
    return err > threshold


@TERMINATION_TERM_REGISTRY.register()
def motion_link_orientation_tracking_error_above(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    links_name: list[str],
    threshold: float,
    reduction: Literal["mean", "max"] = "max",
) -> torch.Tensor:
    """Terminate when link quat-error reduction exceeds ``threshold``."""
    assert_motion_field_present(env, entity_name, "links_quat_w")
    motion_idx = get_motion_link_idx(env, entity_name, links_name)
    robot_idx = get_robot_link_idx(env, entity_name, links_name)
    ref = env.motion_manager.data[entity_name]
    motion_quat = current_motion_frame(ref.links_quat_w, motion_idx)
    robot_quat = env.get_entity(entity_name).get_links_quat(ls_idx_local=robot_idx)
    err = quat_error_magnitude(motion_quat, robot_quat)
    err = err.max(-1).values if reduction == "max" else err.mean(-1)
    return err > threshold


@TERMINATION_TERM_REGISTRY.register()
def motion_dofs_pos_tracking_error_above(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    threshold: float,
    dofs_name: list[str] | None = None,
    reduction: Literal["mean", "max"] = "max",
) -> torch.Tensor:
    """Terminate when per-DOF |q − q_ref| reduction exceeds ``threshold``."""
    assert_motion_field_present(env, entity_name, "dofs_pos")
    entity = env.get_entity(entity_name)
    ref = env.motion_manager.data[entity_name]
    if dofs_name is None:
        dofs_name = env.motion_manager.get_target_dofs_name(entity_name)
        if dofs_name is None:
            raise ValueError(f"Entity '{entity_name}' is not tracked by any motion term.")
        if not dofs_name:
            raise ValueError(
                f"motion_dofs_pos_tracking_error_above: entity '{entity_name}' has no DOFs "
                f"in the motion reference. Use motion_link_position_tracking_error_above / "
                f"motion_base_position_tracking_error_above for rigid or link-only entities, "
                f"or pass an explicit dofs_name list."
            )
    motion_idx = get_motion_dofs_idx(env, entity_name, dofs_name)
    robot_idx = get_robot_dofs_idx(env, entity_name, dofs_name)
    motion_q = current_motion_frame(ref.dofs_pos, motion_idx)
    # `dofs_idx_local` to the getter rather than a gather off a full-width read — see the same
    # line in ``motion_dofs_pos_tracking_exp``: `get_dofs_pos()` is columned in dofs_name order,
    # which `robot_idx` (Genesis-local) does not index.
    robot_q = entity.get_dofs_pos(dofs_idx_local=robot_idx)
    err = (motion_q - robot_q).abs()
    err = err.max(-1).values if reduction == "max" else err.mean(-1)
    return err > threshold


# ---------------------------------------------------------------------------
# Anchor-yaw-warped (heading-invariant) family + clip-end termination
# ---------------------------------------------------------------------------


def _adaptive_pos_threshold(
    motion_anchor_z: torch.Tensor,
    threshold: float,
    threshold_adaptive: bool,
    down_threshold: float,
    root_height_threshold: float,
) -> torch.Tensor | float:
    """Return an effective per-env termination threshold.

    The threshold is optionally scaled up on low-anchor motion frames (crawl/kneel)
    and held at ``threshold`` on upright frames.

    With ``threshold_adaptive=False`` (default) the static ``threshold`` is
    returned as a scalar.

    With ``threshold_adaptive=True`` the threshold is scaled by

    .. code-block:: text

        scale = 1.0 + clamp((root_height_threshold - z) / root_height_threshold, 0, 1) * down_threshold

    so the gate stays tight at the configured ``threshold`` for upright
    motion frames (``z >= root_height_threshold``) and ramps up linearly to
    ``threshold * (1 + down_threshold)`` as the motion's anchor z drops to
    zero.

    Returns a scalar ``float`` when adaptive is off (so the caller can keep
    its hot path scalar-vs-tensor) and a ``(num_envs,)`` tensor when on.
    """
    if not threshold_adaptive:
        return threshold
    if root_height_threshold <= 0:
        raise ValueError(f"root_height_threshold must be > 0 to compute an adaptive ramp; got {root_height_threshold}.")
    ramp = ((root_height_threshold - motion_anchor_z) / root_height_threshold).clamp(min=0.0, max=1.0)
    return threshold * (1.0 + ramp * down_threshold)


@TERMINATION_TERM_REGISTRY.register()
def bad_anchor_pos_z_only(
    env: MimicRLEnvBase,
    *,
    entity_name: str = "robot",
    anchor_link_name: str = "torso_link",
    threshold: float = 0.01,
    threshold_adaptive: bool = False,
    down_threshold: float = 0.0,
    root_height_threshold: float = 0.5,
) -> torch.Tensor:
    """Terminate when anchor link's |z| error vs motion ref exceeds ``threshold``.

    Asserts both ``links_pos_w`` and ``links_quat_w`` because
    :func:`get_anchor_state` reads both regardless of which we consume here.

    ``threshold_adaptive=True`` switches to an adaptive schedule —
    the threshold grows linearly when the motion's anchor z drops below
    ``root_height_threshold`` (default ``0.5`` m). At the floor the
    threshold becomes ``threshold * (1 + down_threshold)`` so the gate is
    more lenient for crawl / kneel clips, while staying tight for upright
    walks. With ``threshold_adaptive=False`` (default) the static
    ``threshold`` is used.
    """
    assert_motion_field_present(env, entity_name, "links_pos_w")
    assert_motion_field_present(env, entity_name, "links_quat_w")
    robot_pos, _, motion_pos, _ = anchor_state_once(env, entity_name, anchor_link_name)
    eff_threshold = _adaptive_pos_threshold(
        motion_pos[:, -1], threshold, threshold_adaptive, down_threshold, root_height_threshold
    )
    return torch.abs(motion_pos[:, -1] - robot_pos[:, -1]) > eff_threshold


@TERMINATION_TERM_REGISTRY.register()
def bad_anchor_ori(
    env: MimicRLEnvBase,
    *,
    entity_name: str = "robot",
    anchor_link_name: str = "torso_link",
    threshold: float = 0.01,
) -> torch.Tensor:
    """Terminate when the anchor's projected-gravity z diverges from the motion ref's.

    Fires when the difference exceeds ``threshold`` (a pitch/roll-only check on the unit gravity direction, so
    an Isaac Lab threshold ports unchanged).

    Asserts both ``links_pos_w`` and ``links_quat_w`` because
    :func:`get_anchor_state` reads both regardless of which we consume here.
    """
    assert_motion_field_present(env, entity_name, "links_pos_w")
    assert_motion_field_present(env, entity_name, "links_quat_w")
    _, robot_quat, _, motion_quat = anchor_state_once(env, entity_name, anchor_link_name)
    return _projected_gravity_z_error(env, robot_quat, motion_quat) > threshold


def _projected_gravity_z_error(env: MimicRLEnvBase, sim_quat: torch.Tensor, motion_quat: torch.Tensor) -> torch.Tensor:
    """``|g_z(motion) - g_z(sim)|``: how far two frames disagree on tilt, blind to yaw. Shape ``(N,)``.

    ``g`` is the unit down vector :func:`~eden.managers.terms.utils.gravity_vec`, Isaac Lab's ``GRAVITY_VEC_W``, so
    the gap is ``|cos(tilt_motion) - cos(tilt_sim)|`` in ``[0, 2]`` and an Isaac Lab threshold means the same tilt
    here. The solver's gravity is in m/s^2; using it would scale every threshold by 9.81.
    """
    gravity_vec_w = gravity_vec(env.num_envs, env.device)
    motion_projected_gravity_b = inv_transform_by_quat(gravity_vec_w, motion_quat)
    sim_projected_gravity_b = inv_transform_by_quat(gravity_vec_w, sim_quat)
    return (motion_projected_gravity_b[:, 2] - sim_projected_gravity_b[:, 2]).abs()


@TERMINATION_TERM_REGISTRY.register()
def bad_base_ori(env: MimicRLEnvBase, *, entity_name: str, threshold: float) -> torch.Tensor:
    """Terminate when a free base's projected-gravity z diverges from the motion ref's by more than ``threshold``.

    The base-pose counterpart of :func:`bad_anchor_ori` — a tilt-only check that ignores yaw, for an entity with no
    anchor link of its own to read: a tracked object that has been knocked over, while spinning it about the vertical
    is left to the orientation reward. Reads ``get_quat()`` (build the object with ``align=False`` so that is the
    authored body frame the reference records).
    """
    assert_motion_field_present(env, entity_name, "base_quat_w")
    motion_quat = current_motion_frame(env.motion_manager.data[entity_name].base_quat_w)
    return _projected_gravity_z_error(env, env.get_entity(entity_name).get_quat(), motion_quat) > threshold


@TERMINATION_TERM_REGISTRY.register()
class BadMotionLinkPosZOnly(TerminationTerm):
    """Terminate when any tracked link's |z| error vs the motion ref exceeds ``threshold``.

    ``threshold_adaptive=True`` switches to an adaptive schedule —
    the threshold grows linearly when the motion's anchor z drops below
    ``root_height_threshold`` (default ``0.5`` m). At the floor the
    threshold becomes ``threshold * (1 + down_threshold)`` so the gate is
    more lenient for crawl / kneel clips, while staying tight for upright
    walks. With ``threshold_adaptive=False`` (default) the static
    ``threshold`` is used.
    """

    entity_name: str = "robot"
    anchor_link_name: str = "torso_link"
    links_name: list[str] = []
    threshold: float = 0.01
    threshold_adaptive: bool = False
    down_threshold: float = 0.0
    root_height_threshold: float = 0.5

    def build(self) -> None:
        assert_motion_field_present(self._env, self.entity_name, "links_pos_w")
        self._motion_idx = get_motion_link_idx(self._env, self.entity_name, self.links_name)
        self._robot_idx = get_robot_link_idx(self._env, self.entity_name, self.links_name)
        if self.threshold_adaptive:
            # Resolve once at build time so compute() doesn't pay the
            # name-lookup cost per step.
            self._motion_anchor_idx = get_motion_link_idx(self._env, self.entity_name, [self.anchor_link_name])
        else:
            self._motion_anchor_idx = None

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        ref = self._env.motion_manager.data[self.entity_name]
        motion_links_pos_w = current_motion_frame(ref.links_pos_w, self._motion_idx)
        robot_links_pos_w = self.resolve_entity(self.entity_name).get_links_pos(ls_idx_local=self._robot_idx)
        error = torch.abs(motion_links_pos_w[:, :, -1] - robot_links_pos_w[:, :, -1])
        if self._motion_anchor_idx is None:
            return torch.any(error > self.threshold, dim=-1)
        motion_anchor_z = current_motion_frame(ref.links_pos_w, self._motion_anchor_idx)[:, 0, -1]
        eff_threshold = _adaptive_pos_threshold(
            motion_anchor_z,
            self.threshold,
            self.threshold_adaptive,
            self.down_threshold,
            self.root_height_threshold,
        )
        # eff_threshold is (num_envs,); broadcast against the (num_envs, K) error.
        return torch.any(error > eff_threshold[:, None], dim=-1)


@TERMINATION_TERM_REGISTRY.register()
def motion_ends(env: MimicRLEnvBase, *, entity_name: str) -> torch.Tensor:
    """Terminate when the motion buffer's current lookahead frame is invalid.

    Reads :attr:`MotionReferenceData.validity` at index ``[:, 0]`` for
    ``entity_name``. **Note**: with the default
    ``MotionManagerOptions.data_start_from="one_frame_interval"``, frame ``0``
    is the *next* frame to be played, not the current one — so this term
    fires one step *before* the policy actually sees the missing frame. With
    ``data_start_from="current_time"`` it fires on the current frame.
    """
    ref = env.motion_manager.data[entity_name]
    return ~current_motion_frame(ref.validity)


@TERMINATION_TERM_REGISTRY.register()
class MotionBasePoseTrackingBreak(TerminationTerm):
    """Base-pose deviation gate vs the current motion frame.

    Terminates when the entity's world-frame base position error exceeds
    ``pos_threshold`` (m) OR its orientation error exceeds ``rot_threshold``
    (radians). ConTrack's object pose-break termination.

    ``pos_threshold`` / ``rot_threshold`` are where the gate starts, and assigning
    either after build raises. The gate reads its thresholds from the device
    (:attr:`thresholds`), so a curriculum can tighten them online with
    :meth:`set_thresholds`, which writes them in place without a host round trip: pair with
    :class:`~eden.managers.terms.curricula.motion.MotionBasePoseThresholdCurriculum`
    to get ConTrack's success-driven threshold schedule (which is where that
    state belongs — it is then logged under ``Curriculum/*`` and frozen at
    eval). On its own this term is a static gate.
    """

    entity_name: str = "object"
    pos_threshold: float = 0.1
    rot_threshold: float = 1.0

    def __setattr__(self, name: str, value) -> None:
        # Once built, the gate compares against its device thresholds: assigning a configured float would move
        # nothing, so it raises at the assignment rather than leaving the gate where it was. A number is quoted;
        # anything else by its type, as a CUDA tensor's repr would read it back to the host.
        if name in ("pos_threshold", "rot_threshold") and "_pos_threshold" in self.__dict__:
            assigned = repr(value) if isinstance(value, (int, float)) else f"a {type(value).__name__}"
            raise RuntimeError(
                f"{term_subject(self)}: {name} was assigned ({assigned}) after build, but the gate compares against "
                "its device thresholds, set at build from the configured values. Move the gate with "
                "set_thresholds(pos_threshold, rot_threshold) instead of assigning the configured fields."
            )
        super().__setattr__(name, value)

    def build(self) -> None:
        assert_motion_field_present(self._env, self.entity_name, "base_pos_w")
        assert_motion_field_present(self._env, self.entity_name, "base_quat_w")
        # Allocated once and written in place (set_thresholds), so the gate reads one address all run: a CUDA
        # graph of the graded phase (EnvOptions.cuda_graph_options) replays reading them where they live.
        self._pos_threshold = torch.tensor(float(self.pos_threshold), device=self.device)
        self._rot_threshold = torch.tensor(float(self.rot_threshold), device=self.device)

    @property
    def thresholds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The gate's current ``(pos_threshold, rot_threshold)``, as the 0-dim device tensors it compares against."""
        return self._pos_threshold, self._rot_threshold

    def set_thresholds(self, pos_threshold: float | torch.Tensor, rot_threshold: float | torch.Tensor) -> None:
        """Move the gate to ``(pos_threshold, rot_threshold)``, in place (a 0-dim device tensor never syncs)."""
        self._pos_threshold.fill_(pos_threshold)
        self._rot_threshold.fill_(rot_threshold)

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        entity = self.resolve_entity(self.entity_name)
        ref = self._env.motion_manager.data[self.entity_name]
        pos_err = torch.norm(entity.get_pos() - current_motion_frame(ref.base_pos_w), dim=-1)
        rot_err = quat_error_magnitude(entity.get_quat(), current_motion_frame(ref.base_quat_w))
        return (pos_err > self._pos_threshold) | (rot_err > self._rot_threshold)
