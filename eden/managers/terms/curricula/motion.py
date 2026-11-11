"""Motion-tracking curricula (success-driven tolerance tightening)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers.curriculum_manager import (
    CURRICULUM_TERM_REGISTRY,
    CurriculumTerm,
)
from eden.managers.terms.terminations.tracking import MotionBasePoseTrackingBreak
from eden.utils.misc import sum_over_envs
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.envs.base import MimicRLEnvBase


@CURRICULUM_TERM_REGISTRY.register()
class MotionBasePoseThresholdCurriculum(CurriculumTerm):
    """Tighten a :class:`MotionBasePoseTrackingBreak` gate as success rises.

    ConTrack's "threshold curriculum": an EMA of the per-episode success rate linearly
    interpolates the paired termination term's ``pos_threshold`` / ``rot_threshold`` from
    their ``*_max`` (lenient) values to their ``*_min`` (tight) ones as the EMA moves
    across ``[success_rate_low, success_rate_high]``. An episode counts as a success when
    it ends with the motion clip exhausted (``validity[:, 0]`` false at reset time) rather
    than via the pose gate.

    This lives in a curriculum term rather than inside the termination term so it gets the
    three things curriculum state needs and a termination term cannot provide:
    ``CurriculumManager`` logs :meth:`compute`'s return under ``Curriculum/<name>/*``,
    it freezes curricula at eval (``eval_mode=True``) so an eval rollout is
    graded at a fixed tolerance, and the state has one obvious home if/when term
    checkpointing lands (nothing calls ``state_dict()`` on manager terms today, so a
    resumed run still restarts the EMA at 0 — i.e. at the lenient thresholds).

    Declaration order matters only against other curriculum terms; the thresholds this
    writes are consumed by the *next* step's ``termination_manager.compute()``.

    Parameters
    ----------
    termination_term_name: str
        Name of the ``MotionBasePoseTrackingBreak`` term to drive, as declared in
        ``termination_options``.
    entity_name: str
        Entity whose motion-reference ``validity`` defines episode success. Must match
        the termination term's own ``entity_name``.
    pos_threshold_max / pos_threshold_min: float
        Position tolerance (m) at ``success_rate_low`` / ``success_rate_high``.
    rot_threshold_max / rot_threshold_min: float
        Orientation tolerance (rad) at ``success_rate_low`` / ``success_rate_high``.
    success_ema_alpha: float
        Per-**episode** EMA rate. The update is applied once per reset batch with an
        effective rate of ``1 - (1 - alpha) ** n`` for a batch of ``n`` episodes, which is
        exactly ``n`` successive per-episode updates. Without that correction the EMA's
        time constant would scale with ``num_envs`` (at 4096 envs some env resets almost
        every step, so a per-batch alpha converges within a fraction of one episode).
    success_rate_low / success_rate_high: float
        EMA values at which the thresholds sit at ``*_max`` and ``*_min`` respectively.
    """

    termination_term_name: str = "object_pose_break"
    entity_name: str = "object"
    pos_threshold_max: float = 0.1
    pos_threshold_min: float = 0.05
    rot_threshold_max: float = 1.0
    rot_threshold_min: float = 0.5
    success_ema_alpha: float = 0.05
    success_rate_low: float = 0.9
    success_rate_high: float = 0.999

    def build(self) -> None:
        # Fail fast on the degenerate spans: `high == low` would make the interpolation
        # weight 0/0 -> nan, which propagates through clamp into both thresholds and turns
        # `err > threshold` permanently False — the gate would silently never fire.
        if not self.success_rate_high > self.success_rate_low:
            raise ValueError(
                f"{term_subject(self)} requires success_rate_high > success_rate_low; got "
                f"success_rate_low={self.success_rate_low}, success_rate_high={self.success_rate_high}."
            )
        if not 0.0 < self.success_ema_alpha <= 1.0:
            raise ValueError(f"{term_subject(self)} requires 0 < success_ema_alpha <= 1; got {self.success_ema_alpha}.")
        for name in ("pos", "rot"):
            lo, hi = getattr(self, f"{name}_threshold_min"), getattr(self, f"{name}_threshold_max")
            if lo > hi:
                raise ValueError(
                    f"{term_subject(self)}: {name}_threshold_min ({lo}) must not exceed "
                    f"{name}_threshold_max ({hi}) — the curriculum tightens from max to min."
                )
        # The reset grades the episode that ended, whose window is refilled before the manager resets
        # run; a wrong entity_name raises here, not on the first reset.
        self._env.motion_manager.record_ended_episode_validity(self.entity_name)
        self._success_ema = torch.zeros((), device=self.device)
        # The termination manager is built before the curricula (``EnvBase._manager_order``), so
        # the term this curriculum drives resolves here — a wrong name or a mismatched entity
        # raises at env construction rather than on the first compute().
        self._term = self._env.termination_manager.get_term(self.termination_term_name)
        if not isinstance(self._term, MotionBasePoseTrackingBreak):
            raise TypeError(
                f"{term_subject(self)} drives termination term '{self.termination_term_name}', a "
                f"{type(self._term).__name__}, but it can only move a MotionBasePoseTrackingBreak gate (through its "
                "set_thresholds). Point termination_term_name at one."
            )
        if self._term.entity_name != self.entity_name:
            raise ValueError(
                f"{term_subject(self)} drives termination term '{self.termination_term_name}' on entity "
                f"'{self._term.entity_name}' but grades success on '{self.entity_name}'. Point both at "
                f"the same entity, or the tolerance would track a different clip's progress."
            )

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        env: MimicRLEnvBase = self._env
        # Both reads below describe the episode that just ended: the validity the motion manager took
        # as it ended (the window itself already holds the new episode's), and `episode_length_buf`,
        # which `_reset_idx` zeroes after the manager resets. An exhausted clip (the episode ran to
        # the end instead of breaking) reads as validity False.
        success = (~env.motion_manager.ended_episode_validity(self.entity_name)).to(torch.float32)
        # Never-stepped envs (the build-time bootstrap reset) still hold the all-invalid
        # make_empty default, which would read as instant success and bias the EMA.
        stepped = (env.episode_length_buf > 0).to(torch.float32)
        # Kept entirely device-side: no `.item()`/`bool()`/mask-index — the auto-reset hands a bool
        # mask, which as an index is a sync per gather, so the sums reduce through it instead (they
        # count whole numbers, so they match the gathered ones exactly). n == 0 makes `weight` 0,
        # i.e. a no-op update.
        envs_idx = slice(None) if envs_idx is None else envs_idx
        n = sum_over_envs(stepped, envs_idx)
        mean_success = sum_over_envs(success * stepped, envs_idx) / n.clamp(min=1.0)
        weight = 1.0 - (1.0 - self.success_ema_alpha) ** n
        self._success_ema += weight * (mean_success - self._success_ema)

    def compute(self, *args, **kwargs) -> dict[str, float]:
        w = torch.clamp(
            (self._success_ema - self.success_rate_low) / (self.success_rate_high - self.success_rate_low),
            0.0,
            1.0,
        )
        pos_threshold = self.pos_threshold_max * (1.0 - w) + self.pos_threshold_min * w
        rot_threshold = self.rot_threshold_max * (1.0 - w) + self.rot_threshold_min * w
        self._term.set_thresholds(pos_threshold, rot_threshold)
        return {
            "success_ema": self._success_ema,
            "pos_threshold": pos_threshold,
            "rot_threshold": rot_threshold,
        }

    def state_dict(self) -> dict:
        return {"success_ema": self._success_ema.detach().cpu().clone()}

    def load_state_dict(self, state: dict) -> None:
        if (ema := state.get("success_ema")) is not None:
            self._success_ema = torch.as_tensor(float(ema), device=self.device, dtype=torch.float)
