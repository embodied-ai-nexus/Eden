"""FlashDexRetarget's per-episode metrics: completion and mean tracking errors."""

from __future__ import annotations

import torch

from eden.constants import MetricDirection, MetricMode
from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.managers.terms.terminations.common import time_out
from eden.managers.terms.terminations.tracking import motion_ends
from eden.tasks.flashdexretarget.custom_terms.common import HandTermMixin
from eden.utils.geom import quat_error_magnitude


@METRIC_TERM_REGISTRY.register()
class FDREpisodeCompleted(MetricTerm):
    """1 on the step an episode times out (its clip runs out, ``motion_clip_ended``, or the episode time limit,
    ``time_out``), else 0; judged at the reset, so an episode succeeds when no failure term cut it short, the
    completion :class:`FDRAssistCurriculum` gates on, and ``Metrics/Success`` is the completion rate.

    Metrics compute before the terminations, on the same motion frame and episode length they read, so the value on
    an episode's last step is whether a time-out term ended it. On a step where a failure term fires too, the episode
    counts as completed.

    Parameters
    ----------
    entity_name : str
        The entity whose motion reference the clip's validity is read from.
    """

    entity_name: str = "robot"
    success_threshold: float = 1.0
    metric_mode: MetricMode = MetricMode.RESET

    def compute(self) -> torch.Tensor:
        return (motion_ends(self._env, entity_name=self.entity_name) | time_out(self._env)).float()


class _TrackingErrorMetric(HandTermMixin, MetricTerm):
    """An error against the reference, averaged over the hands (and their fingertips) and over the episode's steps.
    Reported only: success is :class:`FDREpisodeCompleted`'s.
    """

    is_cumulative: bool = True
    direction: MetricDirection = MetricDirection.LIB
    metric_mode: MetricMode = MetricMode.RESET
    contributes_to_success: bool = False

    def build(self) -> None:
        self._build_state()

    def compute(self) -> torch.Tensor:
        return self._error().flatten(1).mean(dim=1)


@METRIC_TERM_REGISTRY.register()
class FDRObjectPosError(_TrackingErrorMetric):
    """Object position error to its reference (m), mesh origin."""

    def _error(self) -> torch.Tensor:
        return (self.state.object_pose()[0] - self.state.ref_object(0)["pos"]).norm(dim=-1)


@METRIC_TERM_REGISTRY.register()
class FDRObjectRotError(_TrackingErrorMetric):
    """Object orientation error to its reference (rad)."""

    def _error(self) -> torch.Tensor:
        return quat_error_magnitude(self.state.object_pose()[1], self.state.ref_object(0)["quat"])


@METRIC_TERM_REGISTRY.register()
class FDRTipPosError(_TrackingErrorMetric):
    """Robot fingertip error to the MANO fingertips (m)."""

    def _error(self) -> torch.Tensor:
        return (self.state.mano_tips(0) - self.state.tips_pos()).norm(dim=-1)
