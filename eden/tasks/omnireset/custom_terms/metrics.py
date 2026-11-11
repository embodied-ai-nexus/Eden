"""How far the insertive object is from assembled: the shared quantity UWLab's ``ProgressContext`` computes."""

from __future__ import annotations

from typing import Literal

import torch

from eden.constants import MetricDirection, MetricMode
from eden.tasks.omnireset.constants import PAIRS
from eden.tasks.omnireset.custom_terms.terminations import assembly_errors
from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm


@METRIC_TERM_REGISTRY.register()
class AssemblyError(MetricTerm):
    """One component of the insertive object's error from its assembled pose in the receptive object.

    ``component="position"`` is the distance (m), ``"tilt"`` is ``|roll| + |pitch|`` (rad); yaw is free. Graded at
    episode end against ``success_threshold`` (lower is better), as UWLab grades a reset; with both components
    contributing, ``Metrics/Success`` is UWLab's success rate. The rewards read the values through the metric manager.
    """

    pair: str = "peg"
    component: Literal["position", "tilt"] = "position"
    insertive_name: str = "insertive_object"
    receptive_name: str = "receptive_object"
    direction: MetricDirection = MetricDirection.LIB
    metric_mode: MetricMode = MetricMode.RESET

    def build(self) -> None:
        if self.pair not in PAIRS:
            raise ValueError(f"AssemblyError: unknown pair {self.pair!r}; pairs: {list(PAIRS)}.")
        if self.component not in ("position", "tilt"):
            raise ValueError(f"AssemblyError: component must be 'position' or 'tilt', got {self.component!r}.")
        self._pair = PAIRS[self.pair]
        for name in (self.insertive_name, self.receptive_name):
            self.resolve_entity(name)

    def compute(self) -> torch.Tensor:
        position, tilt = assembly_errors(self._env, self._pair, self.insertive_name, self.receptive_name)
        return position if self.component == "position" else tilt
