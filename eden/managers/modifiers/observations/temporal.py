"""Stateful observation modifiers that act across control steps: an exponential moving average and a delay."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers.modifiers.base import OBSERVATION_MODIFIER_REGISTRY, ObservationModifier
from eden.utils.buffers.circular_buffer import CircularBuffer
from eden.utils.misc import is_env_mask

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.modifiers import ObservationModifierOptions


@OBSERVATION_MODIFIER_REGISTRY.register()
class ExponentialMovingAverage(ObservationModifier):
    """Exponential moving average, updated once per control step.

    ``y(t) = alpha * x(t) + (1 - alpha) * y(t - 1)``.

    ``alpha`` is the weight of the *newest* frame, as in pandas' ``ewm(alpha=...)``. Some filter
    libraries write the complement, so check that before porting a value. The term's frame must be
    floating-point — the weights are held in its dtype, so an integer one would truncate them to zero;
    :class:`ObservationDelay` is the modifier that carries any dtype.

    A non-finite frame keeps its kind: an infinity entering the average stays an infinity, so a
    ``clip`` on the term still bounds it — a raycast's ``+inf`` for "no hit" reaches the policy as the
    clip's far end, not as a NaN that no clip bounds — and a NaN stays a NaN. Either one stays in the
    average until the env resets. The group's ``nan_policy`` checks the term's output *after* its
    modifiers, so it replaces what the policy receives but cannot clean the average itself. The one
    frame that still averages to NaN is a component that sees both ``+inf`` and ``-inf``, their
    weighted sum being genuinely undefined.

    Parameters
    ----------
    alpha : float
        Weight of the newest frame, in ``(0, 1]``. Required. ``alpha=1`` passes each frame through
        unchanged; a smaller ``alpha`` smooths more.
    backfill : bool
        What the average starts from after a reset (default True). ``True``: the first frame after
        the reset starts the average, so it is delivered unchanged, as if every earlier frame had
        equalled it. ``False``: the average starts from zero, so the first frame delivers
        ``alpha * x``.
    """

    alpha: float | None = None
    backfill: bool = True

    def __init__(self, env: EnvBase, options: ObservationModifierOptions):
        super().__init__(env=env, options=options)
        if self.alpha is None:
            raise ValueError("ExponentialMovingAverage: alpha is required (the weight of the newest frame, in (0, 1]).")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"ExponentialMovingAverage: alpha must be in (0, 1], got {self.alpha}.")
        # The weight a row's first frame after a reset gets. With backfill it is 1, so that frame replaces
        # the zeroed average instead of being blended with it.
        self._first_frame_weight = 1.0 if self.backfill else self.alpha

    def build(self, frame: torch.Tensor) -> None:
        if not frame.is_floating_point():
            # The weights live in the frame's dtype, and ``torch.full(..., 0.2, dtype=torch.int64)`` is 0
            # without raising, so an integer frame would deliver zeros from its second frame on.
            raise ValueError(
                f"ExponentialMovingAverage: needs a floating-point frame, got {frame.dtype}. Average a float "
                f"term — a count, an index or a flag has no meaningful average — or drop the modifier. "
                f"ObservationDelay carries a frame of any dtype unchanged."
            )
        self._average = torch.zeros_like(frame)
        # One weight per env, broadcast over the feature dims: ``_first_frame_weight`` on a row whose
        # first frame since the reset is still to come, ``alpha`` once that frame has been advanced in.
        # ``_hold``, the weight of the stored average, is carried alongside rather than computed as
        # ``1 - _weight`` per read, so the pair is written together everywhere it changes.
        weight_shape = (frame.shape[0],) + (1,) * (frame.dim() - 1)
        self._weight = torch.full(weight_shape, self._first_frame_weight, dtype=frame.dtype, device=frame.device)
        self._hold = torch.full(weight_shape, 1.0 - self._first_frame_weight, dtype=frame.dtype, device=frame.device)

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if is_env_mask(envs_idx):
            # Auto-reset passes a mask every step; ``masked_fill_`` applies it on the device, as
            # ``CircularBuffer.reset`` does.
            rows = envs_idx.view(self._weight.shape)
            self._average.masked_fill_(rows, 0.0)
            self._weight.masked_fill_(rows, self._first_frame_weight)
            self._hold.masked_fill_(rows, 1.0 - self._first_frame_weight)
        else:
            rows = slice(None) if envs_idx is None else envs_idx
            self._average[rows] = 0.0
            self._weight[rows] = self._first_frame_weight
            self._hold[rows] = 1.0 - self._first_frame_weight

    def compute(self, frame: torch.Tensor) -> torch.Tensor:
        # ``(1 - w) * y + w * x``, written out rather than as ``torch.lerp(y, x, w)``: lerp evaluates
        # ``y + w * (x - y)`` (``x - (x - y) * (1 - w)`` for ``w >= 0.5``), so an infinite frame meets
        # ``inf - inf`` and comes out NaN — which ``clip`` passes through, since ``clamp`` does, and
        # which then reaches the policy. This form turns an infinity into an infinity, so the term's
        # clip still bounds it. Its one indeterminate product, ``(1 - w) * y``, needs ``y`` to be zero
        # wherever ``1 - w`` is; :meth:`advance` is what keeps that true.
        return torch.addcmul(self._hold * self._average, self._weight, frame)

    def advance(self, frame: torch.Tensor) -> None:
        if self.alpha < 1.0:
            self._average.copy_(self.compute(frame))
        # ``alpha == 1`` is pass-through: ``_hold`` is zero everywhere and the average is never read
        # again, so it is left zeroed rather than written. Storing an infinite frame in it would make
        # the next ``(1 - w) * y`` a ``0 * inf``, and so a NaN, for a modifier that changes nothing.
        self._weight.fill_(self.alpha)
        self._hold.fill_(1.0 - self.alpha)


@OBSERVATION_MODIFIER_REGISTRY.register()
class ObservationDelay(ObservationModifier):
    """Delivers the frame the term produced ``control_steps`` control steps earlier.

    The delay is counted in pushes of the observation window (see
    :class:`~eden.managers.modifiers.base.ObservationModifier`), which happen once per control step.
    That is the same clock as ``history_length``, so a delayed term with a history window holds the
    delayed frames.

    Parameters
    ----------
    control_steps : int
        How many control steps to delay by, ``>= 1``. Required.
    backfill : bool
        What is delivered while fewer than ``control_steps`` frames have been seen since the reset
        (default True). ``True``: the first frame after the reset. ``False``: zeros. This setting is
        separate from the term's own ``backfill``, which applies only to its history window.
    """

    control_steps: int | None = None
    backfill: bool = True

    def __init__(self, env: EnvBase, options: ObservationModifierOptions):
        super().__init__(env=env, options=options)
        if self.control_steps is None:
            raise ValueError("ObservationDelay: control_steps is required (how many control steps to delay by, >= 1).")
        if isinstance(self.control_steps, bool) or not isinstance(self.control_steps, int) or self.control_steps < 1:
            raise ValueError(
                f"ObservationDelay: control_steps must be an int >= 1, got {self.control_steps!r}. "
                f"A delay of 0 is no delay; drop the modifier instead."
            )

    def build(self, frame: torch.Tensor) -> None:
        # The window holds the current frame and the ``control_steps`` before it, so its oldest slot is the
        # delayed frame. ``CircularBuffer`` owns backfill, per-row reset and the read without an append.
        self._window = CircularBuffer(
            max_len=self.control_steps + 1, batch_size=frame.shape[0], device=frame.device, backfill=self.backfill
        )

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        self._window.reset(batch_ids=envs_idx)

    def compute(self, frame: torch.Tensor) -> torch.Tensor:
        return self._window.peek_oldest(frame)

    def advance(self, frame: torch.Tensor) -> None:
        self._window.append(frame)
