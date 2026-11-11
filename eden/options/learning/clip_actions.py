"""The action bound a runner's options and every env wrapper accept: ``clip_actions``."""

from __future__ import annotations

import math
from typing import Annotated

from pydantic import AfterValidator


def validate_clip_actions(clip_actions: float, field: str = "clip_actions") -> float:
    """Return ``clip_actions`` if it is a finite positive bound, raising otherwise.

    Actions are clamped to ``[-clip_actions, clip_actions]``: zero would clamp every action to zero, a
    negative bound inverts the interval, and an infinite or NaN one bounds nothing while promising a
    bound to a library that reads the action space. ``field`` is where the caller set it, for the message.
    """
    if not (0 < clip_actions < math.inf):
        raise ValueError(
            f"{field}={clip_actions!r} must be a finite positive bound, or None to leave actions unclamped."
        )
    return clip_actions


#: A validated action bound. Declare a ``clip_actions`` field as ``ClipActions | None`` so a zero, negative,
#: infinite or NaN bound fails at config load with :func:`validate_clip_actions`'s message.
ClipActions = Annotated[float, AfterValidator(validate_clip_actions)]
