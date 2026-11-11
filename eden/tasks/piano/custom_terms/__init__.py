"""Task-local manager terms for the piano task.

``commands.py`` is the one to read first: the score, and the once-per-step measurement of the keyboard that every
other term grades against.
"""

from eden.tasks.piano.custom_terms.commands import KEY_DOWN_FRACTION, SongCommand
from eden.tasks.piano.custom_terms.metrics import key_press_f1, key_press_precision, key_press_recall
from eden.tasks.piano.custom_terms.observations import (
    base_offset_from_default,
    base_pose_from_default,
    piano_key_positions,
)
from eden.tasks.piano.custom_terms.rewards import (
    FingertipToKey,
    actuation_energy,
    gaussian_tolerance,
    key_press,
    rest_pose_deviation,
)
from eden.tasks.piano.custom_terms.terminations import song_over

__all__ = [
    "KEY_DOWN_FRACTION",
    "FingertipToKey",
    "SongCommand",
    "actuation_energy",
    "base_offset_from_default",
    "base_pose_from_default",
    "gaussian_tolerance",
    "key_press",
    "key_press_f1",
    "key_press_precision",
    "key_press_recall",
    "piano_key_positions",
    "rest_pose_deviation",
    "song_over",
]
