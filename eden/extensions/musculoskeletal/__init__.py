"""Musculoskeletal Hill-type muscle actuation and MJCF joint-torque bridge for Eden.

Provides exact NumPy/PyTorch ports of MuJoCo's ``mjDYN_MUSCLE`` (Millard activation ODE),
``mjGAIN_MUSCLE`` (active force-length-velocity bump curve), and ``mjBIAS_MUSCLE`` (passive
force-length curve), :func:`strip_muscles_for_genesis` to prepare spatial-tendon MJCF skeletons
(e.g. ``myo_sim`` / MyoSuite ``myoHand``, ``myoArm``, ``myoFullBody``) for ``genesis.morphs.MJCF``,
and :class:`~eden.extensions.musculoskeletal.controller.HillMuscleController` to drive them via
``entity.control_dofs_force``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from eden.extensions.musculoskeletal.hill_muscle import (
    MuscleParams,
    dense_actuator_moment,
    muscle_activation_dot,
    muscle_actuator_force,
    muscle_bias,
    muscle_gain,
    step_muscle_activation,
    strip_muscles_for_genesis,
)

if TYPE_CHECKING:
    from eden.extensions.musculoskeletal.controller import HillMuscleController

__all__ = [
    "HillMuscleController",
    "MuscleParams",
    "dense_actuator_moment",
    "muscle_activation_dot",
    "muscle_actuator_force",
    "muscle_bias",
    "muscle_gain",
    "step_muscle_activation",
    "strip_muscles_for_genesis",
]


def __getattr__(name: str) -> Any:
    if name == "HillMuscleController":
        from eden.extensions.musculoskeletal.controller import HillMuscleController

        return HillMuscleController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
