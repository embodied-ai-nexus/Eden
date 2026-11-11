"""Configuration options for teleoperation operator stacks."""

from __future__ import annotations

from eden.options.options import DynamicConfigurableOptions


class TeleopOptions(DynamicConfigurableOptions):
    """Options for teleoperation operator stacks.

    Carries the fields common to any operator process; a concrete stack declares
    its device-/protocol-specific parameters as class attributes on its
    :class:`~eden.extensions.teleop.base.TeleopOperator` subclass, which are
    mirrored through as extras (this options class is ``extra="allow"``).

    Parameters
    ----------
    robot_host : str
        Host running the robot process this operator drives (its feedback channel).
    port : int
        Port the operator binds to publish its commands / pose / planner stream.
    feedback_port : int
        Port the robot process publishes its debug / feedback state on.
    """

    robot_host: str = "localhost"
    port: int = 5556
    feedback_port: int = 5557
