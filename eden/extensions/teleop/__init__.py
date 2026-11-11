"""Eden teleoperation extensions (operator-side stacks).

Exposes the teleop interfaces every operator stack implements
(:class:`~eden.extensions.teleop.base.TeleopInputSource`,
:class:`~eden.extensions.teleop.base.TeleopStreamer`,
:class:`~eden.extensions.teleop.base.TeleopOperator`). Concrete stacks (e.g.
SONIC under ``teleop/sonic``) import lazily so their heavy deps (genesis, zmq,
msgpack) aren't pulled in just to import the base interfaces.
"""

from eden.extensions.teleop.base import TeleopInputSource, TeleopOperator, TeleopStreamer

__all__ = ["SonicTeleopOperator", "TeleopInputSource", "TeleopOperator", "TeleopStreamer"]


def __getattr__(name: str):
    if name == "SonicTeleopOperator":
        from eden.extensions.teleop.sonic.manager import SonicTeleopOperator

        return SonicTeleopOperator
    raise AttributeError(name)
