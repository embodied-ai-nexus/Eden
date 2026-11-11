"""SONIC whole-body teleoperation operator station (PICO body tracking -> ZMQ).

The :class:`~eden.extensions.teleop.base.TeleopOperator` implementation for the
GEAR-SONIC stack. ``SonicTeleopOperator`` is re-exported lazily so importing this
package doesn't pull the operator's heavy deps (genesis, zmq, msgpack).
"""

__all__ = ["SonicTeleopOperator"]


def __getattr__(name: str):
    if name == "SonicTeleopOperator":
        from eden.extensions.teleop.sonic.manager import SonicTeleopOperator

        return SonicTeleopOperator
    raise AttributeError(name)
