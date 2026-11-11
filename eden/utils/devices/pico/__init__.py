"""PICO teleoperation (XRoboToolkit tracking client + headset video link).

Submodules are imported lazily on purpose: :mod:`protocol` is pure stdlib,
:mod:`video` needs PyAV, and :mod:`xr_client` needs the ``xrobotoolkit_sdk``
extension module, which is built from source (see :mod:`~eden.utils.devices.pico.xr_client`).
"""
