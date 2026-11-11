"""Device-bridge commands: gello, metaquest, pico, wuji.

Handlers lazily import their (optional-dependency) backends so a plain
``eden --help`` never pays for them.
"""

from __future__ import annotations


def _run_gello_calibrate(extra_args: list[str]) -> None:
    from eden.utils.devices.gello.calibration import calibrate_gello_leader

    calibrate_gello_leader(extra_args)


def _run_gello_leader(extra_args: list[str]) -> None:
    from eden.utils.devices.gello.leader import run_gello_leader

    run_gello_leader(extra_args)


def _run_metaquest_bridge(extra_args: list[str]) -> None:
    from eden.utils.devices.metaquest.xr_bridge import main as run_xr_bridge

    run_xr_bridge(extra_args)


def _run_pico_bridge(extra_args: list[str]) -> None:
    from eden.utils.devices.pico.bridge import main as run_pico_bridge

    run_pico_bridge(extra_args)


def _run_wuji_scan(extra_args: list[str]) -> None:
    from eden.utils.devices.wuji.monitor import main as run_wuji

    raise SystemExit(run_wuji(["scan", *extra_args]))


def _run_wuji_monitor(extra_args: list[str]) -> None:
    from eden.utils.devices.wuji.monitor import main as run_wuji

    raise SystemExit(run_wuji(["monitor", *extra_args]))
