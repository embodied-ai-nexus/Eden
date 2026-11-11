"""Handlers for the ``eden teleop`` commands (lazy imports)."""

from __future__ import annotations


def _run_sonic_robot(extra_args: list[str]) -> None:
    from eden.extensions.policies.sonic.runtime import main as run_sonic_robot

    run_sonic_robot(extra_args)


def _run_sonic_operator(extra_args: list[str]) -> None:
    from eden.extensions.teleop.sonic.manager import main as run_sonic_operator

    run_sonic_operator(extra_args)
