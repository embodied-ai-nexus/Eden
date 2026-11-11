"""Command primitives for the ``eden`` CLI: the :class:`Command` shape and the
argparse build/dispatch helpers.

This module is pure mechanism — it knows nothing about which commands exist.
``eden.__main__`` declares the ``COMMANDS`` table (importing the handlers from
the area modules) and drives ``_add_command`` / ``_dispatch`` over it. Keeping
the table out of here means importing these primitives never pulls the command
bodies.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    """One CLI subcommand (or subcommand group). Three mutually exclusive shapes:

    * **forwarding** (``forwards_extra=True``) — declares no flags; the raw
      ``extra_args`` are handed to ``handler(extra_args)`` for the command to
      parse itself (the task commands build their flags from the task's config
      schema). ``--help`` is left to the handler, so argparse auto-help is off.
    * **self-parsed** (``configure`` set, or neither) — declares flags via
      ``configure(parser)`` (omitted for no-arg commands like ``list``); the
      parsed namespace goes to ``handler(args)``.
    * **group** (``subcommands`` set) — a parent like ``gello`` that dispatches
      to one of its children. Children may be any shape (``_dispatch`` recurses).

    ``__post_init__`` enforces the shape contract so a misconfigured entry fails
    at import, not silently at dispatch.
    """

    name: str
    help: str
    handler: Callable | None = None
    configure: Callable[[argparse.ArgumentParser], None] | None = None
    forwards_extra: bool = False
    subcommands: tuple[Command, ...] = ()

    def __post_init__(self) -> None:
        if self.subcommands:
            if self.handler is not None or self.configure is not None or self.forwards_extra:
                raise ValueError(
                    f"Command {self.name!r}: a group (subcommands set) must not also set "
                    "handler/configure/forwards_extra — it only dispatches to children."
                )
        else:
            if self.handler is None:
                raise ValueError(f"Command {self.name!r}: a non-group command needs a handler.")
            if self.forwards_extra and self.configure is not None:
                raise ValueError(
                    f"Command {self.name!r}: a forwarding command parses its own args; "
                    "drop configure (or drop forwards_extra to declare flags here)."
                )

    @property
    def consumes_extra(self) -> bool:
        """Whether leftover args are forwarded (True) or rejected as a typo (False)."""
        return self.forwards_extra or bool(self.subcommands)


def _subcommand_dest(cmd: Command) -> str:
    """Return the argparse ``dest`` holding the chosen subcommand name for ``cmd``.

    Single source for the convention shared by ``_add_command`` (which sets it)
    and ``_dispatch`` (which reads it back).
    """
    return f"{cmd.name}_command"


def _add_command(subparsers, cmd: Command, parsers: dict[Command, argparse.ArgumentParser]) -> None:
    """Register ``cmd`` (and any subcommands) on ``subparsers``.

    Records each command's built parser in ``parsers`` so ``_dispatch`` can show
    a group's help when it's invoked with no subcommand.
    """
    # Forwarding commands own their own --help; everyone else gets argparse's.
    sub = subparsers.add_parser(cmd.name, help=cmd.help, add_help=not cmd.forwards_extra)
    parsers[cmd] = sub
    if cmd.configure is not None:
        cmd.configure(sub)
    if cmd.subcommands:
        child_subparsers = sub.add_subparsers(dest=_subcommand_dest(cmd), help=f"{cmd.name} subcommands")
        for child in cmd.subcommands:
            _add_command(child_subparsers, child, parsers)


def _dispatch(cmd: Command, args, extra_args: list[str], parsers: dict[Command, argparse.ArgumentParser]) -> None:
    """Run the command selected by the parsed ``args``.

    Recurses for groups so each child runs by its own shape, rather than
    assuming children are forwarding.
    """
    if cmd.subcommands:
        selected = getattr(args, _subcommand_dest(cmd), None)
        child = next((c for c in cmd.subcommands if c.name == selected), None)
        if child is None:  # `eden gello` with no subcommand → show the group's help
            parsers[cmd].print_help()
            sys.exit(0)
        _dispatch(child, args, extra_args, parsers)
    elif cmd.forwards_extra:
        cmd.handler(extra_args)
    else:
        cmd.handler(args)
