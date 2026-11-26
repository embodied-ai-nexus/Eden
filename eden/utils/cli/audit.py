"""Asset audit command: ``eden audit <entity-or-file>``.

The checks live in :mod:`eden.utils.asset_audit`; this module resolves the target, runs them on the CPU, prints
the report and sets the exit status (1 when a check fails).
"""

from __future__ import annotations

import argparse
import json


def _resolve_audit_target(target: str, fixed_base: bool | None):
    """Entity options for ``target``: an asset file when it ends in a known suffix, else an entity name.

    ``fixed_base`` pins or frees the base; ``None`` keeps the declaration (an entity's class, or what
    :func:`~eden.utils.asset_audit.options_for_file` reads from a file).
    """
    from eden.options.entities import CompositeEntityOptions
    from eden.utils.asset_audit import ASSET_FILE_SUFFIXES, check_auditable, options_for_file
    from eden.utils.cli.viewers import _declaration_reaches_base, _find_entity_options

    if target.lower().endswith(ASSET_FILE_SUFFIXES):
        return options_for_file(target, fixed_base=fixed_base)

    options_cls = _find_entity_options(target)
    if issubclass(options_cls, CompositeEntityOptions):
        check_auditable(options_cls(), target)  # raises, naming the parts to audit instead
    options = options_cls()
    if fixed_base is None or fixed_base == options.is_fixed_base:
        return options
    if not _declaration_reaches_base(options):
        flag = "--fixed-base" if fixed_base else "--free-base"
        raise ValueError(
            f"eden audit: {flag} cannot change {target!r}: its MJCF ({options.file}) declares the base itself "
            f"(is_fixed_base={options.is_fixed_base}). Drop {flag}, or audit a copy of the file whose root body "
            "has the joint you want."
        )
    # Through the constructor, where a primitive reconciles ``fixed`` with ``is_fixed_base``.
    return options_cls(is_fixed_base=fixed_base)


def _run_audit(args) -> None:
    import eden as en
    from eden.utils.asset_audit import AuditThresholds, audit_entity

    fixed_base = True if args.fixed_base else False if args.free_base else None
    options = _resolve_audit_target(args.target, fixed_base)
    en.init()
    report = audit_entity(options, target=args.target, thresholds=AuditThresholds(seconds=args.seconds))
    print(report.format())
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Report written to {args.json}")
    if report.failed:
        raise SystemExit(1)


def _configure_audit(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "target",
        type=str,
        help="Entity name or registry key (e.g. ur5e, YCB003CrackerBox), or a path to a URDF / MJCF / USD / mesh file",
    )
    base = parser.add_mutually_exclusive_group()
    base.add_argument("--fixed-base", action="store_true", help="Pin the base (a robot arm from a URDF needs this)")
    base.add_argument("--free-base", action="store_true", help="Free the base, so the settle check drops it")
    parser.add_argument(
        "--seconds", type=float, default=2.0, help="Simulated time for the settle and hold checks (default: 2.0)"
    )
    parser.add_argument("--json", type=str, default=None, help="Also write the report as JSON to this path")
