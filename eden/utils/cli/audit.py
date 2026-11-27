"""Asset audit command: ``eden audit [<entity-or-file>] [--task <task> [--entity <name>]]``.

The checks live in :mod:`eden.utils.asset_audit`. This module parses the arguments through the shared task parser
(so ``--task``, ``--config``, ``--override``, the task's variant flags and ``--cpu`` behave as in ``eden smoke``),
picks the entity to audit, runs the checks, prints the report and sets the exit status (1 when a check fails).

The entity comes from one of two places:

- a target: an entity name or registry key, or an asset file. With ``--task`` it is audited at that task's
  physics, without it at ``EnvOptions()``'s.
- ``--task`` alone: the task's own scene entity named ``--entity`` (default ``robot``), at the task's physics, so
  both come from the config the task builds.
"""

from __future__ import annotations

import os

#: The scene entity ``eden audit --task`` audits when ``--entity`` is not given.
DEFAULT_TASK_ENTITY = "robot"


def _resolve_audit_target(target: str, fixed_base: bool | None):
    """Entity options for ``target``: an asset file when it exists or ends in a known suffix, else an entity name.

    ``fixed_base`` pins or frees the base; ``None`` keeps the declaration (an entity's class, or what
    :func:`~eden.utils.asset_audit.options_for_file` reads from a file).
    """
    from eden.options.catalog import base_follows_is_fixed_base, find_entity_or_rig_options
    from eden.options.entities import CompositeEntityOptions
    from eden.utils.asset_audit import ASSET_FILE_SUFFIXES, check_auditable, options_for_file

    if os.path.isfile(target) or target.lower().endswith(ASSET_FILE_SUFFIXES):
        return options_for_file(target, fixed_base=fixed_base)  # an unsupported suffix raises with the list

    options_cls = find_entity_or_rig_options(target)
    if issubclass(options_cls, CompositeEntityOptions):
        check_auditable(options_cls(), target)  # raises, naming the parts to audit instead
    options = options_cls()
    if fixed_base is None or fixed_base == options.is_fixed_base:
        return options
    if not base_follows_is_fixed_base(options):
        flag = "--fixed-base" if fixed_base else "--free-base"
        raise ValueError(
            f"eden audit: {flag} cannot change {target!r}: its MJCF ({options.file}) declares the base itself "
            f"(is_fixed_base={options.is_fixed_base}). Drop {flag}, or audit a copy of the file whose root body "
            "has the joint you want."
        )
    # Through the constructor, where a primitive reconciles ``fixed`` with ``is_fixed_base``.
    return options_cls(is_fixed_base=fixed_base)


def _task_entity_options(config, task: str, entity: str):
    """The options of scene entity ``entity`` in the built task ``config`` (composite rigs expanded into parts)."""
    entities = config.scene_options.named_entities()
    if entity not in entities:
        raise KeyError(
            f"eden audit: task {task!r} has no scene entity {entity!r}; its entities are {sorted(entities)}. "
            f"Pass one with --entity, or name an entity or asset file as the target."
        )
    return entities[entity]


def audit_command(extra_args: list[str]) -> None:
    """Parse ``eden audit``'s arguments, audit the chosen entity, print the report; exit 1 when a check fails."""
    from eden.tasks.parser import get_task_argparser, get_task_config_from_args
    from eden.utils.asset_audit import AuditThresholds, audit_entity, check_auditable

    parser = get_task_argparser(
        description="Build one entity alone on a ground plane and check it for physics problems.",
        argv=extra_args,
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Entity name or registry key (e.g. ur5e, YCB003CrackerBox), or a path to a URDF / MJCF / USD / mesh "
        "file. Omit it with --task to audit the task's own scene entity.",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help=f"With --task and no target: the scene entity to audit (default: {DEFAULT_TASK_ENTITY}).",
    )
    base = parser.add_mutually_exclusive_group()
    base.add_argument("--fixed-base", action="store_true", help="Pin the target's base (a URDF arm needs this).")
    base.add_argument("--free-base", action="store_true", help="Free the target's base, so the settle check drops it.")
    parser.add_argument(
        "--seconds",
        type=float,
        default=2.0,
        help="Simulated time for the settle and hold checks (default: 2.0; at least about 0.85, so the drop and the "
        "landing finish before the graded rest window).",
    )
    parser.add_argument("--json", default=None, help="Also write the report as JSON to this path.")
    args = parser.parse_args(extra_args)

    # Every argument is checked here, before Eden initializes or anything is built.
    from eden.tasks import TASK_REGISTRY

    if args.target is None and not args.task:
        parser.error("name an entity or asset file to audit, or pass --task to audit the task's own entity.")
    if args.task and args.task not in TASK_REGISTRY:
        parser.error(TASK_REGISTRY.missing_task_message(args.task))
    if args.entity is not None and (args.target is not None or not args.task):
        parser.error("--entity picks a scene entity of --task; it cannot be combined with a target.")
    if args.target is None and (args.fixed_base or args.free_base):
        parser.error("--fixed-base / --free-base apply to a target; a task's entity is audited as the task builds it.")
    if args.json and not os.path.isdir(os.path.dirname(os.path.abspath(args.json))):
        parser.error(
            f"--json {args.json!r}: the directory {os.path.dirname(os.path.abspath(args.json))!r} does not exist."
        )
    try:
        thresholds = AuditThresholds(seconds=args.seconds)
    except ValueError as err:
        parser.error(str(err))

    config = get_task_config_from_args(args) if args.task else None
    env_options = config.env_options if config is not None else None
    # Resolving the entity only reads names, files and the config; what it raises is an argument error.
    try:
        if args.target is not None:
            fixed_base = True if args.fixed_base else False if args.free_base else None
            options, label = _resolve_audit_target(args.target, fixed_base), args.target
        else:
            entity = args.entity or DEFAULT_TASK_ENTITY
            options, label = _task_entity_options(config, args.task, entity), f"{args.task}:{entity}"
        check_auditable(options, label)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as err:
        parser.error(err.args[0] if err.args else str(err))

    # ``eden`` is already imported (through ``eden.tasks.parser`` above), so genesis loads after it as required.
    import genesis as gs

    import eden as en

    en.init(backend=gs.cpu if args.cpu else gs.gpu)
    report = audit_entity(options, env_options=env_options, target=label, thresholds=thresholds)
    print(report.render())
    if args.json:
        report.to_json(args.json)
        print(f"Report written to {args.json}")
    if report.failed:
        raise SystemExit(1)
