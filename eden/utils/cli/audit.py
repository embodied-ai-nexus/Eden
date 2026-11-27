"""Asset audit command: ``eden audit [<entity-or-file>] [--task <task> [--entity <name>]]``.

The checks live in :mod:`eden.utils.asset_audit`. This module parses the arguments through the shared task parser
(so ``--task``, ``--config``, ``--override``, the task's variant flags and ``--cpu`` behave as in ``eden smoke``),
picks the entity to audit, runs the checks, prints the report and sets the exit status (1 when a check fails).

The entity comes from one of two places:

- a target: an entity name or registry key, or an asset file. With ``--task`` it is audited at that task's
  physics, without it at ``EnvOptions()``'s.
- ``--task`` alone: the task's own scene entity named ``--entity`` (default ``robot``), at the task's physics, so
  both come from the config the task builds. An entity the task welds to another (a composite rig's gripper, an
  ``attachments_dict`` child) declares a free base for its weld; audited alone, it gets back the base its parent
  gives it in the task (see :func:`_as_held`).
"""

from __future__ import annotations

import os

#: The scene entity ``eden audit --task`` audits when ``--entity`` is not given.
DEFAULT_TASK_ENTITY = "robot"


class AuditArgumentError(ValueError):
    """An argument of ``eden audit`` names nothing it can audit; the command reports it as a usage error.

    Raised only where the arguments are judged (an unknown name, a missing or unsupported file, a composite rig, a
    base the asset cannot take, a scene entity the task lacks). Any other exception keeps its traceback.
    """


def _resolve_audit_target(target: str, fixed_base: bool | None):
    """Entity options for ``target``: an asset file when it exists or ends in a known suffix, else an entity name.

    ``fixed_base`` pins or frees the base; ``None`` keeps the declaration (an entity's class, or what
    :func:`~eden.utils.asset_audit.options_for_file` reads from a file).
    """
    from eden.options.catalog import base_follows_is_fixed_base, find_entity_or_rig_options
    from eden.options.entities import CompositeEntityOptions
    from eden.utils.asset_audit import ASSET_FILE_SUFFIXES, check_asset_file, check_auditable, options_for_file

    if os.path.isfile(target) or target.lower().endswith(ASSET_FILE_SUFFIXES):
        try:
            check_asset_file(target)
        except (FileNotFoundError, ValueError) as err:
            raise AuditArgumentError(err.args[0]) from err
        return options_for_file(target, fixed_base=fixed_base)

    try:
        options_cls = find_entity_or_rig_options(target)
    except ValueError as err:  # the catalog's "no entity options found", with its suggestions
        raise AuditArgumentError(err.args[0]) from err
    if issubclass(options_cls, CompositeEntityOptions):
        try:
            check_auditable(options_cls(), target)
        except TypeError as err:  # always, for a rig: names the parts to audit instead
            raise AuditArgumentError(err.args[0]) from err
    options = options_cls()
    if fixed_base is None or fixed_base == options.is_fixed_base:
        return options
    if not base_follows_is_fixed_base(options):
        flag = "--fixed-base" if fixed_base else "--free-base"
        raise AuditArgumentError(
            f"eden audit: {flag} cannot change {target!r}: its MJCF ({options.file}) declares the base itself "
            f"(is_fixed_base={options.is_fixed_base}). Drop {flag}, or audit a copy of the file whose root body "
            "has the joint you want."
        )
    # Through the constructor, where a primitive reconciles ``fixed`` with ``is_fixed_base``.
    return options_cls(is_fixed_base=fixed_base)


def _welds(scene_options) -> dict[str, str]:
    """Scene entity name -> the entity it is welded to: the scene's ``attachments_dict`` and every rig's own welds."""
    from eden.options.entities import CompositeEntityOptions

    welds = {child: spec[0] for child, spec in scene_options.attachments_dict.items()}
    for value in (scene_options.model_extra or {}).values():
        if isinstance(value, CompositeEntityOptions):
            welds.update({child: spec[0] for child, spec in value.attachments().items()})
    return welds


def _as_held(options):
    """A welded entity's options with the base its parent gives it in the task, and whether that base is fixed.

    A part welded to an arm declares ``is_fixed_base=False`` so the weld can carry it, but in the task its base never
    moves on its own. Built alone that declaration would drop it, or contradict an MJCF whose root has no free joint,
    so it is pinned wherever the flag reaches the asset; a plain MJCF keeps the base its file declares.
    """
    from eden.options.catalog import base_follows_is_fixed_base
    from eden.options.entities import PrimitiveOptions
    from eden.utils.asset_audit import mjcf_has_free_root

    if base_follows_is_fixed_base(options):
        fixed = True
    else:
        fixed = not mjcf_has_free_root(options.fetch_asset(options.resolve_runtime_file()))
    update = {"is_fixed_base": fixed}
    if isinstance(options, PrimitiveOptions):
        update["fixed"] = fixed  # a primitive's own switch, kept in step with is_fixed_base
    return options.model_copy(deep=True, update=update), fixed


def _task_entity_options(config, task: str, entity: str):
    """The options of scene entity ``entity`` in the built task ``config``, and the label to report it under.

    Composite rigs are expanded into their parts. A welded entity comes back as :func:`_as_held` makes it, and its
    label says so.
    """
    entities = config.scene_options.named_entities()
    if entity not in entities:
        raise AuditArgumentError(
            f"eden audit: task {task!r} has no scene entity {entity!r}; its entities are {sorted(entities)}. "
            f"Pass one with --entity, or name an entity or asset file as the target."
        )
    parent = _welds(config.scene_options).get(entity)
    if parent is None:
        return entities[entity], f"{task}:{entity}"
    options, fixed = _as_held(entities[entity])
    base = "fixed" if fixed else "free"
    return options, f"{task}:{entity} (welded to {parent}; audited alone with a {base} base)"


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

    # The shared task parser adds these for the commands that train or replay; the audit has no use for them.
    unused = [flag for flag, value in (("--checkpoint", args.checkpoint), ("--run_name", args.run_name)) if value]
    if unused:
        verb = "does" if len(unused) == 1 else "do"
        parser.error(f"{' and '.join(unused)} {verb} nothing in eden audit, which loads no policy and writes no run.")
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
    try:
        if args.target is not None:
            fixed_base = True if args.fixed_base else False if args.free_base else None
            options, label = _resolve_audit_target(args.target, fixed_base), args.target
        else:
            options, label = _task_entity_options(config, args.task, args.entity or DEFAULT_TASK_ENTITY)
        try:
            check_auditable(options, label)  # a terrain, a non-rigid material, a grouped entity
        except TypeError as err:
            raise AuditArgumentError(err.args[0]) from err
    except AuditArgumentError as err:
        parser.error(str(err))

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
