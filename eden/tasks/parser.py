"""CLI helpers for building task configs from argparse.

This module wires :class:`~eden.tasks.registry.TaskMod` parameters into a
standard ``argparse.ArgumentParser`` so that user-facing scripts can accept
flat flags (``--difficulty hard``, ``--robot g1``) that are translated into
typed mod kwargs and applied via :meth:`TaskRegistry.build`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

from eden.tasks import TASK_REGISTRY, TaskMod
from eden.tasks.registry import cli_fields, literal_choices, unwrap_optional
from eden.utils import distributed as eden_dist
from eden.utils.configs import is_full_config, overrides_from_str, read_config_file, saved_config_qualname
from eden.utils.format import ColorHelpFormatter
from eden.utils.misc import sanitize_run_name

# Flag names owned by :func:`get_task_argparser` itself. A mod field that
# resolves to one of these would otherwise collide with argparse's opaque
# "conflicting option string" error at parser-build time — turn it into a
# clear message pointing at the offending mod.
_RESERVED_CLI_FLAGS: frozenset[str] = frozenset({"task", "run_name", "config", "override", "checkpoint", "cpu"})


def _peek_task_name(argv: list[str], default: str) -> str:
    """Pre-parse ``--task`` so we know which task's mods to add to the full parser."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--task", default=default)
    known, _ = pre.parse_known_args(argv)
    return known.task


def _add_mod_field_to_parser(
    parser: argparse.ArgumentParser,
    mod: TaskMod,
    field_name: str,
    annotation: Any,
    help_text: str | None = None,
) -> None:
    """Translate a single CLI-tunable field on a TaskMod instance into an argparse arg.

    The flag name comes from ``mod.flag_for(field_name)`` (prefix-aware) and
    the default is read off the instance — so per-registration defaults and
    per-instance prefixes both flow through. ``help_text`` (from
    ``Annotated[T, "..."]`` metadata) populates argparse ``--help``.
    """
    default = getattr(mod, field_name)
    flag = mod.flag_for(field_name)

    if flag in _RESERVED_CLI_FLAGS:
        raise ValueError(
            f"TaskMod {type(mod).__name__}.{field_name} resolves to CLI flag "
            f"'--{flag}', which is reserved by get_task_argparser. Rename the "
            f"field or set a distinct prefix via .with_prefix(...). "
            f"Reserved flags: {sorted(_RESERVED_CLI_FLAGS)}."
        )

    add_kwargs: dict[str, Any] = {"default": default}
    if help_text is not None:
        add_kwargs["help"] = help_text

    # Unwrap Optional[T] / T | None to T so path-like fields with default=None
    # work — argparse handles None defaults natively, we just need a concrete
    # type= callable for parsing the supplied value. Shared with the programmatic
    # validator (``_check_cli_value``), which needs the same narrowing to keep
    # checking a widened field at all.
    annotation, _ = unwrap_optional(annotation)

    choices = literal_choices(annotation)
    if choices is not None:
        item_type = type(choices[0])
        parser.add_argument(f"--{flag}", type=item_type, choices=list(choices), **add_kwargs)
        return

    if annotation is bool:
        # BooleanOptionalAction supports both --flag and --no-flag with explicit default.
        parser.add_argument(f"--{flag}", action=argparse.BooleanOptionalAction, **add_kwargs)
        return

    if annotation in (int, float, str):
        parser.add_argument(f"--{flag}", type=annotation, **add_kwargs)
        return

    raise TypeError(
        f"TaskMod {type(mod).__name__} field '{field_name}' has unsupported annotation {annotation!r} for CLI; "
        "supported forms: str, int, float, bool, typing.Literal[...]."
    )


def get_task_argparser(
    description: str,
    default_task_name: str = "",
    argv: list[str] | None = None,
) -> argparse.ArgumentParser:
    """Build an ``ArgumentParser`` with standard flags plus the selected task's mod flags.

    The selected task is determined by pre-parsing ``--task`` from ``argv``
    (defaults to ``sys.argv[1:]``). Modifier flags are added only for the
    selected task, so ``--help`` shows only the relevant options. Pass
    ``argv`` explicitly when invoking this from a wrapper that already has
    its own arg slice (e.g. the ``eden train`` / ``eden inference``
    subcommands forward ``extra_args``) so the peek and the final parse
    agree on input.
    """
    task_name = _peek_task_name(sys.argv[1:] if argv is None else argv, default_task_name)

    parser = argparse.ArgumentParser(description=description, formatter_class=ColorHelpFormatter)
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Run name, put in the log directory name (<timestamp>-<run_name>-<id>) and so in the wandb run name.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=default_task_name,
        help="Task name. May expose additional mod flags depending on the task.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a config file (.json/.yaml/.yml). A saved config (one Eden wrote, carrying "
        "its class metadata) replaces the task's config; anything else layers onto it as a "
        "partial override.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=None,
        metavar="YAML",
        help="Inline config override as a YAML mapping, e.g. --override 'env_options: {num_envs: 64}'. "
        "Repeatable; later ones win. Applied on top of --config.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint file.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU backend.")

    if task_name and task_name in TASK_REGISTRY:
        for mod in TASK_REGISTRY.get_modifiers(task_name):
            for field_name, annotation, _default, help_text in cli_fields(type(mod)):
                _add_mod_field_to_parser(parser, mod, field_name, annotation, help_text)

    return parser


def _collect_modifier_kwargs(args: argparse.Namespace, task_name: str) -> dict[str, Any]:
    """Pull modifier kwargs for ``task_name`` out of an argparse Namespace.

    Returns keys matching each mod's prefixed flag names, ready to pass to
    :meth:`TaskRegistry.build` without translation.
    """
    if not task_name or task_name not in TASK_REGISTRY:
        return {}
    out: dict[str, Any] = {}
    for mod in TASK_REGISTRY.get_modifiers(task_name):
        for field_name, _annotation, _default, _help in cli_fields(type(mod)):
            flag = mod.flag_for(field_name)
            if hasattr(args, flag):
                out[flag] = getattr(args, flag)
    return out


def _apply_run_name(config: Any, run_name: str) -> None:
    """Write a filesystem-safe, length-capped run_name onto ``config.runner_options`` if present."""
    if run_name and getattr(config, "runner_options", None) is not None:
        config.runner_options.run_name = sanitize_run_name(run_name)


def explicit_task_mods(task_name: str, modifier_kwargs: dict[str, Any]) -> list[str]:
    """Return the ``--`` flags for task mods set away from their default.

    The comparison baseline is ``getattr(mod, field_name)`` — the mod *instance's* value,
    which is exactly what :func:`_add_mod_field_to_parser` uses as the argparse default (not
    the ``__init__`` signature default, which can differ when a task registers a mod with an
    overridden instance value). So a flag left unset — or passed at its default — reads as
    *not* explicitly set.

    Used to fail fast wherever mods collide with a config that is authoritative on its own
    (a saved ``--config``, a checkpoint's sibling ``config.json``) and would otherwise be
    silently ignored.

    Parameters
    ----------
    task_name: str
        The registered task whose mods to compare against.
    modifier_kwargs: dict[str, Any]
        Flag-keyed mod values, as produced by :func:`_collect_modifier_kwargs`.

    Returns
    -------
    list[str]
        The dash-prefixed flag names that differ from their default.
    """
    if not task_name or task_name not in TASK_REGISTRY:
        return []
    changed = []
    for mod in TASK_REGISTRY.get_modifiers(task_name):
        for field_name, _annotation, _default, _help in cli_fields(type(mod)):
            flag = mod.flag_for(field_name)
            if flag in modifier_kwargs and modifier_kwargs[flag] != getattr(mod, field_name):
                changed.append(f"--{flag}")
    return changed


def _task_hint_for_config_qualname(qualname: str) -> str:
    """The ``Re-run with --task X`` line for a mismatched saved config, or the fallback.

    A hint must never outrank the error it decorates, so a lookup failure degrades to the
    generic advice rather than propagating — the caller is already raising, and losing the
    identity message to a secondary failure would be strictly worse than losing the task name.
    """
    try:
        owner = TASK_REGISTRY.task_for_config_qualname(qualname)
    except Exception:  # noqa: BLE001 — a broken registry must not eat the real error
        owner = None
    if owner:
        return f"Re-run with --task {owner}."
    return "No registered task builds it — register it as a task, or drop --config and use --override."


def _is_inline_mapping(text: str) -> bool:
    """Whether ``text`` is something ``--override`` would accept.

    Kept as a helper rather than inlined into a nested ``except``: swallowing the probe's own
    exception inside a function restores the outer one, so ``read_config_arg``'s bare ``raise``
    re-raises the file error it means to, not the probe's.
    """
    try:
        overrides_from_str(text)
    except ValueError:
        return False
    return True


def read_config_arg(path: str) -> dict:
    """Read the ``--config`` argument — :func:`read_config_file` plus the CLI's own redirect.

    The two flags each take what the other rejects, so each names the other when handed it:
    ``--override <a path>`` already points at ``--config``, and this is the mirror. The
    redirect fires on a *positive* identification — the value is something ``--override``
    itself would accept — not on a guess about the string, so a genuine path with the wrong
    extension (or unreadable contents) still gets the plain message.

    Parameters
    ----------
    path: str
        The ``--config`` value.

    Returns
    -------
    dict
        The file's contents, as :func:`read_config_file` returns them.

    Raises
    ------
    ValueError
        Naming ``--override`` when ``path`` is an inline mapping; otherwise unchanged from
        :func:`read_config_file`.
    """
    try:
        return read_config_file(path)
    except ValueError as e:
        if not _is_inline_mapping(path):
            raise  # a real path, just not one we can read — its own message is the right one
        raise ValueError(
            f"--config takes a .json/.yaml/.yml path, but {path!r} is an inline mapping. Pass it to --override instead."
        ) from e


def _apply_config_file(config: Any, path: str, *, task_name: str, modifier_kwargs: dict[str, Any]) -> Any:
    """Apply ``--config``: a saved config replaces ``config``, a partial override layers onto it.

    The two forms are told apart by content, not by a separate flag — a config Eden saved
    carries its class and version tags (:func:`~eden.utils.configs.is_full_config`), a
    hand-written partial override does not.

    A saved config must be the one ``--task`` builds, compared by **exact class**. The
    subclass tolerance of :meth:`EdenConfig.load_from_file` is right for a library loader but
    wrong for task identity: several tasks register a config subclassing another's
    (``manipulation_lift_box_rgb`` → ``manipulation_lift_box``, ``extreme_parkour_student`` → ``…_teacher``),
    so accepting a subclass would run one task's config under the other's name — and, worse, under the
    other's ``logs/<task>/`` directory, where ``--resume`` and checkpoint-sibling replay resolve later.
    """
    data = read_config_arg(path)
    if not is_full_config(data):
        return config.with_overrides_from_dict(data)

    ignored = explicit_task_mods(task_name, modifier_kwargs)
    if ignored:
        raise ValueError(
            f"{path} is a saved config, which replaces the whole task build — so {', '.join(ignored)} "
            f"would be silently dropped. Drop the mod flag(s), or express the change as "
            f"--override instead."
        )

    saved = saved_config_qualname(data)
    built = f"{type(config).__module__}.{type(config).__name__}"
    if saved is not None and saved != built:
        raise ValueError(
            f"{path} was saved from '{saved}', but --task {task_name} builds '{built}'. A saved config "
            f"replaces the task's own, so running it under another task's name would put the run in "
            f"logs/{task_name}/ with a different task's config. {_task_hint_for_config_qualname(saved)}"
        )
    return type(config).load_from_data(data, path)


def get_task_config(
    task_name: str,
    *,
    run_name: str = "",
    modifier_kwargs: dict[str, Any] | None = None,
    config_path: str | None = None,
    config_overrides: Sequence[dict[str, Any]] | None = None,
) -> Any:
    """Build a task config programmatically.

    :meth:`TaskRegistry.build`, then — in this order, later winning — an optional
    ``config_path`` file and any ``config_overrides`` dicts, then a sanitized ``run_name``
    written to ``runner_options`` if present.

    Parameters
    ----------
    task_name: str
        Registered task to build.
    run_name: str
        Run name to write onto ``runner_options``; ignored when empty.
    modifier_kwargs: dict[str, Any] | None
        TaskMod kwargs forwarded to :meth:`TaskRegistry.build`.
    config_path: str | None
        A ``.json``/``.yaml``/``.yml`` config file. A saved config replaces the task build; a
        partial one layers onto it. See :func:`_apply_config_file`.
    config_overrides: Sequence[dict[str, Any]] | None
        Override dicts applied on top of everything else, in order — the ``--override`` payload.

    Returns
    -------
    Any
        The task's config instance.
    """
    modifier_kwargs = modifier_kwargs or {}
    config = TASK_REGISTRY.build(task_name, **modifier_kwargs)
    if config_path:
        config = _apply_config_file(config, config_path, task_name=task_name, modifier_kwargs=modifier_kwargs)
    for overrides in config_overrides or ():
        config = config.with_overrides_from_dict(overrides)
    _apply_run_name(config, run_name)
    return config


def overrides_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Parse the repeated ``--override`` arguments on ``args``, in the order given."""
    return [overrides_from_str(text) for text in getattr(args, "override", None) or ()]


def get_task_config_from_args(args: argparse.Namespace, run_name: str = "") -> Any:
    """Build a task config from an argparse Namespace produced by :func:`get_task_argparser`."""
    config = get_task_config(
        args.task,
        run_name=run_name,
        modifier_kwargs=_collect_modifier_kwargs(args, args.task),
        config_path=args.config,
        config_overrides=overrides_from_args(args),
    )

    runner_options = getattr(config, "runner_options", None)
    if runner_options is not None:
        if eden_dist.is_distributed():
            # Each rank sees one isolated GPU as cuda:0 via CUDA_VISIBLE_DEVICES;
            # "auto" resolves to that through gs.device, keeping device selection
            # uniform with the non-DDP path.
            runner_options.device = "auto"
        if args.cpu:
            runner_options.device = "cpu"
    return config
