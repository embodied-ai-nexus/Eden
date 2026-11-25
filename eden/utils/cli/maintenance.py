"""Maintenance commands: list registered tasks, clean logs."""

from __future__ import annotations

import os
import shutil


def clean_up_logs(log_dir: str | None = None):
    """Delete the empty directories under ``log_dir``, and those holding nothing but a ``config.json``."""
    import glob
    import shutil

    base_path = os.path.dirname(os.path.dirname(__file__))
    log_dir = log_dir or os.path.join(base_path, "logs")

    logs = glob.glob(os.path.join(log_dir, "**"), recursive=True)
    empty_dirs = 0
    removed_dirs = 0
    for path in logs:
        # NOTE: remove empty directories
        if os.path.isdir(path) and not os.listdir(path):
            empty_dirs += 1
            os.rmdir(path)
        # NOTE: remove directories with only config.json in it
        if os.path.isdir(path) and len(os.listdir(path)) == 1 and "config.json" in os.listdir(path):
            shutil.rmtree(path)
            removed_dirs += 1
    print(f"Removed {removed_dirs} directories with only config.json in it")
    print(f"Removed {empty_dirs} empty logs")


_TAGS_COLUMN_WIDTH = 16


def print_task_list(tags: tuple[str, ...] = ()):
    """Print available tasks in Eden, sourced from TASK_REGISTRY (auto-discovery + eager registrations).

    ``tags`` keeps the tasks that carry every one of them (keys of ``TASK_TAGS``).
    """
    from eden.tasks import TASK_REGISTRY

    names = TASK_REGISTRY.list_tasks(tags=tags)
    if not names:
        print(f"No task carries every tag of {list(dict.fromkeys(tags))}." if tags else "No tasks registered.")
        return

    # modifier_names(), task_tags() and task_source() stay non-importing so listing can't hit the network.
    # Source is "eden" for a built-in, else the distribution or module the task comes from.
    rows: list[tuple[int, str, str, str, str]] = [
        (
            i,
            name,
            TASK_REGISTRY.task_source(name),
            ", ".join(TASK_REGISTRY.task_tags(name)) or "—",
            ", ".join(TASK_REGISTRY.modifier_names(name)) or "—",
        )
        for i, name in enumerate(names, 1)
    ]
    name_w = max(len("Task Name"), max(len(row[1]) for row in rows))
    source_w = max(len("Source"), max(len(row[2]) for row in rows))
    # Tags wraps inside its cell past this width; a hand-object tracking task carries three tags.
    tags_w = min(_TAGS_COLUMN_WIDTH, max(len("Tags"), max(len(row[3]) for row in rows)))

    try:
        from eden.utils.format import format_table

        # Wrap only the Modifiers column when the terminal is narrower than
        # the natural table width; leave the other columns at natural width
        # so they don't get squeezed into vertical-split single chars.
        term_cols = shutil.get_terminal_size(fallback=(80, 24)).columns
        no_w = max(len("No."), len(str(len(rows))))
        # Budget = terminal − (no_w + name_w + source_w + tags_w) − padding/borders (≈16 chars).
        modifiers_budget = max(20, term_cols - no_w - name_w - source_w - tags_w - 16)
        table = format_table(
            "Available Tasks in Eden",
            ["No.", "Task Name", "Source", "Tags", "Modifiers"],
            rows,
            align={"Task Name": "l", "Source": "l", "Tags": "l", "Modifiers": "l"},
            max_width={"Tags": _TAGS_COLUMN_WIDTH, "Modifiers": modifiers_budget},
        )
        print(table)
    except ImportError:
        # Fall back to plain text so `eden list` works without prettytable.
        # Nothing wraps a cell here, so pad Tags to its widest value rather than the table's cap.
        plain_tags_w = max(len("Tags"), max(len(row[3]) for row in rows))
        for i, name, source, task_tags, mods in rows:
            print(f"  {i:>3}. {name:<{name_w}}  {source:<{source_w}}  {task_tags:<{plain_tags_w}}  {mods}")


def _configure_list(parser) -> None:
    import argparse
    import textwrap

    from eden.tasks.registry import TASK_TAGS

    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        choices=sorted(TASK_TAGS),
        metavar="TAG",
        help="Show only the tasks that carry this tag. Repeat it to require several "
        "(e.g. '--tag manipulation --tag dexterous'). The tags are listed below.",
    )
    parser.description = (
        "List every registered task with its source, tags and task mods, without importing any task module."
    )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    # RawDescriptionHelpFormatter keeps the epilog's line breaks and does no wrapping, so wrap each rule here.
    parser.epilog = "tags:\n" + "\n".join(
        textwrap.fill(rule, width=88, initial_indent=f"  {tag:<16} ", subsequent_indent=" " * 19)
        for tag, rule in TASK_TAGS.items()
    )


def _run_list(args) -> None:
    print_task_list(tuple(args.tag))


def _run_clean(args) -> None:
    clean_up_logs()
