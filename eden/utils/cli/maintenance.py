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


def print_task_list():
    """Print available tasks in Eden, sourced from TASK_REGISTRY (auto-discovery + eager registrations)."""
    from eden.tasks import TASK_REGISTRY

    names = sorted(TASK_REGISTRY.list_tasks())
    if not names:
        print("No tasks registered.")
        return

    # modifier_names() and task_source() stay non-importing so listing can't hit the network.
    # Source is "eden" for a built-in, else the distribution or module the task comes from.
    rows: list[tuple[int, str, str, str]] = [
        (i, name, TASK_REGISTRY.task_source(name), ", ".join(TASK_REGISTRY.modifier_names(name)) or "—")
        for i, name in enumerate(names, 1)
    ]
    name_w = max(len("Task Name"), max(len(name) for _, name, _, _ in rows))
    source_w = max(len("Source"), max(len(source) for _, _, source, _ in rows))

    try:
        from eden.utils.format import format_table

        # Wrap only the Modifiers column when the terminal is narrower than
        # the natural table width; leave the other columns at natural width
        # so they don't get squeezed into vertical-split single chars.
        term_cols = shutil.get_terminal_size(fallback=(80, 24)).columns
        no_w = max(len("No."), len(str(len(rows))))
        # Budget = terminal − (no_w + name_w + source_w) − padding/borders (≈13 chars).
        modifiers_budget = max(20, term_cols - no_w - name_w - source_w - 13)
        table = format_table(
            "Available Tasks in Eden",
            ["No.", "Task Name", "Source", "Modifiers"],
            rows,
            align={"Task Name": "l", "Source": "l", "Modifiers": "l"},
            max_width={"Modifiers": modifiers_budget},
        )
        print(table)
    except ImportError:
        # Fall back to plain text so `eden list` works without prettytable.
        for i, name, source, mods in rows:
            print(f"  {i:>3}. {name:<{name_w}}  {source:<{source_w}}  {mods}")


def _run_list(args) -> None:
    print_task_list()


def _run_clean(args) -> None:
    clean_up_logs()
