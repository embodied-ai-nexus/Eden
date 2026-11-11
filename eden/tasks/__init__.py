"""Eden task configurations.

Tasks self-register via ``@TASK_REGISTRY.register()`` on the config class.
The registry AST-scans ``eden/tasks/**/*.py`` on first lookup and finds the
decorator without importing the module — so writing a new task is a single
file change with no central index to update. See
``eden/tasks/benchmark/reacher/config.py`` for the canonical example.

Use ``TASK_REGISTRY.build("task_name", **modifier_kwargs)`` to construct a
configured instance — it runs the task's registered TaskMods, which is what
``eden train`` / ``eden smoke`` do. ``TASK_REGISTRY.get("task_name")`` returns
the bare config class; instantiating that yourself skips every mod.

How a task package is named and laid out (``<package>_<task>`` from ``<Package><Task>Config`` in a
``config.py``, a docstring-only ``__init__.py``, variants as TaskMods) is the "Task Package Layout"
section of ``docs/guides/task-config.md``; ``tests/tasks/test_init.py`` enforces it.
"""

from eden.tasks.registry import TASK_REGISTRY, TaskMod, TaskRegistry  # noqa: F401
