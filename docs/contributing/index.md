# Contributing

Bug reports, fixes, new tasks, terms, robots and documentation are all welcome. This page covers setting up a
development checkout, the lint and test commands, how to title and describe a pull request, and where each kind of
change lives. Every change also follows the rules in [`AGENTS.md`](https://github.com/embodied-ai-nexus/Eden/blob/main/AGENTS.md)
at the repository root. It is short, and it is what coding agents read first, so read it before your first pull
request.

## Development setup

Eden needs Python 3.12 or 3.13 (`requires-python = ">=3.12,<3.14"`) and [uv](https://docs.astral.sh/uv/). Linux
with an NVIDIA GPU is the primary platform.

```bash
git clone https://github.com/embodied-ai-nexus/Eden.git
cd Eden
uv sync                    # the core dependencies plus the dev group (ruff, pre-commit, pytest)
uv run pre-commit install  # run the hooks on every commit
```

Add the extras your change touches with `--inexact`, for example `uv sync --inexact --extra vis --extra docs`.
Without `--inexact`, `uv sync` installs exactly what you name and **removes** every extra you installed earlier,
so a later bare `uv sync` strips them too. For everything at once, run `uv sync --all-extras --all-groups`: some
extras need a package that is not on PyPI, and it lives in a `<extra>-git` dependency group
(`uv sync --inexact --extra flashsac --group flashsac-git`). The [installation page](../getting-started/installation.md)
lists the extras.

Run everything through `uv run` (or the checkout's `.venv/bin/python`), which keeps the environment in step with
`uv.lock`:

```bash
uv run eden list                                       # the CLI
uv run python examples/<path>/<script>.py              # a script
uv run pytest tests/managers/test_reward_manager.py    # one test file
uv run pre-commit run --files <changed files>          # lint and format what you touched
```

??? note "Working in several git worktrees"
    If you keep one git worktree per branch, give each worktree its own virtual environment
    (`env -u VIRTUAL_ENV uv sync` inside it). Eden is installed in editable mode, so a shared environment imports
    `eden` from whichever checkout synced it last. Two things lead there by accident: `uv run --active` (or a
    `VIRTUAL_ENV` left over from another checkout), which also rewrites that environment's editable finder to point
    at the current worktree; and `uv pip install` without `--python .venv/bin/python`, which installs into
    whatever `VIRTUAL_ENV` names. Check with `.venv/bin/python -c "import eden; print(eden.__file__)"`.

## Lint and format

`.pre-commit-config.yaml` runs four hooks: `ruff-check` and `ruff-format` (ruff 0.14.11), `yamlfmt` for YAML files
and `toml-sort-fix` for TOML files. CI runs the same hooks on every file (`uvx pre-commit run --all-files`), so a
pull request that skips them fails the lint job.

Ruff is configured in `pyproject.toml`:

- the line length is 120;
- on top of ruff's default rules it enables pydocstyle (`D`) with the NumPy convention, so every package needs a
  docstring in its `__init__.py` (`D104`), and any docstring you write must be valid NumPy style and document all of
  its parameters (`D417`). Modules, classes, methods and functions are not required to have one (`D100`-`D103` are
  ignored), though most modules carry one.

Code conventions that ruff cannot check are in `AGENTS.md`: absolute imports outside `__init__.py`, Eden imported
before `genesis` and `torch`, geometry through `eden.utils.geom` rather than re-implemented at the call site,
quaternions in wxyz order, and failing fast with an error message that says how to fix the problem.

## Tests

`tests/` mirrors `eden/`: the tests of `eden/utils/geom.py` live in `tests/utils/test_geom.py`. Tests are Detroit
style: they build real options, managers, entities and envs, and use test doubles only at hardware, network and
clock boundaries. [Testing](../guides/testing.md) covers the layout, what to assert, and how to keep the cost of
scene builds down.

Run the test files your change touches, one file at a time. The full suite builds many Genesis scenes and takes a
long time.

Some checked-in files are generated, and a test fails when one drifts from the code. If you add, rename or
re-parameterize a term, regenerate the term catalog:

```bash
uv run python -m eden.managers.catalog          # rewrites docs/api/terms/catalog.md and eden/resources/term_catalog.md
uv run python -m eden.managers.catalog --check  # exits 1 if either page is stale
```

## Pull requests

- Branch from `main`, open the pull request against `main`, and keep one change per pull request.
- **Tag the title** with one or more of the tags the history uses, then a scope and a summary, for example
  `[BUG FIX] tasks registry: override=True keeps the shadowed task's mods`.

    | Tag | For |
    |---|---|
    | `[FEATURE]` | a new capability |
    | `[BUG FIX]` | a fix to wrong behaviour |
    | `[BREAKING]` | a change to a public API, a config field, a CLI flag or a checkpoint layout; add it next to the main tag |
    | `[PERF]` | a measured speedup; put the numbers in the description |
    | `[REFACTOR]` | restructuring with no change in behaviour |
    | `[TEST]` | tests only, or next to the main tag when the pull request also adds tests |
    | `[DOCS]` | documentation |
    | `[EXAMPLES]` | changes under `examples/` |
    | `[MISC]` | dependency pins, tooling, anything else |

- Fill in the pull request template: describe the change and why it is needed, link the issue it resolves, and say
  how you tested it, with the exact commands.
- A misconfiguration must fail at the earliest well-defined point (Pydantic validation, a term's `configure()` or
  `build()`) with a message that names the bad value and what is accepted. Do not let it fail on the first `step()`,
  and do not fall back silently to a default. [Error Messages](../guides/error-messages.md) says what such a message
  must contain.

<!-- VERIFY: commit sign-off policy (DCO or CLA) — pending the maintainers' decision, so it is not documented here. -->

## Where things live

| You are changing | Where it goes | Read first |
|---|---|---|
| A task | `eden/tasks/<package>/config.py`, a config class registered with `@TASK_REGISTRY.register()` | [Task configuration](../guides/task-config.md), [your first task](../getting-started/first-task.md) |
| A task variant | a `TaskMod` registered with the task | [TaskMods](../guides/task-mods.md) |
| A reusable term | `eden/managers/terms/<kind>/`, registered with `@<KIND>_TERM_REGISTRY.register()` | [Terms](../guides/terms.md) |
| A term only one task uses | the task's own package (`eden/tasks/<package>/custom_terms/` or its `config.py`), registered the same way | [Terms](../guides/terms.md) |
| A robot | `eden/options/robots/<robot>.py`, a `RobotOptions` subclass registered with `@ROBOT_REGISTRY.register()` | [Robots](../guides/robots.md) |
| A sensor or solver Genesis does not ship | `eden/engine/` | [Engine addons](../guides/engine-addons.md) |
| A custom RL algorithm | `eden/learning/wrappers/<library>/<algorithm>/`, never `examples/` | [Training runs](../guides/training-runs.md) |
| A frozen policy | `eden/extensions/policies/<name>/` | [Extending policies](../guides/extending-policies.md) |
| A deployment backend | `eden/extensions/deployment/` | [Deployment](../guides/deployment.md) |
| A CLI subcommand | a handler in `eden/utils/cli/<area>.py` plus an entry in `COMMANDS` in `eden/__main__.py` | [CLI internals](../api/utils/cli.md) |
| A runnable example | `examples/` | `examples/AGENTS.md`, [examples](../examples/index.md) |
| Documentation | `docs/` | [Writing the docs](docs.md) |

Tasks, terms and robots register themselves where they are defined; there is no central index to update. A term is
looked up by its registry key, which is always its snake-cased class or function name, so a term class you write
must be registered to be found.

## Reporting bugs

Open bugs and feature requests with the issue templates. Include the exact command, the full traceback, and the
output of:

```bash
uv run python -c "import eden, genesis, torch; print(torch.__version__, torch.cuda.is_available())"
```

## See also

- [Testing](../guides/testing.md)
- [Error Messages](../guides/error-messages.md)
- [Writing the docs](docs.md)
