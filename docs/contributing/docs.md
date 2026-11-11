# Writing the Docs

This page is for anyone editing Eden's documentation: how to build and preview the site, which pages are generated
or checked by tests, the style the pages follow, and how the site is published.

## Build and preview

The site is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and
[mkdocstrings](https://mkdocstrings.github.io/) from `mkdocs.yml` at the repository root. Install the `docs` extra
and serve the site locally:

```bash
uv sync --inexact --extra docs
uv run mkdocs serve        # live-reloading preview at http://127.0.0.1:8000/Eden/
uv run mkdocs build --strict   # a one-off build into site/, as CI runs it
```

mkdocstrings reads the package source with griffe, which does not import it, so the build needs only the `docs`
extra's packages: CI installs just those, without torch or Genesis, at the versions pinned in
`.github/actions/docs-toolchain/constraints.txt`. In a development
checkout, the `uv sync` above adds them to the environment you already have. `--strict` fails the build on any warning: a link to a page or anchor that does not exist,
or a page missing from `nav:`. The one exception is griffe's docstring and signature warnings, which the tree still
carries by the thousand: `docs/hooks/griffe_warnings.py` lowers them to DEBUG and prints how many there were.
Their number may not grow: a build with more than the hook's `MAX_GRIFFE_WARNINGS` fails. `uv run mkdocs build -v`
shows them, so you can find the ones your change added. When you fix some, lower the ceiling in the same change.
The count depends on the griffe version, so a local build with other versions than CI's pins can differ; bump the
pins and the ceiling together.

## Where pages live

| Directory | Contents |
|---|---|
| `docs/getting-started/` | installation, quickstart, concepts, the first-task tutorial, assets, troubleshooting |
| `docs/guides/` | one guide per subsystem; `AGENTS.md` cites these as the authoritative prose |
| `docs/tasks/` | the task suites |
| `docs/api/` | the API reference: short intros around `:::` blocks |
| `docs/reference/` | the CLI and environment-variable references |
| `docs/contributing/` | this section |
| `docs/assets/` | images, videos and stylesheets |

The navigation is the `nav:` tree in `mkdocs.yml`; a new page needs an entry there.

## Generated and test-pinned content

Some pages, or parts of them, are produced from the code or parsed by tests. Edit them the way the table says, or a
test fails.

| Content | Rule | Checked by |
|---|---|---|
| `docs/api/terms/catalog.md` | **Generated.** Never edit it by hand: run `uv run python -m eden.managers.catalog` after adding, renaming or re-parameterizing a term. It also rewrites `eden/resources/term_catalog.md`, the index that ships in the wheel. `eden terms --markdown` prints the same page. | `tests/resources/test_term_catalog.py` |
| The `<!-- step-order-table -->` block and the manager-order rows of `docs/guides/environments.md` | Parsed and compared with the env classes; keep the columns and their formatting. | `tests/envs/test_base.py` |
| The `inherited_members: [...]` lists in `docs/api/managers.md` | Each must name every field the term kind's mixin declares. | `tests/managers/test_base.py` |
| Every install command in `docs/**/*.md` | `pip install 'eden-world[<extra>]'`, never `eden[<extra>]`; only extras declared in `pyproject.toml`; never pip for an extra in `UV_ONLY_EXTRAS`; every `uv sync` that names an extra or group needs `--inexact`, and an extra that has a `-git` group needs that group too. | `tests/test_pyproject.py` |

Code and tests also cite guide paths and sections (`docs/guides/testing.md`, `docs/guides/error-messages.md`, the
"Task Package Layout" section of `docs/guides/task-config.md`, and others). Before you rename or move a guide, grep
the repository for its path.

## API pages

An API page is a short introduction followed by mkdocstrings blocks:

```markdown
::: eden.managers.terms.rewards.locomotion
    options:
      show_root_heading: false
```

The handler's defaults are in `mkdocs.yml` (NumPy docstrings, private names and Pydantic's `model_*` methods
filtered out, members in source order). The options used most on these pages are `show_root_heading: false` to
render a module's members without a heading for the module itself, `members: [...]` to pick members, and
`show_bases: true` with `inherited_members: false` for a subclass that should not repeat its base class.

Every `:::` path must resolve. Griffe, the library mkdocstrings uses (installed with the `docs` extra), loads the
source statically, so you can check a path without importing Eden:

```bash
uv run python -c "import griffe; griffe.load('eden.envs.base', search_paths=['.'])['EnvBase']"
```

A directory without an `__init__.py` is invisible to that loader under its `eden.` path: `eden/envs/wrappers/`,
`eden/utils/motion_reference/`, `eden/utils/buffers/`, `eden/options/objects/` and a few others. Name their classes
in prose instead of using `:::`, or give the directory an `__init__.py` with a docstring. Render each object on one
page only; a second `:::` for the same object gives it two anchors.

## Style

- Write plainly and directly, in the second person and the present tense. Leave out marketing words. Prefer a
  longer sentence a newcomer can read once over a compressed one they have to decode.
- Open each page with `# Title` and one to three sentences saying what it covers and who needs it. Use `##` and
  `###` sections, and end with a short `## See also` list where it helps.
- Lead with the common case and a working example; put edge cases and internals further down.
- Put code identifiers in backticks. Link the first mention of a major class to its API page. Link between pages
  with relative `.md` paths, never with absolute URLs to the published site.
- Use admonitions for gotchas (`!!! warning`, `!!! note`, `!!! tip`), collapsible `??? note "..."` blocks for long
  asides, content tabs (`=== "uv"` / `=== "pip"`) where two variants matter, and Mermaid diagrams only where a
  picture explains a flow better than prose.
- Use tables only for facts that are really tabular, and only with values you checked.
- Check every class name, import path, field, default, command and flag against the source or `--help` before you
  write it. Write commands as `eden <command> ...`. Snippets use absolute imports and wxyz quaternions; prefer an
  excerpt of real code from `eden/tasks/` or `examples/`, with its path, to an invented example.
- Use only numbers that are in the repository (the README, code comments, tests), and say where they come from.
- Do not reference issue or pull-request numbers, private repositories, people or internal hosts: they mean
  nothing to a reader of the public site.
- Reuse media under `docs/assets/` (from a page in `docs/guides/`, `../assets/readme/<file>.webp`).

`pymdownx.snippets` can include a file into a page with `--8<-- "<path>"`, relative to the repository root; a
missing path fails the build.

## Publishing

The site is published to GitHub Pages at <https://embodied-ai-nexus.github.io/Eden> from a branch named `docs`
of `embodied-ai-nexus/Eden`:

1. Every push to `main` that touches `docs/`, `mkdocs.yml`, `eden/` or the workflow itself runs
   `.github/workflows/docs.yml`. You can also start it by hand from the Actions tab (Docs → Run workflow). Only
   `embodied-ai-nexus/Eden` publishes; in a fork the job is skipped.
2. The workflow installs the `docs` extra and runs `mkdocs gh-deploy --strict --force --remote-branch docs`, which
   builds the site and force-pushes the HTML to `docs` as a single commit. A build that warns is not published: the
   site stays on the last good build, and GitHub reports the failed run to whoever pushed.
3. GitHub Pages serves that branch. To set it up once, open the repository's **Settings → Pages**, choose
   **Deploy from a branch** as the source, and select the `docs` branch and the `/ (root)` folder.

The `docs` branch holds only generated HTML; never edit it by hand. Pull requests do not publish: the `docs` job in
`.github/workflows/ci.yml` runs the same `mkdocs build --strict` without deploying.

## See also

- [Contributing](index.md)
- [Testing](../guides/testing.md)
