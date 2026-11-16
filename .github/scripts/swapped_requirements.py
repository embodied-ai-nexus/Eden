"""Print the requirements ``pyproject.toml`` swaps for what the published wheel does not install, space-separated
(none: nothing).

uv resolves Eden's own project with three settings the wheel's metadata does not carry, so a published wheel would
install a distribution the checkout never ran. ``release.yml`` refuses to publish while this prints anything:

* A ``[tool.uv.sources]`` entry for a published requirement that names a URL, a git repository or a path. A source of a
  development group alone (``[dependency-groups]``) never reaches the wheel, and an ``index`` source installs the same
  release, so neither is printed.
* A direct reference (``name @ url``) in ``[tool.uv]``'s ``override-dependencies`` or ``constraint-dependencies``,
  whatever the package: either one replaces a transitive requirement too (Genesis' own dependencies), which the wheel's
  users resolve from the index.
* Any ``override-dependencies`` entry for a published requirement: an override replaces the requirement the wheel
  publishes (``genesis-world==1.5.0`` over ``genesis-world==1.4.6`` runs 1.5.0 in the checkout alone).

An override's or a constraint's version bound on a transitive requirement is not printed. The wheel's users resolve
that package from the index too, if at another release (the ``trimesh`` override lifts a cap for the checkout's
lockfile): accepted drift between released versions, not an unreleased build.
"""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_RESOLUTION_SETTINGS = ("override-dependencies", "constraint-dependencies")


def swapped_requirements(pyproject: dict) -> list[str]:
    project = pyproject["project"]
    published = [
        *project.get("dependencies", ()),
        *(r for reqs in project.get("optional-dependencies", {}).values() for r in reqs),
    ]
    names = {canonicalize_name(Requirement(req).name) for req in published}
    uv = pyproject.get("tool", {}).get("uv", {})
    swapped = [
        f"{name} ([tool.uv.sources])"
        for name, source in uv.get("sources", {}).items()
        if canonicalize_name(name) in names and _swaps(source)
    ]
    for setting in _RESOLUTION_SETTINGS:
        swapped += [
            f"{req.name} ([tool.uv] {setting})"
            for req in map(Requirement, uv.get(setting, ()))
            if req.url or (setting == "override-dependencies" and canonicalize_name(req.name) in names)
        ]
    return sorted(swapped)


def _swaps(source: dict | list) -> bool:
    """Whether a source installs another distribution than the index would: a URL, a git repository or a path does; an
    `index` alone (torch from a CUDA index, say) installs the same release from elsewhere.
    """
    entries = source if isinstance(source, list) else [source]
    return any(set(entry) - {"index", "marker", "extra", "group"} for entry in entries)


if __name__ == "__main__":
    print(" ".join(swapped_requirements(tomllib.loads(Path("pyproject.toml").read_text()))))
