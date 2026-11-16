"""Print the published requirements ``[tool.uv.sources]`` swaps for another source, space-separated (none: nothing).

A source applies only where uv builds Eden from source, and the wheel keeps the requirement it swaps: a published wheel
would install a distribution the checkout never ran. ``release.yml`` refuses to publish while this prints anything. A source of
a development group alone (``[dependency-groups]``) never reaches the wheel, and an ``index`` source installs the same
release, so neither is printed.
"""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def swapped_requirements(pyproject: dict) -> list[str]:
    project = pyproject["project"]
    published = [
        *project.get("dependencies", ()),
        *(r for reqs in project.get("optional-dependencies", {}).values() for r in reqs),
    ]
    names = {canonicalize_name(Requirement(req).name) for req in published}
    sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
    return sorted(name for name, source in sources.items() if canonicalize_name(name) in names and _swaps(source))


def _swaps(source: dict | list) -> bool:
    """Whether a source installs another distribution than the index would: a URL, a git repository or a path does; an
    `index` alone (torch from a CUDA index, say) installs the same release from elsewhere.
    """
    entries = source if isinstance(source, list) else [source]
    return any(set(entry) - {"index", "marker", "extra", "group"} for entry in entries)


if __name__ == "__main__":
    print(" ".join(swapped_requirements(tomllib.loads(Path("pyproject.toml").read_text()))))
