"""Tests for ``pyproject.toml``: optional extras stay installable on every environment uv resolves for."""

import re
from pathlib import Path
import tomllib

from packaging.markers import Marker
from packaging.requirements import Requirement


def _load_pyproject() -> dict:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return tomllib.loads(pyproject_path.read_text())


def _nyx_requirements() -> dict[str, Requirement]:
    nyx_extra = _load_pyproject()["project"]["optional-dependencies"]["nyx"]
    return {(req := Requirement(entry)).name: req for entry in nyx_extra}


def test_nyx_is_an_extra_not_a_dependency():
    dependencies = {Requirement(entry).name for entry in _load_pyproject()["project"]["dependencies"]}
    assert not dependencies & {"gs-nyx", "gs-nyx-plugin"}


def test_nyx_extra_is_platform_gated():
    """No macOS wheels exist. Asserted by evaluating the marker, not by its formatting."""
    for name, req in _nyx_requirements().items():
        assert not req.marker.evaluate({"platform_system": "Darwin"}), name
        assert req.marker.evaluate({"platform_system": "Linux"}), name


def test_nyx_extra_excludes_the_renderers_that_overlight_from_an_environment_map():
    """0.1.4 and 0.1.5 reflect more environment light than a surface receives; 0.1.3 predates the bridge fixes.

    Those releases cap no Genesis version, so an unfloored extra resolves to them happily.
    """
    nyx = _nyx_requirements()["gs-nyx"].specifier
    for broken in ("0.1.3", "0.1.4", "0.1.5"):
        assert not nyx.contains(broken), broken
    assert nyx.contains("0.1.6")


def test_nyx_extra_floors_the_bridge_plugin_at_the_first_release_admitting_the_pinned_genesis():
    """Every plugin up to 0.1.6 declares `genesis-world < 1.4.0`, while the pin is 1.4.

    uv resolves ONE universal lockfile, so one of those makes `uv sync` fail for every user rather than only for the
    ones who asked for the extra. 0.1.7 admits `genesis-world >= 1.4.0, < 1.5.0`.
    """
    plugin = _nyx_requirements()["gs-nyx-plugin"].specifier
    assert not plugin.contains("0.1.6")


def test_nyx_extra_floors_the_bridge_plugin_at_the_buffer_names_the_mochi_recipes_fill():
    """0.1.10 renamed `_deformable_{uvs,indices}_cuda` to `_deformable_{uvs,indices}_backend`.

    `examples/recipes/mochi/_nyx.py` fills the new names, so under an older plugin the Mochi recipes stop with an
    `AttributeError` on their first render.
    """
    plugin = _nyx_requirements()["gs-nyx-plugin"].specifier
    assert not plugin.contains("0.1.9")
    assert plugin.contains("0.1.10")


def test_nyx_extra_is_capped_below_the_next_minor():
    """A new minor of a private-API-coupled bridge is opt-in, not an automatic upgrade."""
    for name, req in _nyx_requirements().items():
        assert not req.specifier.contains("0.2.0"), name


# `tool.uv.required-environments` spelled out as environments a marker can be evaluated
# against. uv resolves ONE universal lockfile that must be installable on every one of
# them, so an extra whose package has no wheel (and no sdist) for one of these breaks
# `uv sync` / `uv run` for every user, not just the ones who asked for the extra.
REQUIRED_ENVIRONMENTS = {
    "linux-x86_64": {"sys_platform": "linux", "platform_machine": "x86_64"},
    "darwin-arm64": {"sys_platform": "darwin", "platform_machine": "arm64"},
    "win32-AMD64": {"sys_platform": "win32", "platform_machine": "AMD64"},
}


def test_required_environments_are_the_ones_this_module_checks_against():
    """Keep the table above in sync with pyproject, so the gating tests stay meaningful."""
    declared = [Marker(entry) for entry in _load_pyproject()["tool"]["uv"]["required-environments"]]
    matched = {name for name, env in REQUIRED_ENVIRONMENTS.items() if any(m.evaluate(env) for m in declared)}
    assert matched == set(REQUIRED_ENVIRONMENTS)
    assert len(declared) == len(REQUIRED_ENVIRONMENTS), "a required environment was added; extend the table"


def test_wuji_extra_is_gated_to_the_platforms_its_wheels_cover():
    """wuji-sdk ships manylinux wheels (x86_64 / aarch64) and NO sdist.

    Without the marker there is nothing to resolve to on darwin-arm64 / win32-AMD64, and
    because those are required environments the whole project becomes unsatisfiable — the
    failure lands on every `uv run`, with a message about a Windows split that never
    mentions the glove. Asserted by evaluating the marker, not by its formatting.
    """
    wuji_extra = _load_pyproject()["project"]["optional-dependencies"]["wuji"]
    reqs = [Requirement(entry) for entry in wuji_extra]
    assert reqs, "the wuji extra is empty"
    for req in reqs:
        assert req.marker is not None, f"{req.name} needs a platform marker"
        assert req.marker.evaluate(REQUIRED_ENVIRONMENTS["linux-x86_64"]), req.name
        assert not req.marker.evaluate(REQUIRED_ENVIRONMENTS["darwin-arm64"]), req.name
        assert not req.marker.evaluate(REQUIRED_ENVIRONMENTS["win32-AMD64"]), req.name
        # aarch64 Linux (a Jetson host) is covered by the wheels and must stay allowed.
        assert req.marker.evaluate({"sys_platform": "linux", "platform_machine": "aarch64"}), req.name


# --- The published distribution -------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


def _published_requirements() -> list[Requirement]:
    project = _load_pyproject()["project"]
    extras = project["optional-dependencies"].values()
    return [Requirement(entry) for entry in [*project["dependencies"], *(e for extra in extras for e in extra)]]


def test_the_distribution_name_runtime_code_uses_is_the_one_pyproject_publishes():
    from eden.utils.distribution import DISTRIBUTION_NAME

    assert DISTRIBUTION_NAME == _load_pyproject()["project"]["name"]


def test_the_published_metadata_holds_no_direct_reference():
    """PyPI rejects a `Requires-Dist` with a URL, extras included: a git-only package belongs in a `<extra>-git`
    dependency group (never published) and in `UNPUBLISHED_REQUIREMENTS`, not in an extra.
    """
    direct = {req.name for req in _published_requirements() if req.url}
    assert direct == set()


def test_each_unpublished_requirement_is_the_pin_its_extras_git_group_installs():
    """The install command printed outside a checkout and the group a checkout syncs must name the same commit."""
    from eden.utils.distribution import UNPUBLISHED_REQUIREMENTS

    pyproject = _load_pyproject()
    git_groups = {
        name.removesuffix("-git"): tuple(reqs)
        for name, reqs in pyproject["dependency-groups"].items()
        if name.endswith("-git")
    }
    assert git_groups == UNPUBLISHED_REQUIREMENTS
    assert set(UNPUBLISHED_REQUIREMENTS) <= set(pyproject["project"]["optional-dependencies"])
    for reqs in UNPUBLISHED_REQUIREMENTS.values():
        assert all(Requirement(req).url for req in reqs), reqs


def _doc_sources() -> list[Path]:
    """Every file an install command can be written in: the package, the docs, the examples, the README (PyPI's
    project page), and pyproject.toml's comments.
    """
    return [
        REPO_ROOT / "README.md",
        REPO_ROOT / "pyproject.toml",
        *(REPO_ROOT / "eden").rglob("*.py"),
        *(REPO_ROOT / "eden").rglob("*.md"),
        *(REPO_ROOT / "docs").rglob("*.md"),
        *(REPO_ROOT / "examples").rglob("*.py"),
        *(REPO_ROOT / "examples").rglob("*.md"),
    ]


# `pip install 'eden-world[onnx]'`, `uv add eden-world[a,b]`: the distribution and the extras of an install command.
# `-e '.[pose]'` (a checkout) names no distribution and is not matched.
_INSTALL_COMMAND = re.compile(r"(?:pip install|uv add)\s+(?:--?[\w-]+\s+)*['\"]?([A-Za-z][\w.-]*)\[([\w,\s-]+)\]")
# The runtime hints build theirs with `install_hint("<extra>")` (or one of the two commands it joins).
_INSTALL_HELPER = re.compile(r"(?P<helper>install_hint|pip_install_extra|uv_sync_extra)\(['\"](?P<extra>[\w-]+)['\"]\)")
# A checkout's `uv sync ...`, up to the end of its code span, table cell or line.
_UV_SYNC = re.compile(r"uv sync\b[^`|\n]*")
# `eden[<extra>]` anywhere, in a command or in prose ("needs `eden[nyx]`"): pasted, it installs the wrong package.
_IMPORT_NAME_SPEC = re.compile(r"(?<![\w./-])eden\[[\w,\s-]+\]")


def test_every_install_command_in_the_package_docs_and_examples_names_the_distribution_and_a_declared_extra():
    """A hint naming `eden[...]` installs PyPI's unrelated `eden` (a Kivy project); one naming a removed extra
    installs nothing; a pip command for an extra in `UV_ONLY_EXTRAS` fails to resolve. All read as correct until
    somebody pastes them.
    """
    from eden.utils.distribution import DISTRIBUTION_NAME, UV_ONLY_EXTRAS

    extras = set(_load_pyproject()["project"]["optional-dependencies"])
    installable = extras - set(UV_ONLY_EXTRAS)
    wrong = []
    for path in _doc_sources():
        text = path.read_text(encoding="utf-8")
        where = path.relative_to(REPO_ROOT)
        wrong += [f"{where}: {m.group(0)!r}" for m in _IMPORT_NAME_SPEC.finditer(text)]
        for match in _INSTALL_COMMAND.finditer(text):
            named = {e.strip() for e in match.group(2).split(",")}
            if match.group(1) == DISTRIBUTION_NAME and not named <= installable:
                wrong.append(f"{where}: {match.group(0)!r}")
        for match in _INSTALL_HELPER.finditer(text):
            helper, extra = match["helper"], match["extra"]
            if extra not in extras or (helper == "pip_install_extra" and extra in UV_ONLY_EXTRAS):
                wrong.append(f"{where}: {match.group(0)!r}")
    assert not wrong, (
        f"install commands must read `{DISTRIBUTION_NAME}[<declared extra>]`, and pip must not be told to install one "
        f"of {sorted(UV_ONLY_EXTRAS)}: {wrong}"
    )


def test_each_reinstalled_package_is_a_requirement_of_its_extra():
    """The onnx hints reinstall the extra's onnxruntime wheel by name: renamed in the extra, the hint would reinstall
    a package the extra no longer brings.
    """
    from packaging.requirements import Requirement

    from eden.utils.distribution import REINSTALLED_PACKAGES

    extras = _load_pyproject()["project"]["optional-dependencies"]
    for extra, package in REINSTALLED_PACKAGES.items():
        assert package in {Requirement(req).name for req in extras[extra]}, (extra, package)


def test_each_extra_pip_cannot_install_is_declared_and_the_reason_is_still_in_pyproject():
    """`UV_ONLY_EXTRAS` exists because of `[tool.uv] override-dependencies`: with no override left there is nothing
    pip would miss, and the entry would keep printing a uv-only hint for an extra pip now installs.
    """
    from eden.utils.distribution import UV_ONLY_EXTRAS

    pyproject = _load_pyproject()
    assert set(UV_ONLY_EXTRAS) <= set(pyproject["project"]["optional-dependencies"])
    if UV_ONLY_EXTRAS:
        assert pyproject["tool"]["uv"].get("override-dependencies")


def test_every_uv_sync_in_the_package_docs_and_examples_adds_declared_extras_and_completes_them():
    """`uv sync --extra flashsac` alone leaves rsl-rl-flashsac out (it is not on PyPI, so it lives in the
    `flashsac-git` group), as does `uv sync --all-extras` without `--all-groups`, and `--extra mano` names an extra
    that no longer exists. A sync that names extras or groups without `--inexact` uninstalls every other one, which a
    reader pasting it into an environment synced with `--all-extras --all-groups` does not expect. The runtime hints
    build theirs with `uv_sync_extra`; this holds the prose to the same rule.
    """
    from eden.utils.distribution import UNPUBLISHED_REQUIREMENTS

    pyproject = _load_pyproject()
    extras, groups = set(pyproject["project"]["optional-dependencies"]), set(pyproject["dependency-groups"])
    wrong = []
    for path in _doc_sources():
        for match in _UV_SYNC.finditer(path.read_text(encoding="utf-8")):
            command = match.group(0)
            named_extras = set(re.findall(r"--extra[ =]([\w-]+)", command))
            named_groups = set(re.findall(r"--group[ =]([\w-]+)", command))
            missing_groups = {
                f"{extra}-git"
                for extra in named_extras & set(UNPUBLISHED_REQUIREMENTS)
                if "--all-groups" not in command
            } - named_groups
            # `--all-extras` names every extra, so it needs every `-git` group as much as `--extra flashsac` does.
            incomplete_all = "--all-extras" in command and "--all-groups" not in command and UNPUBLISHED_REQUIREMENTS
            exact = bool(named_extras or named_groups) and "--inexact" not in command
            if named_extras - extras or named_groups - groups or missing_groups or incomplete_all or exact:
                wrong.append(f"{path.relative_to(REPO_ROOT)}: {command.strip()!r}")
    assert not wrong, (
        "each uv sync must name declared extras and groups, an extra's `-git` group with it, and `--inexact` when it "
        f"names any: {wrong}"
    )
