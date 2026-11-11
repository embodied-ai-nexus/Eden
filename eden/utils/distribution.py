"""The name ``pip install`` and ``importlib.metadata`` know Eden by (not its import name), and how to install an extra.

Eden imports as ``eden`` and is distributed as ``eden-world``: the PyPI name ``eden`` belongs to an unrelated Kivy
project, so ``pip install eden`` installs the wrong package. ``[project].name`` in ``pyproject.toml`` is the source of
truth; ``tests/test_pyproject.py`` pins :data:`DISTRIBUTION_NAME` to it, and checks that every
``pip install "<name>[<extra>]"`` written in ``eden/``, ``docs/`` and ``examples/`` names this distribution and an
extra it declares.

A message that tells the reader how to install an extra says it with :func:`install_hint`, which names both commands:
whoever reads it may have Eden from PyPI or a clone. An extra in :data:`UV_ONLY_EXTRAS` gets the checkout's command
alone, with the reason pip cannot install it.

This module is the one home for what is particular about installing an extra: each such fact is a table here, which
:func:`pip_install_extra` / :func:`uv_sync_extra` read and ``tests/test_pyproject.py`` ties to ``pyproject.toml``. A
call site never writes its own install command. It imports nothing beyond the standard library, so ``import eden``
can read :func:`get_eden_version` without loading torch.
"""

import json
from importlib import metadata
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

DISTRIBUTION_NAME = "eden-world"


def get_eden_version() -> str:
    """The installed version of Eden's distribution (``eden-world``, e.g. ``0.1.0``), or ``""`` when it is not
    installed under that name (a source tree on ``PYTHONPATH``, say).
    ``eden.__version__`` reads it.

    The same in a checkout as in a wheel, so it suits a stamp that should change only with a release: a checkout's
    commit (:func:`eden.utils.misc.get_eden_commit`) changes with every commit.
    """
    return installed_version(DISTRIBUTION_NAME) or ""


def installed_version(name: str) -> str | None:
    """The installed version of distribution ``name`` (its ``pip install`` name), or ``None`` when it is not
    installed under that name. Never raises, so a message or a provenance record can call it.
    """
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def direct_url_record(dist: metadata.Distribution) -> dict:
    """``dist``'s PEP 610 ``direct_url.json`` (how a ``git+``, local-directory or editable install was made), or ``{}``
    when it has none: an install from an index records none, and a corrupt record says nothing either.
    """
    try:
        record = json.loads(dist.read_text("direct_url.json") or "{}")
    except ValueError:
        return {}
    return record if isinstance(record, dict) else {}


def editable_checkout(dist: metadata.Distribution) -> Path | None:
    """The checkout an editable install of ``dist`` points at, as its ``direct_url.json`` records it, or ``None`` when
    ``dist`` is not an editable install of a local directory. The directory may since have moved.
    """
    record = direct_url_record(dist)
    url = urlparse(str(record.get("url", "")))
    if not (record.get("dir_info", {}).get("editable") and url.scheme == "file"):
        return None
    return Path(url2pathname(url.path))


#: Requirements an extra needs that are not on PyPI. PyPI refuses a wheel whose metadata holds a direct (``git+``)
#: reference, extras included, so the extra leaves them out and its install command names them instead. In a checkout,
#: ``pyproject.toml``'s ``<extra>-git`` dependency group installs the same pins (:func:`uv_sync_extra`);
#: ``tests/test_pyproject.py`` keeps the two equal. An entry goes once its package is published (then it is an
#: ordinary requirement of the extra).
UNPUBLISHED_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "metaquest": ("televuer @ git+https://github.com/Kashu7100/televuer@a1d8d6a0784fb289ddcde9aa108b92c7cf52f41b",),
    "flashsac": (
        "rsl-rl-flashsac @ git+https://github.com/Holiday-Robot/rsl_rl_flashsac@9822b1b7e0b5cce45e7f79d6b3e2c169fa042faa",
    ),
}

#: Extras whose wheel shares its import directory with another extra's: ``onnxruntime`` and ``onnxruntime-gpu`` both
#: own ``onnxruntime/``. Neither command removes the other wheel (``uv sync --inexact`` keeps it, and pip never
#: uninstalls an unrelated package), so the extra's wheel must be installed last: that puts its files on top and
#: repairs a directory a half-removed wheel left empty, which pip and uv still count as installed. ``uv sync``
#: reinstalls it at the locked version. pip reinstalls it only when it is already installed, at that version: the
#: extra's install leaves an installed wheel alone, and installs one it lacks last anyway (``--force-reinstall``
#: alone would fetch the newest from the index, and on macOS fail for ``onnxruntime-gpu``, which the extra skips
#: there). Not an uninstall: ``uv pip uninstall`` follows ``VIRTUAL_ENV``, which may name another environment, while
#: ``uv sync`` always works on the checkout's own. ``tests/test_pyproject.py`` checks that each package is a
#: requirement of its extra.
REINSTALLED_PACKAGES: dict[str, str] = {"onnx": "onnxruntime", "onnx-gpu": "onnxruntime-gpu"}

#: Extras pip cannot install, each with the reason. Their resolution needs ``[tool.uv] override-dependencies`` in
#: ``pyproject.toml``, which only uv reads, and only in a checkout: the published metadata cannot carry an override.
#: :func:`install_hint` names the checkout's command alone for these, and ``tests/test_pyproject.py`` refuses a
#: ``pip install`` of one written anywhere in the package, docs or examples. An entry goes once the conflict is gone
#: upstream (then its override goes too).
UV_ONLY_EXTRAS: dict[str, str] = {
    "soma-retargeter": (
        "eden-soma-retargeter 0.4.0 pins trimesh==4.11.3 while genesis-world needs trimesh>=5, and only the "
        "checkout's uv override reconciles them"
    ),
}


def pip_install_extra(extra: str) -> str:
    """The shell command that installs Eden with ``extra``, e.g. ``pip install "eden-world[vis]"``.

    For an extra with :data:`UNPUBLISHED_REQUIREMENTS` the command installs those too, so it is complete as printed.
    For one in :data:`REINSTALLED_PACKAGES` whose wheel is already installed, it then reinstalls that wheel alone at
    the installed version: ``pip install "eden-world[onnx]" && pip install --force-reinstall --no-deps
    "onnxruntime==1.27.0"`` (``--no-deps``, or pip would reinstall torch with it). So the command reads this
    environment. Each requirement is double-quoted: single quotes reach pip verbatim from Windows ``cmd``, double
    quotes work in every shell.
    """
    command = " ".join(
        [
            "pip install",
            f'"{DISTRIBUTION_NAME}[{extra}]"',
            *(f'"{req}"' for req in UNPUBLISHED_REQUIREMENTS.get(extra, ())),
        ]
    )
    package = REINSTALLED_PACKAGES.get(extra)
    installed = installed_version(package) if package else None
    if installed:
        command += f' && pip install --force-reinstall --no-deps "{package}=={installed}"'
    return command


def uv_sync_extra(extra: str) -> str:
    """The command that adds ``extra`` to a checkout's environment, e.g. ``uv sync --inexact --extra vis``.

    ``--inexact`` because ``uv sync`` is exact by default: it uninstalls every extra and group the command does not
    name, so a reader synced with ``--all-extras --all-groups`` who pasted a sync of ``--extra vis`` alone from an
    error message would lose everything else. ``tests/test_pyproject.py`` holds the docs and examples to the same rule.

    For an extra with :data:`UNPUBLISHED_REQUIREMENTS` it also syncs the extra's ``<extra>-git`` dependency group,
    which holds them: ``uv sync --inexact --extra flashsac --group flashsac-git``. For one in
    :data:`REINSTALLED_PACKAGES` it reinstalls that wheel: ``uv sync --inexact --extra onnx --reinstall-package
    onnxruntime``.
    """
    group = [f"--group {extra}-git"] if extra in UNPUBLISHED_REQUIREMENTS else []
    reinstall = [f"--reinstall-package {REINSTALLED_PACKAGES[extra]}"] if extra in REINSTALLED_PACKAGES else []
    return " ".join(["uv sync --inexact", f"--extra {extra}", *group, *reinstall])


def install_hint(extra: str) -> str:
    """How to install ``extra``, for an error message or a help text, in both the ways a reader may need: with Eden
    from PyPI, and in a clone. For ``vis`` it reads::

        `pip install "eden-world[vis]"`, or `uv sync --inexact --extra vis` in a checkout

    Both commands are complete as printed (see :func:`pip_install_extra` and :func:`uv_sync_extra`). An extra in
    :data:`UV_ONLY_EXTRAS` gets the checkout's command alone, and why pip cannot install it.
    """
    if extra in UV_ONLY_EXTRAS:
        return f"`{uv_sync_extra(extra)}` in a checkout of Eden (pip cannot install it: {UV_ONLY_EXTRAS[extra]})"
    return f"`{pip_install_extra(extra)}`, or `{uv_sync_extra(extra)}` in a checkout"
