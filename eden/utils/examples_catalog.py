"""A queryable catalog of the runnable examples under ``examples/``.

Each example carries a ``---``-fenced YAML metadata block in its module docstring
(see ``examples/AGENTS.md``). This module is the single source of truth for that
contract: it parses the blocks **by AST, never importing the example**, validates
them against a closed schema/vocabulary, and exposes a faceted query API modeled on
the asset manifest (:mod:`eden.options.catalog`).

    from eden.utils.examples_catalog import CATALOG
    CATALOG.names(tier="recipe", capability_kind="sensor")
    CATALOG.search("lidar")
    CATALOG.get("recipes/sensors/sensors_lidar_go2")

Backs ``eden examples list`` and the smoke gate. The parser/validator is also used by
``tests/utils/test_examples_catalog.py``.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from eden.utils.distribution import install_hint

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES = REPO_ROOT / "examples"

TIERS = frozenset({"project", "policy", "recipe"})
HARDWARE = frozenset({"cpu", "gpu", "gpu-fp64", "real-robot"})
CAP_NAMESPACES = frozenset(
    {
        "solver",
        "sensor",
        "robot",
        "teleop",
        "dataset",
        "pose",
        "retarget",
        "term",
        "render",
        "planning",
        "task",
        "scene",
        "deploy",
        "collision",
        "terrain",
        "rl",
    }
)
# The closed `optional_deps` vocabulary. One table binds each tag to the module whose
# presence proves it is installed *and* to the install path a user takes when it is not.
# The header validator (below) and the smoke gate (`eden.utils.example_smoke`) read the
# same entries, so the vocabulary and the probe cannot drift apart, and a skip can always
# say how to un-skip.


@dataclass(frozen=True)
class OptionalDep:
    """One optional-dep tag: the import probe and the human install path.

    ``module`` is the top-level third-party module the tagged examples reach (often
    transitively) — the smoke gate imports it to decide installed vs missing. ``install``
    is the one-line hint :func:`eden.utils.example_smoke.skip_reason` shows next to a
    missing dep; it is also the documented home for the tags that cannot be a pip extra
    (source builds, git URLs, wheels bundled with another package).
    """

    module: str
    install: str


OPTIONAL_DEPS = {
    "libuipc": OptionalDep("uipc", "pip install pyuipc"),
    # gs_madrona is a genesis-world dependency on Linux x86-64 (no wheels elsewhere), so a
    # normal Eden install already has it; missing means an unsupported platform.
    "madrona": OptionalDep("gs_madrona", "pip install gs-madrona (Linux x86-64 only)"),
    "curobo": OptionalDep("curobo", "pip install --no-build-isolation git+https://github.com/NVlabs/curobo.git"),
    "ompl": OptionalDep("ompl", "pip install ompl"),
    # eden-grasp-gen ships the `grasp_gen` module (there is no `graspgen` module to probe).
    "graspgen": OptionalDep("grasp_gen", f"{install_hint('graspgen')}, then `grasp_gen download`"),
    # eden-graspgenx ships the `graspgenx` module.
    "graspgenx": OptionalDep("graspgenx", install_hint("graspgenx")),
    "metaquest": OptionalDep("televuer", install_hint("metaquest")),
    "pico": OptionalDep(
        "xrobotoolkit_sdk",
        f"build from https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind, plus {install_hint('pico')}",
    ),
    # The gello device stack itself is first-party (eden/utils/devices/gello); pyzmq is the
    # third-party module its leader<->sim ZMQ channel reaches.
    "gello": OptionalDep("zmq", install_hint("gello")),
    # wuji-sdk publishes manylinux wheels and no sdist, so the `wuji` extra carries a
    # sys_platform marker and resolves to nothing off Linux.
    "wuji": OptionalDep("wuji_sdk", f"{install_hint('wuji')} (Linux x86-64 / aarch64 only)"),
    # spnav wraps the system spacenav stack; the full recipe lives in the device module.
    "spacemouse": OptionalDep(
        "spnav", "apt install libspnav-dev spacenavd, then the spnav build in eden/utils/devices/spacemouse.py"
    ),
    "viser": OptionalDep("viser", "pip install viser"),
    # The `nyx` extra installs both the renderer (gs-nyx) and its Genesis bridge (gs-nyx-plugin),
    # a matched pair with the Genesis pin; see the `nyx` extra in pyproject.toml.
    "nyx": OptionalDep("gs_nyx_plugin", install_hint("nyx")),
    "onnx": OptionalDep("onnxruntime", f"CPU: {install_hint('onnx')}; GPU: {install_hint('onnx-gpu')}"),
    "sb3": OptionalDep("stable_baselines3", install_hint("sb3")),
    "skrl": OptionalDep("skrl", install_hint("skrl")),
    "metamotivo": OptionalDep(
        "metamotivo", "pip install 'metamotivo[huggingface] @ git+https://github.com/facebookresearch/metamotivo.git'"
    ),
}
REQUIRED = frozenset({"tier", "capabilities", "hardware", "run"})

# Modules under ``examples/`` that are not themselves examples — no header required.
# Two kinds: helper modules an example imports, and offline asset converters that record how a
# hosted asset was produced. Modules inside a *package* (a directory carrying __init__.py, e.g.
# an example-local training stack like extreme_parkour/rma_training) need no entry here:
# entry_points() skips packages wholesale — a package is by definition imported, not run.
EXCLUDE = frozenset(
    {
        "examples/projects/mimic/beyondmimic/config.py",
        "examples/projects/sysid/xhand1/sim_twin_config.py",
        "examples/recipes/mochi/_mochi_common.py",
        "examples/recipes/mochi/_finray.py",
        "examples/recipes/mochi/_nyx.py",
        "examples/recipes/mochi/_rj45.py",
        "examples/recipes/mochi/convert_rj45_asset.py",
        "examples/recipes/retargeting/dex_retargeter/hand_presets.py",
        "examples/recipes/retargeting/dex_retargeter/single_hand_detector.py",
    }
)


def entry_points() -> list[Path]:
    """Every runnable example script (excludes helper/config modules and packages).

    Returns an empty list when ``examples/`` is absent — e.g. a wheel install, where
    only the ``eden`` package ships and this catalog is inert rather than broken.
    """
    if not EXAMPLES.is_dir():
        return []
    out = []
    for p in sorted(EXAMPLES.rglob("*.py")):
        rel = p.relative_to(REPO_ROOT).as_posix()
        if "/custom_terms/" in rel or "/__pycache__/" in rel or rel in EXCLUDE:
            continue
        if (p.parent / "__init__.py").exists():  # package module: imported, never run
            continue
        out.append(p)
    return out


def rel_path(path: Path) -> str:
    """Repo-relative path for messages, falling back to the absolute one when outside the repo."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


class HeaderError(ValueError):
    """A malformed example header, tagged with the file it came from.

    Subclasses ``ValueError`` so ``except ValueError`` callers keep working; ``reason``
    is the bare message for callers that print the path themselves.
    """

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"{rel_path(path)}: {reason}")
        self.path = path
        self.reason = reason


def parse_header(path: Path) -> dict:
    """Parse one example's docstring header into a dict (with ``purpose``); raise HeaderError."""
    doc = ast.get_docstring(ast.parse(path.read_text()))
    if not doc:
        raise HeaderError(path, "no module docstring")
    lines = doc.splitlines()
    try:
        i = next(k for k, ln in enumerate(lines) if ln.strip() == "---")
        j = next(k for k in range(i + 1, len(lines)) if lines[k].strip() == "---")
    except StopIteration:
        raise HeaderError(path, "no '---'-fenced YAML block in docstring")
    purpose = next((ln.strip() for ln in lines[:i] if ln.strip()), "")
    if not purpose:
        raise HeaderError(path, "no one-line summary before the YAML block")
    try:
        meta = yaml.safe_load("\n".join(lines[i + 1 : j])) or {}
    except yaml.YAMLError as e:
        raise HeaderError(path, f"YAML block does not parse: {e}") from e
    if not isinstance(meta, dict):
        raise HeaderError(path, "YAML block is not a mapping")
    meta["purpose"] = purpose
    return meta


def validate(meta: dict) -> list[str]:
    """Return a list of schema/vocabulary violations for a parsed header (empty = valid)."""
    errs = []
    missing = REQUIRED - meta.keys()
    if missing:
        errs.append(f"missing fields: {sorted(missing)}")
    if meta.get("tier") not in TIERS:
        errs.append(f"tier {meta.get('tier')!r} not in {sorted(TIERS)}")
    if meta.get("hardware") not in HARDWARE:
        errs.append(f"hardware {meta.get('hardware')!r} not in {sorted(HARDWARE)}")
    caps = meta.get("capabilities")
    if not isinstance(caps, list) or not caps:
        errs.append("capabilities must be a non-empty list")
    else:
        for c in caps:
            if not isinstance(c, str) or c.count(".") != 1:
                errs.append(f"capability {c!r} is not 'namespace.leaf'")
            elif c.split(".")[0] not in CAP_NAMESPACES:
                errs.append(f"capability {c!r} has unknown namespace")
    for d in meta.get("optional_deps", []) or []:
        if d not in OPTIONAL_DEPS:
            errs.append(f"optional_dep {d!r} not in {sorted(OPTIONAL_DEPS)}")
    if not str(meta.get("run", "")).strip():
        errs.append("run command is empty")
    return errs


@dataclass(frozen=True)
class ExampleSpec:
    """One runnable example. ``key`` is the ``examples/``-relative path without ``.py``."""

    key: str
    path: str  # repo-relative
    tier: str
    purpose: str
    capabilities: tuple[str, ...]
    hardware: str
    run: str
    optional_deps: tuple[str, ...] = field(default_factory=tuple)

    @property
    def capability_kinds(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(c.split(".")[0] for c in self.capabilities))

    @classmethod
    def from_meta(cls, path: Path, meta: dict) -> ExampleSpec:
        rel = path.relative_to(REPO_ROOT).as_posix()
        return cls(
            key=path.relative_to(EXAMPLES).with_suffix("").as_posix(),
            path=rel,
            tier=meta["tier"],
            purpose=meta["purpose"],
            capabilities=tuple(meta["capabilities"]),
            hardware=meta["hardware"],
            run=str(meta["run"]).strip(),
            optional_deps=tuple(meta.get("optional_deps") or ()),
        )


class ExampleCatalog:
    """Lazy, faceted index over the example headers. Query API mirrors ``AssetManifest``."""

    def __init__(self) -> None:
        self._specs: dict[str, ExampleSpec] | None = None

    def _ensure(self) -> dict[str, ExampleSpec]:
        if self._specs is None:
            specs: dict[str, ExampleSpec] = {}
            bad: list[str] = []  # collect them all: one bad header must not mask the rest
            paths = entry_points()
            for path in paths:
                try:
                    meta = parse_header(path)
                except HeaderError as e:
                    bad.append(str(e))
                    continue
                if missing := REQUIRED - meta.keys():  # exactly the fields ExampleSpec indexes
                    bad.append(f"{rel_path(path)}: missing fields: {sorted(missing)}")
                    continue
                spec = ExampleSpec.from_meta(path, meta)
                specs[spec.key] = spec
            if bad:
                listing = "\n".join(f"  - {b}" for b in bad)
                raise ValueError(
                    f"{len(bad)} of {len(paths)} examples have an unusable metadata header "
                    f"(see examples/AGENTS.md for the contract):\n{listing}"
                )
            self._specs = specs
        return self._specs

    def specs(self) -> list[ExampleSpec]:
        return list(self._ensure().values())

    def get(self, key: str) -> ExampleSpec:
        specs = self._ensure()
        if key not in specs:
            import difflib

            hint = difflib.get_close_matches(key, specs, n=1)
            raise KeyError(f"unknown example {key!r}" + (f" — did you mean {hint[0]!r}?" if hint else ""))
        return specs[key]

    def names(
        self,
        *,
        tier: str | None = None,
        capability: str | None = None,
        capability_kind: str | None = None,
        hardware: str | None = None,
        optional_dep: str | None = None,
    ) -> list[str]:
        """Sorted example keys matching every provided filter."""
        out = []
        for s in self._ensure().values():
            if tier is not None and s.tier != tier:
                continue
            if capability is not None and capability not in s.capabilities:
                continue
            if capability_kind is not None and capability_kind not in s.capability_kinds:
                continue
            if hardware is not None and s.hardware != hardware:
                continue
            if optional_dep is not None and optional_dep not in s.optional_deps:
                continue
            out.append(s.key)
        return sorted(out)

    def search(self, query: str) -> list[str]:
        """Case-insensitive substring match over key, purpose, and capabilities."""
        q = query.strip().lower()
        if not q:
            raise ValueError("empty search query")
        hits = [
            s.key
            for s in self._ensure().values()
            if q in s.key.lower() or q in s.purpose.lower() or any(q in c for c in s.capabilities)
        ]
        return sorted(hits)

    def facets(self) -> dict[str, list[str]]:
        """Each filter axis → the values present in the catalog (self-describing)."""
        specs = self._ensure().values()
        kinds, caps, deps = set(), set(), set()
        for s in specs:
            kinds.update(s.capability_kinds)
            caps.update(s.capabilities)
            deps.update(s.optional_deps)
        return {
            "tier": sorted({s.tier for s in specs}),
            "hardware": sorted({s.hardware for s in specs}),
            "capability_kind": sorted(kinds),
            "capability": sorted(caps),
            "optional_dep": sorted(deps),
        }

    def summary(self) -> list[dict]:
        """Flat JSON-ready records, one per example."""
        return [asdict(s) for s in self._ensure().values()]


CATALOG = ExampleCatalog()
