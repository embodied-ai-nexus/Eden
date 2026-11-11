"""Resolving entity Options classes by name — two layers over one ``"module:Class"`` pointer.

Eden assets (``eden.options.objects.*``, ``eden.options.robots.*``) know how to *load* a mesh
(their ``EntityOptions`` carry ``up``/``front``/``scale`` etc.) but the question "which Options
class does this name mean?" is answered here, two ways, both lazily importing only the one
module they need:

* **Auto-scan index** (:func:`find_entity_options` / :func:`entity_options_index`) — exhaustive.
  A filesystem walk of the asset packages, cached on disk and keyed by a source fingerprint,
  maps every class name (``RoboLabYcbBanana``) and family-registry key (``robolab/ycb/banana``)
  to its ``"module:Class"`` pointer. This backs the ``eden view`` / ``eden snap`` CLI, where a
  warm lookup imports only the single defining module.

* **Curated manifest** (:data:`MANIFEST`) — a hand-curated *semantic* subset. An asset is
  curated by decorating its Options class::

        @MANIFEST.curate("sofa", "furniture/seating", tags=("seating",), description="a couch")
        @REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_sofa")
        class FRLApartmentSofa(ReplicaCADObjectOptions): ...

  What's curated is **semantic only** (no ``EntityOptions`` field is re-declared): ``category``
  (``/``-delimited taxonomy path), ``placement`` (floor | surface | wall | ceiling),
  ``support_surface`` (place-on target), ``container`` (place-in target), ``tags``,
  ``description``. Physical pose stays on the Options class (``up``/``front``/``scale``);
  ``AssetSpec.movable`` is *derived* at curation time from the constructed Options
  (``not is_fixed_base``), never hand-curated, so it can't drift from the class.

  Variant families (one Options class serving many mesh variants, e.g. Objaverse) curate each
  variant with the direct-call form and ``make_kwargs`` — the variant *selector* baked into
  :meth:`AssetSpec.make` (explicit ``make()`` overrides still win)::

        MANIFEST.curate("apple_3", "food/fruit", make_kwargs={"variant": 3})(ObjaverseApple)

  Population is **never eager** — so importing Eden stays cheap no matter how many assets exist:
  a ``@curate`` decorator registers its asset only when that *family module* is imported, and
  re-import is idempotent (a true name collision raises); :meth:`AssetManifest.build` imports the
  known families on demand; :meth:`ensure` is the build-if-empty entry point for consumers.

Both layers share one ``"module:Class"`` spelling and one lazy resolver (:func:`_resolve`):
``AssetSpec`` stores ``options_path`` (so :meth:`AssetManifest.summary` rows are JSON-ready and
:meth:`make` imports only the one class it constructs), and the auto-scan index stores the same.

    from eden.options.catalog import MANIFEST, find_entity_options
    MANIFEST.ensure()                                  # populate (imports asset families)
    MANIFEST.facets()                                  # what can I query? axes → valid values
    MANIFEST.names(category="furniture/seating")       # drill down → ['armchair','chair','sofa']
    MANIFEST.search("mug")                             # free text → ['coffee_cup', ...]
    opts = MANIFEST.make("sofa", default_root_pos=(0, 0, 0))   # lazily imports + constructs
    cls = find_entity_options("RoboLabYcbBanana")      # exhaustive auto-scan lookup
"""

from __future__ import annotations

import difflib
import hashlib
import importlib
import json
import os
import pkgutil
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from eden.options.entities import BaseEntityOptions, RobotOptions
from eden.utils.misc import get_cache_dir
from eden.utils.registry import Registry

if TYPE_CHECKING:
    from eden.options.entities import EntityOptions


def _resolve(options_path: str) -> type:
    """Import and return the class named by a ``"module:Class"`` pointer.

    The one spelling both layers here use. TaskRegistry's discovery map and the config
    serialization envelope's ``_option_module_``/``_option_class_`` fields are two more
    forms of the same idea — consolidate those onto this helper before adding a fourth.
    """
    module, _, name = options_path.partition(":")
    return getattr(importlib.import_module(module), name)


# --------------------------------------------------------------------------- #
# Curated semantic manifest
# --------------------------------------------------------------------------- #

Placement = Literal["floor", "surface", "wall", "ceiling"]
_LEAF = "__assets__"  # tree key holding leaf asset names (avoids name/category collisions)

#: Asset family modules whose classes carry ``@MANIFEST.curate`` decorators. ``build()``
#: imports these to populate the manifest; add a family's module here when it's curated
#: in-tree. Out-of-tree families need no edit here — their ``@curate`` decorators fire as
#: soon as the consuming code imports the family module.
ASSET_MODULES: tuple[str, ...] = ("eden.options.objects.replica_cad",)


@dataclass(frozen=True)
class AssetSpec:
    """Semantic metadata for one curated asset (the physical pose lives on the Options class)."""

    name: str
    options_path: str  # "module:ClassName" — resolved lazily by make()
    category: str = "object"  # "/"-delimited taxonomy path, e.g. "furniture/seating"
    placement: Placement = "floor"
    support_surface: bool = False  # exposes a usable flat top (a table does, a chair doesn't)
    container: bool = False  # can hold objects inside (bin, basket, cup, cabinet)
    tags: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""
    make_kwargs: dict = field(default_factory=dict)  # variant selector baked into make() (e.g. variant=3)
    #: Whether :meth:`make` yields a free (non-fixed-base) entity. NOT curated by hand: derived
    #: once at curation time from the constructed Options (so it reflects ``model_post_init``'s
    #: ``fixed``/``is_fixed_base`` unification and can't drift). Some surface props ship fixed
    #: (e.g. the clock); pick-and-place consumers free them with ``make(name, is_fixed_base=False)``.
    movable: bool = True

    @property
    def category_path(self) -> tuple[str, ...]:
        return tuple(p for p in self.category.split("/") if p)

    def make(self, **overrides) -> EntityOptions:
        """Construct the asset's Options (lazily importing its class; canonical pose from it).

        Curated ``make_kwargs`` are applied first; explicit ``overrides`` win.
        """
        return _resolve(self.options_path)(**{**self.make_kwargs, **overrides})


def _is_under(path: tuple[str, ...], category: str) -> bool:
    prefix = tuple(p for p in category.split("/") if p)
    return path[: len(prefix)] == prefix


class AssetManifest(Registry):
    """Tree-structured registry of :class:`AssetSpec`, keyed by friendly name.

    Not populated on import: call :meth:`ensure` (or :meth:`build`) first. :meth:`make`
    imports the chosen asset's class lazily.
    """

    # --- curation (decorator) -------------------------------------------------- #
    def curate(
        self,
        name: str,
        category: str,
        *,
        placement: Placement = "floor",
        support_surface: bool = False,
        container: bool = False,
        tags=(),
        description: str = "",
        make_kwargs: dict | None = None,
    ):
        """Curate the decorated ``EntityOptions`` subclass as an asset.

        Usable as a stacked decorator, or in direct-call form for variant families
        (one class, many curated names): ``curate("apple_3", ..., make_kwargs={"variant": 3})(cls)``.
        """
        kwargs = dict(make_kwargs or {})

        def deco(cls):
            # Construct the Options once, here where the class is in hand, to derive `movable`
            # the way make() actually builds it (running model_post_init's fixed/is_fixed_base
            # unification). This also fail-fasts at the curation site if make_kwargs is wrong,
            # rather than at a downstream make()/query. Options construction is offline (no
            # mesh load — that happens at entity build) and cheap.
            try:
                movable = not cls(**kwargs).is_fixed_base
            except Exception as exc:  # noqa: BLE001 — re-raise with the curation site attached
                raise ValueError(f"curate({name!r}): cannot construct {cls.__name__}(**{kwargs!r}): {exc}") from exc
            self._curate(
                AssetSpec(
                    name=name,
                    options_path=f"{cls.__module__}:{cls.__name__}",
                    category=category,
                    placement=placement,
                    support_surface=support_surface,
                    container=container,
                    tags=tuple(tags),
                    description=description,
                    make_kwargs=kwargs,
                    movable=movable,
                )
            )
            return cls

        return deco

    def register(self, obj=None, name=None):  # noqa: ARG002
        """Disable the inherited register entry point.

        The base :meth:`Registry.register` would store a raw class where an :class:`AssetSpec`
        is expected, silently corrupting every query method. Curate assets (with their semantic
        metadata) via :meth:`curate` instead.
        """
        raise NotImplementedError(
            "AssetManifest is populated via @MANIFEST.curate(name, category, ...), not register()."
        )

    def _curate(self, spec: AssetSpec) -> None:
        existing = self._obj_map.get(spec.name)
        if existing is not None:
            # identity = class pointer + baked constructor kwargs, so two variants of the
            # same class under one name collide rather than silently aliasing.
            if (existing.options_path, existing.make_kwargs) != (spec.options_path, spec.make_kwargs):
                raise ValueError(
                    f"asset name {spec.name!r} already curated as {existing.options_path!r} "
                    f"(make_kwargs={existing.make_kwargs!r}); {spec.options_path!r} "
                    f"(make_kwargs={spec.make_kwargs!r}) collides — pick a distinct name"
                )
            return  # idempotent: same asset re-curated (e.g. its module was re-imported)
        self._obj_map[spec.name] = spec

    def make(self, name: str, **overrides) -> EntityOptions:
        return self.get(name).make(**overrides)

    # --- population ------------------------------------------------------------ #
    def build(self, modules: tuple[str, ...] | None = None) -> AssetManifest:
        """Import the asset families so their ``@curate`` decorators populate the manifest."""
        for module in modules or ASSET_MODULES:
            importlib.import_module(module)
        return self

    def ensure(self, modules: tuple[str, ...] | None = None) -> AssetManifest:
        """Populate from the asset families if empty (idempotent); the consumer entry point."""
        if not self._obj_map:
            self.build(modules)
        return self

    # --- discovery: the registry describes its own query surface --------------- #
    def names(
        self,
        *,
        category: str | None = None,
        placement: str | None = None,
        tag: str | None = None,
        support_surface: bool | None = None,
        container: bool | None = None,
        movable: bool | None = None,
    ) -> list[str]:
        """Asset names matching every given filter. Filter axes/values come from :meth:`facets`."""
        # string axes treat ""/None as "no filter"; the boolean axes need `is not None` since
        # False is a real filter value.
        out = []
        for name, spec in self:
            if category and not _is_under(spec.category_path, category):
                continue
            if placement and spec.placement != placement:
                continue
            if tag and tag not in spec.tags:
                continue
            if support_surface is not None and spec.support_surface != support_surface:
                continue
            if container is not None and spec.container != container:
                continue
            if movable is not None and spec.movable != movable:
                continue
            out.append(name)
        return sorted(out)

    def search(self, query: str) -> list[str]:
        """Find asset names by free text.

        Case-insensitive substring match over each asset's name, category path, tags, and
        description — the loosest entry point for an agent resolving task language
        ("a mug", "trash") to catalog names. An empty query raises (use :meth:`names` or
        :meth:`summary` to list the whole catalog).
        """
        q = query.strip().lower()
        if not q:
            raise ValueError("search() needs a non-empty query; use names() or summary() to list everything")
        return sorted(
            name for name, spec in self if q in " ".join((name, spec.category, *spec.tags, spec.description)).lower()
        )

    def facets(self) -> dict[str, list]:
        """Describe the available queries: each filter axis :meth:`names` accepts → its present values."""
        cats, places, tags, supports, containers, movables = set(), set(), set(), set(), set(), set()
        for _, spec in self:
            cats.add(spec.category)
            places.add(spec.placement)
            tags.update(spec.tags)
            supports.add(spec.support_surface)
            containers.add(spec.container)
            movables.add(spec.movable)
        return {
            "category": sorted(cats),
            "placement": sorted(places),
            "tag": sorted(tags),
            "support_surface": sorted(supports),
            "container": sorted(containers),
            "movable": sorted(movables),
        }

    def summary(self) -> list[dict]:
        """Flat per-asset records (JSON-ready; handy to dump into an agent's context)."""
        return [{**asdict(spec), "tags": list(spec.tags)} for _, spec in sorted(self, key=lambda kv: kv[0])]

    def tree(self) -> dict:
        """Nested taxonomy: ``{node: {child: {...}, '__assets__': [names]}}`` from category paths."""
        root: dict = {}
        for name, spec in self:
            node = root
            for part in spec.category_path:
                node = node.setdefault(part, {})
            node.setdefault(_LEAF, []).append(name)
        return root


MANIFEST = AssetManifest("asset")


# --------------------------------------------------------------------------- #
# Robot registry
# --------------------------------------------------------------------------- #


class RobotRegistry(Registry):
    """Friendly ``name -> RobotOptions`` subclass lookup for robots.

    A plain :class:`~eden.utils.registry.Registry` that only adds a type guard: every
    entry must be a :class:`~eden.options.entities.RobotOptions` subclass. The check
    lives in :meth:`_do_register`, the single chokepoint both the decorator and the
    direct-call registration forms funnel through, so the inherited ``register()`` API is
    extended (not disabled — unlike :class:`AssetManifest`, which forbids it).

    Robots register themselves in (or alongside) their option modules::

        @ROBOT_REGISTRY.register(name="g1")
        class G1Dofs29(RobotOptions): ...

        ROBOT_REGISTRY.register(FrankaEmikaPanda)   # direct-call form

    Keys default to ``to_snake_case(cls.__name__)``; pass ``name=`` for a cleaner alias.
    The exhaustive :func:`find_entity_options` index discovers this registry too and
    surfaces its entries as ``robot/<key>`` (plus a bare ``<key>`` when unambiguous).
    """

    def _do_register(self, name: str, obj: Any) -> None:
        if not (isinstance(obj, type) and issubclass(obj, RobotOptions)):
            raise TypeError(
                f"{self._name!r} registry accepts RobotOptions subclasses; got {obj!r}. "
                "Register robots (subclasses of eden.options.entities.RobotOptions) only."
            )
        super()._do_register(name, obj)


ROBOT_REGISTRY = RobotRegistry("robot", populated_by=("eden.options.robots",))


# --------------------------------------------------------------------------- #
# Exhaustive auto-scan index (cached) for the CLI entry points
# --------------------------------------------------------------------------- #
#
# ``eden view`` / ``eden snap`` resolve an Options class from a name. Lookups are answered
# from an on-disk index mapping name -> ``"module:Class"`` so that a warm lookup imports only
# the single module defining the requested class. The index is built once by walking the asset
# packages (recursively, so subpackages like ``eden.options.objects.robolab`` are included) and
# is keyed by a fingerprint of the scanned source files — editing, adding, or removing any
# options module rebuilds it on the next lookup; only the first lookup pays the walk cost.
#
# Indexed names, in lookup priority:
#
# * class names — ``RoboLabYcbBanana``, ``YCB011Banana``, ``FrankaEmikaPanda``;
# * family-registry entries (``ARCTIC_OBJECTS_REGISTRY``, ``ROBOLAB_REGISTRY``, ...) as
#   ``<registry>/<key>`` (e.g. ``robolab/ycb/banana``), with the bare ``<key>`` also aliased
#   when it is unambiguous across families.

#: Asset packages scanned for Options classes. Add a package here when a new
#: asset family lives outside these roots.
SCAN_PACKAGES: tuple[str, ...] = ("eden.options.objects", "eden.options.robots")

_CACHE_SCHEMA = 1
_index: dict[str, str] | None = None  # in-process cache: name -> "module:Class"


# --- cache fingerprint / location ------------------------------------------ #


def _package_dirs() -> list[Path]:
    return [Path(p) for pkg in SCAN_PACKAGES for p in importlib.import_module(pkg).__path__]


def _fingerprint() -> str:
    """Hash of (path, size, mtime) of every .py under the scanned packages."""
    h = hashlib.sha256(f"schema={_CACHE_SCHEMA}".encode())
    for pkg_dir in _package_dirs():
        for py in sorted(pkg_dir.rglob("*.py")):
            st = py.stat()
            h.update(f"{py}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def _cache_path() -> Path:
    # One cache per checkout — worktrees scan different sources.
    src_tag = hashlib.sha256(str(Path(__file__).resolve()).encode()).hexdigest()[:12]
    return Path(get_cache_dir(f"entity_options_index-{src_tag}.json"))


# --- index build ----------------------------------------------------------- #


def _build_index() -> dict[str, str]:
    index: dict[str, str] = {}
    registry_entries: list[tuple[str, str, str]] = []  # (registry_name, key, target)

    def scan_module(module) -> None:
        for attr in list(vars(module).values()):
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseEntityOptions)
                and attr.__module__ == module.__name__  # definition site only — skip re-exports
            ):
                index.setdefault(attr.__name__, f"{attr.__module__}:{attr.__name__}")
            elif isinstance(attr, Registry):
                for key, obj in attr:
                    if isinstance(obj, type) and issubclass(obj, BaseEntityOptions):
                        registry_entries.append((attr._name.lower(), key, f"{obj.__module__}:{obj.__name__}"))

    for pkg_name in SCAN_PACKAGES:
        pkg = importlib.import_module(pkg_name)
        scan_module(pkg)
        for info in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg_name}.", onerror=lambda name: None):
            try:
                module = importlib.import_module(info.name)
            except Exception as e:  # noqa: BLE001 — one broken pack must not break discovery
                warnings.warn(f"Skipping {info.name} during entity-options discovery: {e}", stacklevel=2)
                continue
            scan_module(module)

    # Registry keys: always namespaced; bare key only while unambiguous.
    bare: dict[str, str | None] = {}  # key -> target, or None once contested
    for registry_name, key, target in registry_entries:
        index.setdefault(f"{registry_name}/{key}", target)
        bare[key] = None if bare.get(key, target) != target else target
    for key, target in bare.items():
        if target is not None:
            index.setdefault(key, target)

    return index


def _load_cached_index() -> dict[str, str] | None:
    path = _cache_path()
    try:
        payload = json.loads(path.read_text())
        if payload.get("fingerprint") == _fingerprint():
            return payload["index"]
    except (OSError, ValueError, KeyError):
        pass
    return None


def _save_index(index: dict[str, str]) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp-{os.getpid()}")
        tmp.write_text(json.dumps({"fingerprint": _fingerprint(), "index": index}))
        tmp.replace(path)
    except OSError as e:
        warnings.warn(f"Could not write entity-options index cache to {path}: {e}", stacklevel=2)


# --- public API ------------------------------------------------------------ #


def entity_options_index(rebuild: bool = False) -> dict[str, str]:
    """Name -> ``"module:Class"`` map of all discoverable entity Options."""
    global _index
    if rebuild:
        _index = None
    if _index is None:
        _index = None if rebuild else _load_cached_index()
        if _index is None:
            _index = _build_index()
            _save_index(_index)
    return _index


def find_entity_options(name: str) -> type[BaseEntityOptions]:
    """Resolve an Options class by class name (``RoboLabYcbBanana``) or registry
    key (``robolab/ycb/banana``). Warm lookups import only the defining module.
    """
    index = entity_options_index()
    target = index.get(name)
    if target is None:
        close = difflib.get_close_matches(name, list(index), n=5)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise ValueError(f"No entity options found for {name}.{hint}")
    return _resolve(target)
