"""CALVIN benchmark object and play-table options."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import ClassVar

from eden.options.entities import EntityOptions, MetadataOptions
from eden.utils.assets import get_asset_path
from eden.utils.misc import get_cache_dir
from eden.utils.registry import Registry


CALVIN_HF_REGISTRY = "Kashu7100"
CALVIN_HF_DATASET = "eden_calvin"
CALVIN_GLOBAL_SCALING = 0.8
# Bump when the sanitizer logic changes so older cached outputs are ignored.
CALVIN_SANITIZED_CACHE_VERSION = "v1"


class CalvinObjectOptions(EntityOptions):
    """Base options for original CALVIN play-table assets."""

    registry: str | None = CALVIN_HF_REGISTRY
    dataset: str | None = CALVIN_HF_DATASET
    assets_subdir: ClassVar[str] = "calvin"
    scale: float = CALVIN_GLOBAL_SCALING
    metadata: ClassVar[MetadataOptions] = MetadataOptions(benchmark="calvin", asset_source="calvin_env")


CALVIN_TABLE_LINKS_TO_KEEP = [
    "base_link",
    "button_link",
    "switch_link",
    "slide_link",
    "drawer_link",
    "led_link",
    "light_link",
    "plank_link",
]


CALVIN_TABLE_DOFS = ["base__slide", "base__drawer", "base__button", "base__switch"]


CALVIN_OBJECT_REGISTRY = Registry("CALVIN_OBJECT")


def _calvin_cache_dir() -> Path:
    """Per-user cache directory for sanitized CALVIN assets."""
    return Path(get_cache_dir("calvin_sanitized", CALVIN_SANITIZED_CACHE_VERSION))


def _strip_obj_texture_coordinates(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        source_path.open("r", encoding="utf-8", errors="ignore") as source_file,
        target_path.open("w", encoding="utf-8", newline="\n") as target_file,
    ):
        for line in source_file:
            if line.startswith("vt "):
                continue
            if line.startswith("f "):
                vertices = [token.split("/", 1)[0] for token in line.split()[1:]]
                target_file.write("f " + " ".join(vertices) + "\n")
                continue
            target_file.write(line)


def _sanitize_calvin_table_urdf(
    relative_path: str, registry: str | None, dataset: str | None, local_dir: str | None
) -> str:
    """Materialize a sanitized copy of the CALVIN table URDF tree.

    Strips ``vt``/face-uv tokens from every OBJ so downstream mesh loaders don't trip
    on the upstream textures. Caching is per-user, atomic-on-rename, and idempotent
    across concurrent DDP ranks: each process stages into a unique sibling directory
    and only the first to finish wins the rename — losers tear their staging down.
    """
    source_path = Path(get_asset_path(file=relative_path, registry=registry, dataset=dataset, local_dir=local_dir))
    table_root = source_path.parents[1]
    cache_root = _calvin_cache_dir() / table_root.name
    target_path = cache_root / "urdf" / source_path.name
    sentinel = cache_root / ".complete"

    if sentinel.exists():
        return str(target_path)

    cache_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{table_root.name}.", dir=cache_root.parent))
    try:
        for src in table_root.rglob("*"):
            if src.is_dir():
                continue
            dst = staging / src.relative_to(table_root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix == ".obj":
                _strip_obj_texture_coordinates(src, dst)
            else:
                shutil.copy2(src, dst)
        (staging / ".complete").touch()
        try:
            staging.rename(cache_root)
        except OSError:
            # Another process won the race; cache_root is already populated.
            shutil.rmtree(staging, ignore_errors=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return str(target_path)


@CALVIN_OBJECT_REGISTRY.register(name="table")
class CalvinPlayTable(CalvinObjectOptions):
    """Original CALVIN play table D with drawer, slider, button, switch, and lights."""

    file: str = "calvin_table_D/urdf/calvin_table_D.urdf"
    is_fixed_base: bool = True
    is_articulated: bool = True
    links_to_keep: list[str] = CALVIN_TABLE_LINKS_TO_KEEP
    # `base_link` is concave (tabletop with cabinet/legs); Eden's articulated default
    # `decompose_robot_error_threshold=inf` collapses it into one convex hull whose top
    # protrudes above the visual tabletop, so blocks placed at the upstream surface z
    # end up wedged inside the collision geometry. Force a tight decomposition.
    decompose_robot_error_threshold: float = 0.0
    coacd_threshold: float = 0.03
    dofs_name: list[str] = CALVIN_TABLE_DOFS
    default_dofs_pos: dict[str, float] = {dof: 0.0 for dof in CALVIN_TABLE_DOFS}

    def model_post_init(self, context):
        super().model_post_init(context)
        self.file = _sanitize_calvin_table_urdf(
            self.file,
            registry=self.registry,
            dataset=self.dataset,
            local_dir=self.resolve_local_dir(),
        )


@CALVIN_OBJECT_REGISTRY.register(name="block_red")
class CalvinRedBlock(CalvinObjectOptions):
    file: str = "blocks/block_red_middle.urdf"


@CALVIN_OBJECT_REGISTRY.register(name="block_blue")
class CalvinBlueBlock(CalvinObjectOptions):
    file: str = "blocks/block_blue_small.urdf"


@CALVIN_OBJECT_REGISTRY.register(name="block_pink")
class CalvinPinkBlock(CalvinObjectOptions):
    file: str = "blocks/block_pink_big.urdf"


def get_calvin_object_options(object_name: str) -> type[EntityOptions]:
    return CALVIN_OBJECT_REGISTRY.get(object_name)


__all__ = [
    "CALVIN_GLOBAL_SCALING",
    "CALVIN_HF_DATASET",
    "CALVIN_HF_REGISTRY",
    "CALVIN_OBJECT_REGISTRY",
    "CALVIN_TABLE_DOFS",
    "CALVIN_TABLE_LINKS_TO_KEEP",
    "CalvinBlueBlock",
    "CalvinObjectOptions",
    "CalvinPinkBlock",
    "CalvinPlayTable",
    "CalvinRedBlock",
    "get_calvin_object_options",
]
