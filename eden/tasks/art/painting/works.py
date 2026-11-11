"""The paintings the painting tasks copy (:data:`WORKS`), and how a planned one is planned and kept.

Works, on an 18 cm sheet:

* ``sunset`` — a sea, a sky and a low sun in 3 paints and 7 strokes, authored here as a
  :class:`~eden.tasks.art.painting.references.Painting`: the dips and the brush sizes of a painting with none of its
  length.
* ``fuji_study`` — a study of Hokusai's *Fine Wind, Clear Morning*: 5 paints and two layers, 16 and 9.4 mm brushes.
* ``fuji`` — the painting itself, as RoboPaint's acrylic works are painted (``acrylic_redfuji`` is one): 10 paints
  and four layers, RoboPaint's 16, 9.4, 5.2 and 3 mm brushes.

The Hokusai works are :class:`PlannedPainting` recipes: the Metropolitan Museum of Art's scan of the woodblock print
(JP2960, which the Met titles *South Wind, Clear Sky*; CC0), in the :data:`PICTURES_REPO` dataset at
:data:`PICTURES_REVISION`, cropped to a square round the mountain, and what the planner
(:mod:`~eden.tasks.art.painting.planner`) is given. A recipe is planned the first time it is asked for, on the CPU
(some 15 s; one process plans, others wait for it), and the plan is kept in Eden's cache, named for the recipe, the
picture and its ``plan_sha256``, the digest of the plan as published. So a work stays as it was planned while the
code changes, and is planned again when its published plan is re-pinned. A plan made on another machine is checked
against the digest, and ``tests/tasks/art/painting/test_works.py`` re-plans each work to check that the code still
plans it.

:func:`painting` is a work by name, authored or planned.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from eden.tasks.art.painting.planner import ACRYLIC_LAYERS, Layer, load_picture, plan_painting
from eden.tasks.art.painting.references import Painting, load_painting, save_painting, sweep
from eden.utils.misc import get_cache_dir
from eden.utils.warning import warn_once

#: The Hugging Face dataset the planned works' pictures are in (its README credits each), and the revision read.
PICTURES_REPO = "Kashu7100/eden_art"
PICTURES_REVISION = "c516699e62a728632ac18eb9f42d26a87393b115"


def fetch_picture(picture: str) -> str:
    """The local path of ``picture``: itself if it is an absolute path (a picture of one's own), otherwise a file of
    :data:`PICTURES_REPO` at :data:`PICTURES_REVISION`, downloaded once.

    Unlike ``eden.utils.assets.get_asset_path``, a relative path is always a file of the dataset, even where a file of
    that name exists locally: a recipe names the same picture wherever it runs.
    """
    if os.path.isabs(picture):
        return picture
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(PICTURES_REPO, picture, repo_type="dataset", revision=PICTURES_REVISION)
    except Exception as error:
        raise RuntimeError(
            f"Could not fetch the picture {picture!r} of the Hugging Face dataset {PICTURES_REPO} at revision "
            f"{PICTURES_REVISION[:12]} ({type(error).__name__}: {error}). A planned painting work needs its picture "
            "once, to plan; its plan is then kept under Eden's cache "
            "(~/.cache/eden/art/plans/) and the picture under Hugging Face's. Fetch it once with network access "
            "(HF_HUB_OFFLINE unset), or give the recipe an absolute path to a local copy."
        ) from error


@cache
def _sha256_of_version(path: str, mtime_ns: int, size: int) -> str:
    """The SHA-256 of the file at ``path``, once per version of it (its modification time and size)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256(path: str) -> str:
    """The SHA-256 of the file at ``path``, hashed again only when the file changes."""
    stat = os.stat(path)
    return _sha256_of_version(path, stat.st_mtime_ns, stat.st_size)


@dataclass(frozen=True)
class PlannedPainting:
    """A work planned from a picture: what :func:`~eden.tasks.art.painting.planner.plan_painting` is given, and the
    work's ``rule`` and ``about`` (as a :class:`~eden.tasks.art.painting.references.Painting`'s).

    ``picture`` is :func:`fetch_picture`'s, and ``crop`` the box of it the sheet shows
    (:func:`~eden.tasks.art.painting.planner.load_picture`). ``plan_sha256`` is the SHA-256 of the plan's file as it
    was published (``None`` for a work of one's own): a plan made otherwise is used, with a warning.
    """

    name: str
    title: str
    picture: str
    crop: tuple[int, int, int, int]
    paints: int
    layers: tuple[Layer, ...] = ACRYLIC_LAYERS
    sheet: tuple[float, float] = (0.18, 0.18)
    cell: float = 0.0005
    plan_sha256: str | None = None
    rule: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}), hash=False)
    about: str = field(default="", compare=False)

    def cache_path(self) -> str:
        """Where the plan is kept: named for the recipe, the picture and ``plan_sha256``. Another recipe or picture is
        planned afresh, and so is a work whose published plan was re-pinned; a change to the code alone is not (the
        work stays as it was planned).

        A picture of the dataset is named by its path and the pinned revision, which fix its content, so a kept plan
        is found without fetching the picture (offline, or where only Eden's cache was copied); a local picture is
        named by its content.
        """
        recipe = repr(
            (self.name, self.title, self.crop, self.paints, self.layers, self.sheet, self.cell, self.plan_sha256)
        )
        if os.path.isabs(self.picture):
            picture = _sha256(self.picture)
        else:
            picture = f"{PICTURES_REPO}@{PICTURES_REVISION}/{self.picture}"
        key = hashlib.sha256(f"{recipe} {picture}".encode()).hexdigest()[:16]
        return get_cache_dir("art", "plans", f"{self.name}-{key}.json")

    def painting(self) -> Painting:
        """The planned :class:`~eden.tasks.art.painting.references.Painting`: read from the cache, or planned (and
        kept) the first time.
        """
        from filelock import FileLock

        path = self.cache_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with FileLock(f"{path}.lock"):  # one process plans (DDP ranks, test workers); the rest wait and read it
            if not os.path.exists(path):
                picture = load_picture(fetch_picture(self.picture), self.sheet, self.cell, self.crop)
                work = plan_painting(
                    picture,
                    name=self.name,
                    title=self.title,
                    paints=self.paints,
                    layers=self.layers,
                    sheet=self.sheet,
                    cell=self.cell,
                )
                save_painting(work, path)
        planned = _sha256(path)
        if self.plan_sha256 is not None and planned != self.plan_sha256:
            warn_once(
                f"PlannedPainting {self.name!r}: the plan made here ({path}, SHA-256 {planned[:16]}...) is not the "
                f"published one (plan_sha256 {self.plan_sha256[:16]}...), so this task's target and stroke count "
                "differ from the published task's. Either this machine plans differently (the planner runs on the "
                "CPU in float32, and another CPU or torch version can split a stroke otherwise), or the planner or "
                "paint model has changed since the plan was pinned. Re-planning here gives the same plan, so "
                "deleting the kept file does not help. A deliberate change is published by setting plan_sha256 to "
                "the new digest (tests/tasks/art/painting/test_works.py's re-planning test prints it) and updating "
                "the documented counts."
            )
        return load_painting(path, rule=self.rule, about=self.about)


_SKY, _SEA, _SUN = 0, 1, 2

SUNSET = Painting(
    "sunset",
    "A low sun over the sea",
    palette=((0.93, 0.55, 0.30), (0.12, 0.27, 0.45), (0.98, 0.86, 0.35)),
    strokes=(
        # The sky, left to right and back, 16 mm sweeps 12 mm apart.
        sweep("sky 1", _SKY, 0.036, -0.05, 0.05, 0.016),
        sweep("sky 2", _SKY, 0.024, 0.05, -0.05, 0.016),
        sweep("sky 3", _SKY, 0.012, -0.05, 0.05, 0.016),
        # The sea under it.
        sweep("sea 1", _SEA, -0.006, 0.05, -0.05, 0.016),
        sweep("sea 2", _SEA, -0.018, -0.05, 0.05, 0.016),
        sweep("sea 3", _SEA, -0.030, 0.05, -0.05, 0.016),
        # The sun on the horizon, one short sweep.
        sweep("sun", _SUN, 0.008, 0.012, 0.026, 0.016),
    ),
    # A small work on a bare sheet: RoboPaint's mean ΔE of 2 is met before the sun is painted (the sun is a twentieth
    # of the crop the colour is read in). The scripted painter's finished copy is at 0.56.
    rule=MappingProxyType({"de_mean": 1.0}),
    about="a sea, a sky and a low sun: 3 paints, 7 strokes",
)

#: Hokusai's print in the pictures' dataset, and the square of it round the mountain (inside the print's margins).
RED_FUJI = "painting/hokusai_red_fuji.jpg"
RED_FUJI_CROP = (1294, 73, 3832, 2611)

FUJI_STUDY = PlannedPainting(
    "fuji_study",
    "Fine Wind, Clear Morning: a study (after Hokusai)",
    picture=RED_FUJI,
    crop=RED_FUJI_CROP,
    paints=5,
    layers=ACRYLIC_LAYERS[:2],
    plan_sha256="bccf00df03c93a6b2adf49b5f470b3a08acf6e0043c29783d8e19bd62b313178",
    about="a study of Hokusai's Fine Wind, Clear Morning: 5 paints, two layers (16 and 9.4 mm)",
)

FUJI = PlannedPainting(
    "fuji",
    "Fine Wind, Clear Morning (after Hokusai)",
    picture=RED_FUJI,
    crop=RED_FUJI_CROP,
    paints=10,
    plan_sha256="b03b1ab813fc8758b62cf9bb97779c05b43cb2945c2a06211c95048fb957e000",
    about="Hokusai's Fine Wind, Clear Morning: 10 paints, four layers (16, 9.4, 5.2 and 3 mm)",
)

#: The works by name. Each has its sheet and ``rule`` as it is (all a config needs), and the painting itself from
#: ``painting()`` (:func:`painting`), planned for a planned work.
WORKS: dict[str, Painting | PlannedPainting] = {work.name: work for work in (SUNSET, FUJI_STUDY, FUJI)}
#: :data:`WORKS`' keys, for a config field's type (``PaintingCopy.work``); ``tests/tasks/art/painting/test_works.py``
#: keeps the two the same.
WorkName = Literal["sunset", "fuji_study", "fuji"]


@cache
def painting(name: str) -> Painting:
    """The work ``name`` (a key of :data:`WORKS`) as a :class:`~eden.tasks.art.painting.references.Painting`, made
    once per process.
    """
    return WORKS[name].painting()
