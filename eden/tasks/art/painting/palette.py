"""The palette: a tray of paint wells beside the sheet, where the brush is dipped.

The tray lies on the desk to the robot's left of the sheet (the sheet frame's ``-u``), its face level with the
paper's, so a well's depth is read as the paper's is: how far below that face the undeformed tip has gone. The wells
stand in columns of :data:`WELLS_PER_COLUMN`, :data:`WELL_PITCH` apart, the first column :data:`TRAY_GAP` from the
sheet's edge; well ``k`` holds paint ``k`` of the work's palette. A dip is the tip pressed more than ``dip_depth``
into a well, within :data:`WELL_RADIUS` of its centre (``PaintingCopy``); the paint is drawn on the tray's face, and
does not collide. The scene's palette is a work's (:class:`PaletteOptions`), so its wells are the ones a copy of that
work dips in.
"""

from __future__ import annotations

import hashlib

import numpy as np
from pydantic import field_validator, model_validator

from eden.options.entities import EntityOptions
from eden.tasks.art import workcell
from eden.utils.misc import get_cache_dir
from eden.utils.procedural_meshes import write_atomically

WELL_RADIUS = 0.011
WELL_PITCH = 0.03
WELLS_PER_COLUMN = 5
#: From the sheet's left edge to the first column of wells' centres (m).
TRAY_GAP = 0.03
#: How far (m) the tray reaches past the wells' rims.
TRAY_MARGIN = 0.008
#: How deep (m) a plan dips the tip into a well.
DIP_DEPTH = 0.004


def well_centres(count: int, sheet: tuple[float, float]) -> np.ndarray:
    """``(count, 2)``: each well's centre ``(u, v)`` in the sheet frame, for a ``sheet = (width, height)`` sheet.

    Column by column away from the sheet, each column top (``+v``) to bottom, centred on ``v = 0``.
    """
    k = np.arange(count)
    column, row = k // WELLS_PER_COLUMN, k % WELLS_PER_COLUMN
    rows = min(count, WELLS_PER_COLUMN)
    u = -(sheet[0] / 2 + TRAY_GAP + column * WELL_PITCH)
    v = ((rows - 1) / 2 - row) * WELL_PITCH
    return np.stack([u, v], axis=-1).astype(np.float64)


def _to_world(u: float, v: float) -> tuple[float, float]:
    x, y, _ = workcell.SHEET_CENTER
    return x + u * workcell.SHEET_U[0] + v * workcell.SHEET_V[0], y + u * workcell.SHEET_U[1] + v * workcell.SHEET_V[1]


def palette_file(paints: list[tuple[float, float, float]], sheet: tuple[float, float]) -> str:
    """Path of the palette's MJCF for ``paints`` beside a ``sheet``-sized sheet, written to Eden's cache.

    One fixed body: the tray (a box on the desk, its face level with the paper's) and a visual disc of each paint.
    """
    centres = well_centres(len(paints), sheet)
    low, high = centres.min(axis=0) - WELL_RADIUS - TRAY_MARGIN, centres.max(axis=0) + WELL_RADIUS + TRAY_MARGIN
    (cx, cy), half_u, half_v = _to_world(*((low + high) / 2)), (high[0] - low[0]) / 2, (high[1] - low[1]) / 2
    half_z = workcell.PAPER_THICKNESS / 2
    z = workcell.DESK_HEIGHT + half_z
    # The tray's extent along world x is its extent along v, along world y its extent along u.
    geoms = [
        f'      <geom name="tray" type="box" size="{half_v} {half_u} {half_z}" pos="{cx} {cy} {z}" '
        'rgba="0.93 0.93 0.91 1"/>'
    ]
    for k, ((u, v), rgb) in enumerate(zip(centres, paints)):
        x, y = _to_world(u, v)
        r, g, b = rgb
        geoms.append(
            f'      <geom name="well_{k}" type="cylinder" size="{WELL_RADIUS} 0.0003" '
            f'pos="{x} {y} {z + half_z + 0.0003}" contype="0" conaffinity="0" rgba="{r} {g} {b} 1"/>'
        )
    body = "\n".join(geoms)
    text = f"""<mujoco model="palette">
  <compiler angle="radian"/>
  <worldbody>
    <body name="palette">
{body}
    </body>
  </worldbody>
</mujoco>
"""
    tag = hashlib.sha1(text.encode()).hexdigest()[:12]  # one file per palette: envs may build different ones
    path = f"{get_cache_dir('art')}/palette_{tag}.xml"
    write_atomically(path, text)
    return path


def _works():
    """The ``works`` module, imported when first needed: it imports this module (by way of ``references``), so this
    one cannot import it at module scope.
    """
    from eden.tasks.art.painting import works

    return works


class PaletteOptions(EntityOptions):
    """The palette tray of the painting ``work`` (a key of :data:`~eden.tasks.art.painting.works.WORKS`): a well of each
    of its paints beside its sheet. One fixed body whose MJCF is generated (:func:`palette_file`) when the env builds
    the entity. A ``PaintingCopy`` of ``work`` dips in it, and refuses a palette of another work.
    """

    work: str
    file: str = "palette.xml"
    is_fixed_base: bool = True
    default_armature: float = 0.0

    @field_validator("work")
    @classmethod
    def _is_a_painting(cls, work: str) -> str:
        works = _works().WORKS
        if work not in works:
            raise ValueError(f"PaletteOptions: work={work!r} is not a painting; the paintings are {list(works)}.")
        return work

    @model_validator(mode="after")
    def _has_no_pose_of_its_own(self) -> PaletteOptions:
        # The tray's geometry is written beside the workcell's sheet in world coordinates (palette_file), where
        # PaintingCopy reads a dip off the work's wells: a pose of the entity's own would move the tray, not the wells.
        pos, quat = tuple(self.default_root_pos), tuple(self.default_root_quat)
        if pos != (0.0, 0.0, 0.0) or quat != (1.0, 0.0, 0.0, 0.0):
            raise ValueError(
                f"PaletteOptions(work={self.work!r}) got default_root_pos={pos}, default_root_quat={quat}; the tray is "
                "placed beside the workcell's sheet by its own geometry, where the copy reads the dips, so it takes "
                "no pose of its own. Leave default_root_pos and default_root_quat at their defaults."
            )
        return self

    def resolve_runtime_file(self) -> str:
        painting = _works().painting(self.work)
        return palette_file(list(painting.palette), painting.sheet)
