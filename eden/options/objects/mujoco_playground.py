"""Object options for assets vendored from ``google-deepmind/mujoco_playground`` (Apache-2.0).

One so far: the DEXTREME reorientation cube that its LEAP-hand in-hand tasks manipulate, which
is also the cube ``wuji-technology/wuji-mjlab`` reorients -- the two ship byte-identical files.
The mesh and texture are used unmodified; the MJCF wrapping them is Eden's, and the provenance
of each file is recorded next to the asset in ``objects/reorient_cube/ATTRIBUTION.md``.
"""

from __future__ import annotations

from eden.options.entities import EntityOptions


class DexCube(EntityOptions):
    """The DEXTREME reorientation cube: 54 mm, 120 g, a different letter on each face.

    The standard prop for in-hand reorientation. Its faces are individually identifiable,
    which is what makes an orientation legible in a render -- and is also why the asset is an
    MJCF rather than a :class:`~eden.options.entities.BoxOptions`. Genesis builds a box
    primitive from an 8-vertex trimesh carrying no UVs, so a box geom cannot hold a face
    texture at all; the asset instead pairs a textured, massless, contact-free visual mesh
    with an ordinary collision box, leaving the contact model box-box.

    Size and mass are the asset's. ``recompute_inertia`` is left ``False`` on purpose: turning
    it on discards the MJCF's explicit 120 g and derives a mass from the collision volume and
    ``material.rho`` instead. A material is still the place to set *friction* -- the reorient
    task authors the cube low-friction because Genesis resolves a contact pair as ``max(a, b)``.

    Used by ``eden/tasks/reorient``; ``eden view DexCube`` renders it standalone.
    """

    registry: str = "Kashu7100"
    dataset: str = "eden_assets"
    file: str = "objects/reorient_cube/reorient_cube.xml"
    is_fixed_base: bool = False
    is_articulated: bool = False
    is_actuated: bool = False
