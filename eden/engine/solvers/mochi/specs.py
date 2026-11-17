"""Entity specs of the bodies the MochiSolver simulates, the Mochi counterparts of ``gs.entities.Rigid`` and friends.

Genesis creates every entity from a spec, whose class decides the solver that simulates it (``Simulator.add_entity``
looks the class up along its MRO). :func:`~eden.engine.solvers.mochi.hosting.install_mochi_solver` maps both classes
below to the MochiSolver of the scene, so a spec of either reaches it through ``Simulator.add_entity``, while a
``gs.entities.Rigid`` still reaches the RigidSolver.
"""

from collections.abc import Iterable
from typing import Any, NoReturn, Self

import genesis as gs
from genesis.options.morphs import Morph
from genesis.options.patches import Patch
from genesis.options.surfaces import Surface
from pydantic import Field, model_validator

from eden.engine.solvers.mochi import materials as mochi_materials


def _refuse_morph_sequence(cls: type, data: Any) -> Any:
    """Refuse a sequence of morphs, a heterogeneous entity, which the MochiSolver does not simulate.

    Ahead of Genesis' own check, whose message names the class as ``gs.entities.<name>``, which a Mochi spec is not.
    """
    morph = data.get("morph") if isinstance(data, dict) else None
    if isinstance(morph, (tuple, list)):
        gs.raise_exception(
            f"The MochiSolver simulates no heterogeneous entity: pass {cls.__name__} one morph, not a sequence of "
            f"{len(morph)}."
        )
    return data


class MochiRigidSpec(gs.entities.Rigid):
    """Spec of a rigid or articulated body simulated by the MochiSolver.

    A ``gs.entities.Rigid`` whose material is a ``MochiRigid``, of one morph: the MochiSolver simulates no
    heterogeneous entity. Contact is a property of the material (``MochiRigid``), not of the geoms, so no option is set
    on a selection of links or geoms; ``MochiEntity.set_contact_params`` changes it per link once the scene is built.

    Only a scene :func:`~eden.engine.solvers.mochi.hosting.install_mochi_solver` was called on knows it: elsewhere it is
    the ``gs.entities.Rigid`` it derives from, which the RigidSolver simulates. Genesis offers no hook to refuse it there
    short of patching its classes, which the hosting never does; Eden's scenes install the solver themselves.
    """

    _admits_variants = False

    material: mochi_materials.MochiRigid = Field(default_factory=mochi_materials.MochiRigid)

    @model_validator(mode="before")
    @classmethod
    def _check_one_morph(cls, data: Any) -> Any:
        return _refuse_morph_sequence(cls, data)

    @model_validator(mode="after")
    def _check_visualize_contact(self) -> Self:
        if self.visualize_contact:
            gs.raise_exception(
                "`visualize_contact` draws the RigidSolver's contacts only; a Mochi entity has none to draw. Read its "
                "contacts with `MochiEntity.get_contacts` instead."
            )
        return self

    def select(self, selector: Any) -> NoReturn:
        gs.raise_exception(
            "The MochiSolver takes the contact parameters of its bodies from their material, not from options set on "
            "links or geoms: set them on `MochiRigid`, or per link with `MochiEntity.set_contact_params` once the "
            "scene is built."
        )

    def _check_patch(self, patch: Patch) -> None:
        self.select(None)


class MochiSoftSpec(gs.entities.Spec["MochiSoftEntity"]):
    """Spec of a deformable body simulated by the MochiSolver: a tetrahedral solid, a thin shell or a rod.

    Parameters
    ----------
    material : Elastic | Shell | Rod
        The material of the body (``eden.engine.solvers.mochi.materials``). A ``Rod`` takes a ``Rod`` morph
        (``eden.engine.solvers.mochi.morphs``), the other two a ``Box``, ``Sphere``, ``Cylinder`` or ``Mesh``.
    """

    material: mochi_materials.Elastic | mochi_materials.Shell | mochi_materials.Rod

    @model_validator(mode="before")
    @classmethod
    def _check_one_morph(cls, data: Any) -> Any:
        return _refuse_morph_sequence(cls, data)

    # Genesis checks that a material belongs to the solver of the spec through a table of its own solvers, which holds
    # none of the Mochi materials and would refuse them all. The field above admits only the MochiSolver's.
    @model_validator(mode="before")
    @classmethod
    def _check_material_solver(cls, data: Any) -> Any:
        return data


def mochi_spec(
    morph: Morph | tuple[Morph, ...],
    material: mochi_materials.Base,
    surface: Surface | None = None,
    visualize_contact: bool | None = None,
    name: str | None = None,
) -> MochiRigidSpec | MochiSoftSpec:
    """The Mochi spec the individual arguments of ``Scene.add_entity`` stand for, as ``get_entity_spec`` of
    ``genesis.options.entities`` returns the spec of a Genesis material.
    """
    spec_cls = MochiRigidSpec if isinstance(material, mochi_materials.MochiRigid) else MochiSoftSpec
    fields = {"morph": morph, "material": material, "surface": surface, "name": name}
    if visualize_contact:
        if spec_cls is MochiSoftSpec:
            gs.raise_exception(
                "`visualize_contact` draws the RigidSolver's contacts only; a Mochi entity has none to draw."
            )
        # `MochiRigidSpec` refuses it with the reason
        fields["visualize_contact"] = visualize_contact
    if isinstance(morph, Iterable) and not isinstance(morph, Morph):
        fields["morph"] = tuple(morph)  # refused by the spec, with the reason
    return spec_cls(**{field: value for field, value in fields.items() if value is not None})
