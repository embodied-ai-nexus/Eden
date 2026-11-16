"""Fields a Genesis options class no longer has, each mapped to how to migrate a value written for it.

The Genesis counterpart of :attr:`eden.options.options.ConfigurableOptions._retired_fields`, which Eden's own options
declare where they are defined: a Genesis class cannot carry Eden's notes, so they live here, keyed by the class's
qualified name. A config saved before a Genesis upgrade carries every field its options had then, set or left at its
default, so it names a field the upgrade retired even when nobody set it. Loading it raises whether or not the field is
listed here (``eden.utils.configs._safe_model_construct``); a note only tells the error what became of the field, as a
retired term parameter's does (``ConfigurableMixin._retired_params``). An entry can go once configs saved before its
upgrade are no longer loaded.
"""

from collections.abc import Mapping
from types import MappingProxyType

_DFSPH_ONLY = "Genesis solves SPH with DFSPH alone now"
_SAP_ONE_CONTACT_MODEL = (
    "the SAP coupler has one contact model now: every kind of pair collides, and the floor touches the vertices of FEM"
    " entities and rigid bodies; remove from the config"
)

GENESIS_RETIRED_FIELDS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        # genesis-world#3807
        "genesis.options.solvers.FEMOptions": {
            "enable_vertex_constraints": "vertex constraints are always available now, up to 'max_vertex_constraints'"
            "; remove from the config",
        },
        # genesis-world#3778
        "genesis.options.solvers.SPHOptions": {"pressure_solver": f"{_DFSPH_ONLY}; remove from the config"},
        # genesis-world#3794
        "genesis.options.solvers.SAPCouplerOptions": {
            "enable_fem_self_tet_contact": "renamed 'enable_fem_self_collision', which leaves distinct FEM entities"
            " colliding whatever its value; set that instead",
            "point_contact_stiffness": "the floor stiffness of a FEM entity follows its material now, bounded by"
            " 'hydroelastic_stiffness'; remove from the config",
            "fem_floor_contact_type": _SAP_ONE_CONTACT_MODEL,
            "rigid_floor_contact_type": _SAP_ONE_CONTACT_MODEL,
            "rigid_rigid_contact_type": _SAP_ONE_CONTACT_MODEL,
            "enable_rigid_fem_contact": _SAP_ONE_CONTACT_MODEL,
        },
    }
)


def retired_fields(cls: type) -> Mapping[str, str]:
    """The retired fields of ``cls`` and their migration notes: the notes of the Genesis classes in its MRO (a subclass
    inherits its base's, and its own note for a name wins), under an Eden option's own declaration.
    """
    retired: dict[str, str] = {}
    for base in reversed(cls.__mro__):
        retired.update(GENESIS_RETIRED_FIELDS.get(f"{base.__module__}.{base.__qualname__}", {}))
    return {**retired, **getattr(cls, "_retired_fields", {})}
