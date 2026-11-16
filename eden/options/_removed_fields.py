"""Fields an options class has lost, each with what became of it, for the error that rejects a config still setting one.

A config saved before a Genesis bump carries every field its options had then, set or left at its default, so it names a
field the bump removed even when nobody set it. Loading it raises (:func:`eden.options.options.unknown_field_message`
words the error, for Eden options and for the Genesis options a config embeds): the field configured physics the
installed Genesis no longer has, so the run would not be the one that was saved. This table lets that error say what
became of the field rather than only that it is unknown. An entry stays as long as a saved config may still carry it.

Keyed by ``(class name, field)``: the Genesis classes keep their names across the modules that hold them.
"""

_DFSPH_ONLY = "genesis-world#3778 made DFSPH the only SPH solver, removing WCSPH"
_SAP_ONE_CONTACT_MODEL = (
    "genesis-world#3794 gave the SAP coupler one contact model: every kind of pair collides and the floor touches the"
    " vertices of FEM entities and rigid bodies, so no option switches a kind of contact off or picks its model"
)

REMOVED_FIELDS: dict[tuple[str, str], str] = {
    ("SPHLiquidMaterialOptions", "stiffness"): f"{_DFSPH_ONLY}, whose equation of state this set; nothing replaces it",
    ("SPHLiquidMaterialOptions", "exponent"): f"{_DFSPH_ONLY}, whose equation of state this set; nothing replaces it",
    ("SPHLiquidMaterialOptions", "gamma"): (
        "genesis-world#3755 replaced this unitless cohesion with 'surface_tension', in N/m (default 0.0728, water); a"
        " gamma value does not convert to it"
    ),
    ("SPHOptions", "pressure_solver"): f"{_DFSPH_ONLY}; every SPH scene now runs DFSPH",
    ("FEMOptions", "enable_vertex_constraints"): (
        "genesis-world#3807 made vertex constraints always available; 'max_vertex_constraints' bounds how many"
    ),
    ("SAPCouplerOptions", "enable_fem_self_tet_contact"): (
        "genesis-world#3794 renamed it 'enable_fem_self_collision', which leaves distinct FEM entities colliding"
        " whatever its value"
    ),
    ("SAPCouplerOptions", "point_contact_stiffness"): (
        "genesis-world#3794 derives the floor stiffness of a FEM entity from its material, bounded by"
        " 'hydroelastic_stiffness'"
    ),
    ("SAPCouplerOptions", "fem_floor_contact_type"): _SAP_ONE_CONTACT_MODEL,
    ("SAPCouplerOptions", "rigid_floor_contact_type"): _SAP_ONE_CONTACT_MODEL,
    ("SAPCouplerOptions", "rigid_rigid_contact_type"): _SAP_ONE_CONTACT_MODEL,
    ("SAPCouplerOptions", "enable_rigid_fem_contact"): _SAP_ONE_CONTACT_MODEL,
}
