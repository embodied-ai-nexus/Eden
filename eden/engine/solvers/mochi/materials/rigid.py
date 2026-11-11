from typing import TYPE_CHECKING, Any, Literal

import genesis as gs
from genesis.engine.materials.rigid import Rigid as RigidMaterial
from genesis.typing import PositiveFloat
from pydantic import StrictBool

from eden.engine.solvers.mochi.materials.base import Base

if TYPE_CHECKING:
    pass

ColliderType = Literal["auto", "plane", "sphere", "box", "sdf", "none"]


# Named apart from `gs.materials.Rigid`, which it extends: Genesis' scene serializer names a material by its bare class
# name and resolves that name across every subclass, so a second 'Rigid' would load back in place of
# `gs.materials.Rigid` and break export/load of every rigid scene in a process that imported this module. The Genesis
# fork the solver was ported from fixes the serializer instead (https://github.com/Kashu7100/Genesis/pull/31); `Rigid`
# below keeps the fork's spelling.
class MochiRigid(Base["MochiEntity"], RigidMaterial):
    """
    Rigid body material simulated by the MochiSolver.

    Parameters
    ----------
    use_visual_raycasting : bool, optional
        See Kinematic. Default is False.
    rho : float, optional
        Density in kg/m^3 used to derive the mass and inertia from the collision geometry when the asset specifies
        none. Overrides per-geometry densities found in the asset. Default is 1000.
    friction, penalty_coefficient, penalty_smoothing_half_distance, penalty_threshold, friction_falloff_vel,
    viscous_friction, normal_viscous_damping, max_alignment_normals, has_gravity, contact_layer : optional
        Contact parameters, see `Base`.
    collider_type : str, optional
        Representation of this body used when other bodies' sample points collide against it: "plane", "sphere" and
        "box" are exact analytic distance fields, "sdf" is the precomputed grid of the collision mesh, "none" makes the
        body collide only through its own sample points (it never acts as a collider), and "auto" selects the analytic
        field for plane, sphere and box primitives and the grid otherwise. An analytic field requires the matching
        primitive. Default is "auto".
    sdf_cell_size : float, optional
        Cell size in SDF grid in meters. Contact resolves the penalty ramp against this grid, so the cell should stay
        well below the ramp width. Default is 0.0025.
    sdf_min_res : int, optional
        Minimum resolution of the SDF grid. Must be at least 16. Default is 32.
    sdf_max_res : int, optional
        Maximum resolution of the SDF grid. Must be >= sdf_min_res. Default is 128.
    """

    rho: PositiveFloat = 1000.0
    collider_type: ColliderType = "auto"
    needs_coup: StrictBool = False
    sdf_cell_size: PositiveFloat = 2.5e-3

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.coup_type is not None or self.coup_links is not None or self.coup_collision_links is not None:
            gs.raise_exception("IPC coupling fields are not supported by Mochi materials.")
        if self.gravity_compensation != 0.0:
            gs.raise_exception("Use `has_gravity` instead of `gravity_compensation` for Mochi materials.")
        if self.needs_coup:
            gs.raise_exception("Mochi materials handle contact internally; `needs_coup` must be False.")


Rigid = MochiRigid
