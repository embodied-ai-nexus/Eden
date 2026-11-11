"""The MochiSolver, hosted by Eden on stock Genesis.

A fully implicit multi-physics solver: every substep solves one nonlinear system over the generalized coordinates of
the rigid and articulated bodies and the vertex positions of the deformable ones (tetrahedral solids, thin shells,
discrete elastic rods), with a smooth signed-distance penalty contact model.

**Provenance.** Ported from ``Kashu7100/Genesis@58d893fa98fdba078a98adec7c25078432e12c68`` (branch
``feat/mochi-solver``, upstream Genesis 50a2aa19 plus the solver), itself derived in part from Meta Platforms' "mochi"
physics library (Apache-2.0; the derived modules say so in their headers). The modules under
``genesis/engine/solvers/mochi``, ``genesis/engine/entities/mochi_entity`` and ``genesis/engine/materials/Mochi`` are
copied with their imports made absolute and pointed here. What the fork added to Genesis itself lives in this package
instead, or in :mod:`~eden.engine.solvers.mochi.hosting`:

- ``MochiOptions`` (:mod:`.options`), the ``Rod`` morph (:mod:`.morphs`), the solver and entity states
  (:mod:`.states`), the solver's error codes (:mod:`.error_codes`), a trilinear SDF value-and-gradient helper
  (:mod:`.sdf`) and the embedded render-mesh fields of a visual geom (:mod:`.vis_geom`);
- a ``kernel_update_geoms`` that places the geoms of a root after an articulated body (:mod:`.kinematics`; the upstream
  kernel passes a root's rank where it takes the root's link);
- in ``MochiSolver``: registering its error flags with the base solver and clearing them when an environment's state
  or configuration is set, recording the control force each step applied (``dofs.qf_applied``, as the rigid solver
  does), and raising the kinematic kernels' parallelization level;
- the serializer fix of https://github.com/Kashu7100/Genesis/pull/31, as the class name ``MochiRigid``
  (:mod:`.materials.rigid`);
- in :mod:`.hosting`: constructing the solver for a scene, adding its entities, drawing them with the rasterizer,
  resetting a scene of deformable bodies alone, and freeing the solver's arrays with its scene.

Left out: ``gs.init(cpu_threads=...)`` (set ``QD_NUM_THREADS`` instead), the interactive viewer's wait for a starved
draw thread (a generic viewer fix, not a Mochi one), and the RayTracer path (refused by
:func:`~eden.engine.solvers.mochi.hosting.install_mochi_solver`).
"""

import importlib
from typing import Any

# Re-exported lazily (PEP 562): `eden.options` imports the light `materials`, `morphs` and `options` modules at its
# own import, and must not load the solver's kernels with them.
_LAZY_NAMES = {
    "MochiEntity": "eden.engine.solvers.mochi.mochi_entity",
    "MochiOptions": "eden.engine.solvers.mochi.options",
    "MochiSoftEntity": "eden.engine.solvers.mochi.mochi_soft_entity",
    "MochiSolver": "eden.engine.solvers.mochi.mochi_solver",
    "Rod": "eden.engine.solvers.mochi.morphs",
    "add_mochi_entity": "eden.engine.solvers.mochi.hosting",
    "create_mochi_scene": "eden.engine.solvers.mochi.hosting",
    "find_mochi_solver": "eden.engine.solvers.mochi.hosting",
    "install_mochi_solver": "eden.engine.solvers.mochi.hosting",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_NAMES.get(name)
    if module_path is None:
        # A submodule (`from eden.engine.solvers.mochi import materials`) is imported by the import system after this.
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_NAMES})


__all__ = sorted(_LAZY_NAMES)
