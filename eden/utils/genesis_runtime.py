"""Genesis runtime helpers — transient kinematics envs for preprocessing.

Shared infrastructure for any Eden preprocessing code that needs to spin up
a Genesis scene just to run forward kinematics or query a Jacobian, without
the full training-loop machinery. Callers include:

* :meth:`eden.datasets.base.DatasetBase._forward_velocity_kinematics` — fills
  in per-link velocity fields after dataset prepare.
* :class:`eden.datasets.modifiers.smooth_start_end.SmoothStartTransition` —
  builds a single-env scene for the smooth-start IK.
* :class:`eden.datasets.modifiers.ground.GroundFirstFrame` — FK of each clip's
  first frame to put the robot on the floor.
* :class:`eden.extensions.teleop.sonic.g1_fk.G1KeyFrameFk` — a scene held open
  for the process's FK queries.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from eden.entities.rigid import RigidEntity
    from eden.envs.base import EnvBase
    from eden.options.entities import EntityOptions


def _ensure_genesis_initialized() -> None:
    """Raise a clear preflight error if ``genesis.init`` hasn't been called.

    ``EnvBase()`` would otherwise fail deep inside Quadrants with a less
    actionable error. Probes the private ``_initialized`` flag because
    Genesis doesn't (yet) expose a public ``is_initialized()`` predicate.
    """
    import genesis as gs

    if not getattr(gs, "_initialized", False):
        raise RuntimeError(
            "Genesis is not initialized. Call ``eden.init(backend=gs.cpu)`` "
            "(or ``gs.init``) before spinning up a transient kinematics env."
        )


@contextmanager
def transient_robot_env(
    robot_options: "EntityOptions",
    *,
    num_envs: int = 1,
    sim_dt: float = 0.005,
) -> "Iterator[tuple[EnvBase, RigidEntity]]":
    """Yield ``(env, robot)`` for a single-shot kinematics scene.

    Destroys the env on exit. Use for preprocessing code that needs FK /
    Jacobian queries but doesn't run physics simulation.

    Parameters
    ----------
    robot_options:
        Eden ``RobotOptions`` (or any ``EntityOptions``) for the robot to
        spawn.
    num_envs:
        Scene env count. Most preprocessing wants 1; ``_forward_velocity_kinematics``
        uses ``len(motion_seq.buffer_length)`` to vectorise over clips.
    sim_dt:
        Doesn't affect kinematics queries, only matters for the constraint
        solver warmup. Set to the clip's dt when relevant, otherwise leave
        at the small default.

    Wall time: ~5–30 s on first call per Python process due to the Quadrants kernel
    compile; subsequent invocations within the same process reuse the
    compile cache.
    """
    _ensure_genesis_initialized()

    # Imports here (not at module scope) because ``EnvBase`` pulls in heavy
    # Genesis machinery — this helper should be cheap to ``from … import``
    # without paying the cost when the consumer doesn't actually need a
    # scene.
    from eden.envs.base import EnvBase
    from eden.options import EnvOptions, SceneOptions

    env = EnvBase(
        env_options=EnvOptions(num_envs=num_envs, sim_dt=sim_dt),
        scene_options=SceneOptions(robot=robot_options),
    )
    env.build()
    try:
        yield env, env.entities["robot"]
    finally:
        env.destroy()


#: The rigid solver's per-step query caches, as ``(owner attribute on the solver, cache attribute on it)``:
#: Genesis answers a repeated query of one step (contacts, equality-constraint info) from these.
_RIGID_QUERY_CACHES = (("collider", "_contact_data_cache"), ("constraint_solver", "_eq_const_info_cache"))


def clear_rigid_query_caches(rigid_solver) -> None:
    """Drop the rigid solver's per-step query caches, so the next query of the step is answered afresh.

    The part of Genesis' ``RigidSolver._restart`` that Eden needs, and only that part: ``_restart`` also re-arms
    every drone's once-per-step propeller guard, which a mid-step call must not do. Genesis exposes no public
    call for this, so the cache attributes are named here, in one place, and a Genesis that renamed one raises
    rather than leaving a cache that would be served stale.
    """
    for owner_name, cache_name in _RIGID_QUERY_CACHES:
        owner = getattr(rigid_solver, owner_name, None)
        if owner is None:  # a solver with nothing of this kind built (no collider in an all-Mochi scene, say)
            continue
        cache = getattr(owner, cache_name, None)
        if not isinstance(cache, dict):
            raise RuntimeError(
                f"Genesis' {type(owner).__name__} has no per-step query cache '{cache_name}' (a dict) any more, "
                f"so eden.utils.genesis_runtime.clear_rigid_query_caches cannot drop it: this Genesis renamed or "
                "restructured it. Update _RIGID_QUERY_CACHES to the new attribute (see RigidSolver._restart in "
                "Genesis); skipping the clear would serve a cached query stale."
            )
        cache.clear()
