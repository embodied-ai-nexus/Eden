"""OMPL-based motion planner."""

from __future__ import annotations
from typing import TYPE_CHECKING
import math

import genesis as gs
from genesis.utils import tensor_to_array
import torch
import numpy as np

import eden as en
from eden.extensions.motion_planning import (
    MotionPlannerBase,
    MOTION_PLANNER_REGISTRY,
)
from eden.options.extensions.motion_planning import PlannerOptions

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


IS_OMPL_AVAILABLE = True
try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    IS_OMPL_AVAILABLE = False


#: Which generation of OMPL's Python bindings is installed; pip offers both. 1.x (Boost.Python) wraps a callback in a
#: type of its own (``ob.StateValidityCheckerFn``, ...) and takes a start or goal as a scoped ``ob.State(space)``. 2.x
#: (nanobind) dropped both: it takes the Python callable itself, and a state the space allocates.
OMPL_BOOST_PYTHON = IS_OMPL_AVAILABLE and hasattr(ob, "StateValidityCheckerFn")


def _bare_callable(fn):
    return fn


# How the installed bindings take each callback, looked up once here rather than by name at the call site.
if OMPL_BOOST_PYTHON:
    _as_validity_checker = ob.StateValidityCheckerFn
    _as_sampler_allocator = ob.ValidStateSamplerAllocator
else:
    _as_validity_checker = _as_sampler_allocator = _bare_callable


def _ompl_state(space, values: np.ndarray):
    """A state of ``space`` holding ``values``, as the installed bindings take a start or goal state.

    Under 2.x the allocation is never fully released: the bindings expose no ``freeState``, and dropping the Python
    object frees the state but not its value array (8 bytes per DOF, measured with ompl 2.0.1): two small arrays per
    ``plan_motion`` call, which only OMPL's bindings can fix.
    """
    state = ob.State(space) if OMPL_BOOST_PYTHON else space.allocState()
    for i, value in enumerate(values):
        state[i] = float(value)
    return state


def _ompl_planner_names() -> list[str]:
    return sorted(
        name for name in dir(og) if isinstance(cls := getattr(og, name), type) and issubclass(cls, ob.Planner)
    )


def _ompl_planner_class(name: str) -> type:
    """The ``ompl.geometric`` planner class called ``name``, or a ``ValueError`` listing the ones there are."""
    cls = getattr(og, name, None)
    if isinstance(cls, type) and issubclass(cls, ob.Planner):
        return cls
    raise ValueError(
        f"planner={name!r} is not an ompl.geometric planner in the installed OMPL. "
        f'Pass one of: {", ".join(_ompl_planner_names())} (the default is planner="RRTConnect").'
    )


@MOTION_PLANNER_REGISTRY.register()
class OMPLPlanner(MotionPlannerBase):
    """Motion planner backed by OMPL.

    Plans in the entity's full ``qpos`` space for one environment at a time, checking collisions
    by setting the entity's ``qpos`` in the scene.

    Parameters
    ----------
    env : EnvBase
        The environment instance.
    options : PlannerOptions
        From ``OMPLPlanner.configure()``.
    """

    def __init__(self, env: EnvBase, options: PlannerOptions) -> None:
        super().__init__(env, options)

        if not IS_OMPL_AVAILABLE:
            raise ImportError(
                "OMPLPlanner needs OMPL's Python bindings, which are not installed. Install them with "
                "`pip install ompl` (OMPL 1.7 and 2.x both work). Note that OMPL is not distributed on Windows OS."
            )

    def plan_motion(
        self,
        qpos_goal: torch.Tensor,
        qpos_start: torch.Tensor | None = None,
        resolution: float = 0.01,
        timeout: float = 5.0,
        max_retry: int = 1,
        smooth_path: bool = True,
        num_waypoints: int = 100,
        ignore_collision: bool = False,
        planner: str = "RRTConnect",
        envs_idx: int = 0,
    ) -> list[torch.Tensor]:
        """Plan a collision-free joint-space path from the start to the goal configuration.

        A start or goal outside the joint limits widens the planning bounds to contain it (with a
        warning), and collision pairs already active in the scene when it is called are ignored
        throughout. The entity's ``qpos`` is set to ``qpos_start`` before returning.

        Parameters
        ----------
        qpos_goal : torch.Tensor
            The goal configuration of the entity, shape ``(n_qs,)``.
        qpos_start : torch.Tensor | None
            The start configuration, shape ``(n_qs,)``. ``None`` uses the entity's current ``qpos``.
        resolution : float
            State-validity checking resolution along a motion, as OMPL's
            ``setStateValidityCheckingResolution`` takes it (a fraction of the space's extent).
        timeout : float
            Time budget of one solve attempt, in seconds.
        max_retry : int
            The maximum number of solve attempts while the planner times out or finds only an
            approximate solution.
        smooth_path : bool
            Whether to shortcut and B-spline-smooth the path.
        num_waypoints : int
            The number of waypoints the path is interpolated to.
        ignore_collision : bool
            Whether to skip collision checking.
        planner : str
            The name of an ``ompl.geometric`` planner class, e.g. ``"RRTConnect"``.
        envs_idx : int
            The environment index. This planner does not support batch processing.

        Returns
        -------
        waypoints : list[torch.Tensor]
            One ``(n_qs,)`` configuration per waypoint; empty when planning is infeasible or no
            path was found.
        """
        if timeout <= 0.0 or not math.isfinite(timeout):
            raise ValueError("Timeout must be a positive finite number.")
        if max_retry <= 0:
            raise ValueError("Max retry must be a positive integer.")
        planner_cls = _ompl_planner_class(planner)

        if qpos_start is None:
            qpos_start = self.entity.get_qpos(envs_idx=envs_idx).squeeze(0)
        qpos_start = tensor_to_array(qpos_start)
        qpos_goal = tensor_to_array(qpos_goal)

        if qpos_start.shape != (self.entity.n_qs,) or qpos_goal.shape != (self.entity.n_qs,):
            raise ValueError(
                f"Invalid shape for `qpos_start` or `qpos_goal`. Expected ({(self.entity.n_qs,)}), got ({qpos_start.shape}, {qpos_goal.shape})."
            )

        ######### process joint limit ##########
        q_limit_lower, q_limit_upper = (tensor_to_array(limit[0]) for limit in self.entity.get_qpos_limit(envs_idx))

        if (qpos_start < q_limit_lower).any() or (qpos_start > q_limit_upper).any():
            en.logger.warning(
                "`qpos_start` exceeds joint limit. Relaxing joint limit to contain `qpos_start` for planning."
            )
            q_limit_lower = np.minimum(q_limit_lower, qpos_start)
            q_limit_upper = np.maximum(q_limit_upper, qpos_start)

        if (qpos_goal < q_limit_lower).any() or (qpos_goal > q_limit_upper).any():
            en.logger.warning(
                "`qpos_goal` exceeds joint limit. Relaxing joint limit to contain `qpos_goal` for planning."
            )
            q_limit_lower = np.minimum(q_limit_lower, qpos_goal)
            q_limit_upper = np.maximum(q_limit_upper, qpos_goal)

        ######### setup OMPL ##########
        ou.setLogLevel(ou.LOG_ERROR)
        space = ob.RealVectorStateSpace(self.entity.n_qs)
        bounds = ob.RealVectorBounds(self.entity.n_qs)

        for i_q in range(self.entity.n_qs):
            bounds.setLow(i_q, float(q_limit_lower[i_q]))
            bounds.setHigh(i_q, float(q_limit_upper[i_q]))
        space.setBounds(bounds)
        ss = og.SimpleSetup(space)

        geoms_idx = tuple(range(self.entity.geom_start, self.entity.geom_start + len(self.entity.geoms)))
        mask_collision_pairs = set(
            (i_ga, i_gb)
            for i_ga, i_gb in self.entity.detect_collision(env_idx=envs_idx)
            if i_ga in geoms_idx or i_gb in geoms_idx
        )
        if not ignore_collision and mask_collision_pairs:
            en.logger.info("Ignoring collision pairs already active for starting pos.")

        def is_ompl_state_valid(state):
            if ignore_collision:
                return True
            qpos = torch.tensor(
                [state[i] for i in range(self.entity.n_qs)],
                dtype=gs.tc_float,
                device=self.env.device,
            )
            self.entity.set_qpos(qpos, zero_velocity=False, envs_idx=envs_idx)
            collision_pairs = set(map(tuple, self.entity.detect_collision(env_idx=envs_idx)))
            return not (collision_pairs - mask_collision_pairs)

        ss.setStateValidityChecker(_as_validity_checker(is_ompl_state_valid))

        si = ss.getSpaceInformation()
        si.setStateValidityCheckingResolution(resolution)

        def allocOBValidStateSampler(si):
            vss = ob.UniformValidStateSampler(si)
            vss.setNrAttempts(100)
            return vss

        si.setValidStateSamplerAllocator(_as_sampler_allocator(allocOBValidStateSampler))

        ss.setPlanner(planner_cls(si))

        ss.setStartAndGoalStates(_ompl_state(space, qpos_start), _ompl_state(space, qpos_goal))

        ######### solve ##########
        waypoints = []
        for i in range(max_retry):
            if ss.getPlanner():
                ss.getPlanner().clear()
            status = ss.solve(timeout)
            status_type = status.getStatus()

            if status_type in (
                ob.PlannerStatus.UNKNOWN,
                ob.PlannerStatus.CRASH,
                ob.PlannerStatus.ABORT,
            ):
                raise ValueError("Unknown error.")
            if status_type in (
                ob.PlannerStatus.INVALID_START,
                ob.PlannerStatus.INVALID_GOAL,
                ob.PlannerStatus.UNRECOGNIZED_GOAL_TYPE,
                ob.PlannerStatus.INFEASIBLE,
            ):
                en.logger.warning("Path planning infeasible. Returning empty path.")
                break

            if status:
                path = ss.getSolutionPath()

                if smooth_path:
                    ps = og.PathSimplifier(si)
                    try:
                        ps.partialShortcutPath(path)
                        ps.ropeShortcutPath(path)
                    except Exception as e:
                        en.logger.warning(f"Path simplification failed: {e}")
                        ps.shortcutPath(path)
                    ps.smoothBSpline(path)

                if num_waypoints is not None:
                    path.interpolate(num_waypoints)

                waypoints = [
                    torch.as_tensor(
                        [state[i] for i in range(self.entity.n_qs)],
                        dtype=gs.tc_float,
                        device=gs.device,
                    )
                    for state in path.getStates()
                ]

            # A timeout or an approximate solution retries until max_retry attempts are used up.
            if status_type in (
                ob.PlannerStatus.TIMEOUT,
                ob.PlannerStatus.APPROXIMATE_SOLUTION,
            ):
                if i + 1 < max_retry:
                    en.logger.warning("Path planning did not converge. Trying again...")
                    continue
                else:
                    if waypoints:
                        en.logger.warning("Path planning did not converge. Returning approximation path.")
                    else:
                        en.logger.warning("Path planning did not converge. Returning empty path.")
                    break
            en.logger.info("Path solution found successfully.")
            break

        ########## restore original state #########
        self.entity.set_qpos(qpos_start, zero_velocity=False, envs_idx=envs_idx)

        return waypoints
