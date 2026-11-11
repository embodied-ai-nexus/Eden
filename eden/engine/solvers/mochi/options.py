"""Options configuring the MochiSolver.

Upstream Genesis declares a solver's options as a field of its ``SceneOptions``; the MochiSolver is hosted by Eden, so
its options class lives here and :func:`eden.engine.solvers.mochi.hosting.install_mochi_solver` hands it to the solver.
"""

from typing import Any, Literal

import genesis as gs
from genesis.options.solvers import GravityMixin, TimeBasedMixin
from genesis.typing import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt, UnitInterval
from pydantic import StrictBool


class MochiOptions(GravityMixin, TimeBasedMixin):
    """
    Options configuring the MochiSolver.

    MochiSolver is a fully-implicit solver: at every substep it solves a single nonlinear system in which the inertia
    of every rigid body and a smooth signed-distance penalty contact model (regularized Coulomb friction, viscous
    normal damping) are assembled together. There is no separate collision response stage: contact is re-detected at
    every Newton iterate and enters the same residual and Hessian as the inertia, which is what keeps large time steps
    stable and is the prerequisite for coupling deformable bodies into the same system without a coupler.

    Note
    ----
    Double precision (`gs.init(precision="64")`) is recommended. The default contact stiffness of 1e9 Pa/m combined
    with a 1 mm activation threshold makes the Newton system ill-conditioned in single precision.

    Parameters
    ----------
    integrator : str, optional
        Time integration scheme: "backward_euler" (first order, strongly damped) or "bdf2" (second order, closer to
        energy-conserving; needs the previous two steps so the first step of a fresh or reset scene falls back to
        backward Euler). Defaults to "backward_euler".
    use_newton_euler_inertia : bool, optional
        Whether the rotational inertia enters as the Newton-Euler residual `I dw/dt + w x I w` instead of the
        variational merit of the rotation. The merit form derives from a potential and is what the line search
        monitors; the Newton-Euler form is exact for gyroscopic effects but its Hessian is approximate. Defaults to
        False.
    n_newton_iterations : int, optional
        Maximum number of Newton iterations per substep. Defaults to 4.
    newton_abs_tol : float, optional
        Absolute tolerance on the mass-weighted residual norm (unit acceleration under gravity gives a norm of order
        one). Defaults to 1e-3.
    newton_rel_tol : float, optional
        Relative tolerance on the mass-weighted residual norm with respect to its value at the first iteration.
        Defaults to 1e-6.
    explosion_control : bool, optional
        Whether a substep whose residual grows beyond `explosion_rel_tol` times its initial value or beyond
        `explosion_abs_tol` is flagged as diverged: the affected environment is reset to its previous pose with zero
        velocity and an error is raised at the next error check. A NaN residual is flagged either way. Defaults to
        True.
    explosion_abs_tol : float, optional
        Absolute residual norm above which the solve is considered diverged. Defaults to 1e9.
    explosion_rel_tol : float, optional
        Residual growth factor above which the solve is considered diverged. Defaults to 1e4.
    linesearch_type : str, optional
        Step acceptance rule: "residual_norm" accepts the first trial whose residual norm does not exceed the current
        one, "armijo" requires a sufficient decrease of the incremental potential (costs an extra energy assembly per
        trial), "none" always takes the full Newton step. Defaults to "residual_norm".
    n_linesearch_iterations : int, optional
        Maximum number of step reductions (by `linesearch_alpha`) per Newton iteration. The last trial is kept even
        if it did not improve. Defaults to 4.
    linesearch_alpha : float, optional
        Step size reduction factor between two line search trials. Defaults to 0.5.
    linesearch_wolfe1 : float, optional
        Sufficient decrease parameter of the Armijo rule. Defaults to 1e-4.
    linear_solver : str, optional
        Linear solver for the Newton system: "ldlt" (dense Cholesky of every simulation island, exact, cubic in the
        number of degrees of freedom of the island), "pcg" (block-Jacobi preconditioned conjugate gradient, linear per
        iteration), or "auto" (dense when the largest island of the environment has at most `dense_solver_max_dofs`
        degrees of freedom, PCG otherwise). Bodies coupled by the contact candidates of the step form an island.
        Defaults to "auto".
    dense_solver_max_dofs : int, optional
        Largest island solved with the dense solver under "auto". Defaults to 50.
    dense_matrix_max_dofs : int, optional
        Largest total number of degrees of freedom for which the dense matrix of the system is allocated (memory
        quadratic in this number per environment); beyond it every environment is solved with PCG. Defaults to 256.
    n_pcg_iterations : int, optional
        Maximum number of conjugate gradient iterations. If None, the number of degrees of freedom capped at 500.
        Defaults to None.
    pcg_rel_tol : float, optional
        Relative tolerance of the conjugate gradient solve. Ignored under the "adaptive" tolerance strategy.
        Defaults to 1e-5.
    pcg_abs_tol : float, optional
        Absolute floor of the conjugate gradient stopping test, on the norm of the preconditioned residual (the
        residual scaled by the inverse of the Hessian diagonal, i.e. a displacement). A solve stops as soon as that
        norm drops below the floor, whatever the relative tolerance asks for; mochi's default. Set to 0 to disable.
        Defaults to 1e-9.
    linear_tolerance_strategy : str, optional
        How tightly each Newton step solves its linear system: "constant" always solves to `pcg_rel_tol`, so the
        accuracy of a substep is set by `pcg_rel_tol` and `n_newton_iterations` alone; "adaptive" starts each substep
        at a loose tolerance and tightens it as the nonlinear residual drops, spending far fewer conjugate gradient
        iterations per Newton step but leaving more truncation error in the step it takes, which can cost an extra
        Newton iteration on scenes that would otherwise converge in one. Prefer "adaptive" (mochi's default policy)
        when the conjugate gradient dominates the substep and a relative accuracy of order 1e-5 is enough, "constant"
        when accuracy per substep matters more than the cost of reaching it. Defaults to "adaptive".
    friction_model : str, optional
        Regularization of the Coulomb friction force around zero sliding velocity: "c1" has compact support (exact
        Coulomb beyond `friction_falloff_vel`), "cinf" is smooth everywhere (never exactly Coulomb, better
        conditioned). Defaults to "c1".
    use_fitted_friction_hessian : bool, optional
        Whether the friction Hessian uses a quadratic fit that is the same in every tangential direction. The exact
        Hessian converges faster close to the solution but can stall the Newton iterations at the stick-slip
        transition. Defaults to True.
    friction_with_collider_normal : bool, optional
        Whether the friction plane is defined by the collider's distance gradient (True) or by the colliding surface
        normal (False). Defaults to True.
    fade_friction : bool, optional
        Whether friction fades out as the colliding surface normal and the collider gradient become aligned, i.e. as a
        sample point passes through the far side of a thin collider. Defaults to True.
    implicit_normal_force_for_dissipation : bool, optional
        Whether friction and damping scale with the normal force evaluated at the current iterate instead of the one
        recovered at the start of the step. The implicit form is required for an accurate coefficient of restitution
        through normal damping; the explicit form is cheaper and smoother. Defaults to False.
    boundary_element_type : str, optional
        Quadrature rule placing contact sample points on the collision triangles: "P1Q1" (centroid), "P1Q3" (3 points
        per triangle, degree 2), "P1Q6" (6 points, degree 4). More points resolve contact patches better at a
        proportional cost. Defaults to "P1Q3".
    equality_stiffness : float, optional
        Stiffness of the penalty enforcing the equality constraints of the articulations (connect, weld and joint
        couplings; loop closures): the constraint violation is penalized by `0.5 * k * |c|^2`. Defaults to 1e6.
    equality_damping : float, optional
        Damping of the equality constraint penalty, `0.5 * (d / dt) * |c - c_prev|^2`. Defaults to 0.
    max_contact_pairs_per_env : int, optional
        Capacity of the list of (link, collider geom) pairs whose bounding boxes overlap within a substep. If None, the
        number of possible pairs. Defaults to None.
    broadphase_margin : float, optional
        Absolute padding of the per-step conservative bounding boxes in meters. Defaults to 0.01.
    spatial_hash_bins_per_item : int, optional
        Bins of the spatial hash that locates the collider spheres of shells and rods per inserted sphere, rounded
        up to a power of two; more bins shorten the chains a query walks at the cost of memory (4 bytes per bin per
        environment). The tetrahedra of solids use a bounding-box hierarchy instead. Defaults to 2.
    max_soft_hits_per_sample : int, optional
        Capacity of the list of contacts between deformable boundary samples and rigid colliders, per sample.
        Exceeding it halts the simulation with an error. Defaults to 2.
    max_deformable_collider_hits_per_query : int, optional
        Capacity of the list of contacts between sample points (rigid or deformable) and the tetrahedra of the
        deformable solid colliders, per sample point. Defaults to 2.
    max_point_cloud_hits_per_query : int, optional
        Capacity of the list of contacts between deformable samples and the collider spheres of shells and rods,
        per deformable sample (rigid samples get `max_soft_hits_per_sample` slots each). If None, 4 when a shell or
        rod has self-contact (a sample then sees the spheres of the opposing layer of its own body) and 2 otherwise;
        the capacity is a per-sample average over a shared list, and exceeding it halts with an error naming this
        option. Defaults to None.
    record_contacts : bool, optional
        Whether individual contact points can be read back through `entity.get_contacts()`; their buffers are
        allocated at the first readback. Defaults to True.
    use_sleeping : bool, optional
        Whether bodies at rest are put to sleep: a sleeping body keeps its pose exactly and costs neither contact nor
        element work until an awake body comes near it or a setter changes its state, its controls, the gravity or the
        contact settings. It pays off for scenes with many settled objects. The cost is physical: slow motion stops for
        good, such as creep on a slope or the slow settling of a soft body, and the trajectory differs from a run
        without sleeping. Bodies under an active controller and rods never sleep. Defaults to False.
    sleep_threshold : float, optional
        A step counts as at rest for a body when the solve reduced its residual by less than this factor from the
        warm start, which an unforced body at equilibrium does not improve on. Higher values sleep sooner. Only steps
        whose solve meets the Newton tolerances count: in single precision, a soft body at rest can stall just above
        the default `newton_abs_tol` and then never sleeps. A looser tolerance such as 5e-3 lets it sleep, at the cost
        of accuracy. Defaults to 0.5.
    sleep_min_steps : int, optional
        Consecutive steps at rest after which every body of an island at rest goes to sleep. Fewer steps sleep sooner
        but can freeze a body at the turning point of a slow swing. Defaults to 20.
    sleep_max_speed : float, optional
        Speed in m/s above which a body never counts as at rest, whatever its residual: it keeps a body coasting
        without forces (whose motion the BDF2 warm start predicts exactly) awake. Defaults to 1e-2.
    step_kernel : str, optional
        How a step is executed: "monolith" runs the whole step of every environment in one kernel (one thread per
        environment, one launch per step, no host round trips), "pipeline" runs each stage as its own kernel with the
        host driving the loops, "graph" runs the step as one graph-launched kernel whose Newton, line-search and
        conjugate-gradient loops run on the device with the stages parallel over items and environments (one launch
        per step). "auto" picks the monolith on the CPU (the pipeline for deformable scenes of at least 3000 vertices
        when several CPU threads run) and, on the GPU, for rigid scenes of at most 64 degrees of freedom, the pipeline
        otherwise. The graph kernel is opt-in: on GPUs without device-side graph conditionals
        (before compute capability 9.0) the runtime replays its loop bodies from the host with one flag readback per
        round, which runs at the pipeline's speed, and the single module compiles several times slower than the
        pipeline's kernels (minutes for a cloth with self-contact and an arm). Defaults to "auto".
    graph_pcg_unroll : int, optional
        Conjugate-gradient iterations per round of the graph step kernel's inner loop (each round costs one flag
        readback on GPUs without device-side graph conditionals; each unrolled iteration is compiled once more).
        Defaults to 1.
    joint_limit_stiffness : float, optional
        Stiffness in N/m (N*m/rad) of the penalty holding revolute and prismatic joints inside their range. Joint
        limits are soft: the violation at rest is the limit torque divided by this stiffness. Higher values reduce the
        violation but stiffen the Newton system. Defaults to 1e4 (the original engine defaults to 100).
    joint_limit_damping : float, optional
        Damping in N*s/m (N*m*s/rad) of the joint limit penalty, resisting the velocity of the violation. Defaults to 0.
    batch_links_info : bool, optional
        Whether to batch link info. Defaults to False.
    batch_joints_info : bool, optional
        Whether to batch joint info. Defaults to False.
    batch_dofs_info : bool, optional
        Whether to batch DOF info. Defaults to False.
    IK_max_targets : int, optional
        Maximum number of simultaneous target links of an inverse-kinematics solve (the scratch buffers are quadratic
        in it and allocated at the first solve). Defaults to 6.
    """

    IK_max_targets: PositiveInt = 6
    integrator: Literal["backward_euler", "bdf2"] = "backward_euler"
    use_newton_euler_inertia: StrictBool = False
    n_newton_iterations: PositiveInt = 4
    newton_abs_tol: PositiveFloat = 1e-3
    newton_rel_tol: PositiveFloat = 1e-6
    explosion_control: StrictBool = True
    explosion_abs_tol: PositiveFloat = 1e9
    explosion_rel_tol: PositiveFloat = 1e4
    linesearch_type: Literal["none", "residual_norm", "armijo"] = "residual_norm"
    n_linesearch_iterations: NonNegativeInt = 4
    linesearch_alpha: PositiveFloat = 0.5
    linesearch_wolfe1: PositiveFloat = 1e-4
    linear_solver: Literal["auto", "ldlt", "pcg"] = "auto"
    dense_solver_max_dofs: PositiveInt = 50
    dense_matrix_max_dofs: PositiveInt = 256
    n_pcg_iterations: PositiveInt | None = None
    pcg_rel_tol: PositiveFloat = 1e-5
    pcg_abs_tol: NonNegativeFloat = 1e-9
    linear_tolerance_strategy: Literal["constant", "adaptive"] = "adaptive"
    friction_model: Literal["c1", "cinf"] = "c1"
    use_fitted_friction_hessian: StrictBool = True
    friction_with_collider_normal: StrictBool = True
    fade_friction: StrictBool = True
    implicit_normal_force_for_dissipation: StrictBool = False
    boundary_element_type: Literal["P1Q1", "P1Q3", "P1Q6"] = "P1Q3"
    equality_stiffness: PositiveFloat = 1e6
    equality_damping: NonNegativeFloat = 0.0
    max_contact_pairs_per_env: PositiveInt | None = None
    spatial_hash_bins_per_item: PositiveInt = 2
    max_soft_hits_per_sample: PositiveInt = 2
    max_deformable_collider_hits_per_query: PositiveInt = 2
    max_point_cloud_hits_per_query: PositiveInt | None = None
    broadphase_margin: NonNegativeFloat = 0.01
    record_contacts: StrictBool = True
    use_sleeping: StrictBool = False
    sleep_threshold: UnitInterval = 0.5
    sleep_min_steps: PositiveInt = 20
    sleep_max_speed: NonNegativeFloat = 1e-2
    step_kernel: Literal["auto", "monolith", "pipeline", "graph"] = "auto"
    graph_pcg_unroll: PositiveInt = 1
    joint_limit_stiffness: PositiveFloat = 1e4
    joint_limit_damping: NonNegativeFloat = 0.0
    batch_links_info: StrictBool = False
    batch_joints_info: StrictBool = False
    batch_dofs_info: StrictBool = False

    def model_post_init(self, context: Any) -> None:
        if not (0.0 < self.linesearch_alpha < 1.0):
            gs.raise_exception("`linesearch_alpha` must be strictly between 0 and 1.")
