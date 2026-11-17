"""Error flags the MochiSolver's kernels raise in the per-environment ``errno`` field of its kinematic tree."""

from enum import auto

from genesis.engine.solvers.base_solver import ErrorCode


class MochiErrorCode(ErrorCode):
    """The error flags of the MochiSolver (see ``ErrnoMixin.register_errors``), as Genesis' ``RigidErrorCode`` are the
    RigidSolver's: each takes a bit of the solver's own ``errno`` above those of the backward pass.
    """

    OVERFLOW_MOCHI_CONTACT_PAIRS = auto()
    OVERFLOW_MOCHI_CONTACTS = auto()
    MOCHI_DIVERGED = auto()
