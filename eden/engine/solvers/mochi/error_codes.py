"""Error flags the MochiSolver's kernels raise in the per-environment ``errno`` field of its kinematic tree."""

from enum import IntEnum


class MochiErrorCode(IntEnum):
    """Bits above every one of ``genesis.utils.array_class.ErrorCode``, so the two never collide in one field.

    ``Solver.register_errors`` refuses a bit the base solver reserves (``INVALID_GRAD_NAN``), so a Genesis bump that
    adds an ``ErrorCode`` must move these above it.
    """

    OVERFLOW_MOCHI_CONTACT_PAIRS = 0b00000000000000000000010000000000
    OVERFLOW_MOCHI_CONTACTS = 0b00000000000000000000100000000000
    MOCHI_DIVERGED = 0b00000000000000000001000000000000
