"""The MochiSolver's error flags against those Genesis' solvers reserve."""

from genesis.engine.solvers.base_solver import ERRNO_BITS, BackwardErrorCode

from eden.engine.solvers.mochi.error_codes import MochiErrorCode


def test_each_mochi_error_owns_a_bit_of_the_errno_apart_from_the_backward_pass():
    # `Solver.register_errors` refuses a bit the base solver already holds (the NaN gradient of the backward pass): a
    # collision reads as an unrelated error raised with the wrong message.
    for code in MochiErrorCode:
        assert code.bit_count() == 1 and int(code) < 1 << ERRNO_BITS, code
        assert not int(code) & int(BackwardErrorCode.INVALID_GRAD_NAN), code
    assert len({int(code) for code in MochiErrorCode}) == len(MochiErrorCode)
