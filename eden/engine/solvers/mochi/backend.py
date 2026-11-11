"""The compute backend facts the MochiSolver sizes its parallelism from."""

import os


def cpu_threads_effective() -> int:
    """Number of Quadrants CPU threads the current ``gs.init`` runs with.

    Upstream ``gs.init`` pins the CPU backend to one thread unless ``QD_NUM_THREADS`` is set, so that variable is the
    only way to raise it, and the count it holds is the count in effect.
    """
    return int(os.environ.get("QD_NUM_THREADS", "1"))
