"""The numerical core of OTRetarget's trust-region SQP: Gauss-Newton accumulation and the dense QP (ProxQP).

Nothing here knows about probes, channels or robots: residual rows arrive in factored form (:class:`Rows`) and leave as
the Hessian, gradient and constraint matrices of one QP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Rows:
    """Residual rows in factored form: ``u_r^T J6[link]`` on the robot columns, ``u_o`` on the object columns."""

    link: np.ndarray | None
    u_r: np.ndarray | None  # (K, 6)
    u_o: np.ndarray  # (K, 6 * n_free_objects)


@dataclass
class Local:
    """One linearisation of the frame problem: the QP data at a configuration and the true cost there."""

    H: np.ndarray
    g: np.ndarray
    C: np.ndarray
    l_in: np.ndarray
    hard_lo: np.ndarray
    hard_hi: np.ndarray
    trust: np.ndarray
    terms: dict[str, float]
    n_active: int
    penetration: float
    frozen: list
    term_grads: dict[str, np.ndarray]
    # Per probe set with non-penetration rows: (kind, channel, probe indices, channel-frame points, distances,
    # distance gradients) of those probes; the SQP cuts off the ones a rejected step drove too deep.
    contacts: list = field(default_factory=list)

    @property
    def cost(self) -> float:
        return float(sum(self.terms.values()))


class GaussNewton:
    """Accumulates ``H = sum w J^T J`` and ``g = sum w J^T r`` for the QP, robot rows link by link.

    ``terms`` names the cost terms (each reported with its value and gradient); ``add`` to another name is an error.
    """

    def __init__(self, n_x: int, n_v: int, link_jac: np.ndarray, terms) -> None:
        self.n_v = n_v
        self.link_jac = link_jac
        self.H = np.zeros((n_x, n_x))
        self.g = np.zeros(n_x)
        self.terms = dict.fromkeys(terms, 0.0)
        self.term_grads = {k: np.zeros(n_x) for k in terms}

    def dense(self, rows: Rows) -> np.ndarray:
        """The rows as a dense ``(K, n_x)`` matrix (for constraints)."""
        out = np.zeros((len(rows.u_o), len(self.g)))
        if rows.u_r is not None:
            out[:, : self.n_v] = np.einsum("ki,kix->kx", rows.u_r, self.link_jac[rows.link])
        out[:, self.n_v :] = rows.u_o
        return out

    def add(self, name: str, rows: Rows, r: np.ndarray, w: np.ndarray) -> None:
        if len(r) == 0:
            return
        n_v = self.n_v
        wr = w * r
        self.terms[name] += float((wr * r).sum())
        g_before = self.g.copy()
        if rows.u_o.shape[1]:
            self.H[n_v:, n_v:] += (rows.u_o * w[:, None]).T @ rows.u_o
            self.g[n_v:] += rows.u_o.T @ wr
        if rows.u_r is None:
            self.term_grads[name] += self.g - g_before
            return
        # Per-link sums of w u_r u_r^T, w r u_r and w u_r u_o^T, then one pass through each link Jacobian.
        links, inverse = np.unique(rows.link, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        bounds = np.flatnonzero(np.r_[True, np.diff(inverse[order]) != 0])
        u = rows.u_r[order]
        ws = w[order]
        m_rr = np.add.reduceat(ws[:, None, None] * u[:, :, None] * u[:, None, :], bounds, axis=0)
        g_r = np.add.reduceat(wr[order][:, None] * u, bounds, axis=0)
        J = self.link_jac[links]  # (L, 6, n_v)
        self.H[:n_v, :n_v] += np.einsum("lia,lij,ljb->ab", J, m_rr, J, optimize=True)
        self.g[:n_v] += np.einsum("lia,li->a", J, g_r)
        if rows.u_o.shape[1]:
            m_ro = np.add.reduceat(ws[:, None, None] * u[:, :, None] * rows.u_o[order][:, None, :], bounds, axis=0)
            cross = np.einsum("lia,lij->aj", J, m_ro)
            self.H[:n_v, n_v:] += cross
            self.H[n_v:, :n_v] += cross.T
        self.term_grads[name] += self.g - g_before

    def add_dense(self, name: str, r: np.ndarray, J: np.ndarray, w: float) -> None:
        if len(r) == 0:
            return
        self.H += w * (J.T @ J)
        self.g += w * (J.T @ r)
        self.term_grads[name] += w * (J.T @ r)
        self.terms[name] += float(w * (r * r).sum())


def pad(J: np.ndarray, n_x: int) -> np.ndarray:
    """``J`` widened with zero columns to ``n_x``."""
    out = np.zeros((len(J), n_x))
    out[:, : J.shape[1]] = J
    return out


def single_threaded_blas():
    """Context limiting the BLAS pools to one thread: faster for the solve's small products, and reproducible."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dependency
        raise ImportError(
            "OTRetargeter pins BLAS to one thread with threadpoolctl: install it with `pip install threadpoolctl` "
            "(the `ot-retargeter` extra of eden)."
        ) from exc
    return threadpool_limits(limits=1, user_api="blas")


def elastic_shift(C: np.ndarray, l_in: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray | None:
    """Smallest per-row relaxation ``t >= 0`` (in the l1 sense) making ``C x >= l_in - t`` feasible in the box.

    One LP: ``min sum t  s.t.  C x + t >= l_in,  lo <= x <= hi,  t >= 0``. The l1 objective relaxes only
    the rows that actually conflict. Returns None if the LP solver fails.
    """
    from scipy.optimize import linprog
    from scipy.sparse import hstack as sparse_hstack
    from scipy.sparse import identity

    n, m = C.shape[1], C.shape[0]
    res = linprog(
        np.r_[np.zeros(n), np.ones(m)],
        A_ub=sparse_hstack([-C, -identity(m)], format="csr"),
        b_ub=-l_in,
        bounds=[*zip(lo, hi), *((0.0, None),) * m],
        method="highs",
    )
    return res.x[n:] if res.status == 0 else None


def solve_qp(H, g, C, l_in, lo, hi) -> tuple[np.ndarray, str]:
    """``min 1/2 x^T H x + g^T x  s.t.  C x >= l_in,  lo <= x <= hi`` with ProxQP (dense).

    Linearised non-penetration rows can contradict each other (probes on both sides of a feature
    thinner than the distance grid), making the QP infeasible. When the plain solve does not converge,
    the SQP elastic relaxation is applied: :func:`elastic_shift` finds the smallest (l1) shift of the
    conflicting rows that restores feasibility, and the shifted QP is solved instead. If neither solve
    yields a finite step, the iterate does not move.

    Returns
    -------
    x : np.ndarray
        The step, clipped to the box (zero when no solve succeeded).
    status : str
        ``"solved"``, ``"relaxed"`` (solved after the elastic shift) or ``"failed"``.
    """
    try:
        from proxsuite import proxqp
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dependency
        raise ImportError(
            "OTRetargeter solves its QPs with ProxQP: install it with `pip install proxsuite` "
            "(the `ot-retargeter` extra of eden)."
        ) from exc
    n = len(g)
    m = len(C)
    H = H + 1e-8 * np.eye(n)

    def attempt(lower: np.ndarray, max_iter: int, max_iter_in: int) -> np.ndarray | None:
        qp = proxqp.dense.QP(n, 0, m, True, proxqp.dense.HessianType.Dense)
        qp.settings.eps_abs = 1e-6  # 1 um on metre-scale steps
        qp.settings.max_iter = max_iter
        qp.settings.max_iter_in = max_iter_in
        qp.settings.verbose = False
        qp.init(H, g, None, None, C if m else None, lower if m else None, np.full(m, 1e20) if m else None, lo, hi)
        qp.solve()
        x = np.asarray(qp.results.x, dtype=np.float64)
        ok = qp.results.info.status == proxqp.QPSolverOutput.PROXQP_SOLVED and np.all(np.isfinite(x))
        return np.clip(x, lo, hi) if ok else None

    # Feasible iterates converge in a few dozen inner iterations; the tight cap makes an infeasible one
    # fail in milliseconds instead of exhausting a large budget (seconds) before the relaxation.
    x = attempt(l_in, 40, 20)
    if x is not None:
        return x, "solved"
    if m:
        shift = elastic_shift(C, l_in, lo, hi)
        if shift is not None:
            # A 10 um margin keeps the shifted problem off its feasibility boundary. A problem the LP finds feasible
            # as posed (no shift) was only slow: its solve counts as "solved", not "relaxed".
            relaxed = bool((shift > 1e-9).any())
            x = attempt(l_in - shift - 1e-5, 300, 50)
            if x is not None:
                return x, "relaxed" if relaxed else "solved"
            if not relaxed:
                x = attempt(l_in, 5000, 500)
                if x is not None:
                    return x, "solved"
    return np.zeros(n), "failed"
