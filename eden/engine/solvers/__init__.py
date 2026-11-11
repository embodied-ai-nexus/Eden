"""Genesis solvers that upstream Genesis does not ship, hosted by Eden on stock Genesis.

Upstream builds its solvers from a fixed list, so a solver here is not installed by importing it: each one ships a
``hosting`` module that adds it to one scene at a time (see :mod:`eden.engine.solvers.mochi.hosting`).

- :mod:`eden.engine.solvers.mochi` — the MochiSolver: a fully implicit solver of rigid, articulated and deformable
  bodies (solids, shells, rods) with smooth penalty contact.
"""
