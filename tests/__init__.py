"""Eden's test suite.

This file is what makes ``tests`` a *regular* package rather than a namespace
one. Without it, Python only synthesizes a ``tests`` namespace package when no
regular ``tests`` package exists anywhere on ``sys.path`` — so any installed
dependency that ships its own top-level ``tests/`` (``ultralytics``, pulled in
transitively by the pose-estimation extra, is one) wins the name outright,
regardless of ``sys.path`` order, and every ``from tests.support.asserts import ...``
inside this suite dies with ``ModuleNotFoundError`` at collection time.
"""
