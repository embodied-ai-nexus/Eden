"""dex-retargeting backend.

Re-exports are lazy (PEP 562 ``__getattr__``, mirroring the parent
``eden.extensions.retargeting`` package) so that importing the package — or a
lightweight submodule like :mod:`~eden.extensions.retargeting.dex_retargeter.presets`
— never pulls the SLSQP solver (``nlopt``, the ``dex-retargeter`` extra).
"""

from typing import TYPE_CHECKING

# Static re-export map for type checkers AND for griffe/mkdocstrings: the
# lazy __getattr__ below is invisible to static analysis, so without this
# block the docs build cannot collect e.g. ``dex_retargeter.DexRetargeter``.
if TYPE_CHECKING:
    from eden.extensions.retargeting.dex_retargeter.constants import OPERATOR2MANO, HandType
    from eden.extensions.retargeting.dex_retargeter.optimizer import (
        DexPilotOptimizer,
        Optimizer,
        PositionOptimizer,
        VectorOptimizer,
    )
    from eden.extensions.retargeting.dex_retargeter.retargeter import DexRetargeter, operator2mano

__all__ = [
    "DexRetargeter",
    "HandType",
    "OPERATOR2MANO",
    "Optimizer",
    "PositionOptimizer",
    "VectorOptimizer",
    "DexPilotOptimizer",
    "operator2mano",
]


def __getattr__(name: str):
    if name in ("HandType", "OPERATOR2MANO"):
        from eden.extensions.retargeting.dex_retargeter import constants

        return getattr(constants, name)
    if name in ("Optimizer", "PositionOptimizer", "VectorOptimizer", "DexPilotOptimizer"):
        from eden.extensions.retargeting.dex_retargeter import optimizer

        return getattr(optimizer, name)
    if name in ("DexRetargeter", "operator2mano"):
        from eden.extensions.retargeting.dex_retargeter import retargeter

        return getattr(retargeter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
