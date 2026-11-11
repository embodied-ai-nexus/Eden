"""Materials of the bodies the MochiSolver simulates: rigid and articulated, tetrahedral solids, shells and rods."""

from .base import Base
from .elastic import Elastic
from .rigid import MochiRigid, Rigid
from .rod import Rod
from .shell import Shell

__all__ = ["Base", "Elastic", "MochiRigid", "Rigid", "Rod", "Shell"]
