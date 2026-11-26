"""The visual geom of a Mochi deformable body."""

from genesis.engine.entities.fem_entity import FEMVisGeom


class MochiVisGeom(FEMVisGeom):
    """A ``FEMVisGeom`` that may also be an embedded (skinned) render mesh.

    Each render vertex of an embedded mesh follows a barycentric combination of the four nodes of one simulated element
    instead of a single simulated vertex (``sim_verts_idx`` is unused then). The render vertices of every deformable
    body of the solver lie in one array (see ``MochiSolver._update_soft_render``), where those of this visual geom
    start at ``vvert_start``; the solver lays them out as it builds.
    """

    def __init__(self, entity, vvert_start, vface_start, vmesh, sim_verts_idx, elems_idx=None, bary=None):
        super().__init__(entity, vmesh, sim_verts_idx)
        self._vvert_start = vvert_start
        self._vface_start = vface_start
        self._elems_idx = elems_idx
        self._bary = bary

    @property
    def vvert_start(self):
        """First render vertex of the visual geom among those of the solver."""
        return self._vvert_start

    @property
    def vvert_end(self):
        """One past the last render vertex of the visual geom among those of the solver."""
        return self._vvert_start + self.n_vverts

    @property
    def vface_start(self):
        """First render face of the visual geom among those of the solver."""
        return self._vface_start

    @property
    def elems_idx(self):
        """Element carrying each render vertex of an embedded render mesh (None for a vertex-mapped one)."""
        return self._elems_idx

    @property
    def bary(self):
        """Barycentric coordinates of each render vertex of an embedded render mesh in its element."""
        return self._bary
