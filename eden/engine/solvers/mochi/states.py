"""Dynamic state queried from the deformable entities of the MochiSolver."""

import genesis as gs
import torch


class MochiSoftEntityState:
    """Dynamic state queried from a MochiSoftEntity: vertex positions and velocities, shape (n_envs, n_vertices, 3)."""

    def __init__(self, entity, s_global):
        self._entity = entity
        self._s_global = s_global
        args = {"dtype": gs.tc_float, "device": gs.device}
        self.pos = torch.zeros((entity.solver.n_envs_, entity.n_vertices, 3), **args)
        self.vel = torch.zeros((entity.solver.n_envs_, entity.n_vertices, 3), **args)

    @property
    def entity(self):
        return self._entity

    @property
    def s_global(self):
        return self._s_global
