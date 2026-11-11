"""Dynamic states queried from the MochiSolver and its deformable entities."""

import genesis as gs
import torch


class MochiSoftEntityState:
    """Dynamic state queried from a MochiSoftEntity: vertex positions and velocities, shape (n_envs, n_vertices, 3)."""

    def __init__(self, entity, s_global):
        self._entity = entity
        self._s_global = s_global
        args = {"dtype": gs.tc_float, "device": gs.device}
        self.pos = torch.zeros((entity.sim._B, entity.n_vertices, 3), **args)
        self.vel = torch.zeros((entity.sim._B, entity.n_vertices, 3), **args)

    @property
    def entity(self):
        return self._entity

    @property
    def s_global(self):
        return self._s_global


class MochiSolverState:
    """
    Dynamic state queried from a MochiSolver: generalized coordinates and velocities, link poses, and the multistep
    history the time integration extrapolates from.
    """

    def __init__(self, solver, s_global):
        self.scene = solver.scene
        self._s_global = s_global

        _B = solver._B
        args = {"dtype": gs.tc_float, "device": gs.device}
        self.qpos = torch.zeros((_B, solver.n_qs), **args)
        self.dofs_vel = torch.zeros((_B, solver.n_dofs), **args)
        self.links_pos = torch.zeros((_B, solver.n_links, 3), **args)
        self.links_quat = torch.zeros((_B, solver.n_links, 4), **args)
        self.qpos_prev = torch.zeros((_B, 2, solver.n_qs), **args)
        self.dofs_vel_prev = torch.zeros((_B, 2, solver.n_dofs), **args)
        self.links_vel = torch.zeros((_B, solver.n_links, 3), **args)
        self.links_ang = torch.zeros((_B, solver.n_links, 3), **args)
        self.links_vsym = torch.zeros((_B, solver.n_links, 3, 3), **args)
        self.links_vel_prev = torch.zeros((_B, 2, solver.n_links, 3), **args)
        self.links_ang_prev = torch.zeros((_B, 2, solver.n_links, 3), **args)
        self.links_vsym_prev = torch.zeros((_B, 2, solver.n_links, 3, 3), **args)
        self.n_hist = torch.zeros((_B,), dtype=gs.tc_int, device=gs.device)
        # Deformable bodies: vertex positions and velocities with their multistep history.
        n_soft_verts = solver.n_soft_verts
        self.soft_pos = torch.zeros((_B, n_soft_verts, 3), **args)
        self.soft_vel = torch.zeros((_B, n_soft_verts, 3), **args)
        self.soft_pos_prev = torch.zeros((_B, 2, n_soft_verts, 3), **args)
        self.soft_vel_prev = torch.zeros((_B, 2, n_soft_verts, 3), **args)

    @property
    def s_global(self):
        return self._s_global
