"""Particle entity wrapper for MPM/SPH/PBD deformables."""

from __future__ import annotations

import eden as en
from eden.entities.base import Entity


class ParticleEntity(Entity):
    """Entity subclass for particle-based materials (MPM, SPH, PBD).

    Particle entities have no DOFs, links, or orientation control.
    Position and rotation must be baked into the morph at creation time.
    """

    def set_pos(self, pos, envs_idx=None) -> None:
        """Particle entities do not support set_pos after build.

        Raises
        ------
        NotImplementedError
            Always. Bake position into the morph (e.g. ``gs.morphs.Mesh(pos=...)``) instead.
        """
        raise NotImplementedError(
            f"ParticleEntity '{self.name}' does not support set_pos. "
            "Bake position into the morph (e.g. gs.morphs.Mesh(pos=...)) instead."
        )

    def set_quat(self, quat, envs_idx=None) -> None:
        """Particle entities do not support set_quat after build.

        Raises
        ------
        NotImplementedError
            Always. Bake rotation into the morph (e.g. ``gs.morphs.Mesh(euler=...)``) instead.
        """
        raise NotImplementedError(
            f"ParticleEntity '{self.name}' does not support set_quat. "
            "Bake rotation into the morph (e.g. gs.morphs.Mesh(euler=...)) instead."
        )

    def get_mass(self):
        """Return the total particle mass from Genesis' ``get_mass``.

        Falls back to 0, with a warning, when the Genesis entity's ``get_mass`` raises ``TypeError``.
        """
        try:
            return self._entity.get_mass()
        except TypeError:
            en.logger.warning(
                f"ParticleEntity '{self.name}': get_mass() not supported by Genesis particle entity. Returning 0."
            )
            return 0

    def get_particles_in_bbox(self, lower, upper):
        """Return a mask of particles whose rest position lies in an axis-aligned box.

        Thin wrapper around the Genesis MPM entity's ``get_particles_in_bbox``. Handy
        for selecting a subset of particles to constrain (see ``set_particle_constraints``).

        Parameters
        ----------
        lower, upper : array_like[float, float, float]
            Lower/upper corners of the bounding box, in world coordinates.
        """
        return self._entity.get_particles_in_bbox(lower, upper)

    def set_particle_constraints(self, mask, link_idx, **kwargs):
        """Attach the masked particles to a rigid link (soft spring constraint).

        Thin wrapper around the Genesis MPM entity's ``set_particle_constraints``, so
        an MPM soft body can be welded to a moving rigid body without reaching through
        ``entity.entity``.

        Parameters
        ----------
        mask : array_like
            Boolean/index mask of particles to constrain (e.g. from ``get_particles_in_bbox``).
        link_idx : int
            Global index of the rigid link to follow (e.g. ``rigid.links[0].idx``).
        **kwargs
            Forwarded to Genesis (e.g. ``stiffness``).
        """
        self._entity.set_particle_constraints(mask, link_idx, **kwargs)
