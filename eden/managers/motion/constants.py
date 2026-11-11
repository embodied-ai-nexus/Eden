"""Shared constants for the motion subsystem.

Leaf module (no imports) so the motion collaborators (``outcome_curriculum``,
``bin_sampler``) can share it without circular-import risk.
"""

# Normalisation floor: clamps a sum/denominator away from 0 before dividing, so an
# all-zero weight/rate vector yields zeros instead of NaN/inf. Used by both the
# per-motion-id curriculum and the adaptive-bin weight normalisation.
_NORMALIZE_EPS: float = 1e-12
