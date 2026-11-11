"""Observation layout of the RMA actor: one flat vector, five chunks.

The actor consumes one flat observation vector laid out as
``[prop | scan | priv_explicit | priv_latent | prop history]``. Every module here (networks,
algorithm, distillation) slices that vector through :class:`RmaObsDims` — the single source of
truth for slot indices, built from the actor options' ``num_*`` fields. A task selecting the RMA
actor must emit its actor observation set in exactly this order;
:class:`~eden.learning.wrappers.rsl_rl.rma.model.RmaActorModel` checks the total width at
construction.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RmaObsDims:
    """Widths of the five observation chunks (defaults match the extreme-parkour teacher)."""

    num_prop: int = 53
    num_scan: int = 132
    num_priv_explicit: int = 9
    num_priv_latent: int = 29
    num_hist: int = 10

    @property
    def total(self) -> int:
        return self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent + self.hist_dim

    @property
    def hist_dim(self) -> int:
        return self.num_prop * self.num_hist

    @property
    def prop_slice(self) -> slice:
        return slice(0, self.num_prop)

    @property
    def scan_slice(self) -> slice:
        start = self.num_prop
        return slice(start, start + self.num_scan)

    @property
    def priv_explicit_slice(self) -> slice:
        start = self.num_prop + self.num_scan
        return slice(start, start + self.num_priv_explicit)

    @property
    def priv_latent_slice(self) -> slice:
        start = self.num_prop + self.num_scan + self.num_priv_explicit
        return slice(start, start + self.num_priv_latent)

    @property
    def hist_slice(self) -> slice:
        start = self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent
        return slice(start, start + self.hist_dim)

    def layout(self) -> str:
        """Human-readable layout string for error messages."""
        return (
            f"[prop {self.num_prop} | scan {self.num_scan} | priv_explicit {self.num_priv_explicit}"
            f" | priv_latent {self.num_priv_latent} | history {self.num_hist}x{self.num_prop}]"
            f" = {self.total}"
        )
