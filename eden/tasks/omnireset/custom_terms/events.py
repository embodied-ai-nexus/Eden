"""Reset each env to one of UWLab's recorded OmniReset states."""

from __future__ import annotations

import os

from eden.tasks.omnireset.constants import RESET_FAMILIES, RIGS, RigName
from eden.tasks.omnireset.datasets import buffer_path
from eden.managers.event_manager import EVENT_TERM_REGISTRY
from eden.managers.terms.events.state_buffer import ResetFromStateBuffer


@EVENT_TERM_REGISTRY.register()
class OmniResetStates(ResetFromStateBuffer):
    """:class:`ResetFromStateBuffer` over ``pair``'s recorded ``families``, on robot ``rig``.

    UWLab trains from all four families and evaluates from ``ObjectAnywhereEEAnywhere`` alone. On UWLab's own robot
    (``rig="uwlab"``) the buffer is converted on first build. On Eden's composite (``rig="ur5e_2f85"``) the rows
    have to be retargeted first, once, with ``python -m eden.tasks.omnireset.retarget``; it holds all
    four families, so pick a subset with ``category_weights``. The buffer file follows from ``pair``, ``families``
    and ``rig``, so ``path`` is refused rather than overwritten.
    """

    pair: str = "peg"
    families: list[str] = list(RESET_FAMILIES)
    rig: RigName = "uwlab"

    def build(self) -> None:
        if self.rig not in RIGS:
            raise ValueError(
                f"OmniResetStates: unknown rig {self.rig!r}; rigs: {list(RIGS)}. Fix: "
                f"OmniResetStates.configure(rig='ur5e_2f85') for Eden's composite, or leave rig as 'uwlab'."
            )
        if self.path is not None:
            raise ValueError(
                f"OmniResetStates got path={self.path!r}, but it restores UWLab's states for pair={self.pair!r} / "
                f"families={list(self.families)} / rig={self.rig!r} from its own converted buffer. Fix: drop path=; "
                f"to restore a buffer of your own, use ResetFromStateBuffer.configure(path={self.path!r}) instead."
            )
        if self.rig == "uwlab":
            self.path = buffer_path(self.pair, tuple(self.families))
        else:
            from eden.tasks.omnireset.retarget import retargeted_path

            if list(self.families) != list(RESET_FAMILIES):
                raise ValueError(
                    f"OmniResetStates(rig='{self.rig}') restores the retargeted buffer of all four families; "
                    f"families={self.families} cannot narrow it. Fix: pass category_weights instead, e.g. "
                    "{'ObjectAnywhereEEAnywhere': 1.0}."
                )
            self.path = retargeted_path(self.pair, self.rig)
            if not os.path.exists(self.path):
                raise FileNotFoundError(
                    f"OmniResetStates(rig='{self.rig}', pair='{self.pair}') needs UWLab's reset states retargeted onto "
                    f"that robot, and {self.path} does not exist yet. Fix: run once\n"
                    f"    python -m eden.tasks.omnireset.retarget --pair {self.pair} --rig {self.rig}"
                )
        super().build()
