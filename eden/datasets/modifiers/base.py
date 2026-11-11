"""Base class and registry for dataset modifiers."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from eden.options.datasets import DatasetModifierOptions
from eden.utils.common import ConfigurableMixin
from eden.utils.registry import Registry

if TYPE_CHECKING:
    from eden.utils.motion_reference.motion_data import BatchMotionSequence


class DatasetModifier(ConfigurableMixin[DatasetModifierOptions]):
    """Base class for composable dataset modifiers.

    Dataset modifiers are offline transforms applied to loaded motion sequences
    (after cache load or raw prepare) rather than per simulation step. They are
    useful for data augmentation and canonicalization — e.g. spatial offsets,
    rotations, time warps, or dropping fields that the downstream task does not
    need. Modifiers are composed via :class:`Compose`, similar to
    :class:`~eden.managers.modifiers.actions.actuators.Compose`::

        modifiers = [
            Translate.configure(offset=(0.0, 0.0, 0.05)),
        ]

    Subclasses override :meth:`apply_to_sequence` to transform a single
    :class:`~eden.utils.motion_reference.motion_data.BatchMotionSequence`.
    The default :meth:`apply` iterates over the configured keys (or every key
    when ``motion_sequence_keys`` is ``None``) and delegates to
    :meth:`apply_to_sequence`.
    """

    motion_sequence_keys: tuple[str, ...] | None = None

    #: Why this modifier cannot keep a sequence's body-model parameters (``body_pose_quat`` / ``body_shape``)
    #: consistent with what it does to the sequence; ``None`` when it can. Such a modifier refuses those sequences.
    _breaks_body_model: ClassVar[str | None] = None

    def apply(
        self,
        motion_sequences: dict[str, "BatchMotionSequence"],
    ) -> dict[str, "BatchMotionSequence"]:
        """Apply this modifier to every selected key and return the result.

        ``motion_sequence_keys=None`` (the default) applies to every key;
        an explicit empty tuple applies to none and is a deliberate no-op —
        use that to temporarily disable a modifier without removing it.
        """
        out = dict(motion_sequences)
        for key in self._selected_keys(motion_sequences):
            out[key] = self.apply_to_sequence(out[key])
        return out

    def _selected_keys(
        self, motion_sequences: dict[str, "BatchMotionSequence"], *, single: bool = False
    ) -> tuple[str, ...]:
        """The keys of ``motion_sequences`` this modifier applies to, checked before anything is applied.

        ``motion_sequence_keys=None`` selects every key. A modifier that drives one robot passes ``single=True``,
        refusing a selection of several keys — on a multi-key dataset (robot + object) it would otherwise treat each
        key as that robot. Call it before any expensive setup, so a misnamed key costs nothing.

        Raises
        ------
        ValueError
            With ``single=True``, when more than one key is selected.
            Also when a selected sequence carries body-model parameters this modifier would leave stale
            (``_breaks_body_model``).
        KeyError
            When a configured key is not in ``motion_sequences``.
        """
        keys = tuple(motion_sequences) if self.motion_sequence_keys is None else tuple(self.motion_sequence_keys)
        name = type(self).__name__
        if single and len(keys) > 1:
            raise ValueError(
                f"{name} expects exactly one motion-sequences key (the robot); got {keys}. Set "
                f"``motion_sequence_keys=(robot_key,)`` on the modifier to scope it explicitly on multi-key datasets."
            )
        missing = [key for key in keys if key not in motion_sequences]
        if missing:
            raise KeyError(f"{name}: key(s) {missing} not in motion_sequences {list(motion_sequences)}.")
        if self._breaks_body_model is not None:
            for key in keys:
                if motion_sequences[key].body_pose_quat is not None:
                    raise ValueError(
                        f"{name} cannot apply to '{key}': it carries body-model parameters (body_model_params=True) "
                        f"and {self._breaks_body_model}. Scope the modifier away from it with motion_sequence_keys, "
                        f"or load the dataset with body_model_params=False."
                    )
        return keys

    def apply_to_sequence(self, motion_seq: "BatchMotionSequence") -> "BatchMotionSequence":
        """Transform a single :class:`BatchMotionSequence`. Override in subclasses."""
        return motion_seq


DATASET_MODIFIER_REGISTRY = Registry("DATASET_MODIFIER")


def build_modifier(options: DatasetModifierOptions | dict) -> DatasetModifier:
    """Build the modifier ``options`` names, by registry name.

    ``options`` may be the options object or its dict form (a config loaded from YAML).
    """
    if isinstance(options, dict):
        options = DatasetModifierOptions(**options)
    return DATASET_MODIFIER_REGISTRY.get(options.name)(options=options)
