"""Config surgery shared by the single-object TaskMods of PickPlace and NutAssembly.

robosuite's ``single_object_mode=2`` variants (``PickPlaceMilk``, ``NutAssemblySquare``, ...) keep
one object and remove the others from the model. The Eden configs are authored with every object
present and one term per object, keyed ``<stage>_<object>`` / ``<object>_<field>``; a variant mod
calls :func:`drop_object` for each object it does not want, which removes the scene entity
(``SceneOptions.remove_entity``) and disables every term that names it (``disable_term``). Success
is the AND of the contributing metrics that remain, so it narrows to the kept object's on its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eden.options.managers.base import ManagerOptions
from eden.options.managers.observations import ObservationGroupOptions

if TYPE_CHECKING:
    from eden.utils.configs import EdenRLConfig


#: Manager entries that hold terms of their own: an observation group, a reward group.
_TERM_HOLDERS = (ManagerOptions, ObservationGroupOptions)


def names_object(key: str, name: str) -> bool:
    """True when ``name`` is one underscore-separated token of term key ``key`` (``in_bin_milk`` names ``milk``)."""
    return name in key.split("_")


def _disable_terms_naming(options: ManagerOptions | ObservationGroupOptions | None, name: str) -> int:
    """Disable every term of ``options`` whose key names ``name``, nested holders included; return the count."""
    if options is None:
        return 0
    removed = 0
    keys_naming_object = []
    for key in options.term_keys():
        if names_object(key, name):
            keys_naming_object.append(key)
            continue
        value = getattr(options, key)
        if isinstance(value, _TERM_HOLDERS):  # an observation group, a reward group
            removed += _disable_terms_naming(value, name)
    # After the nested pass: `RewardManagerOptions.disable_term` refuses a key that is still
    # declared both at the top level and inside a group.
    for key in keys_naming_object:
        options.disable_term(key)
    return removed + len(keys_naming_object)


def drop_object(config: EdenRLConfig, name: str) -> None:
    """Remove scene entity ``name`` and every manager term whose key names it (see :func:`names_object`).

    Raises if the scene has no such entity or no term named it — a misspelt object name must not
    silently leave the config untouched.
    """
    config.scene_options.remove_entity(name)
    removed = 0
    for manager in (
        config.observation_options,
        config.reward_options,
        config.metric_options,
        config.termination_options,
        config.event_options,
        config.command_options,
        config.curriculum_options,
    ):
        removed += _disable_terms_naming(manager, name)
    if removed == 0:
        raise KeyError(f"no manager term names object '{name}'; the per-object terms must be keyed on it.")
