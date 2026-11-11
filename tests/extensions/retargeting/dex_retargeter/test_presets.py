"""The bundled ``HAND_PRESETS`` declare the side of the hand they build."""

import pytest

from eden.extensions.retargeting.dex_retargeter.presets import HAND_PRESETS


@pytest.mark.parametrize("name", sorted(HAND_PRESETS))
def test_preset_side_is_the_side_of_the_hand_it_builds(name):
    # A glove teleop refuses a preset whose side differs from the glove's, so a wrong ``side``
    # would let a left glove drive a right hand. The robot options name the hand ``*_L`` / ``*_R``.
    preset = HAND_PRESETS[name]
    robot_suffix = type(preset.robot_options_factory()).__name__.rsplit("_", 1)[-1]
    assert {"L": "left", "R": "right"}[robot_suffix] == preset.side
