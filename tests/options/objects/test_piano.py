"""The ``Piano`` keyboard's layout helpers, checked against its asset."""

import re

import pytest

from eden.options.objects.piano import NUM_KEYS, Piano, is_black_key, key_joint_name, key_of_midi_note, key_y
from eden.utils.assets import get_asset_path


def test_keyboard_layout_helpers():
    assert key_joint_name(0) == "key_0_joint"
    assert (key_of_midi_note(21), key_of_midi_note(60), key_of_midi_note(108)) == (0, 39, NUM_KEYS - 1)
    assert sum(is_black_key(key) for key in range(NUM_KEYS)) == 36
    assert [is_black_key(key) for key in range(5)] == [False, True, False, False, True]  # A0, A#0, B0, C1, C#1


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda: key_of_midi_note(20), "off the keyboard"),
        (lambda: key_of_midi_note(109), "off the keyboard"),
        (lambda: key_joint_name(NUM_KEYS), "keys 0 to 87"),
        (lambda: key_y(NUM_KEYS), "keys 0 to 87"),
    ],
)
def test_a_key_off_the_keyboard_raises(call, match):
    with pytest.raises(ValueError, match=match):
        call()


def test_key_positions_are_the_assets_key_positions():
    """``key_y`` restates geometry that lives in ``piano.xml``; if the asset moves a key, this is how we hear about it.

    Resolved through ``get_asset_path`` so a cold asset cache fetches the file (reading ``eden/assets`` directly
    failed whenever this test ran before anything had built a piano).
    """
    with open(get_asset_path(Piano().file)) as f:
        placed = re.findall(r'<body name="key_(\d+)"[^>]*pos="([^"]+)"', f.read())
    assert len(placed) == NUM_KEYS
    for key, pos in placed:
        assert key_y(int(key)) == pytest.approx(float(pos.split()[1]), abs=1e-9)
