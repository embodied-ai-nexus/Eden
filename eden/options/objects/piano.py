"""An 88-key piano keyboard.

Asset: ``objects/piano/piano.xml`` in the ``Kashu7100/eden_assets`` dataset. The dimensions and joint parameters
are those of the procedurally built piano in `RoboPianist <https://github.com/google-research/robopianist>`_
(Apache-2.0), which follow the Kawai upright regulation manual; ``build_piano.py`` next to the asset regenerates it
and lists the differences.

A fixed base carries 88 hinged keys, ``key_0`` (A0, MIDI note 21) to ``key_87`` (C8, MIDI note 108). Each key is
held up by a joint spring (2 N·m/rad, damping 0.05, armature 0.001) and travels until it meets its joint limit:
10 mm at the tip of a white key, 8 mm for a black one. Keys collide with whatever touches them but never with each
other.

The keyboard is silent by itself. :class:`~eden.engine.sensors.piano_audio.PianoAudio` gives it a voice.
"""

from eden.options.entities import EntityOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.surfaces import DefaultSurfaceOptions

NUM_KEYS = 88
# MIDI note number of `key_0`, the A0 at the far left.
LOWEST_MIDI_NOTE = 21
_BLACK_PITCH_CLASSES = frozenset({1, 3, 6, 8, 10})
# Keyboard geometry, from the asset: white keys on a 23.5 mm pitch, the first of them centred here.
WHITE_KEY_PITCH = 0.0235
_FIRST_KEY_Y = -0.59925


def key_joint_name(key: int) -> str:
    """Name of the joint of key ``key``.

    Parameters
    ----------
    key : int
        Key number, ``0`` (A0) to ``87`` (C8).

    Returns
    -------
    str
        The joint name, which is also its DOF name.
    """
    if not 0 <= key < NUM_KEYS:
        raise ValueError(f"A piano has keys 0 to {NUM_KEYS - 1}, got {key}.")
    return f"key_{key}_joint"


def key_of_midi_note(note: int) -> int:
    """Key number that plays MIDI note ``note``.

    Parameters
    ----------
    note : int
        MIDI note number; middle C is 60.

    Returns
    -------
    int
        The key number, ``0`` to ``87``.
    """
    key = note - LOWEST_MIDI_NOTE
    if not 0 <= key < NUM_KEYS:
        raise ValueError(
            f"MIDI note {note} is off the keyboard, which spans {LOWEST_MIDI_NOTE} to "
            f"{LOWEST_MIDI_NOTE + NUM_KEYS - 1}."
        )
    return key


def is_black_key(key: int) -> bool:
    """Whether key ``key`` is a black key.

    Parameters
    ----------
    key : int
        Key number, ``0`` (A0) to ``87`` (C8).

    Returns
    -------
    bool
        ``True`` for a sharp or flat.
    """
    return (key + LOWEST_MIDI_NOTE) % 12 in _BLACK_PITCH_CLASSES


def key_y(key: int) -> float:
    """Where key ``key`` sits along the keyboard, in the keyboard's own frame.

    White keys are evenly spaced and a black key sits midway between the two whites it is wedged between. This
    restates the asset's geometry so that a hand can be placed over a key without building a scene first;
    ``tests/options/objects/test_piano.py`` pins it against ``piano.xml``.

    Parameters
    ----------
    key : int
        Key number, ``0`` (A0) to ``87`` (C8).

    Returns
    -------
    float
        Offset along ``y`` from the keyboard's origin, in metres.
    """
    if not 0 <= key < NUM_KEYS:
        raise ValueError(f"A piano has keys 0 to {NUM_KEYS - 1}, got {key}.")
    if is_black_key(key):
        return 0.5 * (key_y(key - 1) + key_y(key + 1))
    return _FIRST_KEY_Y + sum(not is_black_key(lower) for lower in range(key)) * WHITE_KEY_PITCH


class Piano(EntityOptions):
    """88-key piano keyboard: a fixed base with one spring-loaded hinge per key."""

    file: str = "objects/piano/piano.xml"

    # The underside of the keys is the underside of the keyboard, and a pressed key dips 10 mm below it. Resting
    # on a floor (z = 0) the keys would meet it after a millimetre of travel and never reach their strike point,
    # so the default leaves them room. Any other placement has to leave that gap too, and stand the keyboard on
    # nothing wider than its base.
    default_root_pos: tuple[float, float, float] = (0.0, 0.0, 0.02)

    # The file's root body has no joint, so Genesis welds the keyboard to the world; the declaration has to say so.
    is_fixed_base: bool = True
    is_articulated: bool = True
    # MJCF declaration order, which is key order: DOF i is key i.
    dofs_name: list[str] = [key_joint_name(key) for key in range(NUM_KEYS)]

    # The keys are balanced, as a real key is by its lead weights. Genesis anchors a joint spring at qpos0 and has
    # no `springref`, so the spring cannot also be preloaded to carry the key: without this each key would sag a
    # fifth of its travel under its own weight. With it a key rests at exactly 0.
    material: RigidMaterialOptions = RigidMaterialOptions(gravity_compensation=1.0)
    # The keys are boxes; interpolated vertex normals would shade them as rounded pebbles.
    surface: DefaultSurfaceOptions = DefaultSurfaceOptions(smooth=False)
