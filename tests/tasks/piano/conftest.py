"""Shared piano fixtures: MIDI files built byte by byte, so the parts a MIDI file can be subtle about are pinned."""

from __future__ import annotations

import pytest

TICKS_PER_BEAT = 480


class Midi:
    """Encoders for the handful of Standard MIDI File pieces the tests need."""

    @staticmethod
    def varlen(value: int) -> bytes:
        """A variable-length quantity: seven bits per byte, high bit set on all but the last."""
        out = [value & 0x7F]
        value >>= 7
        while value:
            out.append((value & 0x7F) | 0x80)
            value >>= 7
        return bytes(reversed(out))

    @classmethod
    def header(cls, n_tracks: int, midi_format: int = 1, division: int = TICKS_PER_BEAT, length: int = 6) -> bytes:
        """An MThd chunk, with every field the reader has to honour open to being made wrong."""
        body = midi_format.to_bytes(2, "big") + n_tracks.to_bytes(2, "big") + division.to_bytes(2, "big")
        return b"MThd" + length.to_bytes(4, "big") + body.ljust(length, b"\x00")

    @classmethod
    def track(cls, *events: bytes) -> bytes:
        """One MTrk chunk around already-encoded, delta-timed events."""
        body = b"".join(events) + cls.varlen(0) + b"\xff\x2f\x00"
        return b"MTrk" + len(body).to_bytes(4, "big") + body

    @classmethod
    def name(cls, text: str) -> bytes:
        """A track-name meta event at delta 0."""
        return cls.varlen(0) + b"\xff\x03" + cls.varlen(len(text)) + text.encode()

    @classmethod
    def tempo(cls, delta: int, microseconds_per_beat: int) -> bytes:
        """A set-tempo meta event."""
        return cls.varlen(delta) + b"\xff\x51\x03" + microseconds_per_beat.to_bytes(3, "big")

    @classmethod
    def note(cls, midi: int, *, delta: int = 0, ticks: int = TICKS_PER_BEAT) -> tuple[bytes, bytes]:
        """A note-on after ``delta`` ticks and its note-off ``ticks`` later, each with its own status byte."""
        return cls.varlen(delta) + bytes([0x90, midi, 0x40]), cls.varlen(ticks) + bytes([0x80, midi, 0x40])

    @classmethod
    def one_note(cls, *extra: bytes) -> bytes:
        """A track holding middle C for one beat, after any events given."""
        return cls.track(*extra, *cls.note(60))

    @classmethod
    def two_hands(cls) -> bytes:
        """Middle C in the right hand for a beat, then C3 in the left for a beat at half speed.

        A tempo change mid-file, a release by zero-velocity note-on under running status, and the track names a
        two-staff export uses to say which hand plays.
        """
        conductor = cls.track(cls.name("tempo"), cls.tempo(0, 500_000), cls.tempo(TICKS_PER_BEAT, 1_000_000))
        right = cls.track(
            cls.name("Piano right"), cls.varlen(0) + b"\x90\x3c\x40", cls.varlen(TICKS_PER_BEAT) + b"\x3c\x00"
        )
        left = cls.track(cls.name("Piano left"), *cls.note(48, delta=TICKS_PER_BEAT))
        return cls.header(3) + conductor + right + left


@pytest.fixture
def midi() -> type[Midi]:
    return Midi


@pytest.fixture
def write_midi(tmp_path):
    """``write_midi(name, data) -> str``: write ``data`` to ``tmp_path / name`` and return the path."""

    def write(name: str, data: bytes) -> str:
        path = tmp_path / name
        path.write_bytes(data)
        return str(path)

    return write


@pytest.fixture
def midi_path(write_midi) -> str:
    """:meth:`Midi.two_hands` on disk, as ``two_hands.mid``."""
    return write_midi("two_hands.mid", Midi.two_hands())
