"""``read_midi``: tempo maps, hands from track names, and malformed files refused by name rather than read short."""

import pytest

from eden.options.objects.piano import key_of_midi_note
from eden.tasks.piano.midi import read_midi
from eden.tasks.piano.songs import Note, get_song

TPB = 480  # ticks per beat, as the ``midi`` fixture writes them


def test_a_score_is_read_with_its_tempo_map_and_its_hands(midi_path):
    song = read_midi(midi_path)
    assert song.bpm == 60.0  # a beat is a second, so a Note's units are seconds either way
    assert song.notes == (Note(60, 0.0, 0.5, "right"), Note(48, 0.5, 1.0, "left"))
    assert song.seconds == pytest.approx(1.5)
    assert song.reach("right") == (key_of_midi_note(60),) * 2
    assert song.reach("left") == (key_of_midi_note(48),) * 2
    assert get_song("twinkle_twinkle").reach("right") is None  # a hand-written tune says nothing about hands


def test_legal_variations_of_an_ordinary_file_are_read(midi, write_midi):
    """A header longer than its six defined bytes, and padding after end-of-track, are both legal."""
    plain = read_midi(write_midi("plain.mid", midi.header(1) + midi.one_note()))
    assert plain.notes == (Note(60, 0.0, 0.5),)
    assert read_midi(write_midi("long.mid", midi.header(1, length=8) + midi.one_note())).notes == plain.notes
    assert read_midi(write_midi("padded.mid", midi.header(1) + midi.one_note() + b"")).notes == plain.notes


def test_striking_a_held_key_again_closes_the_note_it_was_holding(midi, write_midi):
    on, off = midi.note(60)
    restrike = midi.track(on, midi.varlen(TPB) + b"\x90\x3c\x40", off)
    assert read_midi(write_midi("restrike.mid", midi.header(1) + restrike)).notes == (
        Note(60, 0.0, 0.5),
        Note(60, 0.5, 0.5),
    )


@pytest.mark.parametrize(
    ("name", "build", "message"),
    [
        ("not.mid", lambda m: b"RIFF" + bytes(64), "not a MIDI file"),
        ("smpte.mid", lambda m: b"MThd" + (6).to_bytes(4, "big") + b"\x00\x00\x00\x01\xe8\x28" + m.one_note(), "SMPTE"),
        # Truncated mid-file: every short read used to clamp, so the last track silently vanished.
        ("cut.mid", lambda m: (m.header(2) + m.one_note() + b"MTrk" + (99).to_bytes(4, "big"))[:-2], "ran off the end"),
        (
            "cut_track.mid",
            lambda m: m.header(1) + b"MTrk" + (999).to_bytes(4, "big") + b"\x00\x90\x3c\x40",
            "only .* remain",
        ),
        # A meta event reaching past its track but inside the file: the rest of the track used to vanish.
        (
            "overrun.mid",
            lambda m: m.header(2) + m.track(m.varlen(0) + b"\xff\x01\x10" + b"xxxx") + m.one_note(),
            "track 0 ran .* past the .* it declared",
        ),
        # A meta event cancels running status, so note data after one has lost its place.
        (
            "running.mid",
            lambda m: m.header(1) + m.track(m.varlen(0) + b"\xff\x01\x00", m.varlen(0) + b"\x3c\x40"),
            "no running",
        ),
        ("no_status.mid", lambda m: m.header(1) + m.track(m.varlen(0) + b"\x3c\x40"), "no running status"),
        ("zero_division.mid", lambda m: m.header(1, division=0) + m.one_note(), "zero ticks per beat"),
        ("short_header.mid", lambda m: m.header(1, length=4) + m.one_note(), "short of the six"),
        ("format2.mid", lambda m: m.header(1, midi_format=2) + m.one_note(), "format 2"),
        ("miscount.mid", lambda m: m.header(3) + m.one_note(), "declares 3 tracks and the file carries 1"),
        (
            "tempo_len.mid",
            lambda m: m.header(1) + m.one_note(m.varlen(0) + b"\xff\x51\x02\x00\x00"),
            "set-tempo event of 2 bytes",
        ),
        # A note left hanging, and one closed on the tick it opened: both used to be dropped in silence.
        ("unclosed.mid", lambda m: m.header(1) + m.track(m.varlen(0) + b"\x90\x3c\x40"), "never closed by a note-off"),
        (
            "lengthless.mid",
            lambda m: m.header(1) + m.track(m.varlen(0) + b"\x90\x3c\x40", *m.note(60)),
            "closed on the tick it starts",
        ),
        # Off the keyboard: named here, with the file, the track and the time, not later by `key_of_midi_note`.
        (
            "off_keyboard.mid",
            lambda m: m.header(1) + m.track(*m.note(60), *m.note(12, ticks=TPB)),
            "track 0 plays MIDI note 12 at 0.500 s.*spans 21 to 108",
        ),
    ],
)
def test_a_malformed_file_is_refused_naming_the_file(midi, write_midi, name, build, message):
    """Each of these used to return a plausible Song with notes missing, or raise a bare IndexError."""
    with pytest.raises(ValueError, match=message) as raised:
        read_midi(write_midi(name, build(midi)))
    assert name in str(raised.value)
