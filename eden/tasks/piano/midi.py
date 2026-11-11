"""Reading a score out of a standard MIDI file.

A MIDI file is the form real repertoire comes in — the PIG dataset RoboPianist draws its 150 pieces from, and
every public-domain archive. This reader is deliberately small: it takes note-on / note-off pairs and the tempo
map, and ignores everything else a sequencer writes (velocity, pedal, program changes, lyrics), because a
:class:`~eden.tasks.piano.songs.Song` is a list of keys and times and nothing more.

Which hand plays a note comes from the name of the track it is on. A two-staff export names its tracks —
piano-midi.de writes ``Piano right`` / ``Piano left``; a hand marked only by MIDI channel is not read. A file
that does not say is read anyway, with every note unassigned; the task then leaves both hands at their default
rests.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NoReturn

from eden.options.objects.piano import LOWEST_MIDI_NOTE, NUM_KEYS
from eden.tasks.piano.songs import Note, Song

# A score is written in seconds here, so the tempo that turns beats into seconds is the identity.
_BPM = 60.0

# The keyboard this reader is reading for, as MIDI note numbers.
_LOWEST_KEY, _HIGHEST_KEY = LOWEST_MIDI_NOTE, LOWEST_MIDI_NOTE + NUM_KEYS - 1

_HEADER = b"MThd"
_TRACK = b"MTrk"
_DEFAULT_TEMPO = 500_000  # microseconds per quarter note, i.e. 120 bpm, what a file without a tempo event means
_SET_TEMPO = 0x51
_END_OF_TRACK = 0x2F
_TRACK_NAME = 0x03
_NOTE_OFF = 0x80
_NOTE_ON = 0x90
# Channel messages and how many data bytes follow the status byte.
_DATA_BYTES = {0x80: 2, 0x90: 2, 0xA0: 2, 0xB0: 2, 0xC0: 1, 0xD0: 1, 0xE0: 2}

_RIGHT = re.compile(r"\bright\b|\brh\b", re.IGNORECASE)
_LEFT = re.compile(r"\bleft\b|\blh\b", re.IGNORECASE)


class _Reader:
    """A cursor over a bytes buffer, reading the primitives a MIDI file is made of.

    Every read is bounds-checked, and every failure names the file. This is a parser of somebody
    else's bytes, so a short read is the one thing it must not absorb: slicing a ``bytes`` past its
    end yields a shorter string rather than raising, ``int.from_bytes(b"", "big")`` is ``0``, and a
    truncated length field therefore reads as an empty chunk whose whole track then disappears from
    the score with nothing raised anywhere.
    """

    def __init__(self, data: bytes, path: str) -> None:
        self.data, self.path, self.pos = data, path, 0

    def fail(self, what: str) -> NoReturn:
        """Refuse the file, naming it and where in it the reader was.

        Parameters
        ----------
        what : str
            What was wrong, as a phrase completing "is not readable as a MIDI file: ...".
        """
        raise ValueError(
            f"{self.path} is not readable as a MIDI file: {what} (at byte {self.pos} of {len(self.data)}). "
            f"Check the file is complete, and re-export it from a sequencer if it is not."
        )

    def take(self, n: int) -> bytes:
        """Read ``n`` bytes, refusing the file rather than returning fewer.

        Parameters
        ----------
        n : int
            How many bytes to read.

        Returns
        -------
        bytes
            Exactly ``n`` bytes.
        """
        if self.pos + n > len(self.data):
            self.fail(f"a {n}-byte read ran off the end of the file")
        chunk = self.data[self.pos : self.pos + n]
        self.pos += n
        return chunk

    def uint(self, n: int) -> int:
        """Read an ``n``-byte big-endian unsigned integer.

        Parameters
        ----------
        n : int
            Width in bytes.

        Returns
        -------
        int
            The value.
        """
        return int.from_bytes(self.take(n), "big")

    def varlen(self) -> int:
        """Read a variable-length quantity: seven bits per byte, high bit meaning "more to come".

        Returns
        -------
        int
            The value.
        """
        value = 0
        for _ in range(4):  # a VLQ is at most four bytes
            byte = self.take(1)[0]
            value = (value << 7) | (byte & 0x7F)
            if not byte & 0x80:
                return value
        self.fail("a variable-length quantity ran past the four bytes that are its maximum")


def _track_hand(name: str) -> str | None:
    """Which hand a track's name says it is, or ``None`` if it does not say.

    Parameters
    ----------
    name : str
        The track's name meta event.

    Returns
    -------
    str | None
        ``"right"``, ``"left"``, or ``None``.
    """
    if _RIGHT.search(name):
        return "right"
    if _LEFT.search(name):
        return "left"
    return None


def _read_track(
    reader: _Reader, length: int, index: int
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]], str]:
    """Read one track chunk.

    Parameters
    ----------
    reader : _Reader
        Positioned at the first event of the track.
    length : int
        Length of the chunk in bytes.
    index : int
        Which track this is, counting from 0, for the error messages.

    Returns
    -------
    tuple[list[tuple[int, int, int]], list[tuple[int, int]], str]
        Note events as ``(tick, pitch, on)`` with ``on`` 1 or 0, tempo events as ``(tick, microseconds per
        quarter note)``, and the track's name (empty if it has none).
    """
    if reader.pos + length > len(reader.data):
        reader.fail(f"track {index} declares {length} bytes but only {len(reader.data) - reader.pos} remain")
    end = reader.pos + length
    notes: list[tuple[int, int, int]] = []
    tempos: list[tuple[int, int]] = []
    name, tick, status = "", 0, 0
    while reader.pos < end:
        tick += reader.varlen()
        byte = reader.data[reader.pos]
        if byte & 0x80:  # a status byte; otherwise this is running status and `status` still holds
            status = byte
            reader.pos += 1
        elif not status:
            reader.fail(
                f"track {index} has a data byte {byte:#04x} where an event should begin, with no running status "
                f"to carry over — either the track starts mid-event or the reader has lost its place in it"
            )
        if status == 0xFF:
            kind = reader.uint(1)
            n_bytes = reader.varlen()  # read before slicing: it advances the cursor past the length itself
            payload = reader.take(n_bytes)
            if kind == _SET_TEMPO:
                if n_bytes != 3:
                    reader.fail(f"track {index} has a set-tempo event of {n_bytes} bytes, where the format is 3")
                tempos.append((tick, int.from_bytes(payload, "big")))
            elif kind == _TRACK_NAME and not name:
                name = payload.decode("latin-1")
            elif kind == _END_OF_TRACK:
                break  # anything a file pads after this is not ours to interpret
            # A meta event ends any running status: the next event has to state its own.
            status = 0
        elif status in (0xF0, 0xF7):  # sysex, skipped whole
            reader.take(reader.varlen())
            status = 0
        else:
            n_data = _DATA_BYTES.get(status & 0xF0)
            if n_data is None:
                reader.fail(f"track {index} has an unknown status byte {status:#04x}")
            data = reader.take(n_data)
            if status & 0xF0 == _NOTE_ON:
                # A note-on with zero velocity is the common spelling of a note-off.
                notes.append((tick, data[0], int(data[1] > 0)))
            elif status & 0xF0 == _NOTE_OFF:
                notes.append((tick, data[0], 0))
    # No event may reach past the chunk. Snapping the cursor to the boundary without checking would turn every
    # way of losing the plot mid-track — a meta length that overruns it, a malformed event — into a silently
    # short score: the outer loop would resync on the next chunk and the notes after the desync would simply be
    # missing, with nothing raised. (Landing short is fine: that is the padding after an end-of-track event.)
    if reader.pos > end:
        reader.fail(f"track {index} ran {reader.pos - end} bytes past the {length} it declared")
    reader.pos = end
    return notes, tempos, name


def _seconds_at(tick: int, tempos: list[tuple[int, int]], ticks_per_beat: int) -> float:
    """When a tick falls, in seconds, under a tempo map.

    Parameters
    ----------
    tick : int
        The tick to place.
    tempos : list[tuple[int, int]]
        Tempo changes as ``(tick, microseconds per quarter note)``, sorted, starting at tick 0.
    ticks_per_beat : int
        The file's division.

    Returns
    -------
    float
        Seconds from the start of the file.
    """
    seconds, last_tick, tempo = 0.0, 0, tempos[0][1]
    for change_tick, change_tempo in tempos:
        if change_tick >= tick:
            break
        seconds += (change_tick - last_tick) * tempo / (ticks_per_beat * 1e6)
        last_tick, tempo = change_tick, change_tempo
    return seconds + (tick - last_tick) * tempo / (ticks_per_beat * 1e6)


def read_midi(path: str | Path) -> Song:
    """Read a standard MIDI file as a score in seconds.

    Parameters
    ----------
    path : str | Path
        The ``.mid`` file.

    Returns
    -------
    Song
        The notes, at ``bpm`` 60 so that a beat is a second, with :attr:`~eden.tasks.piano.songs.Note.hand` set
        from the track names where they say which hand plays.
    """
    data = Path(path).read_bytes()
    reader = _Reader(data, str(path))
    if reader.data[:4] != _HEADER:
        raise ValueError(f"{path} is not a MIDI file: it does not start with {_HEADER.decode()}.")
    reader.pos = 4
    header_length = reader.uint(4)
    if header_length < 6:
        reader.fail(f"its header declares {header_length} bytes, short of the six every MIDI header carries")
    header_end = reader.pos + header_length
    midi_format = reader.uint(2)
    n_declared = reader.uint(2)
    division = reader.uint(2)
    # A header may carry more than its six defined bytes; the surplus is not ours to read.
    reader.pos = header_end
    if division & 0x8000:
        raise ValueError(f"{path} is timed in SMPTE frames, which this reader does not read; re-export it in ticks.")
    if division == 0:
        reader.fail("its header says zero ticks per beat, which gives no note a time")
    if midi_format == 2:
        raise ValueError(
            f"{path} is a format 2 MIDI file, whose tracks are independent sequences meant to be played one "
            f"after another rather than parts of one piece. This reader plays every track at once, which would "
            f"stack them on top of each other into a piece that was never written; it will not guess how they "
            f"were meant to follow one another. Re-export it as format 1."
        )

    tracks = []
    tempos: list[tuple[int, int]] = []
    n_found = 0
    while reader.pos < len(data):
        chunk = reader.take(4)
        length = reader.uint(4)
        if chunk != _TRACK:  # an unknown chunk type is skipped whole, as the spec requires
            reader.take(length)
            continue
        notes, track_tempos, name = _read_track(reader, length, n_found)
        tempos += track_tempos
        if notes:
            tracks.append((n_found, notes, _track_hand(name)))
        n_found += 1
    if n_found != n_declared:
        reader.fail(f"its header declares {n_declared} tracks and the file carries {n_found}")
    if not tracks:
        raise ValueError(f"{path} has no notes.")
    tempos = sorted(tempos) or [(0, _DEFAULT_TEMPO)]
    if tempos[0][0] != 0:
        tempos.insert(0, (0, _DEFAULT_TEMPO))

    out: list[Note] = []
    unclosed, lengthless = 0, 0

    def close(pitch: int, start: int, stop: int, hand: str | None) -> None:
        """Emit the note a key held from ``start`` to ``stop`` sounded, or count it as lengthless."""
        nonlocal lengthless
        seconds = _seconds_at(start, tempos, division)
        length = _seconds_at(stop, tempos, division) - seconds
        if length > 0.0:
            out.append(Note(pitch, seconds, length, hand))
        else:
            lengthless += 1

    for index, notes, hand in tracks:
        down: dict[int, int] = {}
        for tick, pitch, on in sorted(notes, key=lambda event: (event[0], event[2])):
            if not _LOWEST_KEY <= pitch <= _HIGHEST_KEY:
                # Checked here rather than left to `key_of_midi_note`, which knows nothing of MIDI files: there
                # the same note surfaces as a bare pitch from `Song.reach` or from `piano_roll`, whichever the
                # task reaches first, naming neither the file it came out of nor where in it to look.
                raise ValueError(
                    f"{path}: track {index} plays MIDI note {pitch} at {_seconds_at(tick, tempos, division):.3f} s, "
                    f"which an 88-key piano has no key for — it spans {_LOWEST_KEY} to {_HIGHEST_KEY}. A part "
                    f"written for another instrument, or a pedal or cue marked as an out-of-range pitch, reads "
                    f"like this. Export the piano staves alone, or transpose that part into range."
                )
            if not on:
                if (start := down.pop(pitch, None)) is not None:
                    close(pitch, start, tick, hand)
            else:
                # A key already down is struck again: a real one has to come back up first, so the held note
                # ends here rather than the two merging into one press the score never asked to be repeated.
                if (start := down.get(pitch)) is not None:
                    close(pitch, start, tick, hand)
                down[pitch] = tick
        unclosed += len(down)
    if unclosed or lengthless:
        # Dropping these quietly would take notes out of the score the policy is graded against — and a hole
        # in the grading looks exactly like a policy that missed them.
        faults = [
            f"{count} {noun}"
            for count, noun in ((unclosed, "never closed by a note-off"), (lengthless, "closed on the tick it starts"))
            if count
        ]
        raise ValueError(
            f"{path} does not pair its notes up: it has {' and '.join(faults)}. This reader will not guess at "
            f"either — every note it dropped would be a key the policy is graded on never pressing. Re-export "
            f"the file from a sequencer, which closes its notes and gives each one a length."
        )
    if not out:
        raise ValueError(f"{path} has no notes with a length.")
    return Song(tuple(sorted(out, key=lambda note: (note.start, note.midi))), _BPM)
