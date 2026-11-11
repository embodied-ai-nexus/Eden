"""Scores for the piano task: a handful of public-domain tunes, and how a score becomes a piano roll.

A :class:`Song` is a list of :class:`Note` in beats plus a tempo. :func:`piano_roll` samples it at the control
rate into a ``(steps, 88)`` boolean array, ``roll[t, key]`` meaning "key is down during control step ``t``", which
is what the command term indexes and the policy is graded against.

RoboPianist's repertoire is the PIG dataset, which cannot be redistributed, so the tunes here are written out by
hand. They are short and sit under one hand position each, which is the right size for a first policy. Real
repertoire is a MIDI file away: :func:`get_song` takes the path of one (see :mod:`eden.tasks.piano.midi`), which
is the form PIG and every public-domain archive ship, and such a score usually says which hand plays each note.

No such file ships here — a MIDI file carries its own licence, and the archives' are not ours to pass on.
Bring your own: PIG (https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/, free for academic use
after registering) is what RoboPianist trains on, and piano-midi.de is the classical corpus its pieces are
sequenced from (CC BY-SA 3.0 DE, "for study and research only"; the site rejects scripted downloads, so fetch
it in a browser or from one of its GitHub mirrors).

Put the file in :data:`SCORES_DIR` and name it by filename alone. A score is named in the saved ``config.json``,
so an absolute path there describes one machine's disk and a checkpoint trained against it does not replay
anywhere else; a bare filename resolves against that directory on whichever machine reads the config.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

from eden.options.objects.piano import NUM_KEYS, key_of_midi_note
from eden.utils.misc import get_cache_dir

#: Where a bare score filename is looked up, so that the name in a saved config means the same thing on
#: another machine. Eden never writes here — it is yours to fill.
SCORES_DIR = get_cache_dir("scores")
_MIDI_SUFFIXES = (".mid", ".midi")


class Note(NamedTuple):
    """One key press.

    Parameters
    ----------
    midi : int
        MIDI note number; middle C is 60.
    start : float
        Beat the key goes down on.
    length : float
        Beats the key stays down.
    hand : str | None
        Which hand plays it, ``"right"`` or ``"left"``, where the score says so. A hand-written tune leaves it
        ``None``: what is graded is the keyboard, never the fingering, so this is only used to put a hand where
        it will be needed (:meth:`Song.reach`).
    """

    midi: int
    start: float
    length: float
    hand: str | None = None


class Song(NamedTuple):
    """A score.

    Parameters
    ----------
    notes : tuple[Note, ...]
        The notes, in any order.
    bpm : float
        Tempo in beats per minute.
    """

    notes: tuple[Note, ...]
    bpm: float

    @property
    def beats(self) -> float:
        """Length of the score in beats: the end of its last note."""
        return max(note.start + note.length for note in self.notes)

    @property
    def seconds(self) -> float:
        """Length of the score in seconds."""
        return self.beats * 60.0 / self.bpm

    def reach(self, hand: str) -> tuple[int, int] | None:
        """Lowest and highest key a hand has to reach, or ``None`` if the score does not name that hand.

        Parameters
        ----------
        hand : str
            ``"right"`` or ``"left"``.

        Returns
        -------
        tuple[int, int] | None
            Keyboard indices, low then high.
        """
        keys = [key_of_midi_note(note.midi) for note in self.notes if note.hand == hand]
        return (min(keys), max(keys)) if keys else None


def _melody(pitches: list[int], lengths: list[float], gap: float) -> tuple[Note, ...]:
    """Notes played one after another, each let go ``gap`` beats before the next goes down.

    The gap is what makes a repeated note playable at all: a key has to come back up before it can be struck again.
    """
    notes, beat = [], 0.0
    for midi, length in zip(pitches, lengths, strict=True):
        notes.append(Note(midi, beat, length - gap))
        beat += length
    return tuple(notes)


_C4, _D4, _E4, _F4, _G4, _A4, _B4, _C5 = 60, 62, 64, 65, 67, 69, 71, 72

_SCALE = [_C4, _D4, _E4, _F4, _G4, _F4, _E4, _D4, _C4]
_TWINKLE = [_C4, _C4, _G4, _G4, _A4, _A4, _G4, _F4, _F4, _E4, _E4, _D4, _D4, _C4]
_ODE = [_E4, _E4, _F4, _G4, _G4, _F4, _E4, _D4, _C4, _C4, _D4, _E4, _E4, _D4, _D4]

SONGS: dict[str, Song] = {
    # Five fingers on five keys, up and back: no hand travel, no repeated notes.
    "five_finger_scale": Song(_melody(_SCALE, [1.0] * len(_SCALE), gap=0.25), bpm=80.0),
    # Repeated notes and a reach of a sixth (C to A).
    "twinkle_twinkle": Song(_melody(_TWINKLE, [1.0] * 6 + [2.0] + [1.0] * 6 + [2.0], gap=0.25), bpm=90.0),
    # Two hands: the melody in the right, one held bass note a bar in the left.
    "ode_to_joy": Song(
        _melody(_ODE, [1.0] * 12 + [1.5, 0.5, 2.0], gap=0.25)
        + tuple(Note(midi, 4.0 * bar, 3.75) for bar, midi in enumerate((48, 43, 48, 43))),
        bpm=90.0,
    ),
}


@lru_cache(maxsize=8)
def _read_midi_cached(path: str) -> Song:
    """Read a MIDI score, remembering it: the task asks for the same score once per manager that needs it.

    Parameters
    ----------
    path : str
        The ``.mid`` file.

    Returns
    -------
    Song
        The score.
    """
    from eden.tasks.piano.midi import read_midi  # imported here: midi.py reads this module's Note and Song

    return read_midi(path)


def _resolve_score(name: str) -> Path:
    """Find the MIDI file a score name refers to.

    A path is taken as written. A bare filename is also looked for in :data:`SCORES_DIR`, which is the form
    that survives being written into a run's ``config.json`` and read back on another machine.

    Parameters
    ----------
    name : str
        A path, or the filename of a score in :data:`SCORES_DIR`.

    Returns
    -------
    Path
        The file, which exists.
    """
    given = Path(name).expanduser()
    if given.is_file():
        return given
    in_scores = Path(SCORES_DIR) / given.name
    if given.parent == Path(".") and in_scores.is_file():
        return in_scores
    if not Path(SCORES_DIR).is_dir():
        holds = "That directory does not exist yet."
    elif scores := sorted(path.name for path in Path(SCORES_DIR).iterdir() if path.suffix.lower() in _MIDI_SUFFIXES):
        holds = f"It currently holds: {', '.join(scores)}."
    else:
        holds = "It is currently empty."
    raise FileNotFoundError(
        f"No MIDI file at {name!r}. Eden ships no scores, so this one is yours to put there: drop it in "
        f"{SCORES_DIR} and name it by filename alone (song={given.name!r}). Do not reach for an absolute path "
        f"to make this go away — the name is written into the run's config.json, so an absolute path describes "
        f"this machine's disk and a checkpoint trained against it will not replay anywhere else. {holds}"
    )


def get_song(name: str) -> Song:
    """Look up a built-in song, or read one from a MIDI file.

    Parameters
    ----------
    name : str
        One of :data:`SONGS`; or a ``.mid`` / ``.midi`` file, either as a path or as the filename of one in
        :data:`SCORES_DIR`.

    Returns
    -------
    Song
        The score.
    """
    if name in SONGS:
        return SONGS[name]
    if Path(name).suffix.lower() in _MIDI_SUFFIXES:
        return _read_midi_cached(str(_resolve_score(name).resolve()))
    raise ValueError(
        f"Unknown song {name!r}. Available: {sorted(SONGS)}, or a .mid file — its path, or its filename alone "
        f"if it is in {SCORES_DIR}."
    )


def song_steps(name: str, step_dt: float, lead_in: float = 0.0) -> int:
    """Number of control steps a song takes, lead-in included.

    Parameters
    ----------
    name : str
        One of :data:`SONGS`, or a MIDI file (see :func:`get_song`).
    step_dt : float
        Control step, in seconds.
    lead_in : float
        Seconds of silence before the first beat.

    Returns
    -------
    int
        The length of its piano roll.
    """
    return piano_roll(get_song(name), step_dt, lead_in).shape[0]


def piano_roll(song: Song, step_dt: float, lead_in: float = 0.0) -> np.ndarray:
    """Sample a score at the control rate.

    A note is down on every control step whose start falls inside it, so a note shorter than one step still gets
    the step it starts in.

    Parameters
    ----------
    song : Song
        The score.
    step_dt : float
        Control step, in seconds.
    lead_in : float
        Seconds of silence before the first beat, for the hands to settle over the keys.

    Returns
    -------
    np.ndarray
        ``(steps, 88)`` bool, ``roll[t, key]``: key ``key`` is down during control step ``t``.
    """
    if step_dt <= 0.0:
        raise ValueError(f"step_dt must be positive, got {step_dt}.")
    if not song.notes:
        raise ValueError("The song has no notes.")
    beat = 60.0 / song.bpm
    n_steps = int(np.ceil((lead_in + song.seconds) / step_dt - 1e-9))
    roll = np.zeros((n_steps, NUM_KEYS), dtype=bool)
    for note in song.notes:
        if note.length <= 0.0:
            raise ValueError(f"{note} has no length.")
        first = int(np.ceil((lead_in + note.start * beat) / step_dt - 1e-9))
        last = int(np.ceil((lead_in + (note.start + note.length) * beat) / step_dt - 1e-9))
        roll[first : max(last, first + 1), key_of_midi_note(note.midi)] = True
    return roll
