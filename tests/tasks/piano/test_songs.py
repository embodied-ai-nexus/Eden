"""Scores: sampling onto the control grid, the built-in tunes, and how a song name resolves to a score."""

import numpy as np
import pytest

from eden.options.objects.piano import NUM_KEYS, key_of_midi_note
from eden.tasks.piano import songs
from eden.tasks.piano.midi import read_midi
from eden.tasks.piano.songs import SONGS, Note, Song, get_song, piano_roll, song_steps

STEP_DT = 0.05
C4, G4 = key_of_midi_note(60), key_of_midi_note(67)


def test_piano_roll_puts_each_note_on_the_control_grid():
    # 60 bpm: one beat is one second, 20 control steps.
    roll = piano_roll(Song((Note(60, 0.0, 0.5), Note(67, 1.0, 1.0)), bpm=60.0), STEP_DT, lead_in=0.25)
    assert roll.shape == (5 + 40, NUM_KEYS) and roll.dtype == bool
    assert np.flatnonzero(roll[:, C4]).tolist() == list(range(5, 15))
    assert np.flatnonzero(roll[:, G4]).tolist() == list(range(25, 45))
    assert roll.sum() == 30


def test_a_note_shorter_than_a_control_step_still_gets_one():
    roll = piano_roll(Song((Note(60, 0.0, 0.01), Note(67, 1.0, 1.0)), bpm=60.0), STEP_DT)
    assert np.flatnonzero(roll[:, C4]).tolist() == [0]


@pytest.mark.parametrize("name", sorted(SONGS))
def test_built_in_songs_are_playable(name):
    """Every key comes back up before it is struck again, and no step asks for more keys than there are fingers."""
    roll = piano_roll(get_song(name), STEP_DT, lead_in=1.0)
    assert not roll[:20].any() and roll[20].any()  # the 1 s lead-in is silent
    assert 1 <= roll.sum(axis=1).max() <= 10
    assert song_steps(name, STEP_DT, 1.0) == roll.shape[0]
    notes = get_song(name).notes
    for note in notes:
        again = [n.start for n in notes if n.midi == note.midi and n.start > note.start]
        if again:
            assert min(again) - (note.start + note.length) >= 0.25 - 1e-9


@pytest.mark.parametrize(
    "song, match",
    [
        (Song((Note(60, 0.0, 0.0),), bpm=60.0), "no length"),
        (Song((Note(12, 0.0, 1.0),), bpm=60.0), "off the keyboard"),
    ],
)
def test_a_bad_score_is_refused_when_rolled(song, match):
    with pytest.raises(ValueError, match=match):
        piano_roll(song, STEP_DT)


def test_an_unknown_name_lists_the_songs_and_how_to_name_a_file():
    with pytest.raises(ValueError, match="Unknown song 'chopsticks'.*twinkle_twinkle.*filename alone"):
        get_song("chopsticks")


@pytest.fixture
def scores_dir(tmp_path, monkeypatch):
    """An empty ``SCORES_DIR`` under ``tmp_path``.

    A filesystem boundary: the real one is a per-user cache directory resolved at import, and a test must not
    write there.
    """
    path = tmp_path / "scores"
    path.mkdir()
    monkeypatch.setattr(songs, "SCORES_DIR", str(path))
    return path


def test_a_path_is_read_and_a_bare_filename_resolves_against_the_scores_dir(scores_dir, midi, midi_path):
    """A bare filename is what keeps a saved config portable across machines."""
    assert get_song(midi_path) == read_midi(midi_path)
    (scores_dir / "portable.mid").write_bytes(midi.two_hands())
    assert get_song("portable.mid") == read_midi(midi_path)


def test_a_path_that_exists_wins_over_a_same_named_score_in_the_scores_dir(scores_dir, midi, write_midi):
    left_only = midi.header(1) + midi.track(midi.name("Piano left"), *midi.note(48))
    (scores_dir / "same.mid").write_bytes(left_only)
    explicit = write_midi("same.mid", midi.two_hands())
    assert get_song(explicit).reach("right") is not None
    assert get_song("same.mid").reach("right") is None


def test_a_missing_score_names_the_scores_dir_and_what_is_in_it(scores_dir, midi):
    (scores_dir / "have.mid").write_bytes(midi.two_hands())
    with pytest.raises(FileNotFoundError, match="No MIDI file at 'absent.mid'") as raised:
        get_song("absent.mid")
    assert str(scores_dir) in str(raised.value) and "have.mid" in str(raised.value)
