"""``SongCommand.build``: a score the scene's hands or episode cannot play is refused at build, not trained on.

Placement and episode length come from the ``--song`` mod, so any other way of setting the score keeps the old
hands and episode. The checks read only the built env and the term's ``song``, so each test re-runs the real
term's ``build()`` against one shared env, with the song swapped and then restored.
"""

import pytest

from eden.options.objects.piano import NUM_KEYS
from eden.tasks import TASK_REGISTRY
from eden.tasks.piano.config import hand_placement
from eden.tasks.piano.songs import SONGS, get_song, song_steps
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis

TPB = 480


@pytest.fixture(scope="module")
def song_term():
    """The ``song`` term of a twinkle env: the hands on their fitted rests, a 230-step episode."""
    cfg = TASK_REGISTRY.build("piano", song="twinkle_twinkle")
    cfg.env_options.num_envs = 1
    return build_env(cfg).command_manager.get_term("song")


@pytest.fixture
def rebuild(song_term):
    """``rebuild(song)``: re-run ``song_term.build()`` for ``song``; the twinkle build is restored afterwards."""

    def run(song: str):
        song_term.song = song
        song_term.build()
        return song_term

    yield run
    song_term.song = "twinkle_twinkle"
    song_term.build()


@pytest.mark.parametrize("name", sorted(SONGS))
def test_every_built_in_tune_is_within_reach_of_the_fitted_rests(rebuild, name):
    """``ode_to_joy``'s bass G is 3.4 cm past the left wrist's travel and reached by the little finger."""
    assert rebuild(name).n_steps == song_steps(name, 0.05, 1.0)


def test_a_part_the_hands_are_not_placed_for_is_refused(rebuild, midi, write_midi):
    """C7 in the right hand and C1 in the left: octaves away from where the fitted rests put the hands."""
    far = midi.header(3) + midi.track(midi.name("tempo"), midi.tempo(0, 500_000))
    for track_name, note in (("Piano right", 96), ("Piano left", 24)):
        far += midi.track(midi.name(track_name), *midi.note(note))
    with pytest.raises(ValueError, match="right hand is not placed to play"):
        rebuild(write_midi("far_apart.mid", far))


def test_notes_naming_no_hand_are_held_to_what_the_fingers_can_touch(rebuild, midi, write_midi):
    """Most files name no hand, so they cannot be placed; what is left to check is bare reachability."""
    path = write_midi("out_of_reach.mid", midi.header(1) + midi.track(*midi.note(60), *midi.note(84)))
    with pytest.raises(ValueError, match="Its notes name no hand this scene places.*outside any hand's span"):
        rebuild(path)


def test_an_unnamed_staff_is_checked_even_when_another_staff_names_its_hand(rebuild, midi, write_midi):
    """The checks are per note: the unnamed staff's notes reach neither hand's part, so the fallback must see them.

    The named part spans the right hand's rest notes, C4 to G4, so the fitted rest is placed for it.
    """
    named = midi.track(midi.name("Piano right"), *midi.note(60), *midi.note(67))
    unnamed = midi.track(midi.name("Staff 2"), *midi.note(96))  # C7, far outside either hand's span
    with pytest.raises(ValueError, match="1 of its notes names no hand this scene places") as raised:
        rebuild(write_midi("one_staff_named.mid", midi.header(2) + named + unnamed))
    assert "key 75" in str(raised.value)


def test_a_score_longer_than_the_episode_is_refused(rebuild, midi, write_midi):
    """Cut off where the episode ends, the policy would train on a prefix and the metrics would look healthy."""
    # 24 beats at the default 120 bpm is 12 s, 260 control steps with the lead-in, against the 230-step episode.
    path = write_midi("long.mid", midi.header(1) + midi.track(*midi.note(60, ticks=24 * TPB)))
    with pytest.raises(ValueError, match=r"episode ends before .* does.*260 control steps.*is 230"):
        rebuild(path)


@pytest.mark.slow
def test_a_two_handed_score_the_mod_placed_builds(midi_path):
    """The accepting side: both hands named and placed by the mod, the episode sized to the score."""
    cfg = TASK_REGISTRY.build("piano", song=midi_path)
    cfg.env_options.num_envs = 1
    env = build_env(cfg)
    song = env.command_manager.get_term("song")
    assert song.n_steps == env.max_episode_length
    assert song.goal.shape == (1, NUM_KEYS)
    for side in ("right", "left"):
        wrist_y = float(env.entities[f"{side}_hand"].default_root_pos[0, 1])
        assert wrist_y == pytest.approx(hand_placement(get_song(midi_path), side)[0])
