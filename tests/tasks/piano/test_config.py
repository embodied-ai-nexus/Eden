"""The piano task config: the ``--song`` mod sizes the episode and places each hand over its part."""

import pytest

from eden.options.objects.piano import key_of_midi_note, key_y
from eden.tasks import TASK_REGISTRY
from eden.tasks.piano.config import (
    _REST_NOTES,
    _WRIST_POSITION_SCALE,
    _WRIST_X,
    _WRIST_Y,
    _WRIST_Z,
    HANDS,
    _keys_centre,
    fingertip_links,
    hand_placement,
    hand_rest_pose,
    hand_rest_quat,
)
from eden.tasks.piano.songs import SONGS, Note, Song, song_steps

STEP_DT = 0.05


def _lean(side: str) -> float:
    """How far a resting wrist sits off the centre of the keys its fingers rest on."""
    return _WRIST_Y[side] - _keys_centre(*_REST_NOTES[side])


@pytest.mark.parametrize("song", sorted(SONGS))
def test_a_built_in_tune_sets_the_episode_and_leaves_the_hands_on_their_fitted_rests(song):
    cfg = TASK_REGISTRY.build("piano", song=song)
    assert cfg.command_options.song.song == song
    assert cfg.env_options.sim_dt * cfg.env_options.decimation == pytest.approx(STEP_DT)
    assert cfg.env_options.episode_length_s == pytest.approx(song_steps(song, STEP_DT, 1.0) * STEP_DT)
    for side in HANDS:
        assert getattr(cfg.scene_options, f"{side}_hand").default_root_pos == pytest.approx(
            (_WRIST_X, _WRIST_Y[side], _WRIST_Z)
        )
        assert getattr(cfg.action_options, f"{side}_wrist").position_scale == pytest.approx(_WRIST_POSITION_SCALE)


def test_an_unknown_song_is_rejected_when_the_config_is_built():
    with pytest.raises(ValueError, match="Unknown song"):
        TASK_REGISTRY.build("piano", song="chopsticks")


def test_a_two_handed_score_puts_each_hand_over_its_own_part(midi_path):
    """One note per hand is no reach at all, so the travel stays what a built-in tune gets."""
    cfg = TASK_REGISTRY.build("piano", song=midi_path)
    assert cfg.runner_options.run_name == "two_hands"
    for side, note in (("right", 60), ("left", 48)):
        assert getattr(cfg.scene_options, f"{side}_hand").default_root_pos[1] == pytest.approx(
            key_y(key_of_midi_note(note)) + _lean(side)
        )
        assert getattr(cfg.action_options, f"{side}_wrist").position_scale == pytest.approx(_WRIST_POSITION_SCALE)


def test_a_part_wider_than_a_hand_span_centres_the_hand_and_widens_the_travel_to_reach_both_ends():
    low, high = key_y(key_of_midi_note(48)), key_y(key_of_midi_note(72))
    wrist_y, travel = hand_placement(
        Song((Note(48, 0.0, 0.5, "right"), Note(72, 1.0, 0.5, "right")), bpm=60.0), "right"
    )
    assert wrist_y - travel == pytest.approx(low + _lean("right"))
    assert wrist_y + travel == pytest.approx(high + _lean("right"))


def test_each_hand_flies_on_its_own_controller_and_every_joint_is_driven():
    cfg = TASK_REGISTRY.build("piano")
    assert not cfg.scene_options.attachments_dict  # nothing carries a hand
    for side in HANDS:
        hand = getattr(cfg.scene_options, f"{side}_hand")
        assert hand.metadata.side == side
        # A free body held up by gravity compensation and its wrist controller.
        assert not hand.is_fixed_base
        assert hand.material.gravity_compensation == 1.0
        assert hand.default_root_quat == hand_rest_quat(side)
        assert set(hand_rest_pose(side)) == set(hand.dofs_name)  # `default_dofs_pos` needs every DOF exactly once
        assert getattr(cfg.action_options, f"{side}_hand").dofs_name == list(hand.dofs_name)
        wrist = getattr(cfg.action_options, f"{side}_wrist")
        assert wrist.entity_name == f"{side}_hand" and not wrist.dofs_name
        assert all(link.startswith(side[0]) for link in fingertip_links(side))
