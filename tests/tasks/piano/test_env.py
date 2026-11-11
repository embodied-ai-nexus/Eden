"""The piano task built and stepped: timing, what the reward and F1 pay for, and the hands' control.

What a piano task can get silently wrong is timing: that the step shown as "now" is the step graded, and that
pressing the right key at the right time is what pays. One twinkle env serves the module; every test resets it.
"""

import pytest
import torch

from eden.managers.terms.actions import FreeBasePoseController
from eden.options.objects.piano import NUM_KEYS, key_of_midi_note
from eden.tasks import TASK_REGISTRY
from eden.tasks.piano import custom_terms as ct
from eden.tasks.piano.config import HANDS, KEYBOARD_Z, fingertip_links
from eden.tasks.piano.songs import get_song, piano_roll
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis

STEP_DT = 0.05
LEAD_IN_STEPS = 20  # the task's 1 s lead-in
WINDOW = 11  # "now" plus the ten-step lookahead
# `FreeBasePoseController` takes no `dofs_name`: its six columns are a translation then a rotation vector.
WRIST_AXES = ["x", "y", "z", "rx", "ry", "rz"]
C4, G4 = key_of_midi_note(60), key_of_midi_note(67)
PRESS = 0.6  # joint torque that takes a key to the bottom of its stroke against its 2 N·m/rad spring


@pytest.fixture(scope="module")
def env():
    cfg = TASK_REGISTRY.build("piano", song="twinkle_twinkle")
    cfg.env_options.num_envs = 2
    return build_env(cfg)


def _roll(env) -> torch.Tensor:
    return torch.as_tensor(piano_roll(get_song("twinkle_twinkle"), STEP_DT, 1.0), device=env.device)


def _window(env) -> torch.Tensor:
    """Env 0's command window, ``(WINDOW, NUM_KEYS)`` bool."""
    return env.command_manager.get_term("song").command.view(env.num_envs, WINDOW, NUM_KEYS)[0].bool()


def _step(env, n_steps: int, key_torques: dict[int, float] | None = None) -> None:
    """Step with zero actions, holding ``key_torques`` (``{key: N·m}``) on the keyboard's own joints."""
    piano = env.entities["piano"]._entity
    torques = torch.zeros((env.num_envs, NUM_KEYS), device=env.device)
    for key, torque in (key_torques or {}).items():
        torques[:, key] = torque
    for _ in range(n_steps):
        piano.control_dofs_force(torques)
        env.step(torch.zeros_like(env.action_manager.action))


def _act(env, n_steps: int, **targets: dict[str, float]) -> None:
    """Step with a constant action, ``right_hand={"r_pinky_pip": 1.0}`` style, zero for everything unnamed."""
    action = {}
    for term in ("right_wrist", "right_hand", "left_wrist", "left_hand"):
        dofs_name = getattr(env.config.action_options, term).dofs_name or WRIST_AXES
        values = torch.zeros((env.num_envs, len(dofs_name)), device=env.device)
        for dof, value in targets.get(term, {}).items():
            values[:, dofs_name.index(dof)] = value
        action[term] = values
    for _ in range(n_steps):
        env.step(action)


def test_spaces(env):
    env.reset()
    assert env.action_manager.action.shape == (2, 52)  # 2 x (20 finger joints + a 6-axis wrist)
    # The score window + the keys + the last action + per hand: joint positions and velocities, and the wrist's
    # pose, linear velocity and angular velocity.
    assert env.observation_manager.compute()["policy"].shape == (2, WINDOW * NUM_KEYS + NUM_KEYS + 52 + 2 * (40 + 12))


def test_hands_rest_over_the_keys_they_start_on(env):
    env.reset()
    _step(env, 5)
    piano = env.entities["piano"]._entity
    key_y = {key: float(piano.get_link(f"key_{key}").get_pos()[0, 1]) for key in (C4, key_of_midi_note(48))}
    tips = {}
    for side in HANDS:
        hand = env.entities[f"{side}_hand"]._entity
        tips[side] = hand.get_links_pos()[0, [hand.get_link(name).idx_local for name in fingertip_links(side)]]
    # Right thumb on middle C, left little finger on the C below, both within half a key.
    assert float(tips["right"][0, 1]) == pytest.approx(key_y[C4], abs=0.012)
    assert float(tips["left"][4, 1]) == pytest.approx(key_y[key_of_midi_note(48)], abs=0.012)
    for side in HANDS:
        # Over the front half of the white keys (x in 0..0.075), above them (top 22.5 mm up), not on them.
        assert torch.all((tips[side][:, 0] > 0.0) & (tips[side][:, 0] < 0.075))
        height = tips[side][:, 2] - KEYBOARD_Z
        assert torch.all((height > 0.0225) & (height < 0.06))
    assert env.command_manager.get_term("song").key_pos.max() < 0.02


@pytest.mark.parametrize("side", HANDS)
def test_the_wrist_goes_where_its_action_says(env, side):
    """The free base is PD-driven to an offset from its rest pose: 10 cm along the keyboard, 0.3 rad per unit."""
    outward = 1.0 if side == "right" else -1.0  # away from the other hand, only 13 cm off
    env.reset()
    _act(env, 15, **{f"{side}_wrist": {"y": outward, "rz": -1.0}})
    moved, turned = ct.base_offset_from_default(env.entities[f"{side}_hand"])
    assert float(moved[0, 1]) == pytest.approx(0.10 * outward, abs=0.01)
    assert float(turned[0, 2]) == pytest.approx(-0.3, abs=0.03)
    assert moved[0, [0, 2]].abs().max() < 0.005
    assert turned[0, :2].abs().max() < 0.03


def test_a_hand_asked_for_nothing_holds_its_rest_pose(env):
    env.reset()
    _step(env, 40)
    for side in HANDS:
        moved, turned = ct.base_offset_from_default(env.entities[f"{side}_hand"])
        assert float(moved.abs().max()) < 1.0e-3  # a millimetre
        assert float(turned.abs().max()) < 1.0e-2


@pytest.mark.parametrize(
    "side, finger, midi",
    [("right", "thumb", 60), ("right", "index_finger", 62), ("right", "pinky", 67)]
    + [("left", "thumb", 55), ("left", "index_finger", 53), ("left", "pinky", 48)],
)
def test_each_finger_presses_the_key_it_rests_over_and_only_that_key(env, side, finger, midi):
    p = side[0]
    press = {
        "thumb": {f"{p}_thumb_cmc_abd": -1.0},
        # The knuckle flexes while the middle joint extends: the fingertip goes down rather than back.
        "index_finger": {f"{p}_index_finger_mcp_flex": 1.0, f"{p}_index_finger_pip": -1.0},
        "pinky": {f"{p}_pinky_mcp_flex": 1.5, f"{p}_pinky_pip": -1.0},
    }[finger]
    env.reset()
    song = env.command_manager.get_term("song")
    ever_down = torch.zeros_like(song.key_down[0])
    for _ in range(12):
        _act(env, 1, **{f"{side}_hand": press})
        ever_down |= song.key_down[0]
    assert ever_down[key_of_midi_note(midi)]
    assert int(ever_down.sum()) == 1


def test_the_step_shown_as_now_is_the_step_that_is_graded(env):
    env.reset()
    song, roll = env.command_manager.get_term("song"), _roll(env)
    assert song.n_steps == roll.shape[0]
    for step in range(LEAD_IN_STEPS + 3):
        shown_as_now = _window(env)[0].clone()
        assert torch.equal(shown_as_now, roll[step])
        _step(env, 1)
        assert torch.equal(song.goal[0], shown_as_now)
    now = LEAD_IN_STEPS + 3
    assert torch.equal(_window(env), roll[now : now + WINDOW])


def test_pressing_the_due_key_is_what_pays(env):
    """Twinkle opens on middle C at step 20. Play it, play a wrong key, and play nothing."""
    scores = {}
    for name, key_torques in {"right": {C4: PRESS}, "wrong": {G4: PRESS}, "nothing": {}}.items():
        env.reset()
        _step(env, LEAD_IN_STEPS)
        _step(env, 6, key_torques)  # into the first note, which lasts 10 steps
        song = env.command_manager.get_term("song")
        assert song.goal[0, C4] and song.goal[0].sum() == 1
        scores[name] = {
            "reward": float(ct.key_press(env)[0]),
            "f1": float(ct.key_press_f1(env)[0]),
            "precision": float(ct.key_press_precision(env)[0]),
            "recall": float(ct.key_press_recall(env)[0]),
        }
    assert scores["right"] == pytest.approx({"reward": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0}, abs=1e-3)
    # A wrong key forfeits the "nothing else is down" half and earns next to nothing for the key still up.
    assert scores["wrong"]["f1"] == 0.0 and scores["wrong"]["reward"] < 0.1
    # Nothing keeps that half, and with no key down its precision is vacuously perfect.
    assert scores["nothing"] == pytest.approx({"reward": 0.5, "f1": 0.0, "precision": 1.0, "recall": 0.0}, abs=0.06)


def test_silence_played_as_silence_is_perfect(env):
    env.reset()
    _step(env, 5)
    assert float(ct.key_press_f1(env)[0]) == 1.0
    assert float(ct.key_press(env)[0]) == pytest.approx(0.5)


def test_the_logged_f1_is_the_mean_over_the_episode(env):
    """A non-cumulative metric logs its last-step value at reset, which for a song is its final step."""
    env.reset()
    _step(env, 30)  # 20 steps of silence (F1 = 1 each), then 10 of an unplayed note (F1 = 0 each)
    logged = env.metric_manager.reset()
    assert float(logged["Metrics/key_press_f1"]) == pytest.approx(20 / 30, abs=0.02)
    assert float(logged["Metrics/key_press_recall"]) == pytest.approx(20 / 30, abs=0.02)
    assert float(logged["Metrics/key_press_precision"]) == pytest.approx(1.0, abs=0.02)
    assert "Metrics/Success" not in logged


def test_fingertip_shaping_rewards_being_over_the_key_that_is_due(env):
    shaping = env.reward_manager.get_term("fingertip_to_key")
    env.reset()
    _step(env, 5)
    assert torch.all(shaping.compute() == 0.0)  # nothing due within the horizon yet

    _step(env, LEAD_IN_STEPS - 5 - 3)  # middle C is now three steps away
    over_the_key = shaping.compute()[0]
    hand = env.entities["right_hand"]  # a hand's width along the keyboard, away from middle C
    hand.set_pos(hand.get_pos(relative=True) + torch.tensor([0.0, 0.25, 0.0], device=env.device), relative=True)
    assert over_the_key > 0.5
    assert shaping.compute()[0] < 0.5 * over_the_key


def test_the_episode_ends_with_the_score_and_restarts_it(env):
    env.reset()
    song, roll = env.command_manager.get_term("song"), _roll(env)
    _step(env, song.n_steps - 1)
    assert not song.song_over.any()
    env.step(torch.zeros_like(env.action_manager.action))
    # The auto-reset has already run: the clock is back at zero and the policy is shown the opening again.
    assert torch.all(env.episode_length_buf == 0)
    assert torch.equal(_window(env)[0], roll[0])
    assert not song.key_down.any() and not song.goal.any()
    # Ending with the score is a truncation, not a failure, so the value bootstrap is kept.
    assert bool(env.reset_timeouts.all()) and not bool(env.reset_terminated.any())


def test_two_wrist_controllers_on_one_hand_are_refused_at_build():
    """A free base is six DOFs of one joint that no term may name, so the overlap check must key on indices."""
    cfg = TASK_REGISTRY.build("piano")
    cfg.env_options.num_envs = 1
    cfg.action_options.right_wrist_again = FreeBasePoseController.configure(entity_name="right_hand")
    with pytest.raises(ValueError, match="overlap on the DOFs") as raised:
        build_env(cfg)
    message = str(raised.value)
    assert "'right_wrist'" in message and "'right_wrist_again'" in message
    assert "DOF 0" in message and "only fix" in message  # named by index, since the term has no name for them
