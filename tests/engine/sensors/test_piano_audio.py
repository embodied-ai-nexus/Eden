"""``eden.engine.sensors.piano_audio``: ``PianoAudio``, the sensor that voices the ``Piano`` keyboard.

Most tests share one built keyboard and start from ``env.reset()``.
"""

from __future__ import annotations

import itertools
import math

import genesis as gs
import pytest
import torch

from eden.engine.sensors import PianoAudio, SpatialAudio
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import EntityOptions, PlaneOptions, SphereOptions
from eden.options.objects.piano import NUM_KEYS, Piano, is_black_key, key_joint_name, key_of_midi_note
from eden.options.sensors import SensorOptions, SensorsOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

# 0.005 s steps split into 80 samples: a 16 kHz audio rate.
SIM_DT = 0.005
AUDIO_SUBSTEPS = 80
DT_SUB = SIM_DT / AUDIO_SUBSTEPS
NUM_ENVS = 2

A4 = key_of_midi_note(69)
C4 = key_of_midi_note(60)
# Joint torques on a key. Its spring (2 N·m/rad) needs 0.12 N·m to reach the strike point at all.
HARD, SOFT = 0.6, 0.15


def _build(scene, sensors, num_envs=NUM_ENVS):
    config = EdenConfig(
        env_options=EnvOptions(num_envs=num_envs, sim_dt=SIM_DT, decimation=1),
        scene_options=scene,
        sensors_options=SensorsOptions(**sensors),
    )
    return build_env(config)


def _voice(**options):
    options.setdefault("audio_substeps", AUDIO_SUBSTEPS)
    return SensorOptions(sensor=PianoAudio(**options), attach_entity_name="piano")


@pytest.fixture(scope="module")
def piano_env():
    """One keyboard with its voice and a microphone 0.5 m away, shared by the tests that only play it.

    ``pair`` is a second voice of just two of the keys, named with their own notes.
    """
    return _build(
        SceneOptions(floor=PlaneOptions(), piano=Piano(default_root_pos=(0.0, 0.0, 0.5))),
        {
            "strings": _voice(),
            "pair": _voice(key_notes={key_joint_name(C4): 60, key_joint_name(A4): 69}),
            "mic": SensorOptions(sensor=SpatialAudio(audio_substeps=AUDIO_SUBSTEPS, pos_offset=(0.5, 0.0, 0.5))),
        },
    )


def _record(env, presses, n_steps, sensors, env_idx=0):
    """Reset, then step with joint torques on some keys, and return env ``env_idx``'s 1-D recording of each sensor.

    ``presses`` maps ``key -> (torque, first_step, last_step)``: the torque is applied, in every environment, on
    steps ``first_step <= step < last_step``.
    """
    env.reset()
    piano = env.entities["piano"]._entity
    blocks = {sensor: [] for sensor in sensors}
    for step in range(n_steps):
        torques = torch.zeros((NUM_ENVS, NUM_KEYS), dtype=gs.tc_float, device=gs.device)
        for key, (torque, first, last) in presses.items():
            if first <= step < last:
                torques[:, key] = torque
        piano.control_dofs_force(torques)
        env.step()
        for sensor, chunks in blocks.items():
            chunks.append(env.sensors[sensor].read()[env_idx].reshape(-1).clone())
    return {sensor: torch.cat(chunks) for sensor, chunks in blocks.items()}


def _play(env, presses, n_steps, sensor="strings", env_idx=0):
    """:func:`_record` of one sensor."""
    return _record(env, presses, n_steps, (sensor,), env_idx)[sensor]


def _peak_hz(signal):
    spectrum = torch.fft.rfft(signal * torch.hann_window(signal.numel(), device=signal.device)).abs()
    return float(torch.fft.rfftfreq(signal.numel(), d=DT_SUB)[spectrum.argmax()])


def _seconds(start, stop):
    return slice(round(start / DT_SUB), round(stop / DT_SUB))


# ============================================ the keyboard ============================================


def test_keys_rest_level_and_only_the_pressed_key_moves(piano_env):
    """Without a preload the key spring could not carry the key; gravity compensation is what holds it at zero."""
    piano = piano_env.entities["piano"]._entity
    assert piano.n_dofs == NUM_KEYS
    lower, upper = piano.get_dofs_limit()
    assert torch.all(lower == 0.0)
    white_travel, black_travel = math.atan(0.01 / 0.15), math.atan(0.008 / 0.09)
    expected = torch.tensor([black_travel if is_black_key(key) else white_travel for key in range(NUM_KEYS)])
    torch.testing.assert_close(upper.reshape(-1, NUM_KEYS)[0].cpu(), expected.to(upper.dtype), rtol=1e-4, atol=0.0)

    _play(piano_env, {}, n_steps=40)
    assert piano.get_dofs_position().abs().max() < 1e-5
    # The black keys are solid boxes sunk into the white ones, so any key-to-key collision would show up here.
    contacts = piano_env.scene.rigid_solver.collider.get_contacts(as_tensor=True, to_torch=True)
    assert contacts["link_a"].numel() == 0

    _play(piano_env, {A4: (HARD, 0, 40)}, n_steps=40)
    position = piano.get_dofs_position()[0]
    assert position[A4] >= upper.reshape(-1, NUM_KEYS)[0, A4] * 0.99
    position[A4] = 0.0
    assert position.abs().max() < 1e-5


# ============================================ the sound ============================================


@pytest.mark.parametrize("midi, hertz", [(21, 27.5), (69, 440.0), (108, 4186.0)])
def test_a_key_sounds_its_own_pitch(piano_env, midi, hertz):
    heard = _play(piano_env, {key_of_midi_note(midi): (HARD, 0, 200)}, n_steps=200)
    # One second of signal resolves 1 Hz; the inharmonicity sharpens a fundamental by only 0.02%.
    assert _peak_hz(heard) == pytest.approx(hertz, abs=1.5)


def test_a_faster_key_is_louder_and_a_crept_key_is_silent(piano_env):
    hard = _play(piano_env, {A4: (HARD, 0, 60)}, n_steps=60).abs().max()
    soft = _play(piano_env, {A4: (SOFT, 0, 60)}, n_steps=60).abs().max()
    assert hard > 1.5 * soft > 0.0

    # Ease the key all the way down over two seconds: it ends up just as far down, and no hammer is thrown.
    env = piano_env
    env.reset()
    piano = env.entities["piano"]._entity
    crept = []
    for step in range(400):
        torques = torch.zeros((NUM_ENVS, NUM_KEYS), dtype=gs.tc_float, device=gs.device)
        torques[:, A4] = 0.2 * step / 400
        piano.control_dofs_force(torques)
        env.step()
        crept.append(env.sensors["strings"].read()[0].clone())
    assert env.sensors["strings"].sensor.keys_down[0, A4]
    assert torch.cat(crept).abs().max() < 0.02 * hard


def test_note_rings_while_held_is_damped_on_release_and_the_pedal_keeps_it(piano_env):
    # Down at step 0, up at step 100 (0.5 s).
    heard = _play(piano_env, {A4: (HARD, 0, 100)}, n_steps=200)
    held = heard[_seconds(0.3, 0.45)].abs().max()
    released = heard[_seconds(0.75, 0.9)].abs().max()
    assert held > 0.3 * heard.abs().max()
    assert released < 0.01 * held

    piano_env.reset()
    piano_env.sensors["strings"].sensor.set_sustain(True, envs_idx=[0])
    piano = piano_env.entities["piano"]._entity
    blocks = {0: [], 1: []}
    for step in range(200):
        torques = torch.zeros((NUM_ENVS, NUM_KEYS), dtype=gs.tc_float, device=gs.device)
        torques[:, A4] = HARD if step < 100 else 0.0
        piano.control_dofs_force(torques)
        piano_env.step()
        for env_idx, chunks in blocks.items():
            chunks.append(piano_env.sensors["strings"].read()[env_idx].clone())
    pedalled, dry = (torch.cat(blocks[env_idx]) for env_idx in (0, 1))
    assert not piano_env.sensors["strings"].sensor.keys_down[:, A4].any()
    # The pedal is per environment: with it down the released note rings on, without it the damper takes it.
    assert pedalled[_seconds(0.75, 0.9)].abs().max() > 0.3 * held
    assert dry[_seconds(0.75, 0.9)].abs().max() < 0.01 * held


def _place_key_and_emit(env, key, fraction):
    """Put ``key`` at ``fraction`` of its stroke in every environment and run one synthesis step, without physics."""
    piano = env.entities["piano"]._entity
    _, upper = piano.get_dofs_limit()
    position = torch.zeros((NUM_ENVS, NUM_KEYS), dtype=gs.tc_float, device=gs.device)
    position[:, key] = fraction * upper.reshape(-1, NUM_KEYS)[0, key]
    piano.set_dofs_position(position)
    voice = env.sensors["strings"].sensor._voice
    voice.emit()
    return voice


def test_a_key_strikes_once_per_press_and_rearms_only_below_the_release_point(piano_env):
    piano_env.reset()
    # Strike point 0.87, release point 0.5. The long stay at 0.30 lets the damper kill the first note, so that the
    # second strike is not masked by landing out of phase with what is left of it.
    fractions = [0.0, 0.95, 0.80, 0.95, 0.60, 0.95] + [0.30] * 40 + [0.95]
    energy = []
    for fraction in fractions:
        voice = _place_key_and_emit(piano_env, A4, fraction)
        energy.append(float(torch.hypot(voice._c[0, A4, 0], voice._s[0, A4, 0])))
    grew = [after > 1.01 * before for before, after in itertools.pairwise(energy)]
    # Struck on first passing the strike point; hovering between the two points does nothing; 0.30 re-arms.
    assert grew == [True] + [False] * 44 + [True]


def test_keys_down_is_a_snapshot(piano_env):
    """A caller can keep ``keys_down`` as last step's state: the voice never writes into what it handed out."""
    piano_env.reset()
    sensor = piano_env.sensors["strings"].sensor
    _place_key_and_emit(piano_env, A4, 0.95)
    pressed = sensor.keys_down
    _place_key_and_emit(piano_env, A4, 0.0)
    assert pressed[:, A4].all() and not sensor.keys_down[:, A4].any()


def test_block_synthesis_matches_the_analytic_decaying_partials(piano_env):
    """A struck string is a sum of ``exp(-d t) sin(2 pi f t)`` partials, sample for sample over five blocks.

    Three with the key held, under the ring envelope, then two after it is let go, under the damper's.
    """
    piano_env.reset()
    options = PianoAudio()
    # From just short of the strike point to just past it in one step: a gentle strike, below full scale.
    _place_key_and_emit(piano_env, A4, 0.80)
    blocks = [_place_key_and_emit(piano_env, A4, 0.90).block[0, 0].clone() for _ in range(3)]
    blocks += [_place_key_and_emit(piano_env, A4, 0.0).block[0, 0].clone() for _ in range(2)]
    heard = torch.cat(blocks).double().cpu()

    _, upper = piano_env.entities["piano"]._entity.get_dofs_limit()
    speed = 0.10 * float(upper.reshape(-1, NUM_KEYS)[0, A4]) / SIM_DT
    level = min(speed / options.velocity_ref, 1.0)
    t = torch.arange(1, 5 * AUDIO_SUBSTEPS + 1, dtype=torch.float64) * DT_SUB
    released_at = 3 * AUDIO_SUBSTEPS * DT_SUB
    expected = torch.zeros_like(t)
    for n in range(1, options.n_partials + 1):
        freq = n * 440.0 * math.sqrt(1.0 + options.inharmonicity * n * n)
        ring = options.ring_decay * (1.0 + options.partial_decay_growth * (n - 1))
        damped = max(ring, options.damped_decay)
        decayed = torch.where(t <= released_at, ring * t, ring * released_at + damped * (t - released_at))
        expected += level * n**-options.partial_rolloff * torch.exp(-decayed) * torch.sin(2.0 * math.pi * freq * t)
    assert 0.0 < level < 1.0
    torch.testing.assert_close(heard, expected, rtol=0.0, atol=2e-5)


def test_chosen_keys_keep_their_own_notes(piano_env):
    """``key_notes`` voices a subset with each key at its own pitch, not renumbered from ``lowest_midi_note``.

    ``pair`` holds C4 and A4 only, as a second voice in the same scene: A4 pressed must sound exactly as it does
    in the full keyboard's voice, which also has each voice reading its own rows of the shared read cache.
    """
    heard = _record(piano_env, {A4: (HARD, 0, 60)}, n_steps=60, sensors=("strings", "pair"))
    assert heard["strings"].abs().max() > 0.0
    torch.testing.assert_close(heard["pair"], heard["strings"])
    keys_down = piano_env.sensors["pair"].sensor.keys_down
    assert keys_down.shape == (NUM_ENVS, 2)
    assert keys_down[:, 1].all() and not keys_down[:, 0].any()


def test_partial_reset_silences_only_the_reset_environment(piano_env):
    _play(piano_env, {A4: (HARD, 0, 60)}, n_steps=60)
    voice = piano_env.sensors["strings"].sensor._voice
    assert voice._c.abs().amax(dim=(1, 2)).min() > 0.0
    piano_env.reset(envs_idx=torch.tensor([0], device=gs.device))
    ringing = torch.hypot(voice._c, voice._s).amax(dim=(1, 2))
    assert ringing[0] == 0.0 and ringing[1] > 0.0
    assert not voice.keys_down[0].any()


def test_the_microphone_hears_the_piano_late_and_quieter(piano_env):
    direct = _play(piano_env, {A4: (HARD, 0, 60)}, n_steps=60)
    through_air = _play(piano_env, {A4: (HARD, 0, 60)}, n_steps=60, sensor="mic")

    def first(signal):
        return int(torch.nonzero(signal.abs() > 1e-4 * signal.abs().max())[0])

    # 0.5 m at 343 m/s is 1.46 ms, 23 samples.
    assert first(through_air) - first(direct) == pytest.approx(23, abs=2)
    assert _peak_hz(through_air) == pytest.approx(440.0, abs=5.0)


def test_a_recording_keeps_every_block_between_reads(piano_env):
    """With one ``read()`` per simulation step the recording is exactly the reads, end to end."""
    sensor = piano_env.sensors["strings"].sensor
    with pytest.raises(gs.GenesisException, match="without start_recording"):
        sensor.take_recording()

    piano_env.reset()
    piano = piano_env.entities["piano"]._entity
    torques = torch.zeros((NUM_ENVS, NUM_KEYS), dtype=gs.tc_float, device=gs.device)
    torques[:, A4] = HARD
    sensor.start_recording()
    assert sensor.take_recording().shape == (NUM_ENVS, 0)
    reads, takes = [], []
    for step in range(30):
        piano.control_dofs_force(torques)
        piano_env.step()
        reads.append(piano_env.sensors["strings"].read().clone())
        if step % 10 == 9:
            takes.append(sensor.take_recording())
    assert [take.shape for take in takes] == [(NUM_ENVS, 10 * AUDIO_SUBSTEPS)] * 3
    torch.testing.assert_close(torch.cat(takes, dim=-1), torch.cat(reads, dim=-1), rtol=0.0, atol=0.0)
    assert torch.cat(takes, dim=-1).abs().max() > 0.1

    sensor.stop_recording()
    # Stopped, not never-started: the code can tell the two apart, so the message has to.
    with pytest.raises(gs.GenesisException, match="after stop_recording"):
        sensor.take_recording()


def test_the_recording_refuses_a_second_start_and_a_stop_with_nothing_to_stop(piano_env):
    """The same contract as ``Camera``: neither mistake is silently absorbed.

    A second start could only throw away what the first one kept, and a stop with nothing open is a mistake in
    the calling loop far more often than an intent.
    """
    sensor = piano_env.sensors["strings"].sensor
    with pytest.raises(gs.GenesisException, match="nothing to stop"):
        sensor.stop_recording()

    piano_env.reset()
    sensor.start_recording()
    piano_env.step()
    with pytest.raises(gs.GenesisException, match="already recording, with 1 block"):
        sensor.start_recording()
    # Refused, so the block it was holding is still there.
    assert sensor.take_recording().shape == (NUM_ENVS, AUDIO_SUBSTEPS)

    sensor.stop_recording()
    with pytest.raises(gs.GenesisException, match="nothing to stop"):
        sensor.stop_recording()
    sensor.start_recording()  # and a fresh one opens cleanly after a stop
    assert sensor.take_recording().shape == (NUM_ENVS, 0)


# ============================================ fail fast ============================================


def _build_expecting(scene, sensors, match):
    with pytest.raises(gs.GenesisException, match=match):
        _build(scene, sensors, num_envs=1)


def _keyboard():
    return SceneOptions(floor=PlaneOptions(), piano=Piano())


def test_options_that_could_never_work_are_rejected_at_construction():
    with pytest.raises(gs.GenesisException, match="release_fraction < strike_fraction"):
        PianoAudio(strike_fraction=0.5, release_fraction=0.5)
    with pytest.raises(gs.GenesisException, match="empty `key_notes`"):
        PianoAudio(key_notes={})
    with pytest.raises(gs.GenesisException, match="lowest_midi_note=60, which would be ignored"):
        PianoAudio(key_notes={key_joint_name(A4): 69}, lowest_midi_note=60)
    with pytest.raises(gs.GenesisException, match="less than or equal to 127"):
        PianoAudio(key_notes={key_joint_name(A4): 128})
    # The default is not "set": a saved config writes every field back, and must load again.
    assert PianoAudio(key_notes={key_joint_name(A4): 69}, lowest_midi_note=21).key_notes == {"key_48_joint": 69}


def test_unattached_voice_raises():
    _build_expecting(_keyboard(), {"strings": SensorOptions(sensor=PianoAudio())}, match="entity_idx is still -1")


def test_link_attachment_raises():
    sensor = SensorOptions(sensor=PianoAudio(), attach_entity_name="piano", attach_link_name="key_3")
    _build_expecting(_keyboard(), {"strings": sensor}, match="radiates from its base link.*link_idx_local")


def test_unknown_key_joint_raises_naming_it():
    strings = _voice(key_notes={"key_0_joint": 21, "key_88_joint": 109})
    _build_expecting(_keyboard(), {"strings": strings}, match="key_88_joint")


def test_more_keys_than_midi_notes_raises():
    _build_expecting(_keyboard(), {"strings": _voice(lowest_midi_note=60)}, match="past the top of the MIDI range")


def test_an_entity_without_hinges_raises():
    scene = SceneOptions(floor=PlaneOptions(), piano=SphereOptions(radius=0.05, default_root_pos=(0.0, 0.0, 0.5)))
    _build_expecting(scene, {"strings": _voice()}, match="no single-DOF joints")


def test_a_key_without_a_joint_range_raises(tmp_path):
    mjcf = tmp_path / "limitless.xml"
    mjcf.write_text(
        '<mujoco><worldbody><body name="base"><geom type="box" size="0.1 0.1 0.01"/>'
        '<body name="key" pos="0 0 0.05"><joint name="key_joint" type="hinge" axis="0 1 0"/>'
        '<geom type="box" size="0.05 0.01 0.01"/></body></body></worldbody></mujoco>'
    )
    scene = SceneOptions(
        floor=PlaneOptions(), piano=EntityOptions(file=str(mjcf), is_articulated=True, dofs_name=["key_joint"])
    )
    _build_expecting(scene, {"strings": _voice()}, match="no finite joint range")


def test_keys_the_audio_rate_cannot_carry_are_reported(eden_caplog):
    """At the other audio sensors' default rate (4 kHz) the strings stop at 1800 Hz, so the top octaves are mute."""
    env = _build(_keyboard(), {"strings": _voice(audio_substeps=20)}, num_envs=1)
    warnings = [r.getMessage() for r in eden_caplog.records if "are mute" in r.getMessage()]
    assert len(warnings) == 1
    # A6 (1760 Hz) is the last key under 1800 Hz: key 72. The 15 above it are mute, from A#6 (MIDI note 94) up.
    assert "15 of 88 keys are mute, from MIDI note 94 (1865 Hz) up" in warnings[0]
    gains = env.sensors["strings"].sensor._voice._gain
    assert gains[73:].abs().max() == 0.0 and gains[72, 0] == 1.0
