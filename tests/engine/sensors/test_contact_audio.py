"""``eden.engine.sensors.contact_audio``: the contact microphone, its materials, and its fail-fast paths."""

from __future__ import annotations

import genesis as gs
import pytest
import torch

from eden.engine.sensors import ContactAudio, ContactAudioProperties, SpatialAudio
from eden.engine.sensors.contact_audio import _kernel_synthesize_block
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, PlaneOptions, SphereOptions
from eden.options.sensors import SensorOptions, SensorsOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

# 0.005 s steps split into 80 samples: a 16 kHz audio rate, whose band edge (0.45 x Nyquist) is 3600 Hz.
SIM_DT = 0.005
K = 80

# Long, in-contact ring-downs at two well separated pitches, so a spectrum identifies which material sounded.
LOW = ContactAudioProperties(modal_freqs=(600.0,), modal_decays=(5.0,), modal_gains=(1.0,), contact_damping=1.0)
HIGH = ContactAudioProperties(modal_freqs=(1500.0,), modal_decays=(5.0,), modal_gains=(1.0,), contact_damping=1.0)

# A ball released 0.2 m above a plate lands after sqrt(2 * 0.2 / 9.81) = 0.2 s, i.e. about 40 steps.
DROP_HEIGHT = 0.2
LANDED_BY_STEP = 60
PLATE_TOP = 0.04
BALL_RADIUS = 0.025


def _drop_pairs(*plates_x):
    """A floor, plus a fixed plate and a ball above it at each x: ``plate0``/``ball0``, ``plate1``/``ball1``, ..."""
    entities = {"floor": PlaneOptions()}
    for i, x in enumerate(plates_x):
        entities[f"plate{i}"] = BoxOptions(
            size=(0.3, 0.3, PLATE_TOP), default_root_pos=(x, 0.0, PLATE_TOP / 2), fixed=True
        )
        entities[f"ball{i}"] = SphereOptions(
            radius=BALL_RADIUS, default_root_pos=(x, 0.0, PLATE_TOP + BALL_RADIUS + DROP_HEIGHT)
        )
    return entities


def _contact_mic(body, **options):
    options.setdefault("audio_substeps", K)
    options.setdefault("n_modes", 2)
    return SensorOptions(sensor=ContactAudio(**options), attach_entity_name=body)


def _config(entities, sensors, num_envs=1, decimation=1):
    return EdenConfig(
        env_options=EnvOptions(num_envs=num_envs, sim_dt=SIM_DT, decimation=decimation),
        scene_options=SceneOptions(**entities),
        sensors_options=SensorsOptions(**sensors),
    )


def _record(env, names, n_steps):
    """Step ``n_steps`` times and return each named sensor's env-0 recording as one 1-D tensor."""
    blocks = {name: [] for name in names}
    for _ in range(n_steps):
        env.step()
        for name in names:
            blocks[name].append(env.sensors[name].read()[0].reshape(-1).clone())
    return {name: torch.cat(chunks) for name, chunks in blocks.items()}


def _peak_hz(signal, dt_sub=SIM_DT / K):
    spectrum = torch.fft.rfft(signal * torch.hann_window(signal.numel(), device=signal.device)).abs()
    return float(torch.fft.rfftfreq(signal.numel(), d=dt_sub)[spectrum.argmax()])


def _rms(signal):
    return float(signal.pow(2).mean().sqrt())


@pytest.fixture(scope="module")
def drops():
    """Three balls dropped on three plates, in two envs, recorded falling, landing and at rest.

    ``plate0`` is declared ``LOW`` and ``plate1`` ``HIGH``; ``plate2`` is declared nowhere and the scene has no
    default material. ``ball0`` carries an ungated (``tap0``) and a velocity-gated (``gated``) microphone.
    """
    names = ("tap0", "tap1", "gated", "unlisted", "mic")
    env = build_env(
        _config(
            _drop_pairs(-0.5, 0.5, 1.5),
            {
                "tap0": _contact_mic("ball0", properties={"plate0": LOW}),
                "tap1": _contact_mic("ball1", properties={"plate1": HIGH}),
                "gated": _contact_mic("ball0", properties={"plate0": LOW}, velocity_gate_ref=0.05),
                "unlisted": _contact_mic("ball2", properties={"floor": LOW}),
                "mic": SensorOptions(sensor=SpatialAudio(audio_substeps=K, pos_offset=(0.0, 0.0, 0.3))),
            },
            num_envs=2,
        )
    )
    recordings = {
        "falling": _record(env, names, 20),
        "landed": _record(env, names, LANDED_BY_STEP),
        "settling": _record(env, names, 40),
        "resting": _record(env, names, 10),
    }
    yield env, recordings
    env.destroy()


def test_silent_until_something_touches(drops):
    _, recordings = drops
    assert all(float(signal.abs().max()) == 0.0 for signal in recordings["falling"].values())


def test_a_tap_rings_at_the_struck_materials_pitch(drops):
    """Two identical balls sound like the plate each hit: the material is keyed by the *struck* link.

    Each sensor declares only its own plate, yet the materials merge into one scene-wide table.
    """
    _, recordings = drops
    assert _peak_hz(recordings["landed"]["tap0"]) == pytest.approx(600.0, abs=15.0)
    assert _peak_hz(recordings["landed"]["tap1"]) == pytest.approx(1500.0, abs=15.0)


def test_an_airborne_microphone_hears_every_contact_source(drops):
    _, recordings = drops
    mic = recordings["landed"]["mic"]
    spectrum = torch.fft.rfft(mic).abs()
    freqs = torch.fft.rfftfreq(mic.numel(), d=SIM_DT / K)
    for pitch in (600.0, 1500.0):
        near = (freqs - pitch).abs() < 15.0
        assert float(spectrum[near].max()) > 20.0 * float(spectrum.median())


def test_a_link_no_material_names_is_silent_without_a_default(drops):
    """``ball2`` lands on ``plate2``, which nothing declares; with no ``default_properties`` it makes no sound."""
    _, recordings = drops
    assert float(recordings["landed"]["unlisted"].abs().max()) == 0.0


def test_velocity_gate_quiets_a_body_at_rest(drops):
    """With ``velocity_gate_ref`` set, a ball that has come to rest goes quiet while the ungated plate rings on."""
    _, recordings = drops
    ringing = _rms(recordings["resting"]["tap0"])
    assert ringing > 1e-3, "test premise: the ungated plate is still ringing"
    assert _rms(recordings["resting"]["gated"]) < 0.05 * ringing


def test_reset_silences_the_reset_environment_only(drops):
    """Resetting one environment clears its resonators (its ball is back in the air); its neighbour rings on."""
    env, _ = drops
    env.reset(torch.tensor([0], device=env.device))
    env.step()
    heard = env.sensors["tap0"].read().abs().amax(dim=-1)
    assert float(heard[0]) == 0.0
    assert float(heard[1]) > 0.0


def test_a_tap_between_two_renders_still_sounds():
    """Synthesis follows the simulation, not the render rate: no tap is lost to decimation.

    A tap is an impulse on the one simulation step where the force jumps. Driven by the audio registry, the voice
    is in the same state after the same number of simulation steps whatever the decimation, so the newest block
    rendered at ``decimation=4`` is the one ``decimation=1`` renders on the same simulation step.
    """
    n_sim_steps = 64
    newest = {}
    for decimation in (1, 4):
        env = build_env(
            _config(_drop_pairs(0.0), {"tap": _contact_mic("ball0", properties={"plate0": LOW})}, decimation=decimation)
        )
        try:
            torch.manual_seed(0)
            for _ in range(n_sim_steps // decimation):
                env.step()
            newest[decimation] = env.sensors["tap"].read().clone()
        finally:
            env.destroy()

    assert float(newest[1].abs().max()) > 0.0, "test premise: the ball has landed and the plate is ringing"
    torch.testing.assert_close(newest[4], newest[1], rtol=1e-4, atol=1e-6)


@pytest.fixture(scope="module")
def slide_and_default():
    """A box resting on a 30 deg ramp, then made slippery; and, apart, a ball landing on an undeclared plate.

    The box holds on with its own friction (1.0 > tan 30 deg) until it is set to the ramp's 0.1; Genesis combines a
    pair's friction as ``max``. The ball's microphone declares ``default_properties=HIGH``.
    """
    ramp_quat = (0.9659258, 0.0, 0.2588190, 0.0)  # 30 degrees about +y
    rough = ContactAudioProperties(
        modal_gains=(0.0,), roughness_gain=1.0, roughness_spatial_freq=400.0, roughness_bandwidth=700.0
    )
    entities = {
        **_drop_pairs(1.0),
        "ramp": BoxOptions(
            size=(0.8, 0.3, 0.04),
            default_root_pos=(0.0, 0.0, 0.35),
            default_root_quat=ramp_quat,
            fixed=True,
            default_geoms_friction={".*": 0.1},
        ),
        "slider": BoxOptions(
            size=(0.08, 0.08, 0.05),
            default_root_pos=(-0.2373, 0.0, 0.5440),
            default_root_quat=ramp_quat,
            default_geoms_friction={".*": 1.0},
        ),
    }
    sensors = {
        "scrape": _contact_mic("slider", properties={"ramp": rough}),
        "tap": _contact_mic("ball0", default_properties=HIGH),
    }
    env = build_env(_config(entities, sensors))
    names = ("scrape", "tap")
    settle = _record(env, names, 30)
    held = _record(env, names, 30)
    env.entities["slider"]._entity.set_friction(0.1)
    sliding = _record(env, names, 60)
    yield {
        "held": held["scrape"],
        "sliding": sliding["scrape"],
        "tap": torch.cat([settle["tap"], held["tap"], sliding["tap"]]),
    }
    env.destroy()


def test_sliding_scrapes_and_resting_does_not(slide_and_default):
    """The texture source is driven by slip: the box held still on the ramp is silent, the same box sliding is not."""
    held, sliding = slide_and_default["held"], slide_and_default["sliding"]
    assert _rms(sliding) > 0.1
    assert _rms(held) < 1e-2 * _rms(sliding)
    # A scrape is noise, not a ring-down: once the DC blocker has done its job it has no offset to speak of.
    assert abs(float(sliding.mean())) < 0.05 * float(sliding.std())


def test_default_material_voices_a_link_no_material_names(slide_and_default):
    assert _peak_hz(slide_and_default["tap"]) == pytest.approx(1500.0, abs=15.0)


@pytest.fixture(scope="module")
def band_limited(eden_messages):
    """Two drops at the default ``audio_substeps=20`` (4 kHz, band edge 900 Hz), each plate ringing at 600 and 2000 Hz.

    ``plate0``'s 2 kHz mode is loud; ``plate1``'s has zero gain and so never sounded.
    """

    def material(top_gain):
        return ContactAudioProperties(
            modal_freqs=(600.0, 2000.0), modal_decays=(5.0, 5.0), modal_gains=(1.0, top_gain), contact_damping=1.0
        )

    sensors = {
        "loud": _contact_mic("ball0", audio_substeps=20, properties={"plate0": material(5.0)}),
        "quiet": _contact_mic("ball1", audio_substeps=20, properties={"plate1": material(0.0)}),
    }
    with eden_messages() as messages:
        env = build_env(_config(_drop_pairs(-0.5, 0.5), sensors))
    recording = _record(env, ("loud", "quiet"), LANDED_BY_STEP + 20)
    yield [m for m in messages if "silenced" in m], recording
    env.destroy()


def test_a_mode_above_the_band_edge_is_silenced_with_a_warning_only_if_audible(band_limited):
    warnings, _ = band_limited
    assert warnings, "the audible 2 kHz mode was dropped silently"
    # The table is rebuilt as each sensor joins, so the loud material may be reported more than once.
    assert all("'plate0'" in w and "2000" in w and "900 Hz" in w for w in warnings), warnings


@pytest.mark.parametrize("name", ["loud", "quiet"])
def test_a_silenced_mode_does_not_alias_into_the_signal(band_limited, name):
    """The louder 2 kHz mode would have dominated, in place or aliased; only the 600 Hz one is left."""
    _, recording = band_limited
    assert _peak_hz(recording[name], dt_sub=SIM_DT / 20) == pytest.approx(600.0, abs=15.0)


@pytest.mark.parametrize(
    ("sensors", "match"),
    [
        ({"tap": _contact_mic("ball0", properties={"plaet0": LOW})}, "plaet0.*ball0.*plate0"),
        ({"tap": _contact_mic("ball0", properties={"plate0/no_such_link": LOW})}, "no_such_link.*Its links"),
        (
            {
                "tap_a": _contact_mic("ball0", properties={"plate0": LOW}),
                "tap_b": _contact_mic("ball0", properties={"plate0": HIGH}),
            },
            "'plate0' is declared twice",
        ),
        (
            {"tap_a": _contact_mic("ball0", n_modes=2), "tap_b": _contact_mic("ball0", n_modes=4)},
            "n_modes=4.*n_modes=2",
        ),
        ({"tap": _contact_mic("ball0")}, "declares a material"),
        ({"tap": SensorOptions(sensor=ContactAudio(audio_substeps=K))}, "entity_idx is still -1"),
    ],
    ids=["unknown_entity", "unknown_link", "declared_twice", "mismatched_n_modes", "no_material", "unattached"],
)
def test_misconfigured_contact_audio_fails_at_build(sensors, match):
    """Names that do not resolve, conflicting shared state, a scene that could never sound, or no link, raise."""
    env = EnvBase.from_config(_config(_drop_pairs(0.0), sensors), show_viewer=False)
    try:
        with pytest.raises(gs.GenesisException, match=match):
            env.build()
    finally:
        env.destroy()


@pytest.mark.parametrize(
    ("props", "match"),
    [
        (ContactAudioProperties(modal_freqs=(100.0, 200.0), modal_decays=(1.0,), modal_gains=(1.0,)), "equal length"),
        (ContactAudioProperties(modal_freqs=(1.0,) * 3, modal_decays=(1.0,) * 3, modal_gains=(1.0,) * 3), "n_modes=2"),
    ],
    ids=["ragged_modal_tuples", "more_modes_than_the_bank"],
)
def test_malformed_material_is_rejected_at_construction(props, match):
    """A material that cannot fill the modal bank fails where it is written, before any scene exists."""
    with pytest.raises(gs.GenesisException, match=match):
        ContactAudio(n_modes=2, properties={"plate0": props})
    with pytest.raises(gs.GenesisException, match="default_properties"):
        ContactAudio(n_modes=2, default_properties=props)


def test_block_kernel_matches_the_recurrences_it_fuses():
    """The Quadrants kernel is the modal bank, texture resonator, click and DC block, run sample by sample.

    Random coefficients and state for 2 envs x 3 sensors x 4 modes against the same recurrences in plain torch,
    called twice so the state the kernel writes back is what the next block starts from.
    """
    gen = torch.Generator(device="cpu").manual_seed(0)
    B, S, M, n = 2, 3, 4, 16

    def rand(*shape, scale=1.0):
        return (torch.rand(shape, generator=gen) * scale).to(dtype=gs.tc_float, device=gs.device).contiguous()

    def poles(*shape):  # stable two-pole filters: a1 = 2 r cos(theta), a2 = r^2 with r < 1
        r, theta = rand(*shape, scale=0.98), rand(*shape, scale=3.0)
        return (2.0 * r * torch.cos(theta)).contiguous(), (r * r).contiguous()

    (a1, a2), (at1, at2), (aa1, aa2) = poles(B, S, M), poles(B, S), poles(B, S)
    gains, modal_drive, tex_amp = rand(B, S, M), rand(B, S), rand(B, S)
    hp_r = 0.97
    state = {"modal_y": rand(B, S, M, 2), "tex_y": rand(B, S, 2), "acc_y": rand(B, S, 2), "hp": rand(B, S, 2)}
    es = {name: tensor.clone() for name, tensor in state.items()}

    for _ in range(2):
        impulse, kick = rand(B, S), rand(B, S)
        modal_noise, tex_noise = rand(B, S, M, n) - 0.5, rand(B, S, n) - 0.5
        out = torch.empty((B, S, n), dtype=gs.tc_float, device=gs.device)
        _kernel_synthesize_block(
            a1, a2, gains, modal_drive, modal_noise, tex_noise, impulse, at1, at2, aa1, aa2, tex_amp, kick,
            state["modal_y"], state["tex_y"], state["acc_y"], state["hp"], out, hp_r,
        )  # fmt: skip

        my1, my2 = es["modal_y"][..., 0].clone(), es["modal_y"][..., 1].clone()
        ty1, ty2 = es["tex_y"][..., 0].clone(), es["tex_y"][..., 1].clone()
        ay1, ay2 = es["acc_y"][..., 0].clone(), es["acc_y"][..., 1].clone()
        hpx, hpy = es["hp"][..., 0].clone(), es["hp"][..., 1].clone()
        expected = torch.empty_like(out)
        for k in range(n):
            y = a1 * my1 - a2 * my2 + modal_drive[..., None] * modal_noise[..., k]
            ty = at1 * ty1 - at2 * ty2 + tex_amp * tex_noise[..., k]
            ay = aa1 * ay1 - aa2 * ay2
            if k == 0:
                y, ay = y + impulse[..., None], ay + kick
            my2, my1, ty2, ty1, ay2, ay1 = my1, y, ty1, ty, ay1, ay
            raw = (gains * y).sum(dim=-1) + ty + ay
            hpy = hp_r * hpy + raw - hpx
            hpx = raw
            expected[..., k] = hpy
        es["modal_y"] = torch.stack((my1, my2), dim=-1)
        es["tex_y"], es["acc_y"], es["hp"] = (torch.stack(p, dim=-1) for p in ((ty1, ty2), (ay1, ay2), (hpx, hpy)))

        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)
        for name, tensor in state.items():
            torch.testing.assert_close(tensor, es[name], rtol=1e-4, atol=1e-5)


def test_contact_audio_config_round_trips_through_a_file(tmp_path):
    """A saved config reloads the materials as ``ContactAudioProperties``, the absent default as ``None``."""
    config = _config(
        _drop_pairs(0.0), {"tap": _contact_mic("ball0", properties={"plate0": HIGH}, velocity_gate_ref=0.2)}
    )
    path = str(tmp_path / "contact_audio_config.json")
    config.save_as_json(path, lock_config=False)
    sensor = EdenConfig.load_from_file(path).sensors_options.tap.sensor

    assert isinstance(sensor, ContactAudio) and sensor.velocity_gate_ref == 0.2 and sensor.default_properties is None
    material = sensor.properties["plate0"]
    assert isinstance(material, ContactAudioProperties)
    assert tuple(material.modal_freqs) == (1500.0,) and material.contact_damping == 1.0
