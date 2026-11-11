"""``eden.engine.sensors.audio``: the source registry and the delay line every audio sensor shares.

The ordering and substep tests pin the Genesis behaviour the addon relies on (``SharedSensorContext.update``
runs before the per-sensor-type loop, and on every simulation step), so a Genesis pin bump that breaks it
fails here rather than degrading a scene into silence.
"""

from __future__ import annotations

import math

import genesis as gs
import pytest
import torch
from genesis.utils.geom import transform_by_quat

from eden.engine.sensors import AudioContext, SpatialAudio
from eden.options import BoxOptions, EnvOptions, SceneOptions
from eden.options.sensors import SensorOptions, SensorsOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

# 0.01 s steps split into 20 samples: one block is 10 ms of sound, i.e. 3.43 m of propagation at 343 m/s.
SIM_DT = 0.01
K = 20
TILT = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)  # 90 deg about x: a link-local +z offset points along world -y


def _mic_env(decimation=1, num_envs=1, mic=(0.0, 0.0, 0.0), **mic_kwargs):
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=num_envs, sim_dt=SIM_DT, decimation=decimation),
            scene_options=SceneOptions(
                tilted=BoxOptions(
                    size=(0.1, 0.1, 0.1), default_root_pos=(1.0, 1.0, 0.5), default_root_quat=TILT, fixed=True
                )
            ),
            sensors_options=SensorsOptions(
                mic=SensorOptions(sensor=SpatialAudio(audio_substeps=K, pos_offset=mic, **mic_kwargs))
            ),
        )
    )


@pytest.fixture(scope="module")
def colocated(probe_source, audio_context):
    """A microphone at the origin with a probe there emitting 1, 2, 3, 4, stepped four times.

    Two silent probes are registered alongside, a static one and one riding the tilted box, only to be placed.
    Yields ``(env, heard, probe)``: ``heard[i]`` is the newest sample the microphone read on step ``i``.
    """
    env = _mic_env()
    probe = probe_source(env, [1.0, 2.0, 3.0, 4.0])
    context = audio_context(env)
    context.register(probe, "probe")
    context.register(probe_source(env, [], pos=(3.0, -2.0, 1.0)), "static probe")
    link = env.entities["tilted"].links[0]
    context.register(probe_source(env, [], pos=(0.0, 0.0, 0.3), link_idx=link.idx), "attached probe")
    heard = []
    for _ in range(4):
        env.step()
        heard.append(float(env.sensors["mic"].read().reshape(-1)[-1]))
    yield env, heard, probe
    env.destroy()


def test_sources_emit_before_microphones_render(colocated):
    """A co-located microphone hears exactly the block synthesized *this* step, at unit gain.

    Genesis calls ``SharedSensorContext.update`` before the per-sensor-type loop; without that a microphone would
    render one step stale. Zero distance also pins the ``ref_distance`` clamp: no division by zero, gain 1.
    """
    _, heard, probe = colocated
    assert probe.emit_count == 4
    assert heard == pytest.approx([1.0, 2.0, 3.0, 4.0], rel=1e-4)


def test_static_and_link_attached_sources_are_placed(colocated, audio_context):
    """A static source sits at its offset; an attached one rides its link with the offset in the link's frame."""
    env, _, _ = colocated
    positions = audio_context(env).source_positions()[0]  # (n_src, 3), registration order
    torch.testing.assert_close(positions[1], torch.tensor([3.0, -2.0, 1.0], device=env.device))

    link = env.entities["tilted"].links[0]
    link_pos, link_quat = link.get_pos(relative=False).reshape(3), link.get_quat(relative=False).reshape(4)
    offset = torch.tensor([0.0, 0.0, 0.3], device=env.device)
    torch.testing.assert_close(positions[2], link_pos + transform_by_quat(offset, link_quat))
    assert (positions[2] - (link_pos + offset)).norm() > 0.1


def test_source_registered_after_the_first_step_is_rejected(colocated, probe_source, audio_context):
    """The delay line is sized by the sources present at the first update; a later one raises, not vanishes."""
    env, _, _ = colocated
    with pytest.raises(gs.GenesisException, match="after the delay line was sized"):
        audio_context(env).register(probe_source(env, [1.0]), "late_probe")


@pytest.fixture(scope="module")
def decimated(probe_source, audio_context, eden_messages):
    """``decimation=4``: one ``env.step()`` is four simulation steps but one microphone render.

    Probe ``a`` sits on the microphone and emits 1, 2, 3, 4, then silence. Probe ``b`` stays silent for the first
    control step, then emits 1 while receding from 0.5 m at 0.05 m per simulation step. Yields a dict of what was
    observed after the first and second ``env.step()``.
    """
    env = _mic_env(decimation=4)
    context = audio_context(env)
    a = probe_source(env, [1.0, 2.0, 3.0, 4.0])
    b = probe_source(env, [0.0] * 4 + [1.0] * 4, pos=(0.5, 0.0, 0.0), step_motion=(0.05, 0.0, 0.0))
    context.register(a, "probe a")
    context.register(b, "probe b")
    with eden_messages() as messages:
        env.step()
        # The delay line is a ring; rolling its newest sample to the end puts it in chronological order.
        history = torch.roll(context.src_hist[0, 0], -(context.hist_newest + 1)).clone()
        first = env.sensors["mic"].read().reshape(-1).clone()
        emits_after_first = a.emit_count
        env.step()
        second = env.sensors["mic"].read().reshape(-1).clone()
    yield {"history": history, "first": first, "second": second, "emits": emits_after_first, "messages": messages}
    env.destroy()


def test_delay_line_advances_on_every_substep_not_every_render(decimated):
    """Every block synthesized between two renders is in the history, in order, none overwritten.

    Eden skips sensor updates on intermediate sub-steps (``update_sensors_every_substep=False``) while
    ``context.update()`` still runs, which is why the delay line lives on the context: advanced from the
    microphone's hook it would see one block in four and every propagation delay would come out four times short.
    """
    assert decimated["emits"] == 4
    tail = decimated["history"][-4 * K :].tolist()
    assert tail == pytest.approx([v for v in (1.0, 2.0, 3.0, 4.0) for _ in range(K)], rel=1e-4)
    # The microphone rendered once, so it reports the newest block only.
    assert float(decimated["first"][-1]) == pytest.approx(4.0, rel=1e-4)


def test_decimated_rendering_warns_once(decimated):
    """A gapped ``read()`` is invisible in the data, so it is announced, once: per step would drown a training run."""
    warnings = [m for m in decimated["messages"] if "gapped stream" in m]
    assert len(warnings) == 1
    assert "once per 4 synthesized audio blocks" in warnings[0]


def test_doppler_ramps_across_one_simulation_step_not_one_render(decimated):
    """The last render was four simulation steps back, but the block still ramps over one step of motion.

    After eight emits ``b`` is 0.9 m away and was 0.85 m one simulation step earlier (0.7 m at the last render).
    At ``ref_distance=0.1`` the inverse-law gain must ramp 0.1/0.85 -> 0.1/0.9 across the block; ramping the
    whole 0.7 -> 0.9 m would exaggerate the pitch sweep by the decimation. ``a`` is silent by now.
    """
    ramp = torch.arange(1, K + 1, dtype=torch.float32) / K
    expected = 0.1 / 0.85 + (0.1 / 0.9 - 0.1 / 0.85) * ramp
    torch.testing.assert_close(decimated["second"].cpu(), expected, rtol=1e-4, atol=1e-6)


def test_reset_clears_the_delay_line_of_the_reset_environment_only(probe_source, audio_context):
    """A reset environment starts silent instead of replaying the previous episode's tail; its neighbour does not."""
    env = _mic_env(num_envs=2, mic=(2.0, 0.0, 0.0), enable_doppler=False)
    try:
        audio_context(env).register(probe_source(env, [1000.0] * 4), "probe")
        mic = env.sensors["mic"]
        for _ in range(4):
            env.step()
        # Emissions are exhausted, so anything heard from here on is still in flight in the delay line: one block
        # spans 3.43 m, so a microphone 2 m away is still hearing the last block's tail.
        env.reset(torch.tensor([0], device=env.device))
        env.step()
        heard = mic.read().reshape(2, -1).abs().amax(dim=-1)
        assert float(heard[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(heard[1]) > 0.0

        env.reset()
        env.step()
        assert float(mic.read().abs().max()) == pytest.approx(0.0, abs=1e-6)
    finally:
        env.destroy()


def test_mismatched_audio_substeps_is_rejected_at_build():
    """One scene, one sample rate: a disagreeing participant fails at build, naming both values."""
    from eden.envs.base import EnvBase

    config = EdenConfig(
        env_options=EnvOptions(num_envs=1, sim_dt=SIM_DT),
        scene_options=SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1), fixed=True)),
        sensors_options=SensorsOptions(
            mic20=SensorOptions(sensor=SpatialAudio(audio_substeps=20)),
            mic32=SensorOptions(sensor=SpatialAudio(audio_substeps=32)),
        ),
    )
    env = EnvBase.from_config(config, show_viewer=False)
    try:
        with pytest.raises(gs.GenesisException, match="audio_substeps=32.*audio_substeps=20"):
            env.build()
    finally:
        env.destroy()


def test_inactive_context_refuses_reads_but_no_ops_its_hooks():
    """Per the ``SharedSensorContext`` contract: reads raise, lifecycle hooks do nothing."""
    context = AudioContext(sim=None)
    with pytest.raises(gs.GenesisException, match="inactive"):
        _ = context.sources
    context.update()
    context.reset(None)
