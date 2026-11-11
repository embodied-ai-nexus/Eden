# Portions Copyright 2026 the Tactile Genesis authors. Derived from Tactile Genesis
# (https://github.com/neuroagents-lab/tactile-genesis, commit 3023bcd), licensed under the Apache License,
# Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Modified by the Eden authors.
"""Airborne point microphone: the rendering stage of the audio pipeline.

Provenance
----------
Ported from ``neuroagents-lab/tactile-genesis`` @ ``3023bcd``
(``Genesis/genesis/engine/sensors/spatial_audio.py``), where it is a Genesis in-tree sensor reading a scene-level
``AudioManager``. Here it renders :class:`~eden.engine.sensors.audio.AudioContext` instead, which needs no
Genesis change; see :mod:`eden.engine.sensors.audio`.

Deviations from upstream: ``enable_occlusion`` is dropped (upstream declares it and never reads it);
``attenuation`` is a ``Literal`` rather than a free ``str`` whose typos fell back to ``inverse``; the sample
history lives on the context, not the microphone; and the link attachment goes through Genesis'
``KinematicSensorMixin``, so ``pos_offset`` is link-local and ``set_pos_offset`` works as on any other sensor.

Limitation: free-field propagation only — no occlusion, no reflections, no room response. A microphone behind a
wall hears the source as if the wall were not there.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import genesis as gs
import torch
from genesis.engine.sensors.base_sensor import (
    KinematicSensorMetadataMixin,
    KinematicSensorMixin,
    SimpleSensor,
    SimpleSensorMetadata,
)
from genesis.options.sensors.options import KinematicSensorOptionsMixin, SimpleSensorOptions
from genesis.typing import PositiveFloat, PositiveInt
from genesis.utils.geom import transform_by_quat
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array
from pydantic import StrictBool

from eden.engine._surface import require_members
from eden.engine.sensors.audio import AudioContext

if TYPE_CHECKING:
    from genesis.engine.sensors.sensor_manager import SensorManager
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.vis.rasterizer_context import RasterizerContext

require_members(
    __name__,
    {
        SimpleSensor: ("_update_raw_data", "_get_return_format", "_get_cache_dtype", "reset", "build", "_draw_debug"),
        KinematicSensorMixin: ("_register_link", "set_pos_offset"),
    },
)


class SpatialAudio(KinematicSensorOptionsMixin["SpatialAudioSensor"], SimpleSensorOptions["SpatialAudioSensor"]):
    """A mono point microphone that hears every audio source registered with the scene.

    ``read()`` returns ``(B, audio_substeps)``: this step's block of samples, one microphone signal per
    environment. Sources are summed with ``ref_distance``-referenced distance attenuation and delayed
    by ``distance / speed_of_sound`` through a fractional delay line.

    Parameters
    ----------
    entity_idx : int
        Global entity index the microphone rides. ``-1`` pins it in the world at ``pos_offset``.
    link_idx_local : int
        Local link index on that entity.
    pos_offset : array-like[float, float, float]
        Microphone position: in the link's frame when attached, world when static.
    euler_offset : array-like[float, float, float]
        Accepted for parity with other link-attached sensors; a mono point microphone has no direction, so it
        does not change what is heard.
    audio_substeps : int
        Samples rendered per physics step. Must match every source in the scene; the effective sample
        rate is ``audio_substeps / sim_dt``.
    speed_of_sound : float
        Propagation speed in m/s, setting the delay per meter. Default ``343.0`` (air, ~20 °C).
    ref_distance : float
        Distance in meters at which a source is heard at unit gain. Attenuation is clamped closer than
        this, so a source passing through the microphone stays finite. Default ``0.1``.
    attenuation : {'inverse', 'inverse_square'}
        Falloff with distance. ``'inverse'`` is ``ref/r`` (sound *pressure*, the physical free-field
        law and the usual choice); ``'inverse_square'`` is ``(ref/r)**2``, falling off faster.
    enable_doppler : bool
        Ramp delay and attenuation across each block from last step's value to this step's, which
        pitch-shifts a source in relative motion. Costs nothing extra; disable for a static rig.
    max_delay : float
        Longest propagation delay to keep history for, in seconds. Sources farther than
        ``max_delay * speed_of_sound`` are heard at that clamped delay rather than their true one.
        Sizes the delay line, so raise it only as far as the scene needs. Default ``0.03`` (~10 m).

    Examples
    --------
    A microphone on a robot's head, hearing its own motors:

    >>> from eden.options import SensorOptions
    >>> from eden.engine.sensors.spatial_audio import SpatialAudio
    >>> mic = SensorOptions(
    ...     sensor=SpatialAudio(audio_substeps=20, ref_distance=0.2),
    ...     attach_entity_name="robot",
    ...     attach_link_name="head_link",
    ... )
    """

    audio_substeps: PositiveInt = 20
    speed_of_sound: PositiveFloat = 343.0
    ref_distance: PositiveFloat = 0.1
    attenuation: Literal["inverse", "inverse_square"] = "inverse"
    enable_doppler: StrictBool = True
    max_delay: PositiveFloat = 0.03


@dataclass
class SpatialAudioSensorMetadata(KinematicSensorMetadataMixin, SimpleSensorMetadata):
    """Per-class state of every :class:`SpatialAudioSensor`, batched so one pass renders them all.

    The listener parameters are grown one column per sensor at build, alongside the link buckets and offsets of
    ``KinematicSensorMetadataMixin``. ``prev_dist`` also depends on the source count, which is only final once
    every source has registered, so it is allocated on the first render instead. The sample history itself is
    not here: it belongs to :class:`~eden.engine.sensors.audio.AudioContext`, which advances it every
    simulation step whether or not a microphone renders.
    """

    # Per-listener parameters, shape (n_listeners,).
    speed_of_sound: torch.Tensor = make_tensor_field((0,))
    ref_distance: torch.Tensor = make_tensor_field((0,))
    inverse_square: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: torch.bool)
    doppler: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: torch.bool)

    # Position of each sample within a block, shape (1, 1, 1, K): its ramp fraction and its age in samples.
    ramp: torch.Tensor = make_tensor_field((0,))
    age: torch.Tensor = make_tensor_field((0,))

    prev_dist: torch.Tensor = make_tensor_field((0, 0, 0))  # (B, n_listeners, n_src) last render's distances
    last_render: int = 0  # AudioContext.n_updates at the last render


class SpatialAudioSensor(KinematicSensorMixin, SimpleSensor[SpatialAudio, AudioContext, SpatialAudioSensorMetadata]):
    """Renders :class:`~eden.engine.sensors.audio.AudioContext` into one microphone signal per listener.

    For each (listener, source) pair the current distance gives an attenuation and a delay in samples;
    the delay indexes the per-source sample history with linear interpolation between the two
    straddling samples, and the results are summed over sources. Interpolating the delay across the
    block from its value one simulation step ago to its current one is what produces Doppler: a shrinking
    delay reads the history faster than real time, raising the pitch.
    """

    def __init__(
        self,
        options: SpatialAudio,
        idx: int,
        shared_context: AudioContext,
        shared_metadata: SpatialAudioSensorMetadata,
        manager: "SensorManager",
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)
        self.debug_object: "Mesh | None" = None

    def build(self):
        """Claim the scene's sample rate and append this listener's column."""
        super().build()

        sm = self._shared_metadata
        context = self._shared_context
        # A microphone alone in the scene is valid (it renders silence), so activate rather than
        # register: activation is what a *consumer* of the registry declares.
        context.activate()
        context.require_audio_substeps(self._options.audio_substeps, type(self._options).__name__)

        sm.speed_of_sound = concat_with_tensor(sm.speed_of_sound, self._options.speed_of_sound)
        sm.ref_distance = concat_with_tensor(sm.ref_distance, self._options.ref_distance)
        sm.inverse_square = concat_with_tensor(sm.inverse_square, self._options.attenuation == "inverse_square")
        sm.doppler = concat_with_tensor(sm.doppler, self._options.enable_doppler)

        K = self._options.audio_substeps
        if sm.ramp.numel() == 0:
            sm.ramp = (torch.arange(1, K + 1, device=gs.device, dtype=gs.tc_float) / K).view(1, 1, 1, K)
            sm.age = torch.arange(K - 1, -1, -1, device=gs.device, dtype=gs.tc_float).view(1, 1, 1, K)

        # The delay line must span this listener's longest delay plus the block it renders.
        dt_sub = self._manager._sim.dt / K
        delay_samples = int(self._options.max_delay / dt_sub) + 2
        context.require_history(K + delay_samples)

    def _get_return_format(self) -> tuple[int, ...]:
        return (self._options.audio_substeps,)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def reset(cls, shared_metadata: SpatialAudioSensorMetadata, shared_ground_truth_cache: torch.Tensor, envs_idx):
        """Clear the Doppler history so a reset environment renders no spurious pitch sweep.

        Parameters
        ----------
        shared_metadata : SpatialAudioSensorMetadata
            Per-class state to clear.
        shared_ground_truth_cache : torch.Tensor
            The class's ground-truth cache slice, handled by the base implementation.
        envs_idx : array-like | None
            Environments being reset.
        """
        super().reset(shared_metadata, shared_ground_truth_cache, envs_idx)
        if shared_metadata.prev_dist.numel() > 0:
            shared_metadata.prev_dist[envs_idx] = 0.0

    @classmethod
    def _listener_positions(cls, shared_metadata: SpatialAudioSensorMetadata) -> torch.Tensor:
        """``(B, n_listeners, 3)`` world position of every microphone.

        An attached microphone rides its link with ``pos_offset`` rotated into the link's frame; a static one sits
        at ``pos_offset``. One bulk pose read per solver, over the listeners' links only.

        Parameters
        ----------
        shared_metadata : SpatialAudioSensorMetadata
            Per-class state holding the offsets and the per-solver link buckets.
        """
        pos = shared_metadata.offsets_pos.clone()
        for group in shared_metadata.solver_groups:
            links_pos = group.solver.get_links_pos(links_idx=group.links_idx)
            links_quat = group.solver.get_links_quat(links_idx=group.links_idx)
            if group.solver.n_envs == 0:
                links_pos, links_quat = links_pos[None], links_quat[None]
            cols = group.sensor_cols.long()
            pos[:, cols] = links_pos + transform_by_quat(pos[:, cols], links_quat)
        return pos

    @classmethod
    def _update_raw_data(
        cls, shared_context: AudioContext, shared_metadata: SpatialAudioSensorMetadata, raw_data_T: torch.Tensor
    ):
        """Render every registered source into every listener's block for this step.

        Parameters
        ----------
        shared_context : AudioContext
            The source registry. Already refreshed for this step by ``SensorManager.step``.
        shared_metadata : SpatialAudioSensorMetadata
            Per-class listener parameters and propagation state.
        raw_data_T : torch.Tensor
            Transposed write target, ``(n_listeners * audio_substeps, B)``.
        """
        sm = shared_metadata
        n_blocks = shared_context.blocks_since(sm.last_render, SpatialAudio.__name__)
        sm.last_render = shared_context.n_updates

        src_hist = shared_context.src_hist
        # Nothing radiating (no sources registered, or none with emission points): the mic is silent.
        if src_hist.numel() == 0:
            raw_data_T[:] = 0.0
            return
        K = shared_context.audio_substeps
        B, n_src, H = src_hist.shape
        n_listeners = sm.n_sensors

        # The source count is only final once every source has registered, so the Doppler history is
        # sized on first use rather than at build.
        if sm.prev_dist.shape != (B, n_listeners, n_src):
            sm.prev_dist = torch.zeros((B, n_listeners, n_src), dtype=gs.tc_float, device=gs.device)

        listener_pos = cls._listener_positions(sm)  # (B, nL, 3)
        src_pos = shared_context.source_positions()  # (B, n_src, 3)
        r = (listener_pos[:, :, None, :] - src_pos[:, None, :, :]).norm(dim=-1)  # (B, nL, n_src)

        # Doppler ramps delay and gain across the block, from the distance one simulation step ago to the
        # current one. The last render may be several steps back (Eden renders once per control step by
        # default), so that distance is interpolated rather than taken from the last render as is — ramping a
        # whole control step's travel across one block would exaggerate the pitch shift by the decimation.
        # On the first render after a reset (prev_dist == 0) or with Doppler off, hold the current value so the
        # block carries no spurious sweep.
        hold = (sm.prev_dist <= 0) | ~sm.doppler.view(1, -1, 1)
        r_start = torch.where(hold, r, r - (r - sm.prev_dist) / max(n_blocks, 1))
        sm.prev_dist.copy_(r)

        ref = sm.ref_distance.view(1, -1, 1)
        inverse_square = sm.inverse_square.view(1, -1, 1)
        samples_per_meter = 1.0 / (sm.speed_of_sound.view(1, -1, 1) * (shared_context._sim.dt / K))
        # The oldest sample a block may read is its own oldest plus the delay, which must stay inside the ring.
        max_delay = float(H - K - 1)

        def gain_and_delay(dist):
            gain = ref / dist.clamp(min=ref)
            return torch.where(inverse_square, gain * gain, gain), (dist * samples_per_meter).clamp(max=max_delay)

        gain_start, delay_start = gain_and_delay(r_start)
        gain_end, delay_end = gain_and_delay(r)
        delay_k = torch.lerp(delay_start[..., None], delay_end[..., None], sm.ramp)  # (B, nL, n_src, K)
        gain_k = torch.lerp(gain_start[..., None], gain_end[..., None], sm.ramp)

        # Fractional age of what block sample k hears, mapped onto the ring with linear interpolation.
        age_k = sm.age + delay_k
        age_floor = age_k.floor()
        frac = age_k - age_floor
        newer = (shared_context.hist_newest - age_floor.long()) % H
        hist = src_hist[:, None, :, :].expand(B, n_listeners, n_src, H)  # broadcast over listeners
        sample = torch.lerp(hist.gather(3, newer), hist.gather(3, (newer - 1) % H), frac)

        out = (gain_k * sample).sum(dim=2)  # (B, nL, K)
        raw_data_T[:] = out.permute(1, 2, 0).reshape(n_listeners * K, B)

    def _draw_debug(self, context: "RasterizerContext"):
        """Draw a sphere at the microphone whose radius grows with the loudness of the last block.

        Parameters
        ----------
        context : RasterizerContext
            The viewer context to draw into.
        """
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None
        pos = self._listener_positions(self._shared_metadata)[0 if env_idx is None else env_idx, self._idx]
        amp = float(self.read(env_idx).reshape(-1).abs().max())

        if self.debug_object is not None:
            context.clear_debug_object(self.debug_object)
            self.debug_object = None
        radius = 0.02 + min(amp, 1.0) * 0.05
        self.debug_object = context.draw_debug_sphere(
            pos=tensor_to_array(pos), radius=radius, color=(0.2, 0.6, 1.0, 0.6)
        )
