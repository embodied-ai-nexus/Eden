"""Observation terms for the Extreme Parkour tasks (teacher + student).

Fixed slot table of the single ``policy`` group (753 dims total)::

    prop[0:53] | scan[53:185] | priv_explicit[185:194] | priv_latent[194:223] | hist[223:753]

The student (depth-distillation) task adds two extra observation groups on top:
:class:`ParkourDepthImage` (the processed forward depth frame) and
:class:`ParkourDeltaYawOk` (the yaw-substitution gate consumed by the
distillation runner).

Three terms carry per-step state (the proprio contact latch / delta-yaw cache / history push,
the rate-limited height scan, the two-slot depth frame pair). Each advances it in
``ObservationTerm.post_physics_refresh`` — phase 7 of ``RLEnvBase.step``, once per control
step, on the post-physics scene and above the auto-reset — zeroes rows in ``reset()`` and
returns from its buffers in ``compute()``, which is a read (``docs/guides/terms.md``,
"``compute()`` is a read, not a tick"). The ``update_interval`` cadences are rates on
``common_step_counter`` inside the hook, not re-entrancy guards. Sibling terms
(``ParkourProprioHistory``, ``ParkourDeltaYawOk``) read the owner's buffers directly: the
manager ticks every class term before any ``compute()``, so the order the groups compute in
is irrelevant (the owner must still be *declared* first — the siblings find it at build).

What a freshly reset env observes is what the respawn determines, rebuilt in ``post_reset``
(phase 14, and after a manual ``env.reset()``): a cleared contact latch (the contact-force
buffer still holds the ended episode until the next physics step, so it is not read), a zero
history on the reset step with the spawn frame seeded behind it (repeated from the next step
on), and — on a due step, counter 0 of a run included — a delta-yaw, scan and depth frame taken
against the spawn pose. On a non-due step the delta-yaw and scan read zero and the depth frame
is the pre-reset one until the next due tick, exactly as the source's rate-limited caches do.
Because the state advances in the phase-7 hook, ``EnvOptions.record_final_observations`` does
not change what a reset env observes.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import numpy as np
import torch
from genesis.utils.geom import quat_to_xyz

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.terms.observations.locomotion import GridScanDots
from eden.managers.terms.utils import resolve_link_idx
from eden.tasks.extreme_parkour.custom_terms.commands import ParkourCommand
from eden.utils.geom import wrap_to_pi
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.entities.rigid import RigidEntity


def _resolve_parkour_command(term: ObservationTerm, command_name: str) -> ParkourCommand:
    """Resolve ``command_name`` to a :class:`ParkourCommand`, failing fast with the available names."""
    manager = term._env.command_manager
    if manager is None:
        raise ValueError(f"{type(term).__name__} requires a command manager with a '{command_name}' term.")
    try:
        command = manager.get_term(command_name)
    except KeyError:
        raise ValueError(
            f"{type(term).__name__}: command term '{command_name}' not found. Available: {manager.active_terms}"
        ) from None
    if not isinstance(command, ParkourCommand):
        raise TypeError(
            f"{type(term).__name__}: command term '{command_name}' is a {type(command).__name__}, "
            "expected ParkourCommand."
        )
    return command


@OBSERVATION_TERM_REGISTRY.register()
class ParkourProprio(ObservationTerm):
    """53-dim Extreme Parkour proprioception block.

    Layout::

        [0:3]   base angular velocity (body frame) * 0.25
        [3:5]   wrapped roll, pitch
        [5]     0.0 (yaw slot, zeroed as in the source)
        [6]     delta_yaw       = target_yaw - yaw       (NOT wrapped; refreshed every
        [7]     delta_next_yaw  = next_target_yaw - yaw   `update_interval` steps, else cached)
        [8:10]  zeros (vy / wz command slots, zeroed as in the source)
        [10]    commanded vx
        [11]    (~env_is_flat)  — parkour-course indicator
        [12]    env_is_flat     — flat-course indicator
        [13:25] dofs_pos - default_dofs_pos
        [25:37] dofs_vel * 0.05
        [37:49] last raw action
        [49:53] contact_filt - 0.5, contact_filt = (|F_foot| > threshold) | previous-step contact

    The term also owns the 10-frame proprio history consumed by
    :class:`ParkourProprioHistory`: at each step's tick (:meth:`post_physics_refresh`) it
    snapshots the PRE-append buffer into ``_history_out`` (so the exposed history excludes
    the current frame) and then pushes the current 53-vector with the delta-yaw slots
    ``[6:8]`` zeroed. :meth:`post_reset` backfills a respawned env's buffer with its spawn
    frame and an episode's first tick (``episode_length_buf == 1``) backfills it again with
    that step's frame, so the second and third observations of an episode each see one frame
    repeated (the source's ``obs_history_buf``). The contact latch and the delta-yaw cache
    advance in the same tick; ``compute()`` rebuilds the 53-vector from the live entity and
    these buffers.

    ``history_steps`` is deliberately NOT named ``history_length`` — that name is a
    reserved post-compute modifier on ``ObservationTermOptions`` and would make the
    observation manager wrap this term's output in a second history buffer.
    """

    entity_name: str = "robot"
    command_name: str = "parkour_command"
    feet_link_names: list[str] = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    contact_force_threshold: float = 2.0
    update_interval: int = 5
    history_steps: int = 10

    #: Per-env handshake registry so ParkourProprioHistory can find its sibling instance at
    #: build time (``env.observation_manager`` is not yet assigned while the manager is
    #: constructing its terms, so the sibling cannot be resolved through the manager there).
    _by_env: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def build(self) -> None:
        entity: RigidEntity = self._env.entities[self.entity_name]
        self._command = _resolve_parkour_command(self, self.command_name)
        self._feet_idx = resolve_link_idx(
            entity, self.feet_link_names, self.device, term=type(self).__name__, field="feet_link_names"
        )
        # Stricter than the helper's empty check: this term's layout is one block per named foot.
        if self._feet_idx.numel() != len(self.feet_link_names):
            raise ValueError(
                f"{term_subject(self)}: feet_link_names {self.feet_link_names} resolved to "
                f"{self._feet_idx.numel()} links."
            )
        num_envs, num_feet = self.num_envs, self._feet_idx.numel()
        self._last_contacts = torch.zeros(num_envs, num_feet, dtype=torch.bool, device=self.device)
        self._contact_filt = torch.zeros_like(self._last_contacts)
        self._delta_yaw = torch.zeros(num_envs, device=self.device)
        self._delta_next_yaw = torch.zeros(num_envs, device=self.device)
        self._obs_dim = 53
        self._history_buffer = torch.zeros(num_envs, self.history_steps, self._obs_dim, device=self.device)
        self._history_out = torch.zeros_like(self._history_buffer)
        ParkourProprio._by_env[self._env] = self

    def post_physics_refresh(self) -> None:
        # All persistent buffers are updated with copy_ (never rebound): this runs under
        # torch.inference_mode, and a rebound inference tensor could no longer be mutated by
        # reset() outside it.
        entity: RigidEntity = self.resolve_entity(self.entity_name)
        forces = entity.get_links_net_contact_force()[:, self._feet_idx]  # (N, n_feet, 3)
        contact = torch.norm(forces, dim=-1) > self.contact_force_threshold
        self._contact_filt.copy_(contact | self._last_contacts)
        self._last_contacts.copy_(contact)

        if self._env.common_step_counter % self.update_interval == 0:
            yaw = entity.get_heading()
            # NOT wrapped — matches the source.
            self._delta_yaw.copy_(self._command.target_yaw - yaw)
            self._delta_next_yaw.copy_(self._command.next_target_yaw - yaw)

        # Pre-append snapshot: the exposed history excludes the current frame.
        self._history_out.copy_(self._history_buffer)
        frame = self._build_obs()  # a fresh tensor — safe to edit in place
        frame[:, 6:8] = 0.0  # the history carries no delta-yaw hints
        # The episode's first tick: the counter is incremented before phase 7, and reset() zeroed it.
        fresh = (self._env.episode_length_buf == 1).view(-1, 1, 1)
        rolled = torch.cat([self._history_buffer[:, 1:], frame.unsqueeze(1)], dim=1)
        backfilled = frame.unsqueeze(1).expand(-1, self.history_steps, -1)
        self._history_buffer.copy_(torch.where(fresh, backfilled, rolled))

    def post_reset(self, envs_idx: slice | torch.Tensor) -> None:
        """Rebuild the reset envs' state from the post-reset scene (phase 14, and after a manual reset).

        The hook for what the respawn determines (``ManagerTermBase.post_reset``): the delta-yaw
        cache is re-aimed at the fresh goal on a due step (the command's own ``reset()`` already
        recomputed ``target_yaw`` against the spawn pose), and the history buffer is backfilled
        with the spawn frame, so the episode's second observation carries that frame
        ``history_steps`` times — the source's ``obs_history_buf`` semantics. The reset step
        itself still exposes the zero history ``reset()`` left in ``_history_out``: the next tick
        is what publishes the buffer.

        The contact latch is deliberately NOT rebuilt here. Genesis writes
        ``links.contact_force`` only inside the constraint-solver step — a scene reset clears
        the collider's contact data but not that buffer — so until the next physics step
        ``get_links_net_contact_force()`` still returns the *ended* episode's forces for a
        respawned env. Reading it here would carry the old episode's contact into the new one
        (through ``_last_contacts``, for two observations). ``reset()`` cleared the latch; a
        respawned env reads no contact until its first tick measures one.
        """
        entity: RigidEntity = self.resolve_entity(self.entity_name)
        if self._env.common_step_counter % self.update_interval == 0:
            yaw = entity.get_heading()
            self._delta_yaw[envs_idx] = (self._command.target_yaw - yaw)[envs_idx]
            self._delta_next_yaw[envs_idx] = (self._command.next_target_yaw - yaw)[envs_idx]
        frame = self._build_obs()
        frame[:, 6:8] = 0.0
        self._history_buffer[envs_idx] = frame[envs_idx].unsqueeze(1).expand(-1, self.history_steps, -1)

    def compute(self) -> torch.Tensor:
        return self._build_obs()

    def _build_obs(self) -> torch.Tensor:
        entity: RigidEntity = self.resolve_entity(self.entity_name)
        command = self._command
        num_envs = self.num_envs

        rpy = quat_to_xyz(entity.get_quat(), rpy=True)
        zeros1 = torch.zeros(num_envs, 1, device=self.device)
        return torch.cat(
            [
                entity.get_ang(frame="body") * 0.25,  # [0:3]
                wrap_to_pi(rpy[:, 0:1]),  # [3]
                wrap_to_pi(rpy[:, 1:2]),  # [4]
                zeros1,  # [5]
                self._delta_yaw.unsqueeze(1),  # [6]
                self._delta_next_yaw.unsqueeze(1),  # [7]
                torch.zeros(num_envs, 2, device=self.device),  # [8:10]
                command.command[:, 0:1],  # [10]
                (~command.env_is_flat).float().unsqueeze(1),  # [11]
                command.env_is_flat.float().unsqueeze(1),  # [12]
                entity.get_dofs_pos() - entity.default_dofs_pos,  # [13:25]
                entity.get_dofs_vel() * 0.05,  # [25:37]
                self._env.action_manager.action,  # [37:49]
                self._contact_filt.float() - 0.5,  # [49:53]
            ],
            dim=1,
        )

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._history_buffer[envs_idx] = 0.0
        self._history_out[envs_idx] = 0.0
        self._last_contacts[envs_idx] = False
        self._contact_filt[envs_idx] = False
        self._delta_yaw[envs_idx] = 0.0
        self._delta_next_yaw[envs_idx] = 0.0


# Registry key: "parkour_scan_dots", derived from the class name (term registries take no name=).
@OBSERVATION_TERM_REGISTRY.register()
class ParkourScanDots(GridScanDots):
    """132-dim (12x11) height scan, Extreme Parkour normalization.

    ``GridScanDots`` with ``offset_from_base=True`` returns ``terrain_z - base_z``;
    the Extreme Parkour scan is ``clip(base_z - terrain_z - 0.3, -1, 1)``, i.e.
    ``clip(-(parent output) - 0.3, -1, 1)``. The scan is refreshed at the tick of every
    ``update_interval``-th control step and the cached tensor is returned in between.
    ``reset()`` zeroes a restarting env's row; on a due step :meth:`post_reset` re-scans it
    around the spawn pose, otherwise the env observes the zero row until the next refresh
    (the source's semantics — its scan is a rate-limited cache too).
    """

    measured_points_x: list[float] = [0.375 + v for v in np.linspace(-0.825, 0.825, 12).tolist()]
    measured_points_y: list[float] = np.linspace(-0.75, 0.75, 11).tolist()
    offset_from_base: bool = True
    update_interval: int = 5

    def build(self) -> None:
        super().build()
        scan_dim = len(self.measured_points_x) * len(self.measured_points_y)
        # Pre-allocated and updated with copy_ (never rebound) so reset() can mutate it
        # outside inference mode after the tick wrote it inside.
        self._scan = torch.zeros(self.num_envs, scan_dim, device=self.device)

    def post_physics_refresh(self) -> None:
        if self._env.common_step_counter % self.update_interval == 0:
            raw = super().compute()  # terrain_z - base_z
            self._scan.copy_(torch.clip(-raw - 0.3, -1.0, 1.0))

    def compute(self, *args, **kwargs) -> torch.Tensor:
        return self._scan

    def post_reset(self, envs_idx: slice | torch.Tensor) -> None:
        if self._env.common_step_counter % self.update_interval == 0:
            raw = super().compute()
            self._scan[envs_idx] = torch.clip(-raw - 0.3, -1.0, 1.0)[envs_idx]

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._scan[envs_idx] = 0.0


@OBSERVATION_TERM_REGISTRY.register()
class ParkourPrivExplicit(ObservationTerm):
    """9-dim explicit-privileged block: ``cat(base_lin_vel_body * 2.0, zeros(6))``.

    Fixed slot ``[185:194]`` of the policy group. At deployment/distillation time
    the RMA estimator overwrites this slot at act time with its prediction; during
    teacher training the simulator ground truth (plus six zero slots kept for
    layout compatibility with the source) is fed directly.
    """

    entity_name: str = "robot"

    def build(self) -> None:
        _ = self._env.entities[self.entity_name]
        self._zeros = torch.zeros(self.num_envs, 6, device=self.device)

    def compute(self) -> torch.Tensor:
        base_lin_vel = self.resolve_entity(self.entity_name).get_vel(frame="body")
        return torch.cat([base_lin_vel * 2.0, self._zeros], dim=1)


def process_parkour_depth(
    depth: torch.Tensor,
    *,
    near_clip: float,
    far_clip: float,
    resized_hw: tuple[int, int],
    crop_bottom: int = 2,
    crop_sides: int = 4,
) -> torch.Tensor:
    """Extreme Parkour depth preprocessing: crop -> clip -> bicubic resize -> normalize.

    ``depth`` is a raw ``(N, H, W)`` metric depth batch. The pipeline (order matters —
    clipping before the resize keeps bicubic overshoot local to real edges, matching the
    source): drop ``crop_bottom`` rows from the image bottom and ``crop_sides`` columns
    from each side, clamp to ``[near_clip, far_clip]`` (no-hit / sky pixels arrive as
    ``+inf`` from the batch renderer and land on ``far_clip``), bicubic-resize to
    ``resized_hw``, then map to ``value / (far - near) - 0.5`` (so ``[-0.5, 0.5]`` up to
    bicubic overshoot, which the source also keeps).
    """
    frames = depth[:, : depth.shape[1] - crop_bottom, crop_sides : depth.shape[2] - crop_sides]
    frames = torch.nan_to_num(frames, nan=far_clip, posinf=far_clip, neginf=near_clip)
    frames = frames.clamp(near_clip, far_clip).float()
    frames = torch.nn.functional.interpolate(
        frames.unsqueeze(1), size=resized_hw, mode="bicubic", align_corners=False
    ).squeeze(1)
    return (frames - near_clip) / (far_clip - near_clip) - 0.5


#: The raw camera resolution the source-fixed crop margins (2 bottom rows, 4 side columns)
#: were designed for; the camera wired to :class:`ParkourDepthImage` must render exactly this.
#: Single source of the 106x60 contract — the student config imports it for its camera.
_DEPTH_RAW_RESOLUTION = (106, 60)  # (width, height)


@OBSERVATION_TERM_REGISTRY.register()
class ParkourDepthImage(ObservationTerm):
    """Processed forward depth frame for the Extreme Parkour student, shape ``(N, 58, 87)``.

    Renders the batched depth camera ONLY at the tick of every ``update_interval``-th
    control step (batch renders are the dominant cost of student training; there are no
    off-cadence renders and no per-step device syncs) and pushes the processed frame (see
    :func:`process_parkour_depth`) into a two-slot buffer; ``compute`` returns the OLDER
    slot, so the policy always sees the previous interval's frame — the source's simulated
    one-interval (100 ms) camera latency. Between renders the cached frame is returned
    unchanged.

    Reset envs are marked pending in :meth:`reset` — which never renders — and have BOTH
    slots backfilled with a post-respawn frame: by :meth:`post_reset` when the reset lands
    on a due step (one extra batch render on such a step, after the auto-reset — the initial
    ``env.reset()`` at counter 0 included, so the first observation of a run is a real frame),
    otherwise at the next due tick. Until then a reset env's group value still shows its
    pre-reset view, so an off-cadence consumer sees up to ``update_interval - 1`` steps of
    pre-teleport imagery (source-consistent: its reset envs keep the stale frame between
    refreshes too). The distillation runner and the play policy encode a new frame only on
    the due steps, where a backfill has always landed.

    The raw render is metric plane depth (z-depth) from the gs-madrona batch renderer —
    the convention of the original Isaac Gym implementation (the IsaacLab port's ray-cast
    camera returns euclidean ray length instead; the difference is largest in the image
    corners and bounded by the far clip).
    """

    camera_name: str = "depth"
    update_interval: int = 5
    near_clip: float = 0.0
    far_clip: float = 2.0
    resized_hw: tuple[int, int] = (58, 87)

    def build(self) -> None:
        camera = self._env.cameras.get(self.camera_name)
        if camera is None:
            raise ValueError(
                f"ParkourDepthImage: camera '{self.camera_name}' not found. "
                f"Available cameras: {sorted(self._env.cameras)}"
            )
        # The crop margins are absolute pixel counts tuned for the source sensor; a
        # different camera resolution would crop/resize into geometrically wrong frames
        # with the right output shape (bicubic maps anything to resized_hw) — fail here.
        if tuple(camera._options.cam_resolution) != _DEPTH_RAW_RESOLUTION:
            raise ValueError(
                f"ParkourDepthImage: camera '{self.camera_name}' renders "
                f"{tuple(camera._options.cam_resolution)}, but the source-fixed crop expects "
                f"{_DEPTH_RAW_RESOLUTION} (width, height)."
            )
        h, w = self.resized_hw
        # Buffers updated with copy_ (never rebound) — the tick runs under
        # torch.inference_mode and the buffers must stay mutable outside it.
        self._prev_frame = torch.zeros(self.num_envs, h, w, device=self.device)
        self._curr_frame = torch.zeros_like(self._prev_frame)
        self._pending_backfill = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def post_physics_refresh(self) -> None:
        if self._env.common_step_counter % self.update_interval != 0:
            return
        frame = self._render_frame()
        # Roll the pair; pending (reset) envs get the fresh frame in both slots. Sync-free:
        # torch.where with an all-false mask is a no-op, so the mask never touches the host.
        mask = self._pending_backfill.view(-1, 1, 1)
        self._prev_frame.copy_(torch.where(mask, frame, self._curr_frame))
        self._curr_frame.copy_(frame)
        self._pending_backfill.zero_()

    def compute(self) -> torch.Tensor:
        return self._prev_frame

    def post_reset(self, envs_idx: slice | torch.Tensor) -> None:
        if self._env.common_step_counter % self.update_interval != 0:
            return  # the pending mark lands at the next due tick (no off-cadence render)
        # A due step's tick rendered before the reset; render the respawned scene once more
        # and give the reset envs (exactly the pending ones) that frame in both slots. No roll:
        # the cadence pair of every other env is untouched.
        frame = self._render_frame()
        self._prev_frame[envs_idx] = frame[envs_idx]
        self._curr_frame[envs_idx] = frame[envs_idx]
        self._pending_backfill[envs_idx] = False

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._pending_backfill[envs_idx] = True

    def _render_frame(self) -> torch.Tensor:
        depth = self._env.cameras[self.camera_name].render_depth()
        if not torch.is_tensor(depth):
            depth = torch.as_tensor(depth, device=self.device)
        if depth.ndim == 4:  # (N, H, W, 1)
            depth = depth.squeeze(-1)
        elif depth.ndim == 2:  # unbatched (H, W)
            depth = depth.unsqueeze(0)
        if depth.shape[0] != self.num_envs:
            raise RuntimeError(
                f"ParkourDepthImage: camera '{self.camera_name}' rendered {depth.shape[0]} envs, "
                f"expected {self.num_envs}. All envs must be rendered (rendered_envs_idx default)."
            )
        return process_parkour_depth(
            depth.to(self.device), near_clip=self.near_clip, far_clip=self.far_clip, resized_hw=self.resized_hw
        )


@OBSERVATION_TERM_REGISTRY.register()
class ParkourDeltaYawOk(ObservationTerm):
    """Yaw-substitution gate for depth distillation: ``delta_yaw < threshold`` as ``(N, 1)`` float.

    During student training the runner replaces the observed delta-yaw slots ``[6:8]`` with
    the depth encoder's yaw prediction only where this gate is 1 (the true goal heading is
    hidden from the deployed policy, but a badly disoriented robot keeps the real signal).
    The comparison is one-sided (NOT ``abs``) and unwrapped — faithful to both reference
    implementations. Reads the delta-yaw cache of the sibling :class:`ParkourProprio` term
    (refreshed at its tick every ``update_interval`` steps) so the gate and slot ``[6]``
    never diverge. The manager ticks every class term before any ``compute()``, so the
    cache is current whichever group computes first; the sibling is found at build, so
    ``ParkourProprio`` must be declared before this term (terms build in declaration order).
    """

    threshold: float = 0.6

    def build(self) -> None:
        proprio = ParkourProprio._by_env.get(self._env)
        if proprio is None:
            raise RuntimeError(
                "ParkourDeltaYawOk requires a ParkourProprio term declared in the same env's "
                "observation groups — none was built for this env."
            )
        self._proprio: ParkourProprio = proprio

    def compute(self) -> torch.Tensor:
        return (self._proprio._delta_yaw < self.threshold).float().unsqueeze(1)


@OBSERVATION_TERM_REGISTRY.register()
class ParkourProprioHistory(ObservationTerm):
    """530-dim (10 x 53) proprio history, read from the sibling :class:`ParkourProprio`.

    The history buffer itself is owned by ``ParkourProprio`` (it is the pre-append
    snapshot, so the newest frame is excluded) and advanced at its tick, before any
    ``compute()`` of the step. This term only flattens it. The sibling is resolved through
    ``ParkourProprio._by_env`` at build time because ``env.observation_manager`` is not yet
    assigned while the manager constructs its terms — which is why ``proprio`` must still
    be declared before this term (terms build in declaration order).
    """

    def build(self) -> None:
        proprio = ParkourProprio._by_env.get(self._env)
        if proprio is None:
            raise RuntimeError(
                "ParkourProprioHistory requires a ParkourProprio term declared earlier in the same "
                "observation group — none was built for this env."
            )
        self._proprio: ParkourProprio = proprio

    def compute(self) -> torch.Tensor:
        return self._proprio._history_out.reshape(self.num_envs, -1)
