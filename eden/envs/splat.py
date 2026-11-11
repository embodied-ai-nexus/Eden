"""Gaussian-splat environments (non-RL ``SplatEnvBase`` and RL ``SplatRLEnvBase``), one gsplat backend.

Both render a photoreal splat background with the single
:class:`~eden.utils.batch_splat_renderer.BatchSplatRenderer`, composited over the sim pass by
segmentation mask on each camera that declares a
:class:`~eden.entities.camera_overlays.SplatComposite` overlay (a camera that declares none shows
the plain sim pass). They differ only in the RL surface — mirroring ``EnvBase`` / ``RLEnvBase`` —
so the shared plumbing lives in :class:`_SplatEnvMixin`. Whether a view is rendered once and
broadcast (``SplatCamera``) or per-env (``BatchSplatCamera``) is a *camera* choice
(``SplatOptions.camera_mode``), orthogonal to the RL split. Nyx
(:class:`~eden.envs.nyx_splat.NyxSplatEnvBase`) is the separate path-traced backend.
"""

from __future__ import annotations

import torch
import genesis as gs
from genesis.options.renderers import RendererOptions

from eden.envs.base import EnvBase, RLEnvBase
from eden.options import (
    ActionManagerOptions,
    CameraOptions,
    CamerasOptions,
    CommandManagerOptions,
    CurriculumManagerOptions,
    EnvOptions,
    EventManagerOptions,
    MetricManagerOptions,
    ObservationManagerOptions,
    RecorderManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    SensorsOptions,
    SplatOptions,
    TerminationManagerOptions,
)
from eden.utils.batch_splat_renderer import BatchSplatRenderer, SplatEntry
from eden.utils.configs import EdenSplatConfig, EdenSplatRLConfig
from eden.utils.misc import resolve_named_entity


class _SplatEnvMixin:
    """Shared gsplat plumbing for :class:`SplatEnvBase` (non-RL) and :class:`SplatRLEnvBase` (RL).

    Holds no ``__init__`` so it stays transparent in the cooperative ``super().__init__`` chain
    (each env's own ``__init__`` routes to ``EnvBase`` / ``RLEnvBase``). It is only ever mixed into
    an ``EnvBase`` subclass, so it relies on ``num_envs`` / ``device`` / ``entities`` /
    ``cameras_options`` being present at runtime.

    **Must be listed first** in each env's bases (before ``EnvBase`` / ``RLEnvBase``): it *overrides*
    :meth:`_setup_cameras` and :meth:`_load_managers`, so the MRO has to reach the mixin's versions
    before ``EnvBase``'s. Put it last and ``EnvBase``'s versions win: the env builds plain
    ``Camera``s (a declared ``SplatComposite`` then raises that the camera is not a splat env's) and
    never builds the renderer. That is why the ordering is stated here and at each class.
    """

    splat_options: SplatOptions
    _splat_renderer: BatchSplatRenderer | None
    _splat_body_bindings: list[tuple[int, str, int]]

    def _init_splat(self, splat_options: SplatOptions | None) -> None:
        """Validate ``splat_options`` and null out the deferred renderer. Call before ``super().__init__``.

        Checked here (not just in ``from_config``) so direct construction is covered too. The
        renderer itself is built later, by :meth:`_load_managers`: resolving the entity/link handles
        that back dynamic (body-attached) splats needs the built scene.
        """
        if splat_options is None:
            raise ValueError(
                f"`{type(self).__name__}` requires `splat_options` (it owns a splat renderer). Set "
                "`splat_options=SplatOptions(...)` on the config, or build a task without splats "
                "through `env_class_for_config(cfg).from_config(cfg)`."
            )
        if not isinstance(splat_options, SplatOptions):
            raise TypeError(
                f"`{type(self).__name__}` renders through the gsplat backend, so `splat_options` must "
                f"be a `SplatOptions`, not a `{type(splat_options).__name__}`. For `NyxSplatOptions`, "
                f"build the config through `NyxSplatEnvBase`."
            )
        # Held so `env.config` can report it live (a declared field of the config leaf).
        self.splat_options = splat_options
        self._splat_renderer = None
        self._splat_body_bindings = []

    # ----------------------------------------------------------------- renderer

    def _load_managers(self):
        """Build the splat renderer, then load the managers — in that order.

        The renderer must exist *before* ``super()._load_managers()``, because
        ``ObservationManager._prepare_terms`` sizes every term by calling ``compute()`` once, and a
        camera observation term (``observations.rgb_image``) computes by *rendering* — which reaches
        into ``env._splat_renderer``. Load the managers first and that probe dies on a still-``None``
        renderer with ``AttributeError: 'NoneType' object has no attribute 'per_env_gaussians'``, so
        a splat env could not carry a camera observation at all. Reward, termination and metric terms
        that render are covered by the same ordering.

        This runs from ``EnvBase.post_build_setup``, after every entity's ``post_build()``, so the
        built-scene handles that :meth:`_build_splat_renderer` resolves for entity-attached splats
        are ready. A ``register_post_build_hook`` registration would be too late:
        ``post_build_setup`` drains those hooks *below* ``_load_managers()``. Same idiom as
        :meth:`~eden.envs.base.MimicRLEnvBase._load_managers`, for the same reason — a term may
        reference this during its own init.
        """
        self._build_splat_renderer()
        super()._load_managers()

    def _build_splat_renderer(self):
        """Load the splat PLYs and resolve any entity-attached (dynamic) splats."""
        entries: list[SplatEntry] = []
        self._splat_body_bindings = []
        next_body_row = 0
        for name, pcd in self.splat_options.pcds.items():
            body_row: int | None = None
            if pcd.attach_entity is not None:
                entity = resolve_named_entity(
                    self.entities, pcd.attach_entity, hint=f"Splat PCD '{name}' attaches to it via `attach_entity`."
                )
                link_idx = entity.get_link(pcd.attach_link).idx_local if pcd.attach_link else 0
                body_row = next_body_row
                next_body_row += 1
                self._splat_body_bindings.append((body_row, pcd.attach_entity, link_idx))
            entries.append(
                SplatEntry(ply_path=pcd.path, pos=pcd.pos, quat=pcd.quat, scale=pcd.scale, body_row=body_row)
            )

        if self._splat_body_bindings and not self.splat_options.per_env_gaussians:
            raise ValueError(
                f"SplatOptions has entity-attached (dynamic) splats "
                f"{[name for _, name, _ in self._splat_body_bindings]}, whose per-env motion needs "
                "per-env Gaussians. Set SplatOptions(per_env_gaussians=True)."
            )

        self._splat_renderer = BatchSplatRenderer(
            entries,
            num_envs=self.num_envs,
            per_env_gaussians=self.splat_options.per_env_gaussians,
            device=self.device,
            minibatch=self.splat_options.minibatch,
        )

    def _read_splat_body_poses(self):
        """Stack the current per-env world poses of every dynamic splat body."""
        n_bodies = max(self._splat_renderer.n_bodies, 1)
        pos = torch.zeros(self.num_envs, n_bodies, 3, device=self.device)
        quat = torch.zeros(self.num_envs, n_bodies, 4, device=self.device)
        quat[..., 0] = 1.0
        for body_row, entity_name, link_idx in self._splat_body_bindings:
            entity = self.entities[entity_name]
            # relative=False (the get_links_pos default) is the *solver* link pose. For a free prop Genesis
            # re-origins (a DOF-less body) it sits on the centre of mass and principal axes, not on
            # default_root_pos, which places the authored frame (relative=True); for a fixed or
            # articulated body the two coincide. Either frame is rigid with the body, so the splat tracks it
            # as long as PCDOptions.pos/quat are calibrated in the frame read here.
            pos[:, body_row] = entity.get_links_pos(ls_idx_local=[link_idx]).reshape(self.num_envs, 3)
            quat[:, body_row] = entity.get_links_quat(ls_idx_local=[link_idx]).reshape(self.num_envs, 4)
        return pos, quat

    def _refresh_splat_gaussians(self):
        """Refresh per-env Gaussians from the *current* sim poses (per-env mode; no-op otherwise).

        Called on every ``render_rgb`` rather than cached per control step: renders are pull-based
        and a caller may move entities between renders (e.g. ``set_pos`` outside ``step()``), so
        keying on the step counter would render a stale background. The update is a per-body pose
        read plus a rigid transform — far cheaper than the rasterization it precedes.
        """
        if not self.splat_options.per_env_gaussians:
            return
        pos, quat = self._read_splat_body_poses()
        self._splat_renderer.update(pos, quat)

    def splat_debug_points(self):
        """Return the xyz coordinates of the splat points in the world frame for ``draw_debug_points()``."""
        return self._splat_renderer.world_xyz

    # ------------------------------------------------------------------- cameras

    def _splat_camera_class(self, camera_options: CameraOptions):
        """Pick the camera class that renders this camera's splat (see ``SplatOptions.camera_mode``).

        ``BatchSplatCamera`` renders every env's camera (needed when the pose differs per env — an
        attached/following camera — or when Gaussians differ per env); ``SplatCamera`` renders one
        env-invariant view and broadcasts it (the cheap path for a fixed camera over a shared scene).
        """
        from eden.entities.camera import BatchSplatCamera, SplatCamera

        mode = self.splat_options.camera_mode
        per_env = self.splat_options.per_env_gaussians
        if mode == "single":
            if per_env:
                raise ValueError(
                    "SplatOptions.camera_mode='single' cannot render per_env_gaussians=True (each "
                    "env's Gaussians differ, so a single broadcast view is wrong); use 'auto' or 'batched'."
                )
            return SplatCamera
        if mode == "batched":
            return BatchSplatCamera
        # auto: per-env whenever the Gaussians or the camera pose vary across envs.
        if per_env:
            return BatchSplatCamera
        per_env_pose = bool(camera_options.attach_entity_name) or bool(camera_options.follow_entity_name)
        return BatchSplatCamera if per_env_pose else SplatCamera

    @gs.assert_unbuilt
    def _setup_cameras(self):
        self.cameras: dict[str, object] = {}
        for camera_name in self.cameras_options.keys():
            camera_options: CameraOptions = getattr(self.cameras_options, camera_name)
            cam_cls = self._splat_camera_class(camera_options)
            self.cameras[camera_name] = cam_cls(key=camera_name, env=self, options=camera_options)


# _SplatEnvMixin must stay first so its _setup_cameras / _load_managers override EnvBase's
# (see the mixin docstring).
class SplatEnvBase(_SplatEnvMixin, EnvBase[EdenSplatConfig]):
    """Non-RL environment with a gsplat splat background (viewer / teleop / inference / data collection).

    A sibling of :class:`SplatRLEnvBase` with no reward/termination/command/curriculum surface, as
    ``EnvBase`` is of ``RLEnvBase``. The ``EnvBase[EdenSplatConfig]`` base re-binds the config leaf.
    """

    def __init__(
        self,
        env_options: EnvOptions | None = None,
        scene_options: SceneOptions | None = None,
        splat_options: SplatOptions | None = None,
        observation_options: ObservationManagerOptions | None = None,
        event_options: EventManagerOptions | None = None,
        action_options: ActionManagerOptions | None = None,
        metric_options: MetricManagerOptions | None = None,
        recorder_options: RecorderManagerOptions | None = None,
        cameras_options: CamerasOptions | None = None,
        sensors_options: SensorsOptions | None = None,
        renderer_options: RendererOptions | None = None,
        *,
        show_viewer: bool = False,
        eval_mode: bool = False,
        **kwargs,
    ):
        self._init_splat(splat_options)
        super().__init__(  # -> EnvBase (no RL managers)
            env_options=env_options,
            scene_options=scene_options,
            observation_options=observation_options,
            event_options=event_options,
            action_options=action_options,
            metric_options=metric_options,
            recorder_options=recorder_options,
            cameras_options=cameras_options,
            sensors_options=sensors_options,
            renderer_options=renderer_options,
            show_viewer=show_viewer,
            eval_mode=eval_mode,
            **kwargs,
        )


# _SplatEnvMixin must stay first so its _setup_cameras / _load_managers override EnvBase's
# (see the mixin docstring).
class SplatRLEnvBase(_SplatEnvMixin, RLEnvBase, EnvBase[EdenSplatRLConfig]):
    """RL environment with a gsplat splat background (splat scene + full RL surface).

    The RL counterpart of :class:`SplatEnvBase`, sharing its splat plumbing via
    :class:`_SplatEnvMixin`; the extra ``EnvBase[EdenSplatRLConfig]`` base re-binds the config leaf,
    the same way :class:`~eden.envs.base.MimicRLEnvBase` does.
    """

    def __init__(
        self,
        env_options: EnvOptions | None = None,
        scene_options: SceneOptions | None = None,
        splat_options: SplatOptions | None = None,
        observation_options: ObservationManagerOptions | None = None,
        reward_options: RewardManagerOptions | None = None,
        termination_options: TerminationManagerOptions | None = None,
        event_options: EventManagerOptions | None = None,
        curriculum_options: CurriculumManagerOptions | None = None,
        command_options: CommandManagerOptions | None = None,
        action_options: ActionManagerOptions | None = None,
        metric_options: MetricManagerOptions | None = None,
        recorder_options: RecorderManagerOptions | None = None,
        cameras_options: CamerasOptions | None = None,
        sensors_options: SensorsOptions | None = None,
        renderer_options: RendererOptions | None = None,
        *,
        show_viewer: bool = False,
        eval_mode: bool = False,
        **kwargs,
    ):
        self._init_splat(splat_options)
        super().__init__(  # -> RLEnvBase (full RL surface)
            env_options=env_options,
            scene_options=scene_options,
            observation_options=observation_options,
            reward_options=reward_options,
            termination_options=termination_options,
            event_options=event_options,
            curriculum_options=curriculum_options,
            command_options=command_options,
            action_options=action_options,
            metric_options=metric_options,
            recorder_options=recorder_options,
            cameras_options=cameras_options,
            sensors_options=sensors_options,
            renderer_options=renderer_options,
            show_viewer=show_viewer,
            eval_mode=eval_mode,
            **kwargs,
        )
