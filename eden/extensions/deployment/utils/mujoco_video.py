"""An offscreen MuJoCo render of a sim-to-sim rollout to an mp4, for either MuJoCo backend.

:class:`MujocoVideo` takes a compiled model and its data, the bodies to frame and the env's
kinematic markers (goal markers MuJoCo does not simulate, drawn where the env has them). Recording
starts at the policy's first START and from then on writes every tick, a STOP's holds included, so
the video never skips. Headless, MuJoCo needs ``MUJOCO_GL=egl`` (or ``osmesa``).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np

import eden as en
from eden.options.entities import SphereOptions


class MujocoVideo:
    """Offscreen MuJoCo render of a rollout to an mp4: two views side by side, framed on ``framed_bodies``.

    ``framed_bodies`` are body ids whose starting positions the views look at, not the whole model,
    whose extent a floor and a table stretch. ``markers`` are ``(name, options)`` of the scene's
    kinematic spheres and boxes; :meth:`tick` takes their poses.
    """

    _HEIGHT, _WIDTH = 480, 640
    #: (azimuth, elevation) of the two views, degrees.
    _VIEWS = ((150.0, -25.0), (90.0, -10.0))

    def __init__(
        self,
        model,
        data,
        path: str,
        *,
        fps: float,
        framed_bodies: list[int],
        markers: Sequence[tuple[str, Any]] = (),
    ) -> None:
        import mujoco
        from genesis.utils.video_encoder import VideoEncoder

        self._mujoco = mujoco
        self._data = data
        self.path = path
        self.markers = list(markers)
        #: Frames written so far: once one is, every tick is.
        self.frames = 0
        headlight = model.vis.headlight  # the compiled default renders a robot dark
        headlight.ambient[:] = (0.45, 0.45, 0.45)
        headlight.diffuse[:] = (0.65, 0.65, 0.65)
        try:
            self._renderer = mujoco.Renderer(model, self._HEIGHT, self._WIDTH)
        except Exception as e:
            raise RuntimeError(
                f"record_video={path!r}: MuJoCo cannot open an offscreen renderer ({e}). On a machine without a "
                "display set MUJOCO_GL=egl (or osmesa) before starting the run."
            ) from e
        points = data.xpos[framed_bodies]
        low, high = points.min(axis=0), points.max(axis=0)
        self._cameras = []
        for azimuth, elevation in self._VIEWS:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = 0.5 * (low + high)
            camera.distance = max(1.8 * float(np.linalg.norm(high - low)), 0.5)
            camera.azimuth, camera.elevation = azimuth, elevation
            self._cameras.append(camera)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._encoder = VideoEncoder(path, int(round(fps)), is_threaded=False)

    def tick(self, policy_active: bool, marker_poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        """Write this tick's frame once the policy has been started: from the first START on, every tick."""
        if policy_active or self.frames:
            self.frame(marker_poses)

    def frame(self, marker_poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        mujoco = self._mujoco
        views = []
        for camera in self._cameras:
            self._renderer.update_scene(self._data, camera)
            scene = self._renderer.scene
            for name, options in self.markers:
                if scene.ngeom >= scene.maxgeom:
                    break
                pos, quat = marker_poses[name]
                rotation = np.zeros(9)
                mujoco.mju_quat2Mat(rotation, np.asarray(quat, dtype=np.float64))
                if isinstance(options, SphereOptions):
                    kind, size = mujoco.mjtGeom.mjGEOM_SPHERE, [float(options.radius), 0.0, 0.0]
                else:
                    kind, size = mujoco.mjtGeom.mjGEOM_BOX, [0.5 * float(v) for v in options.size]
                rgba = np.array((list(marker_color(options)) + [1.0])[:4], dtype=np.float32)
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom], kind, np.array(size), np.asarray(pos), rotation, rgba)
                scene.ngeom += 1
            views.append(self._renderer.render().copy())
        self._encoder.write(np.concatenate(views, axis=1))
        self.frames += 1

    def close(self) -> None:
        """Finish the mp4 and say where it is."""
        self._encoder.close()
        self._renderer.close()
        en.logger.info(f"[deploy] MuJoCo rollout video: {self.path} ({self.frames} frames)")


def marker_color(options) -> tuple[float, ...]:
    """A marker's colour as the env renders it: its own ``color``, else its surface's, else a translucent green."""
    for color in (getattr(options, "color", None), getattr(getattr(options, "surface", None), "color", None)):
        if color is not None:
            return tuple(float(c) for c in color)
    return (0.1, 0.9, 0.1, 0.6)
