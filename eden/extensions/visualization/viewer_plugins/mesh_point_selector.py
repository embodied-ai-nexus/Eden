"""Interactive viewer plugin for selecting points on rigid meshes.

Ported from the Genesis ``examples/viewer_plugin/mesh_point_selector.py`` demo.
Left-click a rigid mesh in the Genesis viewer to toggle a selected point; the
point is stored in the clicked link's local frame so it tracks the link as it
moves. On viewer close the selections are written to a CSV file.
"""

import csv
import os
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from pyglet.window import key, mouse
from typing_extensions import override

import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array
from genesis.vis.viewer_plugins import EVENT_HANDLE_STATE, RaycasterViewerPlugin

import eden as en

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidLink
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node


class SelectedPoint(NamedTuple):
    """A point selected on a rigid mesh surface.

    Attributes
    ----------
    link : RigidLink
        The rigid link that the point belongs to.
    local_position : np.ndarray, shape (3,)
        The position of the point in the link's local coordinate frame.
    local_normal : np.ndarray, shape (3,)
        The surface normal at the point in the link's local coordinate frame.
    """

    link: "RigidLink"
    local_position: np.ndarray  # shape (3,)
    local_normal: np.ndarray  # shape (3,)


class MeshPointSelectorPlugin(RaycasterViewerPlugin):
    """Viewer plugin that lets mouse clicks select points on rigid meshes.

    A left *click* toggles the point under the cursor; a left *drag* still orbits
    the camera (middle drag pans, right drag / scroll zooms), so selection and
    navigation share the left button. Selected points are stored in link-local
    coordinates, so they follow the link when the entity moves.

    Parameters
    ----------
    sphere_radius : float
        Radius of the debug sphere drawn at a selected point.
    sphere_color : tuple
        RGBA color of a selected point.
    hover_color : tuple
        RGBA color of the hover preview under the cursor.
    grid_snap : tuple of float, shape (3,)
        Grid a selected point snaps to, per axis in the link's local frame. A negative
        value leaves that axis unsnapped.
    output_file : str
        CSV the selections are written to when the viewer closes.
    use_visual_geom : bool
        Select points on the visual meshes rather than the collision meshes, so a click
        lands on what is drawn on screen. Only entities whose material sets
        ``use_visual_raycasting=True`` are then hittable, and every visual triangle is
        rescanned each step, which costs several times the collision meshes on detailed
        geometry. Defaults to False. See ``RaycasterViewerPlugin``.
    """

    # A left press that moves more than this many pixels before release is an
    # orbit drag, not a selection click.
    _CLICK_DRAG_TOLERANCE_PX = 5

    def __init__(
        self,
        sphere_radius: float = 0.005,
        sphere_color: tuple = (0.1, 0.3, 1.0, 1.0),
        hover_color: tuple = (0.3, 0.5, 1.0, 1.0),
        grid_snap: tuple[float, float, float] = (-1.0, -1.0, -1.0),
        output_file: str = "selected_points.csv",
        use_visual_geom: bool = False,
    ) -> None:
        super().__init__(use_visual_geom)
        self.sphere_radius = sphere_radius
        self.sphere_color = sphere_color
        self.hover_color = hover_color
        self.grid_snap = grid_snap
        self.output_file = output_file

        self.selected_points: dict[int, SelectedPoint] = {}
        self._prev_mouse_pos: tuple[int, int] = (0, 0)
        self._press_pos: tuple[int, int] | None = None  # armed left click, committed on release
        self._closed = False  # guard so the CSV is written exactly once
        self._warned_non_rigid = False  # explain a refused visual-only hit exactly once

    def build(self, viewer, camera: "Node", scene: "Scene"):
        super().build(viewer, camera, scene)

        # Resolved here rather than at module scope: importing RigidLink pulls
        # genesis.utils.array_class, which requires gs.init() to have run. Cached so the
        # per-frame hover check in on_draw does not re-resolve it.
        from genesis.engine.entities.rigid_entity.rigid_link import RigidLink

        self._rigid_link_cls = RigidLink

        # A screen-space click maps to a single world ray, so the selection is only
        # well defined for one environment. Eden scenes are always vectorized
        # (``n_envs >= 1``); a raw Genesis scene is unbatched (``n_envs == 0``). Both
        # are fine, but multiple envs would make the clicked point ambiguous.
        if scene.n_envs > 1:
            raise ValueError(
                f"{type(self).__name__} supports single-environment scenes only "
                f"(n_envs <= 1); got n_envs={scene.n_envs}. Rebuild the env with num_envs=1."
            )

        # Fail loudly now if the output directory is not writable, rather than losing a
        # whole interactive session's selections when the CSV is written on close.
        out_dir = os.path.dirname(os.path.abspath(self.output_file)) or "."
        if not os.access(out_dir, os.W_OK):
            raise ValueError(
                f"{type(self).__name__} cannot write output_file '{self.output_file}': "
                f"directory '{out_dir}' is not writable."
            )

        self._prev_mouse_pos = (self.viewer._viewport_size[0] // 2, self.viewer._viewport_size[1] // 2)

    @staticmethod
    def _link_world_pose(link: "RigidLink") -> tuple[np.ndarray, np.ndarray]:
        """World-frame (position, wxyz-quaternion) of a link, for the single environment.

        ``relative=False`` selects the world frame -- Genesis defaults to the user
        frame, which strips the up/front alignment (see the sibling ``viser.py``).
        Eden's getters carry a leading env axis of length 1; ``atleast_2d(...)[0]``
        reduces that to a flat ``(3,)``/``(4,)`` and also accepts the already-flat
        shape returned by an unbatched Genesis scene.
        """
        pos = tensor_to_array(link.get_pos(relative=False))
        quat = tensor_to_array(link.get_quat(relative=False))
        return np.atleast_2d(pos)[0], np.atleast_2d(quat)[0]

    def _is_selectable(self, ray_hit) -> bool:
        """Whether a hit lands on a link this plugin can record, explaining a refusal once.

        Only a rigid link qualifies. A visual cast (``use_visual_geom=True``) also reaches
        the kinematic solver, whose links are numbered from that solver's own counter — so
        a kinematic link and a rigid link can carry the same ``link.idx``, which the CSV's
        single ``link_idx`` column cannot tell apart. Kinematic entities are visual-only
        ghosts and markers, so refusing them costs nothing and keeps the output honest.
        """
        if ray_hit is None or ray_hit.geom is None:
            return False
        if isinstance(ray_hit.geom.link, self._rigid_link_cls):
            return True
        if not self._warned_non_rigid:
            self._warned_non_rigid = True
            en.logger.warning(
                f"[{type(self).__name__}] ignoring a hit on '{ray_hit.geom.link.name}': it belongs to the "
                "kinematic solver, whose link indices are not comparable with the rigid ones the CSV records. "
                "Only rigid entities can be selected."
            )
        return False

    def _get_pos_hash(self, pos: np.ndarray) -> int:
        """Hash a position to a unique identifier (rounded to 6 decimals)."""
        return hash((round(pos[0], 6), round(pos[1], 6), round(pos[2], 6)))

    def _snap_to_grid(self, point: np.ndarray) -> np.ndarray:
        """Snap a point to the grid on each axis whose ``grid_snap`` is non-negative."""
        grid_snap = np.array(self.grid_snap)
        return np.where(grid_snap >= 0, np.round(point / grid_snap) * grid_snap, point)

    @override
    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        self._prev_mouse_pos = (x, y)

    @override
    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        # Arm a selection for an unmodified left press, but do NOT consume the event:
        # the viewer's trackball still gets it, so left-drag orbits, and ctrl/shift/alt
        # + left stay as camera zoom/pan. The selection commits on release only if the
        # press did not turn into a drag (see on_mouse_drag / on_mouse_release).
        if button == mouse.LEFT and not (modifiers & (key.MOD_CTRL | key.MOD_SHIFT | key.MOD_ALT)):
            self._press_pos = (x, y)
        return None

    @override
    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> EVENT_HANDLE_STATE:
        # Once the left button moves past the tolerance it is an orbit drag, not a
        # click -- cancel the pending selection. Never consume: the trackball orbits.
        if self._press_pos is not None and (buttons & mouse.LEFT):
            px, py = self._press_pos
            if abs(x - px) + abs(y - py) > self._CLICK_DRAG_TOLERANCE_PX:
                self._press_pos = None
        return None

    @override
    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if button == mouse.LEFT and self._press_pos is not None:
            self._press_pos = None
            self._toggle_point_at(x, y)
        return None

    def _toggle_point_at(self, x: int, y: int) -> None:
        """Toggle the selected point on the mesh under screen coordinates (x, y)."""
        ray = self._screen_position_to_ray(x, y)
        ray_hit = self._raycaster.cast(*ray)
        if not self._is_selectable(ray_hit):
            return

        link = ray_hit.geom.link
        link_pos, link_quat = self._link_world_pose(link)

        local_pos = gu.inv_transform_by_trans_quat(ray_hit.position, link_pos, link_quat)
        local_normal = gu.inv_transform_by_quat(ray_hit.normal, link_quat)
        local_pos = self._snap_to_grid(local_pos)

        pos_hash = self._get_pos_hash(local_pos)
        if pos_hash in self.selected_points:
            del self.selected_points[pos_hash]  # deselect if already selected
        else:
            self.selected_points[pos_hash] = SelectedPoint(link, local_pos, local_normal)

    @override
    def on_draw(self) -> None:
        super().on_draw()
        if self.scene._visualizer is not None and self.scene._visualizer.is_built:
            self.scene.clear_debug_objects()
            mouse_ray = self._screen_position_to_ray(*self._prev_mouse_pos)

            # Previewed only where a click would actually select, so "no preview" reads as
            # "not selectable" rather than as a click that mysteriously did nothing.
            closest_hit = self._raycaster.cast(*mouse_ray)
            if self._is_selectable(closest_hit):
                snap_pos = self._snap_to_grid(closest_hit.position)
                self.scene.draw_debug_sphere(
                    snap_pos,
                    self.sphere_radius,
                    self.hover_color,
                )
                self.scene.draw_debug_arrow(
                    snap_pos,
                    tuple(n * 0.05 for n in closest_hit.normal),
                    self.sphere_radius / 2,
                    self.hover_color,
                )

            if self.selected_points:
                world_positions = []
                for point in self.selected_points.values():
                    link_pos, link_quat = self._link_world_pose(point.link)
                    local_pos_arr = np.array(point.local_position, dtype=np.float32)
                    current_world_pos = gu.transform_by_trans_quat(local_pos_arr, link_pos, link_quat)
                    world_positions.append(current_world_pos)

                if len(world_positions) == 1:
                    self.scene.draw_debug_sphere(
                        world_positions[0],
                        self.sphere_radius,
                        self.sphere_color,
                    )
                else:
                    positions_array = np.array(world_positions)
                    self.scene.draw_debug_spheres(positions_array, self.sphere_radius, self.sphere_color)

    @override
    def on_close(self) -> None:
        super().on_close()

        # Genesis may invoke on_close more than once during viewer teardown; only
        # write the CSV on the first call.
        if self._closed:
            return
        self._closed = True

        if not self.selected_points:
            en.logger.info("[MeshPointSelectorPlugin] No points selected.")
            return

        output_file = self.output_file
        try:
            with open(output_file, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                writer.writerow(
                    [
                        "point_idx",
                        "link_idx",
                        "local_pos_x",
                        "local_pos_y",
                        "local_pos_z",
                        "local_normal_x",
                        "local_normal_y",
                        "local_normal_z",
                    ]
                )

                for i, point in enumerate(self.selected_points.values(), 1):
                    writer.writerow(
                        [
                            i,
                            point.link.idx,
                            point.local_position[0],
                            point.local_position[1],
                            point.local_position[2],
                            point.local_normal[0],
                            point.local_normal[1],
                            point.local_normal[2],
                        ]
                    )

            en.logger.info(
                f"[MeshPointSelectorPlugin] Wrote {len(self.selected_points)} selected points to '{output_file}'"
            )

        except Exception as e:
            en.logger.error(f"[MeshPointSelectorPlugin] Error writing to '{output_file}': {e}")
