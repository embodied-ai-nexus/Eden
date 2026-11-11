"""ImGui panel for interactively arranging scene entities (``eden arrange``).

Provides per-entity sliders for the 6 base DoFs (xyz position + roll/pitch/yaw)
and every articulated joint, plus arrangement helpers (drop-to-support-surface,
overlap check, physics settling) and a one-click export of the resulting poses as a
``SceneOptions`` snippet ready to paste into a task config.
"""

from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

import eden as en

if TYPE_CHECKING:
    from genesis.vis.viewer import Viewer

    from eden.entities.rigid import RigidEntity

# How often (in update() calls) the overlap check re-runs collision detection.
_OVERLAP_CHECK_PERIOD = 15
# Clearance left under the AABB when dropping an entity onto a support surface.
_DROP_EPS = 1e-3
# Main-loop pacing in kinematic mode (no physics stepping to set the pace).
_KINEMATIC_FRAME_DT = 0.005
# Tolerance for "did something other than our sliders move this entity" (float32 sim state).
_EXTERNAL_CHANGE_ATOL = 1e-5


@dataclasses.dataclass
class _EntityState:
    key: str  # entity name in the scene / exported SceneOptions field name
    options_class: str  # EntityOptions subclass name, used in the exported snippet
    entity: "RigidEntity"
    gs_name: str  # genesis-level entity name, used to re-bind after a scene rebuild
    pos: np.ndarray  # (3,) world position
    rpy: np.ndarray  # (3,) roll/pitch/yaw in degrees
    joint_names: list[str]
    joint_qs_idx: list[int]  # genesis-level q indices, parallel to joint_names
    joint_values: np.ndarray  # (n_joints,)
    joint_limits: np.ndarray  # (n_joints, 2)
    # Last pose written to (or read back from) the sim; used to detect external edits
    # (gizmo drags, built-in single-step/reset) that the sliders should follow.
    applied_pos: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    applied_quat: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(4))
    applied_qpos: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    init_pos: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    init_rpy: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    init_joint_values: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    dirty: bool = False  # a slider changed since the last apply
    drop_requested: bool = False
    reset_requested: bool = False
    n_contacts: int = 0


def _read_sim_pose(state: _EntityState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # The authored frame (``relative=True``) is the one the exported ``default_root_pos`` /
    # ``default_root_quat`` are applied in; the solver frame of a free prop sits on its centre of mass
    # and principal axes, and would land that offset away when the snippet is loaded back.
    pos = state.entity.get_pos(relative=True).squeeze(0).cpu().numpy()
    quat = state.entity.get_quat(relative=True).squeeze(0).cpu().numpy()
    if state.joint_names:
        qpos = state.entity.get_qpos(qs_idx_local=state.joint_qs_idx).squeeze(0).cpu().numpy()
    else:
        qpos = np.zeros(0, dtype=np.float32)
    return pos, quat, qpos


def _build_entity_state(key: str, options_class: str, entity: "RigidEntity") -> _EntityState:
    """Snapshot the entity's current pose/joints into an editable state."""
    import genesis as gs
    from genesis.utils.geom import quat_to_xyz

    joint_names: list[str] = []
    joint_qs_idx: list[int] = []
    joint_limits: list[tuple[float, float]] = []
    for joint in entity.entity.joints:
        if joint.n_qs == 0 or joint.type in (gs.JOINT_TYPE.FIXED, gs.JOINT_TYPE.FREE):
            continue
        if joint.type == gs.JOINT_TYPE.SPHERICAL:
            en.logger.warning(f"[arrange] skipping spherical joint '{joint.name}' on '{key}' (no slider mapping).")
            continue
        for i in range(joint.n_qs):
            joint_names.append(joint.name if joint.n_qs == 1 else f"{joint.name}[{i}]")
            joint_qs_idx.append(int(joint.qs_idx_local[i]))
            lo, hi = float(joint.desc.dofs_limit[i, 0]), float(joint.desc.dofs_limit[i, 1])
            lo = lo if np.isfinite(lo) else -np.pi
            hi = hi if np.isfinite(hi) else +np.pi
            joint_limits.append((lo, hi))

    state = _EntityState(
        key=key,
        options_class=options_class,
        entity=entity,
        gs_name=entity.entity.name,
        pos=np.zeros(3),
        rpy=np.zeros(3),
        joint_names=joint_names,
        joint_qs_idx=joint_qs_idx,
        joint_values=np.zeros(len(joint_names)),
        joint_limits=np.asarray(joint_limits, dtype=np.float64).reshape(-1, 2),
    )

    pos, quat, qpos = _read_sim_pose(state)
    state.pos[:] = pos
    state.rpy[:] = quat_to_xyz(quat.astype(np.float64), rpy=True, degrees=True)
    if joint_names:
        state.joint_values[:] = np.clip(qpos, state.joint_limits[:, 0], state.joint_limits[:, 1])
    state.applied_pos, state.applied_quat, state.applied_qpos = pos, quat, qpos
    state.init_pos = state.pos.copy()
    state.init_rpy = state.rpy.copy()
    state.init_joint_values = state.joint_values.copy()
    return state


class _StripDefaultOpenImgui:
    """Imgui proxy that drops extra args from ``collapsing_header`` so headers start collapsed."""

    def __init__(self, imgui):
        self._imgui = imgui

    def __getattr__(self, attr):
        return getattr(self._imgui, attr)

    def collapsing_header(self, label, *args, **kwargs):
        return self._imgui.collapsing_header(label)


def _make_overlay_plugin():
    from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin

    class _ArrangeOverlayPlugin(ImGuiOverlayPlugin):
        """Overlay whose built-in entity browser headers start collapsed.

        The arrange panel is the primary control surface, so the per-entity
        sections of the stock browser (which default to open) only add clutter.
        """

        def _render_entity_browser(self):
            imgui = self._imgui
            self._imgui = _StripDefaultOpenImgui(imgui)
            try:
                super()._render_entity_browser()
            finally:
                self._imgui = imgui

    return _ArrangeOverlayPlugin()


class ArrangeGUI:
    """ImGui side panel for arranging entities with base 6-DoF + joint sliders.

    The panel callback runs on the viewer thread and only mutates plain Python
    state; :meth:`update` must be called from the main loop and is the only
    place that touches the simulation.

    ``update()`` drives ``scene.step()`` every frame so the overlay's built-in
    controls (play/pause/step/reset and Rebuild Scene) work: their intent is
    queued on the plugin's ``InteractiveScene`` and applied by a scene pre-step
    callback. While paused the physics advance is skipped, so the sliders place
    entities kinematically; the "Simulate physics" checkbox simply resumes the
    InteractiveScene (it is the same state as the built-in Play button).

    Parameters
    ----------
    scene:
        The built Eden scene (``env.scene``). Its viewer must be alive.
    entities: dict[str, tuple[str, RigidEntity]]
        Mapping from scene entity name to ``(options_class_name, entity)``.
    output_path: str | None
        If given, the config snippet is also written to this file on export.
    """

    def __init__(self, scene, entities: dict[str, tuple[str, "RigidEntity"]], output_path: str | None = None):
        from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin

        self._scene = scene
        self._viewer: Viewer = scene.viewer
        self._output_path = output_path
        self._states = [_build_entity_state(key, cls_name, entity) for key, (cls_name, entity) in entities.items()]

        self._simulate = False
        self._check_overlap = True
        self._export_requested = False
        self._frame = 0

        plugin = next((p for p in self._viewer.plugins if isinstance(p, ImGuiOverlayPlugin)), None)
        if plugin is None:
            plugin = self._viewer.add_plugin(_make_overlay_plugin())
        self._plugin = plugin
        plugin.register_panel(self._draw_panel, section="side")
        # Start paused: arrangement is kinematic until the user enables Simulate physics / Play.
        interactive = getattr(plugin, "interactive_scene", None)
        if interactive is not None:
            interactive.pause()
        en.logger.info("ArrangeGUI ImGui panel registered.")

    @property
    def _interactive(self):
        return getattr(self._plugin, "interactive_scene", None)

    # ------------------------------------------------------------------
    # viewer thread: panel drawing (mutates python state only)
    # ------------------------------------------------------------------

    def _draw_panel(self, imgui):
        imgui.text("Arrange Entities")
        changed, simulate = imgui.checkbox("Simulate physics##arrange_sim", self._simulate)
        if changed:
            self._simulate = simulate
            interactive = self._interactive
            if interactive is not None:
                interactive.resume() if simulate else interactive.pause()
        _, self._check_overlap = imgui.checkbox("Live overlap check##arrange_overlap", self._check_overlap)
        imgui.separator()

        for i, state in enumerate(self._states):
            if not imgui.collapsing_header(f"{state.key} ({state.options_class})##arrange_{i}"):
                continue

            if state.n_contacts > 0:
                imgui.text_colored((1.0, 0.4, 0.4, 1.0), f"{state.n_contacts} contact(s)")

            changed, new_pos = imgui.drag_float3(f"pos##arrange_pos_{i}", list(state.pos), 0.005, -100.0, 100.0, "%.3f")
            if changed:
                state.pos[:] = new_pos
                state.dirty = True
            for axis, (label, lim) in enumerate((("roll", 180.0), ("pitch", 90.0), ("yaw", 180.0))):
                changed, new_val = imgui.slider_float(
                    f"{label}##arrange_rpy_{i}_{axis}", float(state.rpy[axis]), -lim, lim, "%.1f deg"
                )
                if changed:
                    state.rpy[axis] = new_val
                    state.dirty = True

            for j, name in enumerate(state.joint_names):
                changed, new_val = imgui.slider_float(
                    f"{name}##arrange_joint_{i}_{j}",
                    float(state.joint_values[j]),
                    float(state.joint_limits[j, 0]),
                    float(state.joint_limits[j, 1]),
                    "%.3f",
                )
                if changed:
                    state.joint_values[j] = new_val
                    state.dirty = True

            if imgui.button(f"Drop to surface##arrange_drop_{i}"):
                state.drop_requested = True
            imgui.same_line()
            if imgui.button(f"Reset##arrange_reset_{i}"):
                state.reset_requested = True

        imgui.separator()
        if imgui.button("Print config snippet##arrange_export"):
            self._export_requested = True

    # ------------------------------------------------------------------
    # main thread: simulation side
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Advance one GUI frame. Call this from the main loop."""
        self._frame += 1

        interactive = self._interactive
        if interactive is not None:
            # The built-in Play/Pause buttons share the InteractiveScene paused state with our checkbox.
            self._simulate = not interactive.paused
            # Always step: while paused this skips the physics advance but applies queued
            # interactive intent (Rebuild Scene / Reset / single Step) and keeps the viewer fresh.
            self._scene.step()
            if self._rebuild_happened():
                self._rebind_after_rebuild()
        elif self._simulate:
            self._scene.step()

        for state in self._states:
            if state.reset_requested:
                state.reset_requested = False
                state.pos[:] = state.init_pos
                state.rpy[:] = state.init_rpy
                state.joint_values[:] = state.init_joint_values
                state.dirty = True
            if self._simulate:
                state.dirty = False
                self._sync_from_sim(state, force=True)
                continue
            if state.dirty:
                state.dirty = False
                self._apply_to_sim(state)
            else:
                # Follow edits made outside the sliders (gizmo drags, built-in step/reset).
                self._sync_from_sim(state)
            if state.drop_requested:
                state.drop_requested = False
                self._drop_to_surface(state)

        if not self._simulate:
            self._scene.visualizer.update(force=True)
            time.sleep(_KINEMATIC_FRAME_DT)

        if self._check_overlap and self._frame % _OVERLAP_CHECK_PERIOD == 0:
            for state in self._states:
                state.n_contacts = len(state.entity.detect_collision(env_idx=0))

        if self._export_requested:
            self._export_requested = False
            self.export()

    def _rebuild_happened(self) -> bool:
        """Whether the overlay's Rebuild Scene re-created the genesis entities under us."""
        if not self._states:
            return False
        gs_entity = self._states[0].entity.entity
        return not any(e is gs_entity for e in self._scene.entities)

    def _rebind_after_rebuild(self) -> None:
        """Re-point the Eden wrappers and slider states at the rebuilt scene's entities.

        ``InteractiveScene.rebuild()`` re-creates every genesis entity (same names, new
        objects), so the wrappers' underlying ``_entity`` handles go stale.
        """
        entities_by_name = {e.name: e for e in self._scene.entities}
        states = []
        for state in self._states:
            gs_entity = entities_by_name.get(state.gs_name)
            if gs_entity is None:
                en.logger.warning(f"[arrange] entity '{state.key}' removed by scene rebuild; dropping its sliders.")
                continue
            state.entity._entity = gs_entity
            states.append(_build_entity_state(state.key, state.options_class, state.entity))
        self._states = states
        en.logger.info("[arrange] re-bound sliders to the rebuilt scene.")

    def _apply_to_sim(self, state: _EntityState) -> None:
        from genesis.utils.geom import xyz_to_quat

        device = state.entity.device
        quat = xyz_to_quat(state.rpy, rpy=True, degrees=True)
        # Authored frame, orientation first — the order the build write uses (see _read_sim_pose).
        state.entity.set_quat(torch.as_tensor(quat, dtype=torch.float32, device=device).unsqueeze(0), relative=True)
        state.entity.set_pos(torch.as_tensor(state.pos, dtype=torch.float32, device=device).unsqueeze(0), relative=True)
        if state.joint_names:
            qpos = torch.as_tensor(state.joint_values, dtype=torch.float32, device=device).unsqueeze(0)
            state.entity.set_qpos(qpos, qs_idx_local=state.joint_qs_idx)
        # Snapshot the float32 sim state we just wrote so external edits remain detectable.
        state.applied_pos, state.applied_quat, state.applied_qpos = _read_sim_pose(state)

    def _sync_from_sim(self, state: _EntityState, force: bool = False) -> None:
        from genesis.utils.geom import quat_to_xyz

        pos, quat, qpos = _read_sim_pose(state)
        changed = (
            force
            or not np.allclose(pos, state.applied_pos, atol=_EXTERNAL_CHANGE_ATOL)
            or not np.allclose(quat, state.applied_quat, atol=_EXTERNAL_CHANGE_ATOL)
            or not np.allclose(qpos, state.applied_qpos, atol=_EXTERNAL_CHANGE_ATOL)
        )
        if not changed:
            return
        state.pos[:] = pos
        state.rpy[:] = quat_to_xyz(quat.astype(np.float64), rpy=True, degrees=True)
        if state.joint_names:
            state.joint_values[:] = qpos
        state.applied_pos, state.applied_quat, state.applied_qpos = pos, quat, qpos

    def _drop_to_surface(self, state: _EntityState) -> None:
        """Rest the entity on the highest support surface under its XY footprint.

        Candidates are the AABB tops of every other entity that overlaps the target's
        footprint and lies no higher than the target's AABB center — an object hovering
        above a table lands on the tabletop, one already under the table drops to the
        floor, and arranged entities stack on each other. The ground plane (z=0) is the
        fallback.
        """
        aabb = self._entity_aabb(state.entity.entity)
        center_z = 0.5 * (aabb[0, 2] + aabb[1, 2])
        support_z = 0.0
        for gs_entity in self._scene.entities:
            if gs_entity is state.entity.entity or not hasattr(gs_entity, "get_AABB"):
                continue
            other = self._entity_aabb(gs_entity)
            top = float(other[1, 2])
            if not np.isfinite(top) or top > center_z or top <= support_z:
                continue
            overlap_xy = (
                aabb[0, 0] <= other[1, 0]
                and aabb[1, 0] >= other[0, 0]
                and aabb[0, 1] <= other[1, 1]
                and aabb[1, 1] >= other[0, 1]
            )
            if overlap_xy:
                support_z = top
        state.pos[2] += support_z + _DROP_EPS - float(aabb[0, 2])
        self._apply_to_sim(state)

    @staticmethod
    def _entity_aabb(gs_entity) -> np.ndarray:
        """Entity AABB as ``[[min_xyz], [max_xyz]]``, collapsing any batch dimension."""
        pts = gs_entity.get_AABB().detach().cpu().numpy().reshape(-1, 3)
        return np.stack([pts.min(axis=0), pts.max(axis=0)])

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------

    def config_snippet(self) -> str:
        """Build a ``SceneOptions`` snippet with the current poses."""
        from genesis.utils.geom import xyz_to_quat

        lines = ["scene_options=SceneOptions("]
        for state in self._states:
            quat = xyz_to_quat(state.rpy, rpy=True, degrees=True)
            lines.append(f"    {state.key}={state.options_class}(")
            lines.append(f"        default_root_pos=({state.pos[0]:.4f}, {state.pos[1]:.4f}, {state.pos[2]:.4f}),")
            lines.append(f"        default_root_quat=({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}),")
            if state.joint_names:
                lines.append("        default_dofs_pos={")
                for name, value in zip(state.joint_names, state.joint_values):
                    lines.append(f'            "{name}": {value:.4f},')
                lines.append("        },")
            lines.append("    ),")
        lines.append(")")
        return "\n".join(lines)

    def export(self) -> None:
        snippet = self.config_snippet()
        print(snippet, flush=True)
        if self._output_path is not None:
            with open(self._output_path, "w") as f:
                f.write(snippet + "\n")
            en.logger.info(f"[arrange] config snippet written to {self._output_path}")
