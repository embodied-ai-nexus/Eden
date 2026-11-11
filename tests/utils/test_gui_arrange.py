"""Tests for ``eden.utils.gui_arrange``: ``eden arrange`` exports the authored frame, not the solver's.

``ArrangeGUI.__init__`` needs a live viewer (the GUI boundary); applying and exporting do not, so the GUI
object is created without it.
"""

from __future__ import annotations


import genesis.utils.geom as gu
import numpy as np

from eden.options import EntityOptions, EnvOptions, PlaneOptions, SceneOptions
from eden.utils.configs import EdenConfig
from eden.utils.gui_arrange import ArrangeGUI, _build_entity_state
from eden.utils.misc import get_assets_dir
from tests.support.envs import build_env

# A mesh whose centre of mass is off its origin, so its solver and authored frames differ.
_HAND_STL = get_assets_dir("grippers", "franka_hand", "assets", "hand.stl")


def _prop_env(pos, quat):
    obj = EntityOptions(file=_HAND_STL, default_root_pos=tuple(pos), default_root_quat=tuple(quat), convexify=True)
    config = EdenConfig(env_options=EnvOptions(num_envs=1), scene_options=SceneOptions(plane=PlaneOptions(), obj=obj))
    return build_env(config).entities["obj"]


def test_a_prop_rebuilt_from_the_exported_pose_lands_where_it_was_arranged():
    pos, yaw90 = (
        (0.3, 0.2, 0.6),
        tuple(float(v) for v in gu.xyz_to_quat(np.array([0.0, 0.0, 90.0]), rpy=True, degrees=True)),
    )
    obj = _prop_env(pos, yaw90)
    state = _build_entity_state("obj", "EntityOptions", obj)
    np.testing.assert_allclose(state.pos, pos, atol=1e-4)  # the sliders start on the default pose

    gui = ArrangeGUI.__new__(ArrangeGUI)
    gui._states = [state]
    state.pos[:] = (0.5, -0.1, 0.4)
    state.rpy[:] = (20.0, 0.0, 30.0)
    gui._apply_to_sim(state)
    assert "default_root_pos=(0.5000, -0.1000, 0.4000)" in gui.config_snippet()
    arranged_pos, arranged_quat = obj.get_pos()[0].cpu().numpy(), obj.get_quat()[0].cpu().numpy()
    assert np.linalg.norm(arranged_pos - state.pos) > 1e-3  # the solver frame genuinely differs

    rebuilt = _prop_env(state.pos, gu.xyz_to_quat(state.rpy, rpy=True, degrees=True))
    np.testing.assert_allclose(rebuilt.get_pos()[0].cpu().numpy(), arranged_pos, atol=1e-4)
    np.testing.assert_allclose(
        gu.quat_to_R(rebuilt.get_quat()[0].cpu().numpy()), gu.quat_to_R(arranged_quat), atol=1e-4
    )
