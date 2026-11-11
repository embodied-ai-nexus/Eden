"""Tests for ``eden.extensions.retargeting.soma_retargeter.retargeter.SomaRetargeter`` against a real scene.

The library runs in a worker subprocess; here that worker is real and imports the fake installation from
``conftest.py``, which records what reached it. The DOF permutation is what keeps qpos independent of
Genesis' internal DOF order (BFS -> DFS in genesis 1.2.0): the library labels DOFs by name.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eden.extensions.retargeting.soma_retargeter import SomaRetargeter
from eden.extensions.retargeting.soma_retargeter.retargeter import build_dof_permutation
from eden.options import BoxOptions, EnvOptions, PlaneOptions, SceneOptions
from eden.options.robots.franka_emika_panda import FrankaEmikaPanda
from eden.options.robots.limx_oli import OLI_DOFS_NAME
from eden.options.robots.unitree_g1 import G1Dofs29
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

NUM_FRAMES = 3


@pytest.fixture(scope="module")
def env():
    """A free-base G1 to retarget onto, a fixed-base Panda, a cube welded to its hand, and a free primitive box."""
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(
                plane=PlaneOptions(),
                robot=G1Dofs29(),
                arm=FrankaEmikaPanda(default_root_pos=(2.0, 0.0, 0.0)),
                cube=BoxOptions(size=(0.05, 0.05, 0.05), default_root_pos=(2.0, 0.0, 1.0)),
                box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(-2.0, 0.0, 0.05)),
                attachments_dict={"cube": ("arm", "hand")},
            ),
        ),
        reset=True,
    )


def _retargeter(env, **options):
    return SomaRetargeter(env=env, options=SomaRetargeter.configure(entity_name="robot", device="cpu", **options))


def _clip(env, root_z=0.8):
    """What the fake library echoes back: the robot's DOFs in reverse order, with the CSV ``_dof`` suffix."""
    names = [f"{name}_dof" for name in reversed(env.entities["robot"].dofs_name)]
    root_pos = np.zeros((NUM_FRAMES, 3), np.float32)
    root_pos[:, 2] = root_z + 0.1 * np.arange(NUM_FRAMES)
    return {
        "source_fps": 30.0,
        "joint_names": names,
        "dof_pos": np.tile(np.arange(len(names), dtype=np.float32) * 0.01, (NUM_FRAMES, 1)),
        "root_pos": root_pos,
        "root_quat": np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (NUM_FRAMES, 1)),
    }


class TestBuildDofPermutation:
    @staticmethod
    def _apply(perm, values):
        out = np.empty_like(values)
        out[:, perm] = values
        return out

    @pytest.mark.parametrize(
        ("source", "entity"),
        [
            (["c", "a", "d", "b"], ["a", "b", "c", "d"]),
            # The library's URDF order puts Oli's head last; OLI_DOFS_NAME has it mid-list.
            ([n for n in OLI_DOFS_NAME if "head" not in n] + [n for n in OLI_DOFS_NAME if "head" in n], OLI_DOFS_NAME),
        ],
        ids=["letters", "oli"],
    )
    def test_each_source_value_lands_in_its_named_entity_slot(self, source, entity):
        values = np.arange(len(source), dtype=np.float32)[None]
        out = self._apply(build_dof_permutation(source, entity), values)
        assert [out[0, list(entity).index(name)] for name in source] == values[0].tolist()

    def test_the_csv_dof_suffix_is_stripped(self):
        perm = build_dof_permutation(
            ["waist_yaw_joint_dof", "left_hip_pitch_joint_dof"], ["left_hip_pitch_joint", "waist_yaw_joint"]
        )
        assert perm.tolist() == [1, 0]

    @pytest.mark.parametrize(
        ("source", "entity", "message"),
        [(["a", "b"], ["a", "b", "c"], "emits 2 DOFs but entity has 3"), (["a", "x"], ["a", "b"], "not in entity")],
    )
    def test_a_mismatch_is_refused_listing_both_sides(self, source, entity, message):
        with pytest.raises(ValueError, match=message):
            build_dof_permutation(source, entity)


class TestConstruction:
    @pytest.mark.parametrize(
        ("entity", "reason"), [("arm", "is fixed-base"), ("cube", "is welded onto another entity")]
    )
    def test_an_entity_without_base_dofs_is_refused(self, env, entity, reason):
        """Retargeted qpos carries [base_pos, base_quat, joints]; a welded child is not fixed-base, only immobile."""
        with pytest.raises(ValueError, match=reason):
            SomaRetargeter(env=env, options=SomaRetargeter.configure(entity_name=entity))

    def test_the_backend_is_vetted_for_the_configured_body_model_before_a_worker_starts(self, env, soma_install):
        """Examples construct the retargeter directly, so the CLI's check is not enough.

        Had the worker started first, the failure would be its init error, not this ImportError.
        """
        recorded = soma_install(py_soma_x=None)
        with pytest.raises(ImportError, match="body_model='smplh'"):
            _retargeter(env, body_model="smplh")
        assert recorded() == []

    def test_a_procedural_flag_in_retargeter_kwargs_is_refused(self, env, soma_install):
        """It would win over the option silently, since ``retargeter_kwargs`` splats last."""
        recorded = soma_install()
        with pytest.raises(ValueError, match="SomaRetargeter option"):
            _retargeter(env, retargeter_kwargs={"enable_procedural_transforms": True})
        assert recorded() == []


@pytest.fixture(scope="module")
def default_worker(env, soma_installed):
    """One retargeter on the default options, plus ``fit_chunk``; yields it and the fake's record."""
    retargeter = _retargeter(env, retargeter_kwargs={"fit_chunk": 128})
    yield retargeter, soma_installed
    retargeter.close()


@pytest.fixture(scope="module")
def snapping_worker(env, soma_installed):
    retargeter = _retargeter(env, snap_to_ground=True)
    yield retargeter
    retargeter.close()


class TestWorker:
    def test_the_init_message_opts_out_of_procedural_transforms_and_forwards_the_rest(self, default_worker):
        """The library defaults the flag on, which the public SOMA-X assets cannot build."""
        _, recorded = default_worker
        init = next(entry["init"] for entry in recorded() if "init" in entry and "fit_chunk" in entry["init"])
        assert init["enable_procedural_transforms"] is False
        assert (init["body_model"], init["robot"], init["fit_chunk"]) == ("smplx", "unitree_g1", 128)
        assert "robot_model_path" not in init  # the library bundles the G1

    def test_qpos_comes_back_in_the_entitys_dof_order_through_a_default_soma_layer(self, env, default_worker):
        retargeter, recorded = default_worker
        clip = _clip(env)
        qpos = retargeter.retarget_motion(**clip)

        assert qpos.shape == (NUM_FRAMES, 7 + len(env.entities["robot"].dofs_name))
        np.testing.assert_array_equal(qpos[:, :3], clip["root_pos"])
        np.testing.assert_array_equal(qpos[0, 7:], clip["dof_pos"][0, ::-1])  # the reversed names, undone
        assert recorded()[-1] == {"layer_procedural": False}

    def test_opting_in_to_procedural_transforms_reaches_the_soma_layer(self, env, soma_install):
        """Untouched, so the library raises its own FileNotFoundError for the unpublished rig definition."""
        recorded = soma_install()
        retargeter = _retargeter(env, enable_procedural_transforms=True)
        retargeter.retarget_motion(**_clip(env))
        retargeter.close()

        assert recorded()[-1] == {"layer_procedural": True}

    def test_a_robot_without_a_bundled_model_is_given_the_entitys_own_file(self, env, soma_install):
        """Native SOMA input forwards the procedural flag too: the library accepts it and only a fit reads it."""
        recorded = soma_install()
        _retargeter(env, robot="limx_oli", body_model="soma").close()

        init = recorded()[0]["init"]
        assert init["robot_model_path"] == env.entities["robot"].morph.file
        assert (init["body_model"], init["enable_procedural_transforms"]) == ("soma", False)

    def test_an_entity_without_a_model_file_is_refused_for_such_a_robot(self, env, soma_install):
        recorded = soma_install()
        with pytest.raises(ValueError, match="file-based morph"):
            SomaRetargeter(env=env, options=SomaRetargeter.configure(entity_name="box", robot="limx_oli", device="cpu"))
        assert recorded() == []


@pytest.mark.parametrize("root_z", [0.3, 1.5], ids=["sunk", "floating"])
def test_snap_to_ground_puts_the_lowest_collision_point_on_the_floor(env, snapping_worker, root_z):
    """One constant vertical offset per clip; the entity is left as it was found."""
    robot = env.entities["robot"]
    before = robot.get_qpos().clone()
    clip = _clip(env, root_z=root_z)
    qpos = snapping_worker.retarget_motion(**clip)

    assert torch.equal(robot.get_qpos(), before)
    lowest = []
    for frame in torch.as_tensor(qpos, device=robot.device):
        robot.set_qpos(frame)
        lowest.append(robot.entity.get_verts()[..., 2].min().item())
    robot.set_qpos(before)
    assert min(lowest) == pytest.approx(0.0, abs=1e-5)
    shift = qpos[:, 2] - clip["root_pos"][:, 2]
    np.testing.assert_allclose(shift, shift[0], atol=1e-6)  # the whole clip moved together
