"""Eden wrapper around GraspGen: diffusion-based 6-DoF grasp generation, one model per gripper.

GraspGen (`NVlabs/GraspGen <https://github.com/NVlabs/GraspGen>`_) is consumed through the packaged
``eden-grasp-gen`` (import name ``grasp_gen``), installed by the ``graspgen`` extra;
``grasp_gen download`` fetches its per-gripper models
into ``$GRASPGEN_CHECKPOINT_DIR``. For a gripper GraspGen has no model of, use
:class:`~eden.extensions.grasp_sampling.graspgenx.GraspGenXSampler`, whose single model takes the
gripper by name.

**Frames.** A GraspGen pose is its grasp frame (z approach, x closing), with its origin at the
gripper's base. That is no link of an Eden robot: the Franka's ``hand`` link closes its fingers
along y, so it is the grasp frame turned 90° about z, and a hand driven onto the bare grasp frame
closes across the wrong axis. Configure ``end_effector=`` (:attr:`GraspGenSampler.END_EFFECTORS`) for
poses of the Eden link instead.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
from genesis.utils.misc import tensor_to_array

from eden.extensions.grasp_sampling.base import GRASP_SAMPLER_REGISTRY, EndEffectorPairing, GraspSamplerBase
from eden.utils.distribution import install_hint

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.extensions.grasp_sampling import GraspSamplerOptions

_CHECKPOINT_DIR_ENV = "GRASPGEN_CHECKPOINT_DIR"

#: The Eden robots :attr:`GraspGenSampler.END_EFFECTORS` pairs a GraspGen model with (``end_effector=``).
GraspGenEndEffector = Literal["FrankaEmikaPanda", "FrankaHand"]

#: The Franka ``hand`` link in GraspGen's grasp frame: a quarter turn about z (wxyz).
_FRANKA_HAND_IN_GRASP = ((0.0, 0.0, 0.0), (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)))

_INSTALL_HINT = (
    "GraspGen (`grasp_gen`, shipped by `eden-grasp-gen`) is not installed. Install it with "
    f"{install_hint('graspgen')}. Then run `grasp_gen download` and export GRASPGEN_CHECKPOINT_DIR as it prints. "
    "See docs/guides/grasp-sampling.md."
)


def _require_grasp_gen():
    """Import GraspGen's entry points, turning a missing install into an actionable error."""
    try:
        from grasp_gen.grasp_server import GraspGenSampler as Model
        from grasp_gen.grasp_server import load_grasp_cfg
        from grasp_gen.robot import get_gripper_info
        from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps
    except ImportError as exc:
        raise ImportError(f"{_INSTALL_HINT}\nImport failed with: {exc}") from exc
    return Model, load_grasp_cfg, get_gripper_info, filter_colliding_grasps


@GRASP_SAMPLER_REGISTRY.register()
class GraspGenSampler(GraspSamplerBase):
    """Sample grasps with GraspGen, for a gripper it ships a model of.

    The shared options are documented on
    :class:`~eden.options.extensions.grasp_sampling.GraspSamplerOptions`. ``grasp_gen download``
    ships models of ``franka_panda``, ``robotiq_2f_140`` and ``suction_cup_30mm``. Each pose is that
    model's grasp frame, or with ``end_effector`` the Eden robot's link (:attr:`pose_link_name`); see
    the module docstring. Unlike GraspGenX's, GraspGen's grippers carry collision meshes, so
    ``filter_collisions`` works.

    Parameters
    ----------
    gripper_config : str
        The model's gripper config, e.g. ``"graspgen_franka_panda.yml"``: an existing path, or a path
        relative to ``$GRASPGEN_CHECKPOINT_DIR``. Required unless ``end_effector`` picks it.
    end_effector : GraspGenEndEffector | None
        The Eden robot that executes the grasps (see
        :class:`~eden.options.extensions.grasp_sampling.GraspSamplerOptions`). It picks
        ``gripper_config``, and each pose then places that robot's link: one of :attr:`END_EFFECTORS`.
    """

    gripper_config: str = ""
    end_effector: GraspGenEndEffector | None = None

    #: Checked against the Eden Franka by ``tests/extensions/grasp_sampling/test_end_effectors.py`` (its
    #: pads straddle the TCP along the closing axis) and in simulation on YCB objects: placed at the bare
    #: grasp frame, the hand collides with the target on every one of the top 10 grasps of a cracker box
    #: and a mustard bottle; turned the quarter turn, 7 and 9 of them clear and the best pinches.
    END_EFFECTORS: ClassVar[dict[str, EndEffectorPairing]] = {
        "FrankaEmikaPanda": EndEffectorPairing("graspgen_franka_panda.yml", "hand", _FRANKA_HAND_IN_GRASP),
        "FrankaHand": EndEffectorPairing("graspgen_franka_panda.yml", "hand", _FRANKA_HAND_IN_GRASP),
    }
    _gripper_option: ClassVar[str] = "gripper_config"

    def __init__(self, env: EnvBase, options: GraspSamplerOptions) -> None:
        super().__init__(env, options)
        Model, load_grasp_cfg, get_gripper_info, self._filter_colliding_grasps = _require_grasp_gen()

        grasp_cfg = load_grasp_cfg(str(self._resolve_gripper_config()))
        self._model = Model(grasp_cfg)
        gripper = get_gripper_info(grasp_cfg.data.gripper_name)
        self._collision_mesh = gripper.collision_mesh
        # The grasp frame is GraspGen's base-link frame, so its tool TCP is the grasp's TCP.
        tcp = np.asarray(gripper.transform_from_base_link_to_tool_tcp, dtype=np.float64)[:3, 3]
        self._set_pose_frame(None, np.eye(4), tcp)

    def _resolve_gripper_config(self) -> Path:
        if not self.gripper_config:
            raise ValueError(
                "GraspGenSampler needs gripper_config=, the model's gripper config — e.g. "
                "gripper_config='graspgen_franka_panda.yml', a path relative to $GRASPGEN_CHECKPOINT_DIR — or "
                f"end_effector=, the Eden robot that executes the grasps, which picks it: one of "
                f"{sorted(self.END_EFFECTORS)}."
            )
        path = Path(self.gripper_config).expanduser()
        if path.is_file():
            return path
        checkpoint_dir = os.environ.get(_CHECKPOINT_DIR_ENV)
        if not checkpoint_dir:
            raise FileNotFoundError(
                f"gripper_config='{self.gripper_config}' is not a file, and ${_CHECKPOINT_DIR_ENV} — which a "
                f"relative config resolves against — is not set. Fix: run `grasp_gen download` and export "
                f"{_CHECKPOINT_DIR_ENV}=<the directory it downloaded to>, or pass an absolute path."
            )
        path = Path(checkpoint_dir) / self.gripper_config
        if not path.is_file():
            available = sorted(p.name for p in Path(checkpoint_dir).glob("*.yml"))
            raise FileNotFoundError(
                f"gripper_config='{self.gripper_config}' is not under ${_CHECKPOINT_DIR_ENV}={checkpoint_dir}, "
                f"which has {available}."
            )
        return path

    def _infer(self, object_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        grasps, confidences = self._model.run_inference(
            object_points,
            self._model,
            grasp_threshold=self.grasp_threshold,
            num_grasps=self.num_grasps,
            topk_num_grasps=-1,  # the base class applies max_grasps, after any filtering
        )
        if len(grasps) == 0:
            return np.zeros((0, 4, 4)), np.zeros(0, dtype=np.float32)
        grasps = np.array(tensor_to_array(grasps), dtype=np.float64).reshape(-1, 4, 4)
        grasps[:, 3, 3] = 1.0  # GraspGen leaves the homogeneous corner unset
        return grasps, np.asarray(tensor_to_array(confidences), dtype=np.float32).reshape(-1)

    def _collision_free(self, grasps: np.ndarray, scene_points: np.ndarray) -> np.ndarray:
        return np.asarray(
            self._filter_colliding_grasps(
                scene_pc=scene_points,
                grasp_poses=grasps,
                gripper_collision_mesh=self._collision_mesh,
                collision_threshold=self.collision_threshold,
            ),
            dtype=bool,
        )
