"""Custom action term replicating MuJoCo's ``general`` actuator with affine bias.

The HumEnv SMPL humanoid uses MuJoCo ``general`` actuators with
``biastype="affine"`` and ``ctrlrange=[-1, 1]``.  The force mapping is::

    force = clamp(gain * ctrl + bias[0] + bias[1] * q + bias[2] * qdot, forcerange)

The per-joint gain, bias, and forcerange parameters for the SMPL humanoid
are stored as class variables on this action term.

.. note::
    This term requires the Genesis SMPL humanoid to be loaded from the
    **original HumEnv Y-up MuJoCo XML** with ``up=(0,1,0)`` so that
    DOF axes match HumEnv exactly (no axis permutation needed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import torch

from eden.managers.action_manager import ACTION_TERM_REGISTRY, ActionTerm

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.entities.rigid import RigidEntity
    from eden.options.managers.actions import ActionTermOptions


@ACTION_TERM_REGISTRY.register()
class MujocoSMPLActuator(ActionTerm):
    """Emulates MuJoCo ``general`` actuators with ``biastype="affine"`` in Genesis.

    Actions are expected in ``[-1, 1]`` (matching MuJoCo's ``ctrlrange``).
    The resulting joint torques are clamped to ``forcerange`` and applied
    via ``control_dofs_force``.

    Parameters
    ----------
    entity_name : str
        Name of the SMPL humanoid entity.
    dofs_name : list[str]
        DOF names matching keys in the actuator gain/bias dicts.
    """

    entity_name: str = ""
    dofs_name: str | list[str] = []

    # fmt: off
    ACTUATOR_GAIN: ClassVar[dict[str, float]] = {
        "L_Hip_x": 287.330064, "L_Hip_y": 219.911486, "L_Hip_z": 186.374984,
        "L_Knee_x": 278.10949, "L_Knee_y": 18.001326, "L_Knee_z": 18.001326,
        "L_Ankle_x": 72.005304, "L_Ankle_y": 76.497781, "L_Ankle_z": 7.194247,
        "L_Toe_x": 71.99902, "L_Toe_y": 5.755398, "L_Toe_z": 5.755398,
        "R_Hip_x": 287.330064, "R_Hip_y": 219.911486, "R_Hip_z": 186.374984,
        "R_Knee_x": 278.10949, "R_Knee_y": 18.001326, "R_Knee_z": 18.001326,
        "R_Ankle_x": 72.005304, "R_Ankle_y": 76.497781, "R_Ankle_z": 7.194247,
        "R_Toe_x": 71.99902, "R_Toe_y": 5.755398, "R_Toe_z": 5.755398,
        "Torso_x": 174.002345, "Torso_y": 240.017679, "Torso_z": 144.010607,
        "Spine_x": 174.002345, "Spine_y": 240.017679, "Spine_z": 144.010607,
        "Chest_x": 174.002345, "Chest_y": 240.017679, "Chest_z": 144.010607,
        "Neck_x": 27.001989, "Neck_y": 25.201856, "Neck_z": 25.201856,
        "Head_x": 18.001326, "Head_y": 16.801238, "Head_z": 16.801238,
        "L_Thorax_x": 16.200146, "L_Thorax_y": 48.003536, "L_Thorax_z": 75.000289,
        "L_Shoulder_x": 81.597633, "L_Shoulder_y": 79.197456, "L_Shoulder_z": 76.79728,
        "L_Elbow_x": 13.678494, "L_Elbow_y": 123.898131, "L_Elbow_z": 66.23734,
        "L_Wrist_x": 21.249733, "L_Wrist_y": 3.600265, "L_Wrist_z": 4.800354,
        "L_Hand_x": 3.600265, "L_Hand_y": 10.800796, "L_Hand_z": 19.19932,
        "R_Thorax_x": 16.200146, "R_Thorax_y": 48.003536, "R_Thorax_z": 75.000289,
        "R_Shoulder_x": 81.597633, "R_Shoulder_y": 79.197456, "R_Shoulder_z": 76.79728,
        "R_Elbow_x": 13.678494, "R_Elbow_y": 123.898131, "R_Elbow_z": 66.23734,
        "R_Wrist_x": 21.249733, "R_Wrist_y": 3.600265, "R_Wrist_z": 4.800354,
        "R_Hand_x": 3.600265, "R_Hand_y": 10.800796, "R_Hand_z": 19.19932,
    }
    ACTUATOR_BIAS: ClassVar[dict[str, list[float]]] = {
        "L_Hip_x": [-215.32476, -180.0, -2.0], "L_Hip_y": [0.0, -180.0, -2.0], "L_Hip_z": [96.368355, -180.0, -2.0],
        "L_Knee_x": [261.898872, -180.0, -2.0], "L_Knee_y": [3.612832, -180.0, -2.0], "L_Knee_z": [-3.612832, -180.0, -2.0],
        "L_Ankle_x": [0.0, -90.0, -2.0], "L_Ankle_y": [-40.495129, -90.0, -2.0], "L_Ankle_z": [0.0, -90.0, -2.0],
        "L_Toe_x": [-43.196899, -72.0, -0.8], "L_Toe_y": [0.0, -72.0, -0.8], "L_Toe_z": [0.0, -72.0, -0.8],
        "R_Hip_x": [-215.32476, -180.0, -2.0], "R_Hip_y": [0.0, -180.0, -2.0], "R_Hip_z": [-96.368355, -180.0, -2.0],
        "R_Knee_x": [261.898872, -180.0, -2.0], "R_Knee_y": [-3.612832, -180.0, -2.0], "R_Knee_z": [3.612832, -180.0, -2.0],
        "R_Ankle_x": [0.0, -90.0, -2.0], "R_Ankle_y": [40.495129, -90.0, -2.0], "R_Ankle_z": [0.0, -90.0, -2.0],
        "R_Toe_x": [-43.196899, -72.0, -0.8], "R_Toe_y": [0.0, -72.0, -0.8], "R_Toe_z": [0.0, -72.0, -0.8],
        "Torso_x": [66.015334, -240.0, -4.0], "Torso_y": [0.0, -240.0, -4.0], "Torso_z": [0.0, -240.0, -4.0],
        "Spine_x": [66.015334, -240.0, -4.0], "Spine_y": [0.0, -240.0, -4.0], "Spine_z": [0.0, -240.0, -4.0],
        "Chest_x": [66.015334, -240.0, -4.0], "Chest_y": [0.0, -240.0, -4.0], "Chest_z": [0.0, -240.0, -4.0],
        "Neck_x": [9.000663, -36.0, -0.8], "Neck_y": [0.0, -36.0, -0.8], "Neck_z": [0.0, -36.0, -0.8],
        "Head_x": [6.000442, -24.0, -0.8], "Head_y": [0.0, -24.0, -0.8], "Head_z": [0.0, -24.0, -0.8],
        "L_Thorax_x": [-6.607817, -120.0, -1.6], "L_Thorax_y": [-12.000884, -120.0, -1.6], "L_Thorax_z": [9.005899, -120.0, -1.6],
        "L_Shoulder_x": [0.0, -48.0, -1.6], "L_Shoulder_y": [-12.000884, -48.0, -1.6], "L_Shoulder_z": [-4.800354, -48.0, -1.6],
        "L_Elbow_x": [-7.923097, -72.0, -0.8], "L_Elbow_y": [-102.29654, -72.0, -0.8], "L_Elbow_z": [-60.481942, -72.0, -0.8],
        "L_Wrist_x": [-16.449379, -12.0, -0.8], "L_Wrist_y": [0.0, -12.0, -0.8], "L_Wrist_z": [0.0, -12.0, -0.8],
        "L_Hand_x": [0.0, -12.0, -0.8], "L_Hand_y": [0.0, -12.0, -0.8], "L_Hand_z": [0.0, -12.0, -0.8],
        "R_Thorax_x": [-6.607817, -120.0, -1.6], "R_Thorax_y": [12.000884, -120.0, -1.6], "R_Thorax_z": [-9.005899, -120.0, -1.6],
        "R_Shoulder_x": [0.0, -48.0, -1.6], "R_Shoulder_y": [12.000884, -48.0, -1.6], "R_Shoulder_z": [4.800354, -48.0, -1.6],
        "R_Elbow_x": [-7.923097, -72.0, -0.8], "R_Elbow_y": [102.29654, -72.0, -0.8], "R_Elbow_z": [60.481942, -72.0, -0.8],
        "R_Wrist_x": [-16.449379, -12.0, -0.8], "R_Wrist_y": [0.0, -12.0, -0.8], "R_Wrist_z": [0.0, -12.0, -0.8],
        "R_Hand_x": [0.0, -12.0, -0.8], "R_Hand_y": [0.0, -12.0, -0.8], "R_Hand_z": [0.0, -12.0, -0.8],
    }
    ACTUATOR_FORCERANGE: ClassVar[dict[str, tuple[float, float]]] = {
        "L_Hip_x": (-360, 360), "L_Hip_y": (-360, 360), "L_Hip_z": (-360, 360),
        "L_Knee_x": (-360, 360), "L_Knee_y": (-360, 360), "L_Knee_z": (-360, 360),
        "L_Ankle_x": (-180, 180), "L_Ankle_y": (-180, 180), "L_Ankle_z": (-180, 180),
        "L_Toe_x": (-144, 144), "L_Toe_y": (-144, 144), "L_Toe_z": (-144, 144),
        "R_Hip_x": (-360, 360), "R_Hip_y": (-360, 360), "R_Hip_z": (-360, 360),
        "R_Knee_x": (-360, 360), "R_Knee_y": (-360, 360), "R_Knee_z": (-360, 360),
        "R_Ankle_x": (-180, 180), "R_Ankle_y": (-180, 180), "R_Ankle_z": (-180, 180),
        "R_Toe_x": (-144, 144), "R_Toe_y": (-144, 144), "R_Toe_z": (-144, 144),
        "Torso_x": (-360, 360), "Torso_y": (-360, 360), "Torso_z": (-360, 360),
        "Spine_x": (-360, 360), "Spine_y": (-360, 360), "Spine_z": (-360, 360),
        "Chest_x": (-360, 360), "Chest_y": (-360, 360), "Chest_z": (-360, 360),
        "Neck_x": (-72, 72), "Neck_y": (-72, 72), "Neck_z": (-72, 72),
        "Head_x": (-48, 48), "Head_y": (-48, 48), "Head_z": (-48, 48),
        "L_Thorax_x": (-240, 240), "L_Thorax_y": (-240, 240), "L_Thorax_z": (-240, 240),
        "L_Shoulder_x": (-96, 96), "L_Shoulder_y": (-96, 96), "L_Shoulder_z": (-96, 96),
        "L_Elbow_x": (-144, 144), "L_Elbow_y": (-144, 144), "L_Elbow_z": (-144, 144),
        "L_Wrist_x": (-24, 24), "L_Wrist_y": (-24, 24), "L_Wrist_z": (-24, 24),
        "L_Hand_x": (-24, 24), "L_Hand_y": (-24, 24), "L_Hand_z": (-24, 24),
        "R_Thorax_x": (-240, 240), "R_Thorax_y": (-240, 240), "R_Thorax_z": (-240, 240),
        "R_Shoulder_x": (-96, 96), "R_Shoulder_y": (-96, 96), "R_Shoulder_z": (-96, 96),
        "R_Elbow_x": (-144, 144), "R_Elbow_y": (-144, 144), "R_Elbow_z": (-144, 144),
        "R_Wrist_x": (-24, 24), "R_Wrist_y": (-24, 24), "R_Wrist_z": (-24, 24),
        "R_Hand_x": (-24, 24), "R_Hand_y": (-24, 24), "R_Hand_z": (-24, 24),
    }
    # fmt: on

    if TYPE_CHECKING:
        entity: RigidEntity

    def __init__(self, env: EnvBase, options: ActionTermOptions):
        super().__init__(env=env, options=options)
        self._gain: torch.Tensor | None = None
        self._bias: torch.Tensor | None = None
        self._force_lo: torch.Tensor | None = None
        self._force_hi: torch.Tensor | None = None

    def build(self) -> None:
        super().build()
        self._build_actuator_params()

    def _build_actuator_params(self) -> None:
        """Build gain, bias, and forcerange tensors from class variables."""
        gains = [self.ACTUATOR_GAIN[name] for name in self.dofs_name]
        biases = [self.ACTUATOR_BIAS[name] for name in self.dofs_name]
        lo = [self.ACTUATOR_FORCERANGE[name][0] for name in self.dofs_name]
        hi = [self.ACTUATOR_FORCERANGE[name][1] for name in self.dofs_name]

        self._gain = torch.tensor(gains, dtype=torch.float32, device=self.device).unsqueeze(0)
        self._bias = torch.tensor(biases, dtype=torch.float32, device=self.device).unsqueeze(0)
        self._force_lo = torch.tensor(lo, dtype=torch.float32, device=self.device).unsqueeze(0)
        self._force_hi = torch.tensor(hi, dtype=torch.float32, device=self.device).unsqueeze(0)

    def compute(self, actions: torch.Tensor) -> None:
        """Clamp actions to [-1, 1] and store."""
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions
        self._processed_action[:] = torch.clamp(actions, -1.0, 1.0)

    def apply_actions(self) -> None:
        """Compute and apply MuJoCo-style affine actuator torques."""
        ctrl = self._processed_action  # (num_envs, n_dofs), values in [-1, 1]

        q = self.entity.get_dofs_pos(dofs_idx_local=self.genesis_dofs_idx)
        qdot = self.entity.get_dofs_vel(dofs_idx_local=self.genesis_dofs_idx)

        force = self._gain * ctrl + self._bias[:, :, 0] + self._bias[:, :, 1] * q + self._bias[:, :, 2] * qdot

        # Clamp to forcerange (matching MuJoCo's actuator force limits)
        force = torch.clamp(force, self._force_lo, self._force_hi)

        self.entity.control_dofs_force(force, dofs_idx_local=self.genesis_dofs_idx)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._prev_action[envs_idx] = 0.0
        self._raw_action[envs_idx] = 0.0
