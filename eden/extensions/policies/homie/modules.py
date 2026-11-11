"""HOMIE frozen-tracker network modules and observation manager for the G1.

Ported from holosoma (``holosoma/modules/homie.py``), which in turn ports the
released OpenHomie walk/stand tracker. HOMIE is a velocity- + height-command
driven lower-body locomotion policy: given a 4-D command ``[vel_x, vel_y,
ang_vel_z, height]`` and proprioception it emits 15 lower-body joint actions
(12 leg + 3 waist). The upper body (14 arm DOFs) is controlled separately and
held at the HOMIE default pose during pure-locomotion inference.

Architecture
------------
``HIMEstimator``
    History encoder → velocity estimate (3) + dynamic latent (32).
``HIMActorCritic``
    Estimator + actor MLP → 15 lower-body actions (critic kept only for
    checkpoint compatibility).
``HomieActorModule``
    Inference-only wrapper (no critic, no gradients).
``HomieTracker``
    Manages observation construction, the 6-step history buffer, walk/stand
    model switching, and the action → joint-target transform.

The network code is reproduced verbatim from the OpenHomie release so released
``model_walk.pt`` / ``model_stand.pt`` checkpoints load unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenHomie default joint positions (rad) — 29 DOFs in HOMIE joint order, which
# matches Eden's ``G1Dofs29.dofs_name`` order exactly (legs 12 → waist 3 →
# arms 14). These MUST match the training environment to produce correct
# observations (``dof_pos - default_pos``) and action targets
# (``default_pos + action * scale``).
# Source: OpenHomie/HomieRL/legged_gym/legged_gym/envs/g1/g1_29dof_config.py
# ---------------------------------------------------------------------------
HOMIE_DEFAULT_DOF_POS: list[float] = [
    # Lower body (12 DOFs) — indices 0-11
    -0.1,  # left_hip_pitch_joint
    0.0,  # left_hip_roll_joint
    0.0,  # left_hip_yaw_joint
    0.3,  # left_knee_joint
    -0.2,  # left_ankle_pitch_joint
    0.0,  # left_ankle_roll_joint
    -0.1,  # right_hip_pitch_joint
    0.0,  # right_hip_roll_joint
    0.0,  # right_hip_yaw_joint
    0.3,  # right_knee_joint
    -0.2,  # right_ankle_pitch_joint
    0.0,  # right_ankle_roll_joint
    # Waist (3 DOFs) — indices 12-14
    0.0,  # waist_yaw_joint
    0.0,  # waist_roll_joint
    0.0,  # waist_pitch_joint
    # Arms (14 DOFs) — indices 15-28
    0.0,  # left_shoulder_pitch_joint
    0.0,  # left_shoulder_roll_joint
    0.0,  # left_shoulder_yaw_joint
    0.0,  # left_elbow_joint
    0.0,  # left_wrist_roll_joint
    0.0,  # left_wrist_pitch_joint
    0.0,  # left_wrist_yaw_joint
    0.0,  # right_shoulder_pitch_joint
    0.0,  # right_shoulder_roll_joint
    0.0,  # right_shoulder_yaw_joint
    0.0,  # right_elbow_joint
    0.0,  # right_wrist_roll_joint
    0.0,  # right_wrist_pitch_joint
    0.0,  # right_wrist_yaw_joint
]

# OpenHomie PD gains (Kp, Kd) per DOF index — lower body + waist.
# Source: g1_29dof_config.py lines 88-111 + legged_robot.py lines 901-904
HOMIE_PD_GAINS: dict[int, tuple[float, float]] = {
    # Left leg
    0: (100.0, 2.0),  # left_hip_pitch
    1: (100.0, 2.0),  # left_hip_roll
    2: (100.0, 2.0),  # left_hip_yaw
    3: (150.0, 4.0),  # left_knee
    4: (40.0, 2.0),  # left_ankle_pitch
    5: (40.0, 2.0),  # left_ankle_roll
    # Right leg
    6: (100.0, 2.0),  # right_hip_pitch
    7: (100.0, 2.0),  # right_hip_roll
    8: (100.0, 2.0),  # right_hip_yaw
    9: (150.0, 4.0),  # right_knee
    10: (40.0, 2.0),  # right_ankle_pitch
    11: (40.0, 2.0),  # right_ankle_roll
    # Waist
    12: (300.0, 5.0),  # waist_yaw
    13: (200.0, 4.0),  # waist_roll
    14: (200.0, 4.0),  # waist_pitch
}

# OpenHomie upper-body PD gains (arms). Source: g1_29dof_config.py lines 901-904
HOMIE_UPPER_BODY_PD_GAINS: dict[int, tuple[float, float]] = {
    15: (200.0, 4.0),  # left_shoulder_pitch
    16: (200.0, 4.0),  # left_shoulder_roll
    17: (200.0, 4.0),  # left_shoulder_yaw
    18: (100.0, 1.0),  # left_elbow
    19: (20.0, 0.5),  # left_wrist_roll
    20: (20.0, 0.5),  # left_wrist_pitch
    21: (20.0, 0.5),  # left_wrist_yaw
    22: (200.0, 4.0),  # right_shoulder_pitch
    23: (200.0, 4.0),  # right_shoulder_roll
    24: (200.0, 4.0),  # right_shoulder_yaw
    25: (100.0, 1.0),  # right_elbow
    26: (20.0, 0.5),  # right_wrist_roll
    27: (20.0, 0.5),  # right_wrist_pitch
    28: (20.0, 0.5),  # right_wrist_yaw
}

# Combined 29-DOF (Kp, Kd) table in HOMIE / G1Dofs29 joint order.
HOMIE_ALL_PD_GAINS: dict[int, tuple[float, float]] = {**HOMIE_PD_GAINS, **HOMIE_UPPER_BODY_PD_GAINS}

# Number of DOFs HOMIE controls (legs 12 + waist 3).
HOMIE_NUM_ACTIONS: int = 15

# Number of DOFs observed by HOMIE (all 29 body DOFs).
HOMIE_DOF_OBS_SIZE: int = 29

# Observation dimensions.
HOMIE_ONE_STEP_OBS: int = 86  # 7 cmd + 3 ang_vel + 3 gravity + 29 dof_pos + 29 dof_vel + 15 prev_actions
HOMIE_HISTORY_LENGTH: int = 6
HOMIE_TOTAL_OBS: int = HOMIE_ONE_STEP_OBS * HOMIE_HISTORY_LENGTH  # 516

# Observation scales (applied during obs construction, baked into the model).
OBS_SCALE_LIN_VEL: float = 2.0
OBS_SCALE_ANG_VEL_CMD: float = 0.5
OBS_SCALE_BASE_ANG_VEL: float = 0.5
OBS_SCALE_DOF_VEL: float = 0.05

# Default base height for the HOMIE height command (absolute, metres).
HOMIE_DEFAULT_HEIGHT: float = 0.74

# HOMIE action scale (raw action → joint-position delta).
HOMIE_ACTION_SCALE: float = 0.25


def _get_activation(act_name: str) -> nn.Module:
    """Return an instantiated activation module by name.

    Parameters
    ----------
    act_name : str
        One of ``elu``, ``relu``, ``selu``, ``tanh``, ``sigmoid``, ``lrelu``.

    Returns
    -------
    torch.nn.Module
        The instantiated activation layer.
    """
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "lrelu": nn.LeakyReLU,
    }
    cls = activations.get(act_name)
    if cls is None:
        raise ValueError(f"Unknown activation: {act_name}")
    return cls()


# ====================================================================
# Network modules (ported from OpenHomie / GR00T homie_modules.py)
# ====================================================================


class HIMEstimator(nn.Module):
    """History-based Implicit Model estimator.

    Takes flattened observation history (``temporal_steps * num_one_step_obs``)
    and produces a velocity estimate (3) plus a normalized dynamic latent
    (``latent_dim``).
    """

    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        num_height_points: int = 0,
        enc_hidden_dims: list[int] | None = None,
        latent_dim: int = 32,
        activation: str = "elu",
        **kwargs,
    ):
        """Build the encoder MLP.

        Parameters
        ----------
        temporal_steps : int
            Number of history steps fed to the encoder.
        num_one_step_obs : int
            Per-step observation dimension.
        num_height_points : int, optional
            Number of terrain height-scan points appended to the input.
        enc_hidden_dims : list[int] | None, optional
            Hidden-layer widths; defaults to ``[256, 256]``.
        latent_dim : int, optional
            Dynamic latent dimension.
        activation : str, optional
            Activation name passed to :func:`_get_activation`.
        **kwargs : Any
            Ignored; absorbs extra config keys from released checkpoints.
        """
        super().__init__()
        if enc_hidden_dims is None:
            enc_hidden_dims = [256, 256]

        act = _get_activation(activation)
        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_latent = latent_dim

        enc_input_dim = temporal_steps * num_one_step_obs + num_height_points
        layers: list[nn.Module] = []
        for hidden_dim in enc_hidden_dims:
            layers += [nn.Linear(enc_input_dim, hidden_dim), act]
            enc_input_dim = hidden_dim
        layers += [nn.Linear(enc_input_dim, 3 + latent_dim)]
        self.encoder = nn.Sequential(*layers)

    def forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an observation history into ``(velocity, dynamic_latent)``.

        Parameters
        ----------
        obs_history : torch.Tensor
            Flattened history of shape ``(..., temporal_steps * num_one_step_obs)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Detached velocity estimate ``(..., 3)`` and L2-normalized dynamic
            latent ``(..., latent_dim)``.
        """
        parts = self.encoder(obs_history.detach())
        vel = parts[..., :3]
        z = F.normalize(parts[..., 3:], dim=-1, p=2)
        return vel.detach(), z.detach()


class HIMActorCritic(nn.Module):
    """Full HOMIE actor-critic model.

    Only the actor + estimator are used for frozen tracking; the critic is
    instantiated solely so released checkpoints load without missing keys.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_one_step_obs: int,
        num_one_step_critic_obs: int,
        actor_history_length: int,
        critic_history_length: int,
        num_actions: int = 15,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        init_noise_std: float = 0.0,
        **kwargs,
    ):
        """Build the estimator, actor MLP, and (unused) critic MLP.

        Parameters mirror the OpenHomie release config; see
        :data:`HOMIE_INIT_DICT` for the released bundle's values.
        """
        super().__init__()
        if actor_hidden_dims is None:
            actor_hidden_dims = [512, 256, 256]
        if critic_hidden_dims is None:
            critic_hidden_dims = [512, 256, 256]

        act = _get_activation(activation)
        self.num_one_step_obs = num_one_step_obs
        self.actor_history_length = actor_history_length
        self.actor_proprioceptive_obs_length = actor_history_length * num_one_step_obs
        self.num_height_points = num_actor_obs - self.actor_proprioceptive_obs_length
        self.actor_use_height = self.num_height_points > 0
        self.num_actions = num_actions
        self.dynamic_latent_dim = 32
        self.terrain_latent_dim = 32

        if self.actor_use_height:
            mlp_input_dim_a = num_one_step_obs + 3 + self.dynamic_latent_dim + self.terrain_latent_dim
        else:
            mlp_input_dim_a = num_one_step_obs + 3 + self.dynamic_latent_dim

        # Estimator
        self.estimator = HIMEstimator(
            temporal_steps=actor_history_length,
            num_one_step_obs=num_one_step_obs,
            num_height_points=0,
            latent_dim=self.dynamic_latent_dim,
        )

        # Terrain encoder (only if height points exist)
        if self.actor_use_height:
            self.terrain_encoder = nn.Sequential(
                nn.Linear(num_one_step_obs + self.num_height_points, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.terrain_latent_dim),
            )

        # Actor
        actor_layers: list[nn.Module] = [nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]), act]
        for i in range(len(actor_hidden_dims)):
            if i == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], num_actions))
            else:
                actor_layers += [nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]), act]
        self.actor = nn.Sequential(*actor_layers)

        # Critic (checkpoint compatibility only)
        mlp_input_dim_c = num_critic_obs
        critic_layers: list[nn.Module] = [nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]), act]
        for i in range(len(critic_hidden_dims)):
            if i == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], 1))
            else:
                critic_layers += [nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]), act]
        self.critic = nn.Sequential(*critic_layers)

        # Action noise (unused at inference)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))


class HomieActorModule(nn.Module):
    """Inference-only HOMIE actor wrapper (no critic, no training)."""

    def __init__(self, full_model: HIMActorCritic):
        """Extract the actor + estimator (+ optional terrain encoder).

        Parameters
        ----------
        full_model : HIMActorCritic
            A constructed actor-critic with weights already loaded.
        """
        super().__init__()
        self.actor = full_model.actor
        self.estimator = full_model.estimator
        self.terrain_encoder = full_model.terrain_encoder if full_model.actor_use_height else None
        self.actor_use_height = full_model.actor_use_height
        self.num_one_step_obs = full_model.num_one_step_obs
        self.num_height_points = full_model.num_height_points
        self.std = full_model.std
        self.num_actions = HOMIE_NUM_ACTIONS

    @torch.no_grad()
    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        """Run the frozen actor.

        Parameters
        ----------
        obs_history : torch.Tensor
            Flattened observation history of shape ``(N, 516)``.

        Returns
        -------
        torch.Tensor
            Raw lower-body actions of shape ``(N, 15)``.
        """
        vel, dynamic_latent = self.estimator(obs_history[..., : self.num_one_step_obs * HOMIE_HISTORY_LENGTH])
        if self.actor_use_height and self.terrain_encoder is not None:
            terrain_latent = self.terrain_encoder(obs_history[..., -(self.num_one_step_obs + self.num_height_points) :])
            current_obs = obs_history[
                ...,
                -(self.num_one_step_obs + self.num_height_points) : -self.num_height_points,
            ]
            actor_input = torch.cat([current_obs, vel, dynamic_latent, terrain_latent], dim=-1)
        else:
            current_obs = obs_history[..., -self.num_one_step_obs :]
            actor_input = torch.cat([current_obs, vel, dynamic_latent], dim=-1)

        return self.actor(actor_input)


# Model init config matching the released model_walk.pt / model_stand.pt bundle.
HOMIE_INIT_DICT: dict = {
    "num_actor_obs": HOMIE_TOTAL_OBS,  # 516
    "num_critic_obs": 89,
    "num_one_step_obs": HOMIE_ONE_STEP_OBS,  # 86
    "num_one_step_critic_obs": 89,
    "actor_history_length": HOMIE_HISTORY_LENGTH,  # 6
    "critic_history_length": 1,
    "num_actions": HOMIE_NUM_ACTIONS,  # 15
    "actor_hidden_dims": [512, 256, 256],
    "critic_hidden_dims": [512, 256, 256],
    "activation": "elu",
    "init_noise_std": 0.0,
}


# ====================================================================
# HomieTracker: high-level observation / inference manager
# ====================================================================


@dataclass
class HomieConfig:
    """Configuration for the HOMIE frozen tracker."""

    walk_model_path: str = ""
    stand_model_path: str = ""
    command_scale: float = 1.0  # Scale applied to raw command before HOMIE
    walk_stand_threshold: float = 0.1  # Command-norm threshold for walk/stand switching
    clip_commands: bool = True
    clip_linvel_x: float = 0.8
    clip_linvel_y: float = 0.5
    clip_angvel: float = 0.8
    clip_height_lower: float = 0.24  # 0.74 - 0.5
    clip_height_upper: float = 0.74


class HomieTracker:
    """Manages HOMIE frozen-tracker models and observation construction.

    Handles loading and freezing the walk/stand models, building the 86-dim
    per-step observation with correct scaling, maintaining the 6-step history
    buffer, switching between walk and stand models based on the velocity
    command norm, and converting raw actions to joint-position targets.
    """

    def __init__(self, config: HomieConfig, num_envs: int, device: torch.device):
        """Initialize buffers and load the walk/stand models.

        Parameters
        ----------
        config : HomieConfig
            Tracker configuration (model paths, command clips, threshold).
        num_envs : int
            Number of parallel environments.
        device : torch.device
            Device for buffers and models.
        """
        self.config = config
        self.num_envs = num_envs
        self.device = device

        # Default DOF positions (OpenHomie training defaults).
        self.default_dof_pos = torch.tensor(HOMIE_DEFAULT_DOF_POS, dtype=torch.float32, device=device).unsqueeze(0)

        # History buffer: (num_envs, HISTORY_LENGTH, ONE_STEP_OBS).
        self.obs_history = torch.zeros(
            num_envs, HOMIE_HISTORY_LENGTH, HOMIE_ONE_STEP_OBS, dtype=torch.float32, device=device
        )

        # Previous HOMIE actions (fed back into the observation).
        self.prev_actions = torch.zeros(num_envs, HOMIE_NUM_ACTIONS, dtype=torch.float32, device=device)

        self.walk_model: HomieActorModule | None = None
        self.stand_model: HomieActorModule | None = None
        self._load_models()

    def _load_models(self) -> None:
        """Load and freeze the walk/stand HOMIE models."""
        if self.config.walk_model_path:
            self.walk_model = self._load_single_model(self.config.walk_model_path)
            logger.info("HOMIE walk model loaded from %s", self.config.walk_model_path)
        if self.config.stand_model_path:
            self.stand_model = self._load_single_model(self.config.stand_model_path)
            logger.info("HOMIE stand model loaded from %s", self.config.stand_model_path)

        if self.walk_model is None and self.stand_model is None:
            raise ValueError("At least one HOMIE model (walk or stand) must be provided")

    def _load_single_model(self, path: str) -> HomieActorModule:
        """Load one checkpoint into a frozen :class:`HomieActorModule`.

        Parameters
        ----------
        path : str
            Path to a ``.pt`` checkpoint (raw state dict or
            ``{"model_state_dict": ...}``).

        Returns
        -------
        HomieActorModule
            The frozen inference module on ``self.device``.
        """
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        # Support both raw state_dict and {model_state_dict: ...} formats.
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        full_model = HIMActorCritic(**HOMIE_INIT_DICT)
        # Checkpoints may carry training-only keys (estimator targets, etc.)
        # that inference does not need — load with strict=False.
        missing, unexpected = full_model.load_state_dict(state_dict, strict=False)
        if unexpected:
            logger.debug("HOMIE checkpoint: %d unexpected keys (training-only, OK)", len(unexpected))
        if missing:
            logger.warning("HOMIE checkpoint: %d missing keys: %s", len(missing), missing)

        module = HomieActorModule(full_model).to(self.device)
        for p in module.parameters():
            p.requires_grad = False
        module.eval()
        return module

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the history buffer and previous actions.

        Parameters
        ----------
        env_ids : torch.Tensor | None, optional
            Environment indices to reset. ``None`` resets all environments.
        """
        if env_ids is None:
            self.obs_history.zero_()
            self.prev_actions.zero_()
        else:
            self.obs_history[env_ids] = 0.0
            self.prev_actions[env_ids] = 0.0

    def build_one_step_obs(
        self,
        commands: torch.Tensor,  # (N, 7) — [vel_x, vel_y, ang_vel_z, height, roll, pitch, yaw]
        base_ang_vel: torch.Tensor,  # (N, 3) — body-frame angular velocity
        projected_gravity: torch.Tensor,  # (N, 3) — gravity in body frame
        dof_pos: torch.Tensor,  # (N, 29) — current joint positions
        dof_vel: torch.Tensor,  # (N, 29) — current joint velocities
    ) -> torch.Tensor:
        """Build one 86-dim observation step with correct scaling.

        Parameters
        ----------
        commands : torch.Tensor
            Full 7-D HOMIE command ``(N, 7)``.
        base_ang_vel : torch.Tensor
            Body-frame base angular velocity ``(N, 3)``.
        projected_gravity : torch.Tensor
            Gravity direction in the body frame ``(N, 3)``.
        dof_pos : torch.Tensor
            Current joint positions ``(N, 29)`` in HOMIE joint order.
        dof_vel : torch.Tensor
            Current joint velocities ``(N, 29)`` in HOMIE joint order.

        Returns
        -------
        torch.Tensor
            One observation step ``(N, 86)``.
        """
        # Commands: apply physical scaling (lin_vel × 2.0, ang_vel × 0.5).
        scaled_commands = commands.clone()
        scaled_commands[:, 0:2] *= OBS_SCALE_LIN_VEL
        scaled_commands[:, 2] *= OBS_SCALE_ANG_VEL_CMD
        # Zero out small velocity commands (norm < 0.1).
        vel_norm = scaled_commands[:, :3].norm(dim=-1)
        scaled_commands[vel_norm < 0.1, :3] = 0.0

        # DOF positions relative to HOMIE default (NOT the env's robot default).
        rel_dof_pos = dof_pos - self.default_dof_pos

        scaled_ang_vel = base_ang_vel * OBS_SCALE_BASE_ANG_VEL
        scaled_dof_vel = dof_vel * OBS_SCALE_DOF_VEL

        return torch.cat(
            [
                scaled_commands,  # 7
                scaled_ang_vel,  # 3
                projected_gravity,  # 3
                rel_dof_pos,  # 29
                scaled_dof_vel,  # 29
                self.prev_actions,  # 15
            ],
            dim=-1,
        )  # Total: 86

    def _shift_history(self, new_obs: torch.Tensor) -> None:
        """Drop the oldest history step and append a new one.

        Parameters
        ----------
        new_obs : torch.Tensor
            Newest per-step observation ``(N, 86)``.
        """
        self.obs_history = torch.cat([self.obs_history[:, 1:, :], new_obs.unsqueeze(1)], dim=1)

    def process_commands(self, raw_commands: torch.Tensor) -> torch.Tensor:
        """Expand 4-D raw commands into the 7-D HOMIE command format.

        Parameters
        ----------
        raw_commands : torch.Tensor
            ``(N, 4)`` commands ``[vel_x, vel_y, ang_vel_z, height_delta]``.

        Returns
        -------
        torch.Tensor
            ``(N, 7)`` commands ``[vel_x, vel_y, ang_vel_z, height, roll,
            pitch, yaw]`` with RPY defaulting to zero and the absolute height
            offset / clips applied.
        """
        N = raw_commands.shape[0]
        commands_7 = torch.zeros(N, 7, device=self.device)
        commands_7[:, :4] = raw_commands * self.config.command_scale
        # Add default height offset (height command is a delta about the default).
        commands_7[:, 3] += HOMIE_DEFAULT_HEIGHT

        if self.config.clip_commands:
            commands_7[:, 0].clamp_(-self.config.clip_linvel_x, self.config.clip_linvel_x)
            commands_7[:, 1].clamp_(-self.config.clip_linvel_y, self.config.clip_linvel_y)
            commands_7[:, 2].clamp_(-self.config.clip_angvel, self.config.clip_angvel)
            commands_7[:, 3].clamp_(self.config.clip_height_lower, self.config.clip_height_upper)

        return commands_7

    @torch.no_grad()
    def step(
        self,
        raw_commands: torch.Tensor,  # (N, 4)
        base_ang_vel: torch.Tensor,  # (N, 3)
        projected_gravity: torch.Tensor,  # (N, 3)
        dof_pos: torch.Tensor,  # (N, 29)
        dof_vel: torch.Tensor,  # (N, 29)
    ) -> torch.Tensor:
        """Run the full tracker: commands → obs → walk/stand → 15 actions.

        Parameters
        ----------
        raw_commands : torch.Tensor
            ``(N, 4)`` velocity + height-delta command.
        base_ang_vel : torch.Tensor
            Body-frame base angular velocity ``(N, 3)``.
        projected_gravity : torch.Tensor
            Gravity in the body frame ``(N, 3)``.
        dof_pos : torch.Tensor
            Joint positions ``(N, 29)`` in HOMIE joint order.
        dof_vel : torch.Tensor
            Joint velocities ``(N, 29)`` in HOMIE joint order.

        Returns
        -------
        torch.Tensor
            Raw lower-body actions ``(N, 15)`` (apply
            :meth:`actions_to_targets` for joint targets).
        """
        commands_7 = self.process_commands(raw_commands)

        current_obs = self.build_one_step_obs(commands_7, base_ang_vel, projected_gravity, dof_pos, dof_vel)
        self._shift_history(current_obs)

        # Flatten history to (N, 516) — [step_t-5, step_t-4, ..., step_t].
        flat_obs = self.obs_history.reshape(self.num_envs, -1)

        # Walk/stand switching based on the velocity command norm.
        if self.walk_model is not None and self.stand_model is not None:
            walk_actions = self.walk_model(flat_obs)

            # Stand obs: zero the velocity commands (first 3 dims) in every history step.
            stand_obs = flat_obs.clone()
            for step_idx in range(HOMIE_HISTORY_LENGTH):
                start = step_idx * HOMIE_ONE_STEP_OBS
                stand_obs[:, start : start + 3] = 0.0
            stand_actions = self.stand_model(stand_obs)

            cmd_norm = raw_commands[:, :3].norm(dim=-1, keepdim=True)
            walk_mask = cmd_norm > self.config.walk_stand_threshold
            actions = torch.where(walk_mask, walk_actions, stand_actions)
        elif self.walk_model is not None:
            actions = self.walk_model(flat_obs)
        else:
            actions = self.stand_model(flat_obs)

        self.prev_actions = actions.clone()
        return actions

    def actions_to_targets(self, homie_actions: torch.Tensor) -> torch.Tensor:
        """Convert raw HOMIE actions to lower-body joint-position targets.

        ``target = default_pos[:15] + action * HOMIE_ACTION_SCALE``.

        Parameters
        ----------
        homie_actions : torch.Tensor
            Raw lower-body actions ``(N, 15)``.

        Returns
        -------
        torch.Tensor
            Joint-position targets ``(N, 15)`` for DOFs 0-14.
        """
        return self.default_dof_pos[:, :HOMIE_NUM_ACTIONS] + homie_actions * HOMIE_ACTION_SCALE
