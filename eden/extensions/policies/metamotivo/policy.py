"""MetaMotivo policy extension for Eden.

Loads a pretrained MetaMotivo (FBcpr) model and runs it on Eden's
``RLEnvBase`` with an SMPL humanoid.  The policy manages the task-context
vector ``z`` that determines behavior.

The observation term ``humenv_proprio`` replicates HumEnv's egocentric
proprioceptive encoding, and the action term ``MujocoSMPLActuator``
replicates MuJoCo's general affine actuators, so the pretrained model
can be deployed without retraining.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch

from eden.extensions.policies.base import PolicyBase, PolicyOptions, POLICY_REGISTRY
from eden.options.managers.observations import ObservationGroupOptions
from eden.options.robots.smpl_humanoid import SmplHumanoid
from eden.terms import ObsTerm

# Register custom terms by importing them
from eden.extensions.policies.metamotivo.custom_terms.observations import humenv_proprio
from eden.extensions.policies.metamotivo.custom_terms.actions import MujocoSMPLActuator

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions


# HumEnv SMPL humanoid MJCF, resolved from the eden_assets HF dataset. PD gains
# are zeroed below since MujocoSMPLActuator supplies all control torques.
_HUMENV_ROBOT_XML = "humanoids/smpl_humanoid/humenv_smpl_humanoid.xml"


def metamotivo_smpl_robot(**overrides: Any) -> SmplHumanoid:
    """Build the HumEnv SMPL humanoid MetaMotivo was trained against."""
    return SmplHumanoid(
        file=_HUMENV_ROBOT_XML,
        up=(0, 1, 0),  # HumEnv XML is Y-up
        default_root_pos=(0.0, 0.0, 0.94),  # HumEnv T-pose height
        default_dofs_kp={".*": 0.0},
        default_dofs_kd={".*": 0.0},
        **overrides,
    )


@POLICY_REGISTRY.register()
class MetaMotivoPolicy(PolicyBase):
    """MetaMotivo (FBcpr) policy for SMPL humanoid control on Eden.

    The task-context vector ``z`` determines the behavior.  It can be set
    directly via :meth:`set_context`, inferred from a goal via :meth:`compute_goal_context`,
    resampled from the prior via :meth:`sample_random_context`, or left as the random
    sample drawn on the first :meth:`step`.

    Parameters
    ----------
    model_name : str
        Hugging Face model id, loaded directly with ``from_pretrained`` (no Eden checkpoint).
        Variants: ``"facebook/metamotivo-S-1"`` (24.5M) and ``"facebook/metamotivo-M-1"`` (288M).
    """

    # step() reads the observation dict the env publishes, so this policy owns the deploy env.
    reads_env_observations = True

    model_name: str = "facebook/metamotivo-S-1"
    # No checkpoint: the model is loaded from Hugging Face by ``model_name``.
    checkpoint_path: str | None = None
    registry: ClassVar[str | None] = None
    dataset: ClassVar[str | None] = None

    def __init__(self, env: EnvBase, options: PolicyOptions):
        super().__init__(env, options)

        self._z: torch.Tensor | None = None
        self._step_count = 0
        self._model = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the FBcpr model from HuggingFace using ``from_pretrained``."""
        try:
            from metamotivo.fb_cpr.huggingface import FBcprModel
        except ImportError:
            raise ImportError(
                "metamotivo is not installed. Please install it with "
                "`pip install metamotivo[huggingface]@git+https://github.com/facebookresearch/metamotivo.git`."
            )

        self._model = FBcprModel.from_pretrained(self.model_name).to(self.env.device)
        self._model.eval()

    @classmethod
    def robot_options(cls) -> SmplHumanoid:
        """Return the HumEnv SMPL humanoid MetaMotivo was trained against."""
        return metamotivo_smpl_robot()

    @classmethod
    def env_options(cls) -> dict[str, Any]:
        return dict(
            # HumEnv: sim_dt=1/450, action_repeat=15 -> ~30Hz control
            sim_dt=1.0 / 450.0,
            decimation=15,
        )

    @classmethod
    def action_options(cls, entity_name: str = "robot") -> dict[str, ActionTermOptions]:
        return dict(
            smpl_actuator=MujocoSMPLActuator.configure(
                entity_name=entity_name,
                dofs_name=SmplHumanoid().dofs_name,
            ),
        )

    @classmethod
    def observation_options(cls, entity_name: str = "robot") -> dict[str, ObservationGroupOptions]:
        return dict(
            proprio=ObservationGroupOptions(
                concatenate_terms=True,
                history_length=1,
                flatten_history_dim=True,
                proprio=ObsTerm.configure(
                    func=humenv_proprio,
                    params={"entity_name": entity_name},
                ),
            ),
        )

    def step(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Compute action from observation using the MetaMotivo model.

        Parameters
        ----------
        obs : dict
            Observation dict from Eden's observation manager.
            Expected key: ``"proprio"`` with shape ``(num_envs, obs_dim)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Action dict with key ``"smpl_actuator"``, shape ``(num_envs, 69)``.
        """
        with torch.no_grad():
            proprio = obs["proprio"]  # (num_envs, obs_dim)

            # Ensure context z is initialized
            if self._z is None or self._z.shape[0] != proprio.shape[0]:
                self._z = self._model.sample_z(proprio.shape[0], device=str(self.env.device))

            # Get action from the model
            action = self._model.act(
                obs=proprio.float(),
                z=self._z.float(),
                mean=True,
            )

            self._step_count += 1

        return dict(smpl_actuator=action)

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    @property
    def z_dim(self) -> int:
        """Dimensionality of the task-context vector."""
        return self._model.cfg.archi.z_dim

    @property
    def context(self) -> torch.Tensor | None:
        """Current task-context vector z."""
        return self._z

    def set_context(self, z: torch.Tensor) -> None:
        """Set the task-context vector directly.

        Parameters
        ----------
        z : torch.Tensor
            Shape ``(num_envs, z_dim)``; a 1-D ``(z_dim,)`` is unsqueezed to ``(1, z_dim)``.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        self._z = z.to(self.env.device)

    def sample_random_context(self, batch_size: int | None = None) -> torch.Tensor:
        """Sample a random context from the model's z prior.

        Parameters
        ----------
        batch_size : int | None
            If None, uses ``env.num_envs``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, z_dim)``.
        """
        if batch_size is None:
            batch_size = self.env.num_envs
        self._z = self._model.sample_z(batch_size, device=str(self.env.device))
        return self._z

    def compute_goal_context(self, goal_obs: torch.Tensor) -> torch.Tensor:
        """Compute context from a goal observation via the backward map, and make it current.

        Parameters
        ----------
        goal_obs : torch.Tensor
            Target proprioceptive observation, shape ``(1, obs_dim)`` or ``(obs_dim,)``.

        Returns
        -------
        torch.Tensor
            Context vector, shape ``(1, z_dim)``; the current context becomes its
            expansion to ``(num_envs, z_dim)``.
        """
        if goal_obs.dim() == 1:
            goal_obs = goal_obs.unsqueeze(0)
        goal_obs = goal_obs.to(self.env.device).float()
        z = self._model.goal_inference(next_obs=goal_obs)
        self._z = z.expand(self.env.num_envs, -1)
        return z

    def compute_tracking_context(self, target_obs: torch.Tensor) -> torch.Tensor:
        """Compute tracking context from a sequence of target observations.

        Parameters
        ----------
        target_obs : torch.Tensor
            Target observations, shape ``(seq_len, obs_dim)``.

        Returns
        -------
        torch.Tensor
            Context vectors, shape ``(seq_len, z_dim)``. The current context is not changed.
        """
        target_obs = target_obs.to(self.env.device).float()
        z = self._model.tracking_inference(next_obs=target_obs)
        return z
