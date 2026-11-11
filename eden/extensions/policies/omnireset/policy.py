"""OmniReset state expert port (UWLab, ICLR 2026): a UR5e + Robotiq 2F-85 inserting an object into a receptacle.

Reproduces UWLab's stage-1 evaluation (``OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0``) in Genesis.

I/O contract (frozen by the checkpoint; see ``constants.py``):

    obs (43) = last action (6 arm + 1 gripper, raw)
             ++ joint angles (12, Isaac Lab order, the 2F-85's passive linkage included)
             ++ wrist_3_link in base_link            (pos 3 + axis-angle 3)
             ++ insertive object in wrist_3_link      (pos 3 + axis-angle 3)
             ++ receptive object in wrist_3_link      (pos 3 + axis-angle 3)
             ++ insertive object in receptive object  (pos 3 + axis-angle 3)
    network: (obs - mean) / (std + 0.01) -> ELU MLP -> 7 action means
    action: arm (6) -> RelCartesianOSC end-effector delta at 10 Hz, torques at 120 Hz;
            gripper (1) -> finger_joint closed (0.785) when negative, else open.

The observations, the controller and the scene are the ``omnireset`` task's (``eden.tasks.omnireset``), so the
policy only runs the network on the env's ``policy`` group (``reads_env_observations``); its spec methods hand
back the task's pieces (``eden.tasks.omnireset.config``'s spaces and scene on UWLab's robot) for a task-less run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from eden.extensions.policies.base import POLICY_REGISTRY, PolicyBase, PolicyOptions
from eden.extensions.policies.omnireset.constants import EXPERT_CHECKPOINTS
from eden.extensions.policies.omnireset.network import OmniResetActor
from eden.tasks.omnireset import constants as C
from eden.tasks.omnireset.config import action_terms, observation_group, scene_entity_options
from eden.tasks.omnireset.constants import INSERTIVE, RECEPTIVE, RIGS, ROBOT
from eden.options.managers.observations import ObservationGroupOptions

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.entities import EntityOptionsLike, RobotOptions
    from eden.options.managers.actions import ActionTermOptions


@POLICY_REGISTRY.register()
class OmniResetPolicy(PolicyBase):
    """UWLab's state-based OmniReset expert for one object pair (``pair``), acting at 10 Hz."""

    reads_env_observations = True
    pair: str = "peg"
    #: ``None`` runs ``pair``'s released expert (:data:`~.constants.EXPERT_CHECKPOINTS`); a local file overrides it.
    checkpoint_path: str | None = None
    md5_hash: str | None = None

    def __init__(self, env: EnvBase, options: PolicyOptions) -> None:
        super().__init__(env, options)
        if self.pair not in EXPERT_CHECKPOINTS:
            raise ValueError(
                f"OmniResetPolicy: no released expert for pair {self.pair!r}; pairs: {list(EXPERT_CHECKPOINTS)}."
            )
        if self.checkpoint_path is None:
            from huggingface_hub import hf_hub_download

            checkpoint, md5 = EXPERT_CHECKPOINTS[self.pair]
            self.checkpoint_path = hf_hub_download(
                C.UWLAB_REPO, checkpoint, repo_type="dataset", revision=C.DATASETS_REVISION
            )
            self._verify_md5(self.checkpoint_path, md5)
        self._actor = OmniResetActor(self.checkpoint_path, device=env.device)
        for name in (ROBOT, INSERTIVE, RECEPTIVE):
            self.require_entity(name)

    @classmethod
    def robot_options(cls) -> RobotOptions:
        return scene_entity_options(C.PAIRS[cls.pair])[ROBOT]  # the same robot for every pair

    @classmethod
    def scene_entity_options(cls, pair: str = "peg") -> dict[str, EntityOptionsLike]:
        """The workspace and ``pair``'s objects (``pair`` is filled from the policy's own)."""
        if pair not in C.PAIRS:
            raise ValueError(f"OmniResetPolicy: no object pair {pair!r}; pairs: {list(C.PAIRS)}.")
        entities = scene_entity_options(C.PAIRS[pair])
        del entities[ROBOT]
        return entities

    @classmethod
    def env_options(cls) -> dict[str, Any]:
        return dict(sim_dt=C.SIM_DT, sim_substeps=RIGS["uwlab"].sim_substeps, decimation=C.DECIMATION)

    @classmethod
    def action_options(cls) -> dict[str, ActionTermOptions]:
        return action_terms(RIGS["uwlab"])

    @classmethod
    def observation_options(cls) -> dict[str, ObservationGroupOptions]:
        """Return the actor's 43-D group, term for term in ``POLICY_OBSERVATION_LAYOUT`` order.

        See :data:`~eden.extensions.policies.omnireset.constants.POLICY_OBSERVATION_LAYOUT`.
        """
        return dict(policy=observation_group(RIGS["uwlab"]))

    def step(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        action = self._actor(obs["policy"])
        return {"arm": action[:, :6], "gripper": action[:, 6:7]}
