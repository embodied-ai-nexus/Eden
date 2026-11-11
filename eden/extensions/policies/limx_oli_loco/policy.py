"""Eden-trained SE(2) velocity-tracking locomotion policy for the LimX Oli 31-DOF humanoid.

Oli's counterpart to :mod:`eden.extensions.policies.holosoma` (which is G1-only): a frozen
``velocity_tracking_flat`` checkpoint driven by a 3-D velocity command. At **zero command it
is a balance controller** — the thing a motion-imitation deploy hands back to when its clip
runs out (``eden deploy --on_motion_end switch``), and the known-good controller an operator
can always switch to with the remote / keyboard.

Unlike Holosoma (an ONNX graph exported elsewhere) this loads an ``rsl_rl`` checkpoint Eden
trained itself, so the observation layout is not a foreign convention to be reverse-engineered
— it is the ``policy`` observation group of the run's saved ``config.json``, concatenated in
the order the group declares its terms (``ObservationManager._build_group_term_options``
iterates the Pydantic extras in insertion order, so declaration order *is* layout order; it is
NOT sorted, which is the convention Holosoma's foreign graph happens to use).

The layout this class reproduces, 105 floats for 31 DOFs:

===============  ====  =================================================
Slice            Size  Term
===============  ====  =================================================
``0:3``          3     ``base_lin_vel``, body frame
``3:6``          3     ``base_ang_vel``, body frame
``6:9``          3     ``projected_gravity``
``9:40``         31    ``dofs_pos``, offset from the default pose
``40:71``        31    ``dofs_vel``
``71:102``       31    ``last_action`` (this policy's own previous output)
``102:105``      3     ``generated_commands``, the SE(2) velocity command
===============  ====  =================================================

Every observation scale of the training group is ``1.0`` and corruption is off at deploy, so no
per-term scaling appears below. Actions are offsets from the fixed default pose scaled by
``OLI_ACTION_SCALE`` — the same constant the config records — and this class emits the
resulting **absolute** joint targets, as the deploy slot contract requires.

The robot state is re-read from ``env.entities[entity_name]`` on every step rather than cached
in ``__init__``, so after the deployment swaps in its ``RobotStateEntity`` this reads real
robot state on hardware and MuJoCo state under ``--mujoco`` — the same code either way.

There is no released checkpoint for this policy, so ``checkpoint_path`` has no default and
must be pointed at a local ``model_*.pt`` via
``OliVelocityLocoPolicy.use_local_checkpoint(path)`` or the ``checkpoint_path`` option.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import torch

from eden import actions, observations
from eden.constants import ReferenceSource
from eden.extensions.deployment.base import resolve_dofs_values
from eden.extensions.policies.base import POLICY_REGISTRY, PolicyBase
from eden.managers.terms.commands.velocity import ConstantVelocityCommand
from eden.options.managers.actions import ActionTermOptions
from eden.options.managers.commands import CommandTermOptions
from eden.options.managers.observations import ObservationGroupOptions, ObservationTermOptions
from eden.options.robots.limx_oli import OLI_ACTION_SCALE, OLI_DOFS_NAME, Oli
from eden.utils.geom import inv_transform_by_quat

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.extensions.policies import PolicyOptions

NUM_DOFS = len(OLI_DOFS_NAME)
_OBS_DIM = 3 + 3 + 3 + NUM_DOFS + NUM_DOFS + NUM_DOFS + 3  # 105


@POLICY_REGISTRY.register()
class OliVelocityLocoPolicy(PolicyBase):
    """Frozen velocity-tracking loco/stand controller for LimX Oli (rsl_rl backend)."""

    entity_name: str = "robot"

    # No released weights: this is pointed at a local training run, so the HF asset
    # resolution PolicyBase would otherwise apply is disabled.
    checkpoint_path: str | None = None
    md5_hash: str | None = None
    registry: ClassVar[str | None] = None
    dataset: ClassVar[str | None] = None

    def __init__(self, env: EnvBase, options: PolicyOptions) -> None:
        super().__init__(env, options)
        if not self.checkpoint_path:
            raise ValueError(
                "OliVelocityLocoPolicy has no released checkpoint; set checkpoint_path to a local "
                "velocity_tracking_flat model_*.pt trained for Oli (eden deploy --policy_checkpoint)."
            )
        device = env.device
        self.device = device
        self._dtype = torch.float32
        n = env.num_envs

        self._actor, self._obs_mean, self._obs_std = _load_rsl_rl_actor(self.checkpoint_path, device)

        robot = self.require_entity(self.entity_name)
        _, dofs_idx = robot.find_named_dofs_idx_local(OLI_DOFS_NAME, preserve_order=True)
        self._dofs_idx = torch.as_tensor(dofs_idx, dtype=torch.long, device=device)

        defaults = Oli().default_dofs_pos
        self._default_dof_pos = torch.as_tensor(
            resolve_dofs_values(defaults, list(OLI_DOFS_NAME)), dtype=self._dtype, device=device
        )
        self._action_scale = torch.as_tensor(
            resolve_dofs_values(OLI_ACTION_SCALE, list(OLI_DOFS_NAME)), dtype=self._dtype, device=device
        )

        self._commands = torch.zeros(n, 3, device=device, dtype=self._dtype)
        self._last_action = torch.zeros(n, NUM_DOFS, device=device, dtype=self._dtype)
        self._gravity_world = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=self._dtype).expand(n, 3)

    # -- commands ----------------------------------------------------------

    def set_command(self, command: torch.Tensor) -> None:
        """Set the 3-D ``(vx, vy, wz)`` velocity command. All zeros (the default) is a stand."""
        self._commands[:] = torch.as_tensor(command, dtype=self._dtype, device=self.device)

    def reset(self, envs_idx=None) -> None:
        """Clear the action feedback at a handover so the policy re-enters from rest."""
        if envs_idx is None:
            self._last_action.zero_()
        else:
            self._last_action[envs_idx] = 0.0

    # -- spec (task-less runs) --------------------------------------------

    @classmethod
    def robot_options(cls) -> Oli:
        return Oli()

    @classmethod
    def action_options(cls, entity_name: str = "robot") -> dict[str, ActionTermOptions]:
        """An absolute-target controller: :meth:`step` has already applied ``OLI_ACTION_SCALE``.

        The deploy slot contract requires ``reference_source=ZERO`` and ``scale=1.0`` so the
        returned targets are written to the robot unchanged; the training scale lives inside
        :meth:`step`, not here.
        """
        # Through the term class, not a bare ``ActionTermOptions``: ``configure`` stamps the
        # registry key the deploy slot builder resolves the controller by.
        return {
            "dofs_pos_controller": actions.ImplicitPDController.configure(
                entity_name=entity_name,
                dofs_name=list(OLI_DOFS_NAME),
                reference_source=ReferenceSource.ZERO,
                scale=1.0,
                offset=0.0,
            )
        }

    @classmethod
    def command_options(cls, entity_name: str = "robot") -> dict[str, CommandTermOptions]:
        """The ``velocity_command`` term the observation group below reads, held at zero (a stand).

        Declared so the env this policy *owns* (``--policy limx_oli_loco`` alone, or named first)
        can build: ``generated_commands`` resolves its term on the env's command manager at
        construction, and a task-less env has no command manager unless a policy asks for one.
        :meth:`step` reads its own :meth:`set_command` buffer, not this term, so the operator's
        ``--velocity_command`` / ``--base_command`` reach the policy through the deploy loop as
        they do for Holosoma; the term here only has to exist and be quiet.
        """
        return {
            "velocity_command": ConstantVelocityCommand.configure(
                entity_name=entity_name, command_value=(0.0, 0.0, 0.0)
            )
        }

    @classmethod
    def observation_options(cls, entity_name: str = "robot") -> dict[str, ObservationGroupOptions]:
        """The training group, in declaration order — the layout :meth:`step` reproduces."""
        return {
            "policy": ObservationGroupOptions(
                enable_corruption=False,
                base_lin_vel=ObservationTermOptions(
                    name="base_lin_vel", params={"entity_name": entity_name, "frame": "body"}
                ),
                base_ang_vel=ObservationTermOptions(
                    name="base_ang_vel", params={"entity_name": entity_name, "frame": "body"}
                ),
                projected_gravity=ObservationTermOptions(name="projected_gravity", entity_name=entity_name),
                dofs_pos=observations.DofsPos.configure(entity_name=entity_name, offset_from_default=True),
                dofs_vel=observations.DofsVel.configure(entity_name=entity_name),
                actions=ObservationTermOptions(name="last_action"),
                commands=ObservationTermOptions(name="generated_commands", params={"command_name": "velocity_command"}),
            )
        }

    # -- inference ---------------------------------------------------------

    def _build_obs(self, robot) -> torch.Tensor:
        """Assemble the 105-dim actor observation in the training group's declaration order."""
        dofs_pos = robot.get_dofs_pos(dofs_idx_local=self._dofs_idx)
        dofs_vel = robot.get_dofs_vel(dofs_idx_local=self._dofs_idx)
        base_quat = robot.get_quat()
        return torch.cat(
            [
                robot.get_vel(frame="body"),  # base_lin_vel (3)
                robot.get_ang(frame="body"),  # base_ang_vel (3)
                inv_transform_by_quat(self._gravity_world, base_quat),  # projected_gravity (3)
                dofs_pos - self._default_dof_pos,  # dofs_pos (31)
                dofs_vel,  # dofs_vel (31)
                self._last_action,  # actions (31)
                self._commands,  # commands (3)
            ],
            dim=-1,
        )

    def step(self, obs) -> dict[str, torch.Tensor]:
        """Run the frozen actor and return absolute 31-DOF joint targets.

        ``obs`` is ignored: riding along on another task's env, this policy's observation group
        is not the one that env computes, so it reads robot state directly (the Holosoma pattern).
        """
        del obs
        robot = self.env.entities[self.entity_name]
        actor_obs = self._build_obs(robot)
        with torch.no_grad():
            action = self._actor((actor_obs - self._obs_mean) / (self._obs_std + 0.01))
        self._last_action = action
        return {"dofs_pos_controller": self._default_dof_pos + self._action_scale * action}


def _load_rsl_rl_actor(path: str, device) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    """Rebuild the actor MLP and its observation normalizer from an rsl-rl >= 4 checkpoint.

    Reads the shapes straight off ``actor_state_dict`` rather than taking a width/activation
    config, so the loaded graph cannot disagree with the weights. ELU is not read from the
    checkpoint (a state dict records no activations) — it is rsl_rl's default and what every
    Eden ``velocity_tracking_flat`` run trains with; a policy trained with another activation
    would need this told to it.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in ckpt:
        raise ValueError(
            f"{path} carries no 'actor_state_dict' (keys: {sorted(ckpt)}). This loader expects an "
            "rsl-rl >= 4 checkpoint; a 3.x one stores a combined 'model_state_dict'."
        )
    sd = ckpt["actor_state_dict"]
    weights = sorted((int(k.split(".")[1]), k) for k in sd if k.startswith("mlp.") and k.endswith(".weight"))
    layers: list[torch.nn.Module] = []
    for pos, (idx, key) in enumerate(weights):
        out_dim, in_dim = sd[key].shape
        linear = torch.nn.Linear(in_dim, out_dim)
        linear.weight.data.copy_(sd[key])
        linear.bias.data.copy_(sd[f"mlp.{idx}.bias"])
        layers.append(linear)
        if pos < len(weights) - 1:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers).to(device).eval()

    mean = sd["obs_normalizer._mean"].to(device).float()
    std = sd["obs_normalizer._std"].to(device).float()
    if mean.shape[-1] != _OBS_DIM:
        raise ValueError(
            f"{path} expects {mean.shape[-1]} observations but this policy assembles {_OBS_DIM}. "
            "The checkpoint was trained with a different observation group than the one this "
            "class reproduces (see the module docstring for the layout)."
        )
    return actor, mean, std
