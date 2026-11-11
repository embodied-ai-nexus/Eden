"""HUSKY humanoid-skateboarding policy port (Unitree G1 23-DOF + skateboard).

Reproduces the play/deploy pipeline of `TeleHuman/humanoid_skateboarding
<https://github.com/TeleHuman/humanoid_skateboarding>`_ (RSS 2026, CC BY-NC 4.0)
inside an Eden Genesis environment so the released ONNX actor
(``ckpts/test.onnx``, trained with mjlab as ``Mjlab-Skater-Flat-Unitree-G1``)
can be rolled out for sim evaluation.

The scene pairs a 23-active-DOF G1 (wrists welded, 7 collision capsules per
foot — the exact training MJCF) with a free-floating skateboard whose trucks
steer when the deck tilts (MJCF joint-equality couplings; see
:class:`eden.options.objects.husky.HuskySkateboard`). The policy is
phase-conditioned: a 6 s clock sweeps push (right leg kicks the ground) →
transition → steer (both feet on the deck) → transition, and the command is
``[lin_vel_x, relative heading]``.

I/O contract (frozen by the exported ONNX; validated against the repo's
``test_scene/sim.py`` and the ONNX metadata): the actor reads the env's ``policy``
observation group and writes the ``dofs_pos_controller`` action term, both
declared by the training task (:mod:`eden.tasks.husky.config`), so the
released checkpoint and a policy trained in Eden share one layout:

    actor obs (395) = per-term 5-step history blocks, each flattened
    time-major (oldest first), concatenated in term order:
        command(2, x[2,1])  heading(1, x1/pi)  base_ang_vel(3, x0.25)
        projected_gravity(3)  joint_pos_rel(23)  joint_vel(23, x0.05)
        last_action(23)  phase(1)
    actor out -> action (23, MJCF order)

Action transform (matches ``sim.py``): ``target = default_pos + 0.25 *
effort_limit / kp * action``, applied by the action term
(:class:`~eden.actions.ImplicitPDController`, per-joint ``scale``,
``reference_source=DEFAULT``) at 50 Hz, so the env's ``last_action`` is the
actor's raw output, as in training.

Unlike the mjlab play env (which resamples commands every 20 s and re-anchors
the heading each gait cycle), this port holds the user-set command constant —
the same simplification the repo's own MuJoCo deployment script makes. The
command and the phase clock live on the env's ``skate`` command term
(:class:`~eden.tasks.husky.custom_terms.commands.ConstantSkateCommand`); the
heading command is relative to the initial (zero) world heading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch

from eden.extensions.policies.base import POLICY_REGISTRY, PolicyBase, PolicyOptions
from eden.extensions.policies.husky.constants import (
    ACTOR_OBS_DIM,
    CYCLE_TIME_S,
    DEFAULT_DOFS_POS,
    HUSKY_G1_DOFS_SPEC,
    POLICY_JOINT_NAMES,
    PUSH_KEYFRAME_DOFS_POS,
    ROBOT_DEFAULT_ROOT_POS,
    SKATEBOARD_DEFAULT_ROOT_POS,
    STILL_VX_THRESHOLD,
)
from eden.extensions.policies.husky.onnx_wrapper import HuskyActor
from eden.options.entities import EntityOptionsLike, RobotOptions
from eden.options.managers.observations import ObservationGroupOptions
from eden.options.objects.husky import HuskySkateboard

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.actuators import ActuatorSpecOptions
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.commands import CommandTermOptions
    from eden.tasks.husky.custom_terms.commands import ConstantSkateCommand

#: The env observation group the actor reads, and the command term its command and phase come from
#: (the name :func:`eden.tasks.husky.config.actor_observation_terms` reads it under).
OBS_GROUP = "policy"
COMMAND_NAME = "skate"


class HuskyG1(RobotOptions):
    """Skater G1: 23 active DOF (3-DOF waist, wrists welded), training gains and stance.

    ``g1_23dof_skater.xml`` is the training MJCF from humanoid_skateboarding —
    the standard G1 rev_1_0 model with the wrist joints removed and 7 collision
    capsules per foot (the geometry the policy balances the deck with). Gains,
    armature, and effort limits come from the same Unitree motor specs mjlab
    derived them from; the values equal the ONNX metadata digits.
    """

    dofs_spec: ClassVar[dict[str, type[ActuatorSpecOptions]]] = HUSKY_G1_DOFS_SPEC

    file: str = "humanoids/unitree_g1/g1_23dof_skater.xml"

    dofs_name: list[str] = list(POLICY_JOINT_NAMES)
    default_root_pos: tuple[float, float, float] = ROBOT_DEFAULT_ROOT_POS
    default_dofs_pos: dict[str, float] = dict(PUSH_KEYFRAME_DOFS_POS)
    default_dofs_kp: dict[str, float] = {p: s.RAIBERT_HEURISTIC_KP() for p, s in HUSKY_G1_DOFS_SPEC.items()}
    default_dofs_kd: dict[str, float] = {p: s.RAIBERT_HEURISTIC_KD() for p, s in HUSKY_G1_DOFS_SPEC.items()}
    default_dofs_armature: dict[str, float] = {p: s.ARMATURE() for p, s in HUSKY_G1_DOFS_SPEC.items()}
    default_dofs_force_limits: dict[str, float] = {p: s.EFFORT_LIMIT for p, s in HUSKY_G1_DOFS_SPEC.items()}


@POLICY_REGISTRY.register()
class HuskyPolicy(PolicyBase):
    """HUSKY skateboarding controller (G1 23-DOF + skateboard, ONNX backend)."""

    # step() reads the observation dict the env publishes, so this policy owns the deploy env.
    reads_env_observations = True

    entity_name: str = "robot"
    skateboard_name: str = "skateboard"

    # Resolved by PolicyBase / get_asset_path — downloaded from the
    # ``Kashu7100/eden_policies`` HF dataset on first use. Hash pinned against
    # the released ``ckpts/test.onnx``.
    checkpoint_path: str = "husky/skater_flat_g1.onnx"
    md5_hash: str | None = "69549f8d4a4fcc2629e4ba988666f841"

    def __init__(self, env: EnvBase, options: PolicyOptions) -> None:
        super().__init__(env, options)

        self.device = env.device

        robot = self.require_entity(self.entity_name)
        # The policy balances on the board without observing it, so a scene that lacks one runs
        # and falls: fail here instead.
        self.require_entity(self.skateboard_name)
        # The joint observations and the action term read the robot's DOFs in its own order.
        if list(robot.dofs_name) != list(POLICY_JOINT_NAMES):
            raise ValueError(
                f"HuskyPolicy drives the skater G1's 23 DOFs in {list(POLICY_JOINT_NAMES)} order; this robot's "
                f"dofs_name is {robot.dofs_name}. Build the robot from HuskyPolicy.robot_options()."
            )

        # The built robot must carry the training stance in every env: the joint-position
        # observation and the action term are both relative to its default pose.
        expected_default = torch.tensor(DEFAULT_DOFS_POS, device=env.device)
        built_default = robot.get_default_dofs_pos(POLICY_JOINT_NAMES).to(env.device)  # (num_envs, NUM_JOINTS)
        # ``allclose``'s tolerance, per env: the first env whose row is off is the one named.
        mismatched = ~torch.isclose(built_default, expected_default, atol=1e-6).all(dim=1)
        if mismatched.any().item():
            env_idx = int(mismatched.nonzero()[0].item())
            raise ValueError(
                f"Robot default_dofs_pos does not match the HUSKY training keyframe in env {env_idx}: "
                f"built={built_default[env_idx].tolist()} expected={expected_default.tolist()}"
            )

        # The env has to be the one this policy declares: a task env publishes its own
        # ``policy`` group and command terms, which the actor would read as garbage.
        group_dims = env.observation_manager.group_obs_dim
        if group_dims.get(OBS_GROUP) != (ACTOR_OBS_DIM,):
            raise ValueError(
                f"HuskyPolicy reads a {ACTOR_OBS_DIM}-dim '{OBS_GROUP}' observation group, but this env "
                f"publishes {group_dims}. Build the env from HuskyPolicy.observation_options() (a task-less "
                f"`eden inference --policy husky` does), not from another task's config."
            )
        self._skate_command()  # resolves the command term now, so a missing one fails at construction

        self._actor = HuskyActor(self.checkpoint_path, device=str(env.device))

    # ------------------------------------------------------------------
    # Env / action / command / observation specs
    # ------------------------------------------------------------------

    @classmethod
    def robot_options(cls) -> HuskyG1:
        """Skater G1 in the push-stance keyframe, placed to start on the deck."""
        return HuskyG1()

    @classmethod
    def skateboard_options(cls) -> HuskySkateboard:
        """Skateboard placed under the robot's left foot (training placement)."""
        return HuskySkateboard(default_root_pos=SKATEBOARD_DEFAULT_ROOT_POS)

    @classmethod
    def scene_entity_options(cls) -> dict[str, EntityOptionsLike]:
        """The board the policy rides — without it the robot has nothing to balance on."""
        return {cls.skateboard_name: cls.skateboard_options()}

    @classmethod
    def env_options(cls) -> dict[str, Any]:
        # 50 Hz control (decimation 4 x sim_dt 0.005) and the training solver
        # iterations (mjlab: iterations=10, ls_iterations=20).
        return dict(
            sim_dt=0.005,
            decimation=4,
            iterations=10,
            ls_iterations=20,
            max_collision_pairs=64,
            batch_dofs_info=False,
            enable_self_collision=False,
        )

    # The three specs below come from the training task's config. Imported here rather than at
    # module scope because that config imports this module (for ``HuskyG1``).

    @classmethod
    def action_options(cls, entity_name: str = "robot") -> dict[str, ActionTermOptions]:
        """The training task's action term: ``step`` returns the actor's raw (normalized) action."""
        from eden.tasks.husky.config import joint_position_action

        return dict(dofs_pos_controller=joint_position_action(entity_name))

    @classmethod
    def command_options(cls) -> dict[str, CommandTermOptions]:
        """The held ``[lin_vel_x, relative_heading]`` command and the phase clock the actor observes."""
        from eden.tasks.husky.custom_terms.commands import ConstantSkateCommand

        return {
            COMMAND_NAME: ConstantSkateCommand.configure(
                cycle_time_s=CYCLE_TIME_S, still_vx_threshold=STILL_VX_THRESHOLD
            )
        }

    @classmethod
    def observation_options(cls, entity_name: str = "robot") -> dict[str, ObservationGroupOptions]:
        """The training task's actor group, with its 5-step history and without its training noise."""
        from eden.tasks.husky.config import actor_observation_terms

        return {
            OBS_GROUP: ObservationGroupOptions(
                **actor_observation_terms(entity_name=entity_name, noise=False, history=True),
                enable_corruption=False,
            )
        }

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _skate_command(self) -> ConstantSkateCommand:
        manager = getattr(self.env, "command_manager", None)
        if manager is None or COMMAND_NAME not in manager.active_terms:
            raise ValueError(
                f"HuskyPolicy needs the '{COMMAND_NAME}' command term its observations read, and this env has "
                f"{[] if manager is None else manager.active_terms}. Add HuskyPolicy.command_options() to the "
                f"env's command_options (a task-less `eden inference --policy husky` does)."
            )
        return manager.get_term(COMMAND_NAME)

    @property
    def _commands(self) -> torch.Tensor:
        """The held command, ``(num_envs, 2)``; the CLI sizes ``--base_command`` from its width."""
        # Looked up on each call, not cached: a deployment can swap command terms after the policy is built.
        return self._skate_command().command

    def set_command(self, command: torch.Tensor) -> None:
        """Set the skate command ``[lin_vel_x, relative_heading]`` (shared or per-env).

        ``lin_vel_x`` is the target board speed in m/s (training range 0–1.5);
        below 0.1 the phase clock pins at 0 and the policy balances in place.
        ``relative_heading`` (rad, trained on ±pi/4) steers the board relative
        to its initial heading. Takes effect from the next observation the env computes.

        Only an env built from :meth:`command_options` holds a command. On ``--task husky`` the
        training task's :class:`~eden.tasks.husky.custom_terms.commands.SkateCommand` samples its
        own, which is what the policy then observes.
        """
        term = self._skate_command()
        if not hasattr(term, "set_command"):
            raise TypeError(
                f"The env's '{COMMAND_NAME}' command term is a {type(term).__name__}, which samples its own "
                f"command and holds none, so there is nothing to set. Build the env from "
                f"HuskyPolicy.command_options() (a ConstantSkateCommand) to hold one, or leave the command to "
                f"the {type(term).__name__}."
            )
        term.set_command(command)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def step(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {"dofs_pos_controller": self._actor(obs[OBS_GROUP])}
