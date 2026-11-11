"""SimToolReal dexterous tool-manipulation policy port (KUKA iiwa14 + SharPa, 29 DOF).

Reproduces the deploy-time inference pipeline of ``tylerlum/simtoolreal``
(``isaacgymenvs/utils/observation_action_utils_sharpa.py`` +
``deployment/rl_policy_node.py``) inside an Eden Genesis environment so the
released SAPG-LSTM actor can be driven for sim2sim evaluation.

I/O contract (frozen by the checkpoint; see ``constants.py``):

    actor obs (140) = joint_pos(29, unscaled to [-1,1]) ++ joint_vel(29, raw)
        ++ prev_action_targets(29, raw) ++ palm_pos(3) ++ palm_rot(4, xyzw)
        ++ object_rot(4, xyzw) ++ fingertip_pos_rel_palm(15)
        ++ keypoints_rel_palm(12) ++ keypoints_rel_goal(12) ++ object_scales(3)
    network: RunningMeanStd -> +extra_params[0] (172) -> LSTM -> LN -> MLP -> mu
    actor out -> action (29) in [-1, 1]

Action transform (action_utils.apply_action_pipeline):

    arm  (0:7):   t = prev + DOF_SPEED_SCALE * dt * a; clamp; EMA(arm_ma)
    hand (7:29):  t = scale(a, lo, hi); EMA(hand_ma); clamp
    -> absolute joint-position targets fed to Genesis PD (scale=1, ref=ZERO).

Robot: the ``KukaSharpa`` composite — arm (``robot``, 7 DOF, palm frame) welded to
the SharPa hand (``hand``, 22 DOF, fingertips). Obs/action stay in one canonical
29-DOF order; the arm/hand split exists only at the sim boundary. Not the merged
URDF the policy trained on, but equivalent everywhere it can observe (kinematics to
9e-7 m, per-DOF dynamics exact, a 200-step rollout to 0.002 deg).

Parity vs ``tylerlum/simtoolreal``: obs 1.46e-7, action bit-identical, transform
5.96e-8. That harness feeds *synthetic* poses, so it validates the math but never
the read — hence the ``relative=True`` in :meth:`step`.

Scene dependency: reads an ``object`` pose and a goal pose (``goal_pose`` command
term, else the ``set_goal`` buffer) from the env, so it needs a scene that spawns
the table, tool and goal. :meth:`SimToolRealPolicy.scene_entity_options` declares
that workspace, so ``eden inference --policy simtoolreal`` and
``examples/policies/simtoolreal/main.py`` both build it from one source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from genesis.utils.geom import transform_by_quat

from eden import actions, observations
from eden.constants import ReferenceSource
from eden.extensions.policies.base import POLICY_REGISTRY, PolicyBase, PolicyOptions
from eden.extensions.policies.simtoolreal import constants as C
from eden.extensions.policies.simtoolreal.network import SimToolRealActor
from eden.options.entities import GHOST_DEFAULTS, BoxOptions, EntityOptionsLike
from eden.options.materials import RigidMaterialOptions
from eden.options.objects.dextoolbench import DEXTOOLBENCH_REGISTRY
from eden.options.robots.composite.kuka_sharpa import KukaSharpa
from eden.options.managers.observations import ObservationGroupOptions
from eden.utils.geom import wxyz_to_xyzw

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions


@POLICY_REGISTRY.register()
class SimToolRealPolicy(PolicyBase):
    """Object-centric zero-shot tool-manipulation controller (29-DOF, SAPG-LSTM)."""

    entity_name: str = "robot"  # arm: 7 DOF + the palm frame
    hand_name: str = "hand"  # SharPa hand: 22 DOF + the fingertips
    object_name: str = "object"
    goal_command: str = "goal_pose"  # optional CommandTerm exposing the goal pose

    # DexToolBench tool ``scene_entity_options()`` spawns. A different tool has a different grasp
    # bbox, so change ``object_scales`` with it (``C.object_scales_for(...)``) — nothing can catch
    # a mismatch downstream.
    object_asset: str = "long_screwdriver"

    # long_screwdriver bbox; override per tool via ``set_object_scales``.
    object_scales: tuple[float, float, float] = C.DEFAULT_OBJECT_SCALES

    # Downloaded by PolicyBase from the ``Kashu7100/eden_policies`` hub. For custom weights,
    # point ``checkpoint_path`` at a local file and clear ``md5_hash``.
    checkpoint_path: str | None = "simtoolreal/model.pth"
    md5_hash: str | None = "c9f9207125f56b4d4ebaea1e816ab7a8"

    def __init__(self, env: EnvBase, options: PolicyOptions) -> None:
        super().__init__(env, options)
        device = env.device
        self.device = device
        self._dtype = torch.float32
        N = env.num_envs

        self._actor = SimToolRealActor(self.checkpoint_path, device=str(device))
        self._state = self._actor.init_state(N)

        arm = self.require_entity(self.entity_name)
        hand = self.require_entity(self.hand_name)
        # Touched, not cached: ``step()`` re-resolves it (deployment hot-swaps entities), but a
        # scene without it must fail here, not one step into the rollout.
        self.require_entity(self.object_name)

        def _idx(values):
            return torch.as_tensor(values, dtype=torch.long, device=device)

        # Resolved at construction so a renamed joint/link raises here, not as a scrambled
        # obs vector on the first step.
        self._arm_dofs_idx = _idx(arm.find_named_dofs_idx_local(list(C.ARM_DOFS), preserve_order=True)[1])
        self._hand_dofs_idx = _idx(hand.find_named_dofs_idx_local(list(C.HAND_DOFS), preserve_order=True)[1])
        self._palm_link_idx = _idx(arm.find_named_links_idx_local([C.PALM_LINK], preserve_order=True)[1])
        self._ft_link_idx = _idx(hand.find_named_links_idx_local(list(C.FINGERTIP_LINKS), preserve_order=True)[1])

        def _t(x):
            return torch.as_tensor(x, dtype=self._dtype, device=device)

        self._q_lower = _t(C.Q_LOWER_LIMITS)
        self._q_upper = _t(C.Q_UPPER_LIMITS)
        self._palm_offset = _t(C.PALM_CENTER_OFFSET)
        self._ft_offset = _t(C.FINGERTIP_OFFSET)  # (3,) same for all 5
        self._kp_corners = _t(C.KEYPOINT_CORNERS)  # (4, 3)
        self._kp_scale = C.OBJECT_BASE_SIZE * C.KEYPOINT_SCALE / 2.0

        # Absolute target buffer (canonical order); prev_action_targets obs.
        self._prev_targets = torch.zeros(N, C.NUM_JOINTS, device=device, dtype=self._dtype)
        self._initialized = False

        # Per-env buffers; override via the setters or a ``goal_pose`` command term.
        self._object_scales = torch.tensor(self.object_scales, device=device, dtype=self._dtype).expand(N, 3).clone()
        self._goal_pos = torch.tensor(C.GOAL_START_POS, device=device, dtype=self._dtype).expand(N, 3).clone()
        self._goal_quat = torch.tensor(C.GOAL_START_QUAT_WXYZ, device=device, dtype=self._dtype).expand(N, 4).clone()

    def set_goal(self, pos: torch.Tensor, quat_wxyz: torch.Tensor) -> None:
        """Set the per-env object goal pose (world frame, wxyz)."""
        self._goal_pos = pos.to(self.device, self._dtype)
        self._goal_quat = quat_wxyz.to(self.device, self._dtype)

    def set_object_scales(self, scales: torch.Tensor) -> None:
        """Set the per-env object grasp-bbox scale (the object_scales obs)."""
        self._object_scales = scales.to(self.device, self._dtype)

    def _read_q(self, arm, hand) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint positions and velocities in canonical 29-DOF order (arm 0:7, hand 7:29)."""
        q = torch.cat(
            [
                arm.get_dofs_pos(dofs_idx_local=self._arm_dofs_idx),
                hand.get_dofs_pos(dofs_idx_local=self._hand_dofs_idx),
            ],
            dim=-1,
        )
        qd = torch.cat(
            [
                arm.get_dofs_vel(dofs_idx_local=self._arm_dofs_idx),
                hand.get_dofs_vel(dofs_idx_local=self._hand_dofs_idx),
            ],
            dim=-1,
        )
        return q, qd

    @classmethod
    def robot_options(cls) -> KukaSharpa:
        """``KukaSharpa`` at SimToolReal's base placement, fingertips grippier than the palm.

        The composite already defaults to this port's home pose, gains, armature and gravity
        compensation. The fingertip friction reproduces the original's
        ``modify_asset_frictions`` (robot 0.5, fingertips 1.5); the whole hand keeps colliding,
        since the fingers must contact an object all over to wrap it.
        """
        rig = KukaSharpa(base_pos=C.ROBOT_BASE_POS)
        rig.hand.default_geoms_friction = {link: C.FINGER_TIP_FRICTION for link in C.FINGERTIP_LINKS}
        return rig

    @classmethod
    def scene_entity_options(
        cls, *, object_asset: str | None = None, vis_mode: str = "visual"
    ) -> dict[str, EntityOptionsLike]:
        """The workspace the policy's observations are defined against: table, tool, goal ghost.

        The observation is object-centric (tool keypoints relative to the palm and the goal),
        so a robot-only scene cannot host this policy. Frictions and placements are
        SimToolReal's; see ``constants``.

        Parameters
        ----------
        object_asset : str | None
            :data:`~eden.options.objects.dextoolbench.DEXTOOLBENCH_REGISTRY` key to spawn.
            ``None`` reads :attr:`object_asset` — the base hook must stay callable with no
            arguments, and restating the literal here could drift from the attribute.
        vis_mode : str
            ``"collision"`` renders collision geometry instead of the visual meshes.
        """
        tool_cls = DEXTOOLBENCH_REGISTRY.get(object_asset or cls.object_asset)
        return {
            "table": BoxOptions(
                size=C.TABLE_SIZE,
                default_root_pos=C.TABLE_POS,
                fixed=True,
                vis_mode=vis_mode,
                material=RigidMaterialOptions(friction=C.TABLE_FRICTION),
            ),
            cls.object_name: tool_cls(
                default_root_pos=C.OBJECT_START_POS,
                default_root_quat=C.OBJECT_START_QUAT_WXYZ,
                vis_mode=vis_mode,
                material=RigidMaterialOptions(friction=C.OBJECT_FRICTION),
            ),
            # Collision-free and not physics-driven: a marker for the goal pose, re-posed each
            # step by whoever moves the goal (the example's sampler; a fixed goal stays put).
            "goal_ghost": tool_cls(
                default_root_pos=C.GOAL_START_POS,
                default_root_quat=C.GOAL_START_QUAT_WXYZ,
                is_fixed_base=True,
                collision=False,
                **dict(GHOST_DEFAULTS),
            ),
        }

    @classmethod
    def env_options(cls) -> dict[str, Any]:
        # 60 Hz control (120 Hz sim, decimation 2); softer constraints for contact-rich grasps.
        return dict(sim_dt=C.SIM_DT, decimation=C.DECIMATION, constraint_timeconst=0.02)

    @classmethod
    def action_options(cls, entity_name: str = "robot", hand_name: str = "hand") -> dict[str, ActionTermOptions]:
        # One term per entity: a controller drives one entity, so the canonical 29-vector is
        # split here. The PD passes the absolute targets through unchanged.
        def _pd(name: str, dofs: list[str]):
            return actions.ImplicitPDController.configure(
                entity_name=name,
                dofs_name=dofs,
                scale=1.0,
                reference_source=ReferenceSource.ZERO,
            )

        return dict(
            dofs_pos_controller=_pd(entity_name, list(C.ARM_DOFS)),
            hand_dofs_pos_controller=_pd(hand_name, list(C.HAND_DOFS)),
        )

    @classmethod
    def observation_options(cls, entity_name: str = "robot") -> dict[str, ObservationGroupOptions]:
        # The policy reads raw sim state in step(); Eden just needs >=1 obs term to build.
        return dict(
            policy=ObservationGroupOptions(
                concatenate_terms=True,
                enable_corruption=False,
                dofs_pos=observations.DofsPos.configure(entity_name=entity_name),
            ),
        )

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        h, c = self._state
        if envs_idx is None:
            h.zero_()
            c.zero_()
            self._prev_targets.zero_()
            self._initialized = False
        else:
            h[:, envs_idx] = 0.0
            c[:, envs_idx] = 0.0
            self._prev_targets[envs_idx] = 0.0

    def _keypoints(self, pos: torch.Tensor, quat_wxyz: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """World keypoint positions: (N, 3),(N,4 wxyz),(N,3) -> (N, 4, 3)."""
        offsets = self._kp_corners.unsqueeze(0) * self._kp_scale * scales.unsqueeze(1)  # (N,4,3)
        n, k, _ = offsets.shape
        q = quat_wxyz.unsqueeze(1).expand(n, k, 4).reshape(-1, 4)
        world = transform_by_quat(offsets.reshape(-1, 3), q).reshape(n, k, 3)
        return pos.unsqueeze(1) + world

    def _build_obs(self, arm, hand, object_pos, object_quat, goal_pos, goal_quat, object_scales) -> torch.Tensor:
        N = object_pos.shape[0]
        q, qd = self._read_q(arm, hand)
        joint_pos = 2.0 * (q - self._q_lower) / (self._q_upper - self._q_lower) - 1.0

        # Palm center = flange pose + local offset (the weld makes arm and hand one tree, so
        # this frame is the merged rig's iiwa14_link_7).
        palm_pos = arm.get_links_pos(ls_idx_local=self._palm_link_idx)[:, 0]  # (N,3)
        palm_quat = arm.get_links_quat(ls_idx_local=self._palm_link_idx)[:, 0]  # (N,4 wxyz)
        palm_center = palm_pos + transform_by_quat(self._palm_offset.expand(N, 3), palm_quat)

        # Fingertips (5) + local offset, relative to palm center.
        ft_pos = hand.get_links_pos(ls_idx_local=self._ft_link_idx)  # (N,5,3)
        ft_quat = hand.get_links_quat(ls_idx_local=self._ft_link_idx)  # (N,5,4)
        off = self._ft_offset.expand(N, C.NUM_FINGERTIPS, 3).reshape(-1, 3)
        ft_world = ft_pos + transform_by_quat(off, ft_quat.reshape(-1, 4)).reshape(N, C.NUM_FINGERTIPS, 3)
        fingertip_rel_palm = ft_world - palm_center.unsqueeze(1)

        obj_kp = self._keypoints(object_pos, object_quat, object_scales)
        goal_kp = self._keypoints(goal_pos, goal_quat, object_scales)
        keypoints_rel_palm = obj_kp - palm_center.unsqueeze(1)
        keypoints_rel_goal = obj_kp - goal_kp

        return torch.cat(
            [
                joint_pos,
                qd,
                self._prev_targets,
                palm_center,  # absolute palm_pos in world frame (robot at ROBOT_BASE_POS)
                wxyz_to_xyzw(palm_quat),
                wxyz_to_xyzw(object_quat),
                fingertip_rel_palm.reshape(N, -1),
                keypoints_rel_palm.reshape(N, -1),
                keypoints_rel_goal.reshape(N, -1),
                object_scales,
            ],
            dim=-1,
        )

    def _apply_action(self, action: torch.Tensor) -> torch.Tensor:
        prev = self._prev_targets
        targets = prev.clone()

        # Hand: absolute scale of [-1,1] to limits, then EMA, then clamp.
        lo, hi = self._q_lower[7:], self._q_upper[7:]
        hand = 0.5 * (action[:, 7:] + 1.0) * (hi - lo) + lo
        hand = C.HAND_MOVING_AVERAGE * hand + (1.0 - C.HAND_MOVING_AVERAGE) * prev[:, 7:]
        targets[:, 7:] = torch.clamp(hand, lo, hi)

        # Arm: velocity-delta accumulator, clamp, then EMA. Upstream clamps before the EMA and
        # does not re-clamp its output — matched deliberately.
        arm_lo, arm_hi = self._q_lower[:7], self._q_upper[:7]
        arm = prev[:, :7] + C.DOF_SPEED_SCALE * C.CONTROL_DT * action[:, :7]
        arm = torch.clamp(arm, arm_lo, arm_hi)
        targets[:, :7] = C.ARM_MOVING_AVERAGE * arm + (1.0 - C.ARM_MOVING_AVERAGE) * prev[:, :7]

        self._prev_targets = targets
        return targets

    def step(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        # Read off the entity, not an observation group: this policy rides along as an extra deploy
        # slot on another policy's or task's env, whose groups are not its own (see the
        # "Declare the observation" section of docs/guides/policy-porting.md).
        del obs  # raw sim state + object/goal pose read directly
        env = self.env
        arm = env.entities[self.entity_name]
        hand = env.entities[self.hand_name]
        obj = env.entities[self.object_name]

        # LOAD-BEARING ``relative=True``: the default solver frame re-origins a rigid body
        # onto its inertial frame, 3-15 cm down these tools' handles. IsaacGym and MuJoCo
        # report the URDF-root pose, so that is what the policy trained on. See
        # docs/guides/entities.md.
        object_pos = obj.get_pos(relative=True)
        object_quat = obj.get_quat(relative=True)

        # Prefer a ``goal_pose`` command term if the task has one, else the ``set_goal`` buffer.
        try:
            goal_cmd = env.command_manager.get_term(self.goal_command)
            self._goal_pos, self._goal_quat = goal_cmd.goal_pos, goal_cmd.goal_quat
        except Exception:
            pass

        # Seed the EMA from where the robot actually is.
        if not self._initialized:
            self._prev_targets = self._read_q(arm, hand)[0].clone()
            self._initialized = True

        obs_vec = self._build_obs(
            arm, hand, object_pos, object_quat, self._goal_pos, self._goal_quat, self._object_scales
        )
        action, self._state = self._actor(obs_vec, self._state)
        targets = self._apply_action(action)
        # Split the canonical 29-vector back out to the two controllers.
        return {
            "dofs_pos_controller": targets[:, : C.NUM_ARM_JOINTS],
            "hand_dofs_pos_controller": targets[:, C.NUM_ARM_JOINTS :],
        }
