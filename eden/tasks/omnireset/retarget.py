"""UWLab's recorded reset states on Eden's ``ur5e_2f85`` composite instead of the robot they were recorded with.

Everything but the gripper carries over: the composite's arm takes UWLab's joint angles by name (its gripper is
mounted so they put the grasp point where UWLab's did, see :data:`~.constants.COMPOSITE_MOUNT_OFFSET`), and the base,
mount plate and objects keep their poses. The gripper is another mechanism, so its state is made, not mapped: each
row is held at UWLab's arm and object poses in a simulation while the composite's gripper is driven, and its six
joints are recorded where it settles. Where UWLab's gripper had stopped partway (``finger_joint`` strictly between
open and closed: held by the object) the composite's closes onto the object; elsewhere it goes to UWLab's fraction
of the stroke. The family names do not decide this: most ``ObjectRestingEEGrasped`` rows have UWLab's gripper fully
closed beside the object, not holding it. A gripper closed this way is in contact and its linkage consistent, which
an angle mapping between the two linkages would not guarantee.

Run once per pair (cached under Eden's cache dir, next to the converted UWLab buffer)::

    python -m eden.tasks.omnireset.retarget --pair peg
"""

from __future__ import annotations

import argparse
import os

import torch

from eden.tasks.omnireset import constants as C
from eden.tasks.omnireset.config import scene_entity_options
from eden.tasks.omnireset.constants import GRIPPER, INSERTIVE, MOUNT, RECEPTIVE, RIGS, ROBOT
from eden.tasks.omnireset.datasets import buffer_cache_path, buffer_path
from eden.utils.buffers.reset_states import EntityStates, ResetStateBuffer

#: How long the gripper gets to reach its target with everything else held, s.
SETTLE_S = 0.5
#: UWLab's ``finger_joint`` band (rad) that means its gripper was stopped by the object, short of open (0) and of
#: its 0.785 rad closed target.
HELD_BAND = (0.05, 0.77)
#: Version of :func:`retarget`'s output, part of the cached buffer's name (:func:`retargeted_path`). Bump it with any
#: change to how rows are retargeted, or to an asset file the retargeting simulation loads: the name already follows
#: the rig and the scene's entity options, not the files they point to.
RETARGET_VERSION = 1


def held_by_object(finger_joint: torch.Tensor) -> torch.Tensor:
    """Rows whose UWLab gripper had stopped partway, closed onto the object (see :data:`HELD_BAND`)."""
    return (finger_joint > HELD_BAND[0]) & (finger_joint < HELD_BAND[1])


def retargeted_path(pair_name: str, rig_name: str = "ur5e_2f85") -> str:
    """Where :func:`retarget` caches ``pair_name``'s four families for ``rig_name``.

    The file name carries what the rows depend on, so a changed input is retargeted afresh rather than read stale: the
    source buffer's own cache name (a re-converted UWLab buffer), the rig (:data:`~.constants.RIGS`) and every scene
    entity's options the retargeting simulation builds (the robot's mount offset, armature, gripper and friction among
    them), :data:`HELD_BAND`, :data:`SETTLE_S` and :data:`RETARGET_VERSION`.
    """
    import hashlib

    from eden.utils.misc import get_cache_dir

    source = os.path.basename(buffer_cache_path(pair_name))
    scene = {
        name: options.model_dump_json() for name, options in scene_entity_options(C.PAIRS[pair_name], rig_name).items()
    }
    inputs = hashlib.sha1(repr((source, RIGS[rig_name], scene, HELD_BAND, SETTLE_S)).encode()).hexdigest()[:8]
    name = f"{pair_name}-{rig_name}-r{RETARGET_VERSION}-{inputs}.pt"
    return get_cache_dir("omnireset", C.DATASETS_REVISION[:12], name)


def _build_env(pair_name: str, rig_name: str, num_envs: int):
    from eden.envs.base import env_class_for_config
    from eden.options import EnvOptions
    from eden.options.scenes.empty import EmptySceneOptions
    from eden.utils.configs import EdenConfig

    entities = scene_entity_options(C.PAIRS[pair_name], rig_name)
    config = EdenConfig(
        env_options=EnvOptions(num_envs=num_envs, sim_dt=C.SIM_DT, sim_substeps=RIGS[rig_name].sim_substeps),
        scene_options=EmptySceneOptions(robot=entities.pop(ROBOT), **entities),
    )
    env = env_class_for_config(config).from_config(config, show_viewer=False)
    env.build()
    return env


def retarget(pair_name: str, rig_name: str = "ur5e_2f85", num_envs: int = 1024) -> ResetStateBuffer:
    """``pair_name``'s four UWLab families with every row's robot state on ``rig_name`` (see the module docstring)."""
    rig = RIGS[rig_name]
    if rig_name == "uwlab":
        raise ValueError("The UWLab rig is the one the states were recorded on; there is nothing to retarget.")
    source = ResetStateBuffer.load(buffer_path(pair_name))
    env = _build_env(pair_name, rig_name, num_envs)
    arm, gripper = env.entities[rig.arm_entity], env.entities[GRIPPER]
    uw_robot = source.entities[ROBOT]
    uw_joints = list(uw_robot.joints_name)
    arm_columns = [uw_joints.index(name) for name in C.ISAAC_JOINT_ORDER[:6]]
    finger_column = uw_joints.index("finger_joint")
    held = held_by_object(uw_robot.dofs_pos[:, finger_column])

    _, arm_dofs = arm.find_named_dofs_idx_local(list(rig.arm_joints))
    gripper_joints = [joint.name for joint in gripper.joints if joint.n_dofs > 0]
    _, gripper_dofs = gripper.find_named_dofs_idx_local(gripper_joints)
    _, drivers = gripper.find_named_dofs_idx_local(list(rig.gripper_drivers))
    bodies = {name: env.entities[name] for name in (MOUNT, RECEPTIVE, INSERTIVE)}

    gripper_state = torch.zeros(source.num_states, len(gripper_dofs))
    for start in range(0, source.num_states, num_envs):
        rows = torch.arange(start, min(start + num_envs, source.num_states))
        envs = torch.arange(len(rows), device=env.device)
        take = lambda column: column[rows].to(env.device)  # noqa: E731
        arm_pos = take(uw_robot.dofs_pos)[:, arm_columns]
        arm.set_pos(take(uw_robot.root_pos), envs_idx=envs, relative=True)
        arm.set_quat(take(uw_robot.root_quat), envs_idx=envs, relative=True)
        gripper.set_dofs_pos(
            torch.zeros(len(rows), len(gripper_dofs), device=env.device), dofs_idx_local=gripper_dofs, envs_idx=envs
        )
        # Closed onto the object where UWLab's gripper held it, else UWLab's fraction of the stroke.
        fraction = take(uw_robot.dofs_pos)[:, finger_column] / C.GRIPPER_CLOSED
        target = torch.where(held[rows].to(env.device), 1.0, fraction.clamp(0.0, 1.0)) * rig.gripper_closed
        gripper.control_dofs_pos(target[:, None].expand(-1, len(drivers)), dofs_idx_local=drivers, envs_idx=envs)
        for _ in range(round(SETTLE_S / C.SIM_DT)):
            # Hold the arm and every body at the recorded pose; only the gripper moves.
            arm.set_dofs_pos(arm_pos, dofs_idx_local=arm_dofs, envs_idx=envs, zero_velocity=True)
            for name, entity in bodies.items():
                states = source.entities[name]
                entity.set_pos(take(states.root_pos), envs_idx=envs, relative=True)
                entity.set_quat(take(states.root_quat), envs_idx=envs, relative=True)
            bodies[INSERTIVE].set_dofs_vel(torch.zeros(len(rows), 6, device=env.device), envs_idx=envs)
            env.scene.step()
        gripper_state[rows] = gripper.get_dofs_pos(dofs_idx_local=gripper_dofs, envs_idx=envs).cpu()
        print(f"retargeted rows {start}-{rows[-1].item()} of {source.num_states}", flush=True)

    entities = {name: source.entities[name] for name in (MOUNT, RECEPTIVE, INSERTIVE)}
    entities[ROBOT] = EntityStates(
        root_pos=uw_robot.root_pos,
        root_quat=uw_robot.root_quat,
        joints_name=tuple(rig.arm_joints),
        joints_n_dofs=(1,) * 6,
        dofs_pos=uw_robot.dofs_pos[:, arm_columns],
        dofs_vel=uw_robot.dofs_vel[:, arm_columns],
    )
    entities[GRIPPER] = EntityStates(
        joints_name=tuple(gripper_joints), joints_n_dofs=(1,) * len(gripper_joints), dofs_pos=gripper_state
    )
    return ResetStateBuffer(entities, category=source.category, category_names=source.category_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pair", default="peg", choices=sorted(C.PAIRS))
    parser.add_argument("--rig", default="ur5e_2f85", choices=[name for name in RIGS if name != "uwlab"])
    parser.add_argument("--num-envs", type=int, default=1024, help="Rows retargeted in parallel")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    args = parser.parse_args()

    import genesis as gs

    import eden  # noqa: F401  (Eden before Genesis initializes)

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")
    path = retargeted_path(args.pair, args.rig)
    retarget(args.pair, args.rig, args.num_envs).save(path)
    print(path)


if __name__ == "__main__":
    main()
