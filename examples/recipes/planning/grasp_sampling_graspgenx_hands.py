"""GraspGenX grasps on YCB objects for every Eden gripper and hand it describes, each checked in simulation.

---
tier: recipe
capabilities: [planning.graspgenx, robot.franka, robot.robotiq, robot.sharpa, robot.inspire, robot.wuji]
hardware: gpu
optional_deps: [graspgenx]
run: uv run python examples/recipes/planning/grasp_sampling_graspgenx_hands.py
---

One GraspGenX model, five end-effectors, each picked by its Eden robot (``HANDS[...].end_effector``).
Each is a free end-effector, placed at the sampled pose directly with no arm to reach it, so this
exercises the grasps and the frames rather than an arm's reach.

For each object and end-effector, the best grasps are checked in the order GraspGenX ranks them:

1. **Open pose.** The root is placed at the grasp and the fingers set to the open configuration, then
   ``detect_collision`` lists what the hand touches. A collision-free approach touches nothing — not the
   table, not a neighbouring object, and not the target itself (a palm sunk into it is no grasp).
2. **Close**, for the best collision-free grasp. The fingers close under PD toward the closed
   configuration on the free object; then the contacts say what the grasp did: which hand links touch
   the target, whether the closed hand hits anything else, and how far closing pushed the target.
   ``pinched`` is two or more hand links on the target, nothing else touched, pushed under 2 cm.

The pose a sample places is the Eden link ``sampler.pose_link_name``: the pairing in
``GraspGenXSampler.END_EFFECTORS`` maps the description's base link onto it (identity for all but the
Inspire hand, whose Eden root is turned half a turn from the description's base link). The hand's
root is placed so that link lands on the pose.

Snapshots (open and closed, per object) go to ``-o``. Lifting is left out on purpose: whether a closed
grasp holds an object up is the gripper's PD gains and friction (the stock Franka fingers squeeze ~1.5 N),
not the grasp pose.

Usage:
    uv run python examples/recipes/planning/grasp_sampling_graspgenx_hands.py -o /tmp/graspgenx_hands
    uv run python examples/recipes/planning/grasp_sampling_graspgenx_hands.py --hands sharpa_r inspire_r -v
"""

import os
import time
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Callable

import torch

import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.extensions.grasp_sampling.graspgenx import GraspGenXEndEffector, GraspGenXSampler
from eden.options import CameraOptions, CamerasOptions, EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, RobotOptions
from eden.options.objects.ycb import YCB003CrackerBox, YCB005TomatoSoupCan, YCB006MustardBottle, YCB011Banana
from eden.utils.geom import combine_frame_transforms, subtract_frame_transforms

TABLE_TOP = 0.4
#: YCB objects in a row across the table: (options class, y).
OBJECTS = {
    "cracker_box": (YCB003CrackerBox, -0.33),
    "mustard": (YCB006MustardBottle, -0.11),
    "soup_can": (YCB005TomatoSoupCan, 0.11),
    "banana": (YCB011Banana, 0.33),
}
#: Closing may push the target this far [m] and still count as a pinch.
MAX_PUSH = 0.02


@dataclass(frozen=True)
class Hand:
    """An Eden end-effector GraspGenX describes, and how to drive its fingers."""

    #: The Eden robot, by its options class name; it picks GraspGenX's description of it.
    end_effector: GraspGenXEndEffector
    make: Callable[..., RobotOptions]
    #: Open / closed targets by DOF name. ``None`` takes the description's (``sampler.open_dofs_pos`` /
    #: ``close_dofs_pos``), whose joint names the dexterous hands share; the parallel jaws name a single
    #: coupled joint there, so they use the Eden robot's own.
    open_dofs_pos: dict[str, float] | None = None
    close_dofs_pos: dict[str, float] | None = None
    #: The DOFs PD drives when closing; the rest follow through the asset's own couplings. ``None``: all.
    driven_dofs: list[str] | None = None
    notes: str = field(default="", compare=False)


HANDS = {
    "franka_hand": Hand(
        end_effector="FrankaHand",
        make=en.robots.FrankaHand,
        open_dofs_pos=dict(en.robots.FrankaEmikaPanda.open_dofs_pos),
        close_dofs_pos=dict(en.robots.FrankaEmikaPanda.close_dofs_pos),
    ),
    "robotiq_2f85": Hand(
        end_effector="Robotiq2f85",
        make=en.robots.Robotiq2f85,
        open_dofs_pos=dict(en.robots.Robotiq2f85.open_dofs_pos),
        close_dofs_pos=dict(en.robots.Robotiq2f85.close_dofs_pos),
        driven_dofs=list(en.robots.Robotiq2f85.open_dofs_pos),
    ),
    "sharpa_r": Hand(end_effector="SharpaWave_R", make=en.robots.SharpaWave_R),
    "inspire_r": Hand(
        end_effector="InspireRH56_R",
        make=en.robots.InspireRH56_R,
        driven_dofs=list(en.robots.InspireRH56_R.actuated_dofs_name),
        notes="Eden flags its physics as approximate (non-backdrivable, high-friction actuators).",
    ),
    "wuji1_r": Hand(end_effector="WujiHand1_R", make=en.robots.WujiHand1_R),
}


def contacts(env: EnvBase, entity_name: str) -> set[tuple[int, int]]:
    """Geom pairs in contact with ``entity_name`` in its current pose, without stepping."""
    return {tuple(sorted(map(int, pair))) for pair in env.entities[entity_name].detect_collision()}


def owner(env: EnvBase, geom: int) -> str:
    return next(
        name for name, e in env.entities.items() if hasattr(e, "geom_start") and e.geom_start <= geom < e.geom_end
    )


def hand_links_touching(env: EnvBase, hand_name: str, object_name: str) -> tuple[set[str], set[str]]:
    """The hand's links touching ``object_name``, and the other entities the hand touches."""
    hand, target = env.entities[hand_name], env.entities[object_name]
    links, others = set(), set()
    for pair in contacts(env, hand_name):
        mine = [g for g in pair if hand.geom_start <= g < hand.geom_end]
        theirs = [g for g in pair if not hand.geom_start <= g < hand.geom_end]
        if not mine or not theirs:
            continue  # a self-contact
        if target.geom_start <= theirs[0] < target.geom_end:
            links.add(hand.geoms[mine[0] - hand.geom_start].link.name)
        else:
            others.add(owner(env, theirs[0]))
    return links, others


def dofs_target(entity, targets: dict[str, float], mimic: dict | None, device) -> tuple[list[int], torch.Tensor]:
    """``targets`` plus every mimic follower they drive, as (local DOF indices, a (1, n) target)."""
    targets = dict(targets)
    for follower, (driver, multiplier, offset) in (mimic or {}).items():
        if driver in targets:
            targets[follower] = targets[driver] * multiplier + offset
    names, idx = entity.find_named_dofs_idx_local(list(targets))
    return idx, torch.tensor([[targets[name] for name in names]], device=device)


def main():
    parser = ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Open the viewer and pause on each checked grasp")
    parser.add_argument(
        "--hands", nargs="+", type=str, default=sorted(HANDS), choices=sorted(HANDS), help="End-effectors to sample for"
    )
    parser.add_argument("--candidates", type=int, default=10, help="Best grasps per object checked at the open pose")
    parser.add_argument(
        "-o", "--output-dir", type=str, default=None, help="Directory to write the open / closed snapshots to"
    )
    args = parser.parse_args()

    # The model runs on CUDA only. An effective dt of 1 ms keeps the 2F-85's closed linkage stable.
    en.init(backend=gs.gpu)
    hand_names = args.hands[:1] if en.is_smoke() else args.hands
    objects = dict(list(OBJECTS.items())[:1]) if en.is_smoke() else OBJECTS
    # Parked well below the floor until a grasp is checked; fixed bases, so the hand stays where it is put.
    hands = {
        name: HANDS[name].make(is_fixed_base=True, default_root_pos=(-1.0 - i, 0.0, -10.0))
        for i, name in enumerate(hand_names)
    }
    env = EnvBase(
        env_options=EnvOptions(num_envs=1, sim_dt=0.002, sim_substeps=2),
        scene_options=SceneOptions(
            table=BoxOptions(
                size=(0.5, 1.0, TABLE_TOP), default_root_pos=(0.5, 0.0, TABLE_TOP / 2), is_fixed_base=True
            ),
            **{name: cls(default_root_pos=(0.5, y, TABLE_TOP + 0.12)) for name, (cls, y) in objects.items()},
            **hands,
        ),
        cameras_options=CamerasOptions(
            # The sampling view: all four objects, from the front and above.
            camera=CameraOptions(
                cam_pos=(1.3, 0.0, 1.0), cam_lookat=(0.5, 0.0, TABLE_TOP), cam_resolution=(640, 480), cam_fov=50.0
            ),
            **{
                f"view_{name}": CameraOptions(
                    cam_pos=(0.95, y + 0.25, TABLE_TOP + 0.35),
                    cam_lookat=(0.5, y, TABLE_TOP + 0.08),
                    cam_resolution=(480, 400),
                    cam_fov=55.0,
                )
                for name, (_, y) in objects.items()
            },
        ),
        show_viewer=args.vis and not en.is_smoke(),
    )
    env.build()

    def settle():
        env.reset()
        for _ in range(5 if en.is_smoke() else 150):
            env.step()

    def snapshot(object_name: str, label: str):
        env.scene.visualizer.update()
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            env.cameras[f"view_{object_name}"].snapshot(os.path.join(args.output_dir, f"{label}.png"))
        if args.vis and not en.is_smoke():
            time.sleep(1.0)

    summary = []
    for hand_name in hand_names:
        hand_spec, hand = HANDS[hand_name], env.entities[hand_name]
        settle()
        sampler = GraspGenXSampler(env, GraspGenXSampler.configure(end_effector=hand_spec.end_effector))
        open_pos = hand_spec.open_dofs_pos or sampler.open_dofs_pos
        close_pos = hand_spec.close_dofs_pos or sampler.close_dofs_pos
        mimic = getattr(hand_spec.make.metadata, "mimic_dofs", None)
        open_idx, open_target = dofs_target(hand, open_pos, mimic, env.device)
        driven = {k: v for k, v in close_pos.items() if hand_spec.driven_dofs is None or k in hand_spec.driven_dofs}
        close_idx, close_target = dofs_target(hand, driven, None, env.device)
        park = hand.get_pos(relative=True).clone(), hand.get_quat(relative=True).clone()
        hand.set_dofs_pos(open_target, dofs_idx_local=open_idx)
        baseline = contacts(env, hand_name)  # the open hand's own self-contacts, parked
        # Where the hand's root sits in the posed link's frame: identity when the root is that link.
        _, (link,) = hand.find_named_links_idx_local([sampler.pose_link_name])
        root_in_link = subtract_frame_transforms(
            hand.get_links_pos(ls_idx_local=[link])[:, 0],
            hand.get_links_quat(ls_idx_local=[link])[:, 0],
            hand.get_pos(relative=True),
            hand.get_quat(relative=True),
        )

        for object_name in objects:
            result = sampler.sample_grasps(object_name)
            checked, clear = 0, []
            for i in range(min(len(result), 2 if en.is_smoke() else args.candidates)):
                checked += 1
                root = combine_frame_transforms(result.pos[i : i + 1], result.quat[i : i + 1], *root_in_link)
                hand.set_pos(root[0], relative=True)
                hand.set_quat(root[1], relative=True)
                hand.set_dofs_pos(open_target, dofs_idx_local=open_idx)
                hits = {
                    " <-> ".join(sorted(owner(env, g) for g in pair)) for pair in contacts(env, hand_name) - baseline
                }
                if hits:
                    en.logger.debug(f"{hand_name} / {object_name} grasp {i}: open pose collides ({sorted(hits)})")
                else:
                    clear.append(i)

            verdict = "no grasp" if not result.success else f"{len(clear)}/{checked} collision-free at the open pose"
            if clear:
                best = clear[0]
                root = combine_frame_transforms(
                    result.pos[best : best + 1], result.quat[best : best + 1], *root_in_link
                )
                hand.set_pos(root[0], relative=True)
                hand.set_quat(root[1], relative=True)
                hand.set_dofs_pos(open_target, dofs_idx_local=open_idx)
                snapshot(object_name, f"{hand_name}_{object_name}_open")
                obj = env.entities[object_name]
                start = obj.get_pos(relative=True).clone()
                for _ in range(10 if en.is_smoke() else 300):
                    hand.control_dofs_pos(close_target, dofs_idx_local=close_idx)
                    env.step()
                push = float((obj.get_pos(relative=True) - start).norm())
                links, others = hand_links_touching(env, hand_name, object_name)
                pinched = len(links) >= 2 and not others and push < MAX_PUSH
                snapshot(object_name, f"{hand_name}_{object_name}_closed")
                verdict += (
                    f"; best ({best}) closed: {len(links)} links on target, pushed {100 * push:.1f} cm"
                    + (f", also touches {sorted(others)}" if others else "")
                    + (" -> pinched" if pinched else " -> no pinch")
                )
                settle()
                hand.set_dofs_pos(open_target, dofs_idx_local=open_idx)
            summary.append(f"{hand_name:13s} {object_name:12s} {len(result):4d} sampled  {verdict}")
            hand.set_pos(park[0], relative=True)
            hand.set_quat(park[1], relative=True)
        del sampler
        torch.cuda.empty_cache()

    en.logger.info("GraspGenX across Eden's end-effectors:\n" + "\n".join(summary))


if __name__ == "__main__":
    main()
