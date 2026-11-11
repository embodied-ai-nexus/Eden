"""Generate the PHC SMPL humanoid MJCF (neutral, zero betas) used by the ``phc_3`` checkpoint.

---
tier: project
capabilities: [task.mimic, robot.smpl]
hardware: cpu
optional_deps: []
run: python examples/projects/mimic/assistmimic/scripts/generate_smpl_xml.py --smpl-dir <data/smpl> -o smpl_humanoid_phc.xml
---

Runs in a PHC environment (needs ``smpl_sim`` and the SMPL body models), not in Eden's ``uv`` env. It is how
the two MJCFs of ``eden_assets/humanoids/smpl_humanoid/`` were made — ``PhcSmplHumanoid``'s and AssistMimic's
0.7x-mass recipient's:

    python scripts/generate_smpl_xml.py --smpl-dir /path/to/data/smpl -o smpl_humanoid_phc.xml
    python scripts/generate_smpl_xml.py --smpl-dir /path/to/data/smpl -o smpl_humanoid_phc_70pct_mass.xml \
        --mass-scale 0.7

The robot options are PHC's ``robot=smpl_humanoid`` defaults, which is what ``phc_3`` was trained with.
"""

import argparse
import re
import sys
import types

import numpy as np
import torch

# smpl_sim imports cv2 at package import but never uses it for XML generation; a headless box without
# libGL cannot import cv2, so stub it.
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
# chumpy (pulled in by the SMPL parser) still does ``from numpy import bool, int, ...``.
for _alias, _builtin in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if not hasattr(np, _alias):
        setattr(np, _alias, _builtin)

from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smpl-dir", type=str, required=True, help="Directory holding SMPL_NEUTRAL.pkl etc.")
    parser.add_argument("-o", "--output", type=str, default="smpl_humanoid_phc.xml", help="Output MJCF path")
    parser.add_argument(
        "--no-boxes-real-weight",
        action="store_true",
        help="Use real_weight_porpotion_boxes=False (PHC README's phc_comp_3 eval setting)",
    )
    parser.add_argument(
        "--mass-scale",
        type=float,
        default=1.0,
        help="Scale every geom density (hence link mass and inertia), e.g. 0.7 for AssistMimic's recipient",
    )
    args = parser.parse_args()

    robot_cfg = {
        "mesh": False,
        "replace_feet": True,
        "rel_joint_lm": False,
        "upright_start": True,
        "remove_toe": False,
        "freeze_hand": False,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": not args.no_boxes_real_weight,
        "real_weight": True,
        "masterfoot": False,
        "master_range": 30,
        "big_ankle": True,
        "box_body": True,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "model": "smpl",
        "sim": "isaacgym",
    }
    robot = SMPL_Robot(robot_cfg, data_dir=args.smpl_dir)
    gender_beta = np.zeros(17)
    robot.load_from_skeleton(betas=torch.from_numpy(gender_beta[None, 1:]), gender=gender_beta[0:1], objs_info=None)
    robot.write_xml(args.output)
    with open(args.output) as f:
        xml = f.read()
    # The ground plane and light belong to the scene, not the robot: kept in the MJCF they become a static
    # ``world`` link of the humanoid entity (a second floor per humanoid).
    xml = re.sub(r"\s*<geom [^>]*name=\"floor\"[^>]*/>", "", xml)
    xml = re.sub(r"\s*<light [^>]*/>", "", xml)
    # Isaac Gym reads a joint's ``stiffness`` / ``damping`` as its position-drive gains and the motor ``gear`` as
    # its effort limit. Genesis (like MuJoCo) reads them as a passive spring and a torque motor instead, so drop
    # the motors here; PhcSmplHumanoid turns stiffness/damping into kp/kd and zeroes the passive terms.
    xml = re.sub(r"\s*<actuator>.*?</actuator>", "", xml, flags=re.DOTALL)
    if args.mass_scale != 1.0:
        xml = re.sub(r'density="([0-9.eE+-]+)"', lambda m: f'density="{float(m.group(1)) * args.mass_scale:.6f}"', xml)
    with open(args.output, "w") as f:
        f.write(xml)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
