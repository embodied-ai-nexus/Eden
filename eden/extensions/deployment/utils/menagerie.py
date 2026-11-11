"""Mapping from Eden robots to MuJoCo Menagerie models, resolved as an HF-hosted asset.

`MuJoCo Menagerie <https://github.com/google-deepmind/mujoco_menagerie>`_ ships
community-tuned MJCF models — with the joint armature/damping/frictionloss and contact
parameters MuJoCo needs — that Eden's Genesis-authored training MJCFs lack. Using a
Menagerie model as the sim-to-sim target therefore gives a higher-fidelity, more
independent check than replaying the training asset in MuJoCo (e.g. G1 holds a stand
under a plain PD on the Menagerie model, but topples on the training MJCF).

The models are consumed through Eden's normal asset flow (:func:`get_asset_path` against
an HF dataset), so they download and cache like any other asset. Menagerie's per-robot
directories are self-contained (the robot ``.xml`` embeds its meshes and free joint), so
:class:`~eden.extensions.deployment.utils.mujoco_sim.MujocoSim` — which synthesizes its
own ground plane and IMU — targets the robot ``.xml`` directly rather than the wrapping
``scene.xml``.

Menagerie does not cover every robot Eden deploys to. A **vendor** MJCF — the one the
robot's own manufacturer ships — serves the same purpose when it carries the same
detail, and is mirrored into the same dataset under Menagerie's layout; ``limx_oli`` is
one (LimX's ``HU_D04_01.xml``, with per-joint armature and the ankle/waist parallel
linkages as real equality constraints, which Eden's serial URDF does not have).

Hosting: mirror the upstream layout under one HF *dataset* repo (default
``Kashu7100/mujoco_menagerie``), preserving each robot's directory and its ``LICENSE``
(the Unitree models are BSD-3-Clause — redistribution is fine with the notice retained;
a vendor model carries whatever licence its manufacturer ships)::

    mujoco_menagerie/
      unitree_go1/{go1.xml, assets/*, LICENSE}
      unitree_go2/{go2.xml, assets/*, LICENSE}
      unitree_g1/{g1.xml, assets/*, LICENSE}
      limx_oli/{HU_D04_01.xml, assets/*}      # vendor MJCF, not upstream Menagerie

A joint name in the Eden robot's ``dofs_name`` must exist in the Menagerie model;
``MujocoSim`` maps by name and reorders, so extra model joints are tolerated but a
Menagerie model missing a controlled joint is not (hence no G1-23DOF entry — the
Menagerie G1 is the 29-DOF body).
"""

from __future__ import annotations

# Default HF dataset coordinates for the mirrored Menagerie models.
MENAGERIE_REGISTRY = "Kashu7100"
MENAGERIE_DATASET = "mujoco_menagerie"


def _menagerie_map() -> dict[type, str]:
    """Robot options class → Menagerie robot-``.xml`` path, relative to the dataset root.

    Built lazily so importing this module doesn't force the robot option classes to import
    (and so a renamed/removed robot degrades to "no Menagerie model" rather than an import
    error at module load).
    """
    mapping: dict[type, str] = {}
    try:
        from eden.options.robots.unitree_go2 import Go2

        mapping[Go2] = "unitree_go2/go2.xml"
    except ImportError:
        pass
    try:
        from eden.options.robots.unitree_go1 import Go1

        mapping[Go1] = "unitree_go1/go1.xml"
    except ImportError:
        pass
    try:
        from eden.options.robots.unitree_g1 import G1Dofs29, G1Dofs29NoHands

        # Menagerie's G1 is the 29-DOF body (no hands); it covers both variants.
        mapping[G1Dofs29] = "unitree_g1/g1.xml"
        mapping[G1Dofs29NoHands] = "unitree_g1/g1.xml"
    except ImportError:
        pass
    try:
        from eden.options.robots.limx_oli import Oli

        # LimX's own MJCF, not an upstream Menagerie model (Menagerie has no LimX robot).
        # All 31 of Oli's DOFs are present by name; the 13 extra joints are the ankle and
        # waist linkage rods, driven by the model's equality constraints.
        mapping[Oli] = "limx_oli/HU_D04_01.xml"
    except ImportError:
        pass
    return mapping


def menagerie_path_for(robot_options) -> str | None:
    """Return the Menagerie model path for ``robot_options``, or ``None`` if unmapped.

    Walks the robot class MRO so a subclass inherits its parent's mapping.
    """
    mapping = _menagerie_map()
    for ancestor in type(robot_options).__mro__:
        if ancestor in mapping:
            return mapping[ancestor]
    return None
