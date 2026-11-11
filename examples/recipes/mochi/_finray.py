"""Helpers shared by the two Schunk WSG-50 FinRay recipes (not a runnable example).

The model is the FinRay gripper of the LBM eval models
(https://huggingface.co/datasets/toyota-research-institute/lbm_eval_models): an articulated URDF plus, per
finger, a tetrahedral ``.vtk`` simulation mesh and a detailed ``.gltf`` visual mesh. Eden does not ship it,
so both recipes take ``--asset-dir`` (or ``$LBM_EVAL_MODELS``) and skip with a message when it is absent.
"""

import argparse
import os
import xml.etree.ElementTree as ET

import genesis.utils.geom as gu
import numpy as np

from eden.utils.procedural_meshes import derived_file_path, write_atomically

DEFAULT_ASSET_DIR = os.path.join(
    os.environ.get("LBM_EVAL_MODELS", os.path.expanduser("~/datasets/lbm_eval_models")),
    "robots",
    "schunk_grippers",
)
URDF_NAME = "schunk_wsg_50_finray_fr3_mount.urdf"
FINGER_LINKS = ("finray_finger_L", "finray_finger_R")
# The gripper's two prismatic joints, in URDF declaration order. They must be declared as ``dofs_name`` on
# the entity: Eden resolves ``dofs_idx_local`` (and every per-DOF default) by name, so an articulated entity
# that names no DOF gets an *empty* index tensor, and every ``control_dofs_*`` call on it silently addresses
# nothing. Only DRIVE_DOF is driven — the URDF mimic coupling carries the second finger.
GRIPPER_DOFS = ["finger_joint_1", "finger_joint_2"]
DRIVE_DOF = "finger_joint_1"
FINGER_MESH = "finray_finger_MA-049-PT-0005_MA-049-PT-0006_2024_07_23_{side}_low.vtk"
FINGER_VISUAL_MESH = "finray_finger_MA-049-PT-0005_MA-049-PT-0006_2024_07_23_{side}.gltf"
# Base-color map of the pads, the green the FinRay fingers are printed in. Every gltf here points
# its material at a `.ktx2` through `KHR_texture_basisu` and carries the plain `.png` as that
# texture's own `source` fallback; neither Genesis nor Nyx reads the supercompressed form, and both
# drop the texture rather than take the fallback (Genesis says so, "Ignoring texture"; Nyx renders
# the mesh black). A renderer that needs the colour stated outright takes this path.
FINGER_COLOR_TEXTURE = "finray_finger_MA-049-PT-0005_MA-049-PT-0006_2024_07_23_color.png"
# Recentered prismatic coordinates: q = 0 is the open gripper (+-0.045 m from the nominal finger pose),
# positive q closes. The mimic coupling q2 = -q1 keeps the fingers symmetric in the shifted coordinates too.
OPEN_SHIFT = 0.045
# Vertices within 4 mm of the finger's mount plane (link-frame z = 0) are tied to the finger link.
MOUNT_PLANE_DEPTH = 0.004


def add_asset_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--asset-dir",
        type=str,
        default=DEFAULT_ASSET_DIR,
        help=f"Directory holding {URDF_NAME} and its assets/ (default: $LBM_EVAL_MODELS/robots/schunk_grippers)",
    )


def resolve_urdf(asset_dir: str) -> str | None:
    """The gripper URDF, or None (after printing why) when the LBM eval models are not installed."""
    urdf_path = os.path.join(asset_dir, URDF_NAME)
    if not os.path.exists(urdf_path):
        print(
            f"Skipped: '{urdf_path}' not found. Download the LBM eval models "
            f"(https://huggingface.co/datasets/toyota-research-institute/lbm_eval_models) and pass --asset-dir "
            f"or set $LBM_EVAL_MODELS."
        )
        return None
    return urdf_path


def finger_mesh(asset_dir: str, side: str) -> str:
    return os.path.join(asset_dir, "assets", FINGER_MESH.format(side=side))


def finger_visual_mesh(asset_dir: str, side: str) -> str:
    return os.path.join(asset_dir, "assets", FINGER_VISUAL_MESH.format(side=side))


def finger_color_texture(asset_dir: str) -> str:
    """Base-color map of both FinRay pads (one file, shared by L and R)."""
    return os.path.join(asset_dir, "assets", FINGER_COLOR_TEXTURE)


def trim_urdf(urdf_path: str) -> str:
    """Write a gripper URDF whose fingers are left to the deformable bodies, and return its path.

    Strips the finger geometry (simulated as ``MochiElastic`` bodies instead), shrinks the now-empty finger
    link inertia, recenters the finger joints so that q = 0 is the open gripper, and absolutizes the
    remaining mesh paths so the copy can live in Eden's cache (one file per source URDF, rewritten each call).
    """
    ET.register_namespace("drake", "http://drake.mit.edu")
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.iter("link"):
        if link.get("name") not in FINGER_LINKS:
            continue
        for tag in ("visual", "collision"):
            for element in link.findall(tag):
                link.remove(element)
        inertial = link.find("inertial")
        inertial.find("mass").set("value", "0.001")
        inertia = inertial.find("inertia")
        for axis in ("ixx", "iyy", "izz"):
            inertia.set(axis, "1e-7")
        for axis in ("ixy", "ixz", "iyz"):
            inertia.set(axis, "0")
    for joint, shift, lower, upper in (
        ("finger_joint_1", -OPEN_SHIFT, -0.01, OPEN_SHIFT),
        ("finger_joint_2", OPEN_SHIFT, -OPEN_SHIFT, 0.01),
    ):
        element = next(j for j in root.iter("joint") if j.get("name") == joint)
        origin = element.find("origin")
        xyz = [float(v) for v in origin.get("xyz").split()]
        xyz[0] += shift
        origin.set("xyz", " ".join(repr(v) for v in xyz))
        limit = element.find("limit")
        limit.set("lower", repr(lower))
        limit.set("upper", repr(upper))
    for mesh in root.iter("mesh"):
        mesh.set("filename", os.path.join(os.path.dirname(urdf_path), mesh.get("filename")))
    out_path = derived_file_path("mochi_assets", urdf_path, "finray_gripper.urdf")
    write_atomically(out_path, ET.tostring(root, encoding="unicode"))
    return out_path


def link_neutral_pose(link) -> tuple[np.ndarray, np.ndarray]:
    """World pose a parsed link takes at build time, composed from the parse-time link frames.

    A deformable finger must be placed where the link it attaches to stands when the scene is built (an
    attachment anchors each vertex in that frame), and the solver only reports link poses after the build.
    Every joint coordinate of these models starts at zero, so composing the link descriptions is exact.
    """
    pos, quat = gu.zero_pos(), gu.identity_quat()
    while link is not None:
        pos, quat = gu.transform_pos_quat_by_trans_quat(pos, quat, link.desc.pos, link.desc.quat)
        link = link.solver.links[link.parent_idx] if link.parent_idx >= 0 else None
    return pos, quat


def mount_finger(env, gripper, finger, link_name: str, mesh_path: str, visual_mesh_path: str) -> np.ndarray:
    """Place a FinRay finger on its gripper link and tie its mount plane to it. Pre-build only.

    Returns the finger's rest geometry in the link frame, which the caller needs to carry the vertices along
    when the arm is moved to a start pose after the build.
    """
    from eden.engine.solvers.mochi.mochi_soft_entity import load_vtk_tet_files

    verts_local = load_vtk_tet_files(mesh_path)[0]  # the collision mesh is authored in the link frame
    link = gripper.get_link(link_name)
    link_pos, link_quat = link_neutral_pose(link)
    # The model's detailed visual mesh, skinned by the simulation tetrahedra; the URDF's visual tag rotates
    # it by 90 degrees about x into the frame of the collision (simulation) mesh. It is embedded *before*
    # the body is moved: the render mesh is placed in the morph frame and then bound to the tetrahedra by
    # barycentric coordinates, which carry it along when `set_rest_positions` moves the rest geometry.
    finger.set_visual_mesh(file=visual_mesh_path, euler=(90.0, 0.0, 0.0))
    finger.set_rest_positions(gu.transform_by_trans_quat(verts_local, link_pos, link_quat))
    finger.attach_to_link(
        gripper, link_name, np.where(verts_local[:, 2] < MOUNT_PLANE_DEPTH)[0], stiffness=1e6, damping=1.0
    )
    return verts_local
