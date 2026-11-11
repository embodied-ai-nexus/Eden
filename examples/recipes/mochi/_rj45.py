"""The RJ45 connector scene shared by the two ``contacts_rj45_plug`` recipes (not a runnable example).

Geometry, poses, contact tuning and the insertion drive live here; the recipes differ only in how they
are rendered — a rasterized camera, or a Nyx path tracer. The assets are converted from Newton's
``rj45_plug.usd`` by ``convert_rj45_asset.py`` and hosted as ``objects/rj45`` of the ``Kashu7100/eden_assets``
Hub dataset (see its README).
"""

import tempfile

import numpy as np
import torch
import trimesh
from _mochi_common import MOCHI_GRAVITY, write_text

from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import EnvBase
from eden.options import EnvOptions, RodOptions, SceneOptions
from eden.options.entities import EntityOptions, PlaneOptions
from eden.options.materials import MochiRigidMaterialOptions, MochiRodMaterialOptions
from eden.utils.assets import get_asset_path

# The repo directory of the converted assets in the Hub dataset; `get_asset_path` fetches each file on first use.
ASSETS = "objects/rj45"

# Prim origins of Newton's USD stage (metres, z up): where each mesh file's own frame sits in the world.
# The stage is authored in the *mated* state, so these are the poses at full insertion.
SOCKET_POS = (0.0, 0.0, 0.01165375)
PLUG_POS = (0.0, -0.00371498, 0.01085)
LATCH_POS = (0.0, -0.00651222, 0.01487787)
# Newton's PLUG_Y_OFFSET: the plug (and the cable that leaves it) start this far back along the insertion axis.
PLUG_Y_OFFSET = -0.025

CABLE_RADIUS = 0.00325
CABLE_FRICTION = 2.0  # Newton's CABLE_MU
# Self-contact, and the two numbers it turns on. A rod's collider is a sphere per node and a contact is a
# sample *point* against one of them, so a pair closes at `CABLE_RADIUS + penalty_threshold` -- one radius
# short of where two tubes actually meet, which is why the threshold is nearly a radius rather than the
# 1 mm default. What caps it is the exclusion rule: nodes within
# `CABLE_RADIUS * exclusion_ratio + penalty_threshold` of each other *in the rest pose* never collide, and
# the ratio must exceed 1, so the exclusion radius always covers the closing distance. Newton's centerline
# already passes within 6.31 mm of itself (node to node), so the exclusion has to stay under that or it
# swallows the very crossing this is meant to resolve -- which is what the obvious choice of a full radius
# at the default ratio does.
CABLE_SELF_CONTACT_THRESHOLD = 2.9e-3
CABLE_SELF_CONTACT_EXCLUSION_RATIO = 1.03

# Newton's latch tuning, unchanged: a return spring and a soft travel limit on the hinge.
LATCH_LIMIT = (-0.2, 0.3)  # inward / outward deflection [rad]
LATCH_SPRING_KE = 0.15  # [N*m/rad]
LATCH_SPRING_KD = 0.03  # [N*m*s/rad]
# The tab is ~5 mm^2; at the plug's stiffness the return spring pushes it 0.6 mm *through* the socket rather
# than swinging the latch (0.009 rad of deflection instead of 0.155). Against a static collider the pair
# stiffness is the sample side's alone, so stiffening this one link leaves the plug on its 0.1 mm clearance.
LATCH_PENALTY_COEFFICIENT = 1e11

# Contact at the scale of the latch: the penalty ramp must live inside the plug's 0.1 mm clearance, and the
# socket's distance field must resolve the 1 mm latch window.
CONNECTOR_CONTACT = {
    "friction": 0.1,
    "penalty_coefficient": 1e9,
    "penalty_threshold": 1e-5,
    "penalty_smoothing_half_distance": 2e-5,
}

# Segments this many apart along the rod or more count as a crossing rather than the rod's own arc.
STRAND_SKIP = 2

INSERT_STEPS = 180  # steps spent travelling the 25 mm
PLUG_KP, PLUG_KD = 4e4, 4e2  # position drive of the three prismatic joints [N/m], [N*s/m]

CONNECTOR_XML = """<mujoco model="rj45">
  <compiler angle="radian" autolimits="true"/>
  <asset>
    <mesh name="plug" file="{plug}"/>
    <mesh name="latch" file="{latch}"/>
  </asset>
  <worldbody>
    <body name="plug" pos="{plug_pos}">
      <joint name="plug_x" type="slide" axis="1 0 0"/>
      <joint name="plug_y" type="slide" axis="0 1 0"/>
      <joint name="plug_z" type="slide" axis="0 0 1"/>
      <geom type="mesh" mesh="plug" mass="0.005"/>
      <body name="latch" pos="{latch_pos}">
        <joint name="latch_hinge" type="hinge" axis="-1 0 0" range="{lo} {hi}"
               stiffness="{ke}" damping="{kd}"/>
        <geom type="mesh" mesh="latch" mass="0.0002"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def strand_gap(verts: torch.Tensor) -> float:
    """Closest approach (m) between two stretches of the rod that are not neighbours along it.

    Segment to segment, not node to node: the tube is what has to stay apart, and at Newton's 9.76 mm
    nodes a node-wise reading misses a crossing that passes between them. Two tubes touch at
    ``2 * CABLE_RADIUS``. Validated against a brute-force sampling of every segment pair.

    Parameters
    ----------
    verts : torch.Tensor
        The rod's ``(n_nodes, 3)`` node positions.

    Returns
    -------
    float
        The smallest segment-to-segment distance, in metres.
    """
    tail, head = verts[:-1], verts[1:]
    direction = head - tail
    i, j = torch.triu_indices(len(tail), len(tail), offset=STRAND_SKIP, device=verts.device)
    offset = tail[i] - tail[j]
    aa = (direction[i] * direction[i]).sum(-1)
    ab = (direction[i] * direction[j]).sum(-1)
    bb = (direction[j] * direction[j]).sum(-1)
    ar = (direction[i] * offset).sum(-1)
    br = (direction[j] * offset).sum(-1)
    s = torch.zeros_like(aa)
    # Alternate the two clamped projections to convergence: the segments are short and nearly straight,
    # so a few sweeps land on the closest pair of points.
    for _ in range(32):
        t = ((ab * s + br) / bb).clamp(0.0, 1.0)
        s = ((ab * t - ar) / aa).clamp(0.0, 1.0)
    closest = (tail[i] + s.unsqueeze(-1) * direction[i]) - (tail[j] + t.unsqueeze(-1) * direction[j])
    return float(closest.norm(dim=-1).min())


def cable_metrics(cable) -> str:
    """How low the cable lies and how close two of its own strands come, as a step-line fragment.

    Parameters
    ----------
    cable : eden.entities.mochi.MochiSoftEntity
        The rod entity.

    Returns
    -------
    str
        A formatted fragment for the step line.
    """
    verts = cable.get_vertex_positions()[0]
    return f"cable lowest z={float(verts[:, 2].min()) * 1e3:5.2f} mm strand gap={strand_gap(verts) * 1e3:5.2f} mm"


def nodes_inside_plug(points: np.ndarray, plug_pos) -> np.ndarray:
    """Indices of the centerline nodes that start inside the plug shell, to be tied to it.

    Newton hard-codes the first four bodies of its capsule chain and filters them against the connector
    by hand; read from the plug's own extent, the same set falls out and survives a change of cable
    resolution. The bound is the plug mesh's axis-aligned box in its start pose, which the source
    centerline enters exactly once, at its plug end.

    Parameters
    ----------
    points : numpy.ndarray
        The ``(n, 3)`` centerline, in world coordinates.
    plug_pos : array-like[float, float, float]
        World position of the plug mesh's own frame.

    Returns
    -------
    numpy.ndarray
        Indices into ``points``.
    """
    verts = trimesh.load(get_asset_path(f"{ASSETS}/plug.obj"), process=False).vertices + np.asarray(plug_pos)
    inside = ((points >= verts.min(axis=0)) & (points <= verts.max(axis=0))).all(axis=1)
    return np.flatnonzero(inside)


def scene_options(*, with_cable: bool, surfaces: dict, directory: str) -> tuple[SceneOptions, np.ndarray]:
    """The whole scene: the jointed connector, the fixed socket, the ground and (optionally) the cable.

    Parameters
    ----------
    with_cable : bool
        Include the cable rod.
    surfaces : dict
        ``gs.surfaces`` object per entity name (``connector``, ``socket``, ``cable``, ``plane``); a missing
        name takes the Genesis default.
    directory : str
        Directory the connector MJCF is written to. It must still exist at ``env.build()``: Genesis
        parses the model there.

    Returns
    -------
    tuple[SceneOptions, numpy.ndarray]
        The scene options, and the cable node indices to tie to the plug (empty without a cable).
    """
    cable_points = np.load(get_asset_path(f"{ASSETS}/cable_centerline.npy")) + (0.0, PLUG_Y_OFFSET, 0.0)
    plug_pos = (PLUG_POS[0], PLUG_POS[1] + PLUG_Y_OFFSET, PLUG_POS[2])
    anchors = nodes_inside_plug(cable_points, plug_pos) if with_cable else np.zeros(0, dtype=int)

    xml = CONNECTOR_XML.format(
        plug=get_asset_path(f"{ASSETS}/plug.obj"),
        latch=get_asset_path(f"{ASSETS}/latch.obj"),
        plug_pos=" ".join(f"{v:.9g}" for v in plug_pos),
        latch_pos=" ".join(f"{b - a:.9g}" for a, b in zip(PLUG_POS, LATCH_POS)),
        lo=LATCH_LIMIT[0],
        hi=LATCH_LIMIT[1],
        ke=LATCH_SPRING_KE,
        kd=LATCH_SPRING_KD,
    )
    entities = {
        "connector": EntityOptions(
            file=write_text(directory, "rj45.xml", xml),
            is_articulated=True,
            # The plug body hangs off the world by its three slides, so the base carries no DOFs of its
            # own. Eden reads `is_fixed_base` (not the model) to size `qs_idx_local`, and without it
            # `get_qpos` indexes seven phantom root qs past the entity's range. Genesis warns that it
            # cannot forward the flag to `gs.morphs.MJCF`; harmless here, since its parser only adds a
            # free joint to a body that declares none.
            is_fixed_base=True,
            dofs_name=["plug_x", "plug_y", "plug_z", "latch_hinge"],
            default_armature=0.0,  # Newton's model declares none; keep it zero, not the 0.1 Genesis and Eden fill in
            decimate=False,
            convexify=False,
            # The socket is fixed, so it is a collider only and never queries the connector; the cable
            # does, through its node spheres, which is the pair Newton filters by hand.
            material=MochiRigidMaterialOptions(rho=1200.0, collider_type="none", **CONNECTOR_CONTACT),
            surface=surfaces.get("connector"),
        ),
        # A rod collides through samples on its *centerline*, with no cross-section offset, so a cable
        # laid on the ground would rest with its tube half buried. The contact threshold comes from the
        # collider side of a pair, so giving the plane a threshold of one rod radius (over a crisp ramp)
        # lets the cable rest on its own surface. Nothing else touches the ground: the socket is fixed
        # 5 mm above it and the plug never descends. `friction` is Newton's CABLE_MU, which combines
        # with the cable's as a geometric mean.
        "plane": PlaneOptions(
            material=MochiRigidMaterialOptions(
                friction=CABLE_FRICTION,
                penalty_threshold=CABLE_RADIUS,
                penalty_smoothing_half_distance=2e-4,
            ),
            surface=surfaces.get("plane"),
        ),
        "socket": EntityOptions(
            file=get_asset_path(f"{ASSETS}/socket.obj"),
            default_root_pos=SOCKET_POS,
            is_fixed_base=True,
            decimate=False,
            convexify=False,
            material=MochiRigidMaterialOptions(rho=1200.0, sdf_cell_size=1e-4, sdf_max_res=512, **CONNECTOR_CONTACT),
            surface=surfaces.get("socket"),
        ),
    }
    if with_cable:
        entities["cable"] = RodOptions(
            points=cable_points,
            radius=CABLE_RADIUS,
            material=MochiRodMaterialOptions(
                E=2e6,
                nu=0.3,
                rho=1400.0,
                mass_damping=1.0,
                friction=CABLE_FRICTION,
                self_contact=True,
                penalty_threshold=CABLE_SELF_CONTACT_THRESHOLD,
                self_contact_exclusion_ratio=CABLE_SELF_CONTACT_EXCLUSION_RATIO,
            ),
            surface=surfaces.get("cable"),
        )
    return SceneOptions(**entities), anchors


def build(*, with_cable: bool, surfaces: dict, cameras_options=None, show_viewer: bool = False) -> EnvBase:
    """Construct and build the scene, and set up the plug's position drive.

    Parameters
    ----------
    with_cable : bool
        Include the cable rod.
    surfaces : dict
        ``gs.surfaces`` object per entity name; see :func:`scene_options`.
    cameras_options : eden.options.CamerasOptions, optional
        Cameras to add. Defaults to None.
    show_viewer : bool, optional
        Open the interactive viewer. Defaults to False.

    Returns
    -------
    eden.envs.base.EnvBase
        The built environment.
    """
    with tempfile.TemporaryDirectory(prefix="eden_mochi_rj45_") as directory:
        options, anchors = scene_options(with_cable=with_cable, surfaces=surfaces, directory=directory)
        env = EnvBase(
            env_options=EnvOptions(
                num_envs=1,
                sim_dt=1.0 / 60.0,
                mochi_options=MochiOptions(gravity=MOCHI_GRAVITY, n_newton_iterations=12),
            ),
            scene_options=options,
            cameras_options=cameras_options,
            show_viewer=show_viewer,
        )
        connector = env.entities["connector"]
        if with_cable:
            cable = env.entities["cable"]
            # Tied inside the implicit solve, so the cable pulls back on the plug -- Newton's kinematic
            # prefix is one-way. Pre-build, from a hook: the anchors are frozen at build time.
            env.register_pre_build_hook(lambda: cable.attach_to_link(connector, "plug", anchors, stiffness=1e5))
        env.build()  # the XML must still exist here: Genesis parses it at build

    _, slide_dofs = connector.find_named_dofs_idx_local(["plug_x", "plug_y", "plug_z"], preserve_order=True)
    # MochiOptions.batch_dofs_info defaults to False, so the gains are one row shared by every environment.
    connector.set_dofs_kp(torch.full((3,), PLUG_KP, device=env.device), dofs_idx_local=slide_dofs)
    connector.set_dofs_kd(torch.full((3,), PLUG_KD, device=env.device), dofs_idx_local=slide_dofs)
    connector.set_contact_params(links=["latch"], penalty_coefficient=LATCH_PENALTY_COEFFICIENT)
    if with_cable:
        env.entities["cable"].set_vertices_fixed([env.entities["cable"].n_vertices - 1])  # far end clamped
    env.slide_dofs = slide_dofs
    return env


def drive(env, i_step: int) -> None:
    """Command the plug's position one step along the insertion ramp.

    Parameters
    ----------
    env : eden.envs.base.EnvBase
        The environment from :func:`build`.
    i_step : int
        Index of the step about to be taken.
    """
    travel = -PLUG_Y_OFFSET * min(i_step / INSERT_STEPS, 1.0)
    target = torch.tensor([[0.0, travel, 0.0]], device=env.device)
    env.entities["connector"].control_dofs_pos(target, dofs_idx_local=env.slide_dofs)
