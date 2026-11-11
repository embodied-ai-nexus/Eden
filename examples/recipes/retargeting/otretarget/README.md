# OTRetarget

Retarget human motion, hands and manipulated objects included, onto a humanoid with Eden's
`OTRetargeter`, an implementation of *OTRetarget: Joint Robot and Object Motion Retargeting via
Optimal Transport* (Besset, Carn et al., arXiv 2609.36602).

The interaction is read off surfaces rather than skeletons. Dense probes on the human, the robot links
and the objects measure a signed distance, a closest surface point and a direction to every
*channel* (the floor and each object). An entropic optimal-transport plan, solved once per
(human part, robot link group) in a T-pose, carries each robot probe's targets from the human probe it
corresponds to. A per-frame constrained IK then balances those interaction residuals against the
demonstrated style, and solves the robot pose and every object pose together, at native scale. Nothing
is rescaled.

## Prerequisites

```bash
pip install -e ".[ot-retargeter]"        # proxsuite (ProxQP) + threadpoolctl
```

* **OMOMO**: the canonical SMPL-X release (`eden_omomo`: `sequences_canonical/<clip>/{human,object}.npz`
  and `objects/<name>/<name>.obj`) and the SMPL-X model files (`SMPLX_{NEUTRAL,MALE,FEMALE}.npz`).
* **BONES-SEED**: SOMA-rig `.bvh` clips and `SOMA_neutral.npz` from the `nvidia/soma-x` release.

## Usage

```bash
# Hands + body + object, G1 with Wuji hands
python examples/recipes/retargeting/otretarget/retarget_motion.py -c --source omomo \
    --data-dir /path/to/eden_omomo --body-model-path /path/to/smplx \
    --clip sub3_largebox_003 --rig g1_wuji -r

# Body only, LimX Oli
python examples/recipes/retargeting/otretarget/retarget_motion.py -c --source seed \
    --data-dir /path/to/bones_seed_bvh --body-model-path /path/to/SOMA_neutral.npz \
    --clip high_jump_R_001__A277 --rig oli -r
```

| Flag | Description |
|------|-------------|
| `--source` | `omomo` (SMPL-X + object) or `seed` (SOMA BVH) |
| `--data-dir` | `eden_omomo` root, or the folder of `.bvh` files |
| `--body-model-path` | SMPL-X model folder, or `SOMA_neutral.npz` |
| `--clip` | Sequence name (BVH file stem for `seed`) |
| `--rig` | `g1_wuji`, `oli_wuji` (dexterous), `g1`, `oli` (body only) |
| `--num-frames` | Retarget only the first N frames |
| `-o`, `--output-dir` | Output directory (default `results/otretarget`) |
| `-r`, `--record` | Render the demonstration and the robot side by side |
| `-c`, `--cpu` | Genesis CPU backend |

Each run writes `<rig>_<clip>.npz` (`RetargetResult.save`: rig `qpos`, per-entity `qpos`, solved
object poses, per-frame diagnostics) and `<rig>_<clip>_metrics.json`. The metrics follow the paper's
protocol (Sec. IV-A): interaction precision, recall and Jaccard against the demonstration for the
ground and for objects, depth and placement errors, stance clearance, foot skate, penetration, style
orientation error and object drift.

## Using it from code

```python
from eden.extensions.retargeting.otretarget import PRESETS, OTRetargeter, for_body_model
from eden.extensions.retargeting.otretarget.sources import load_omomo_sequence

human, box = load_omomo_sequence(seq_dir, objects_dir, smplx_dir)
retargeter = OTRetargeter(env, OTRetargeter.configure(**PRESETS["g1_wuji"].retargeter_options()))
result = retargeter.retarget_motion(human, [box])     # robot + box solved jointly
env.entities["robot"].set_qpos(result.entity_qpos["robot"][t])
```

A new embodiment needs a `RigPreset`: its T-pose joint values, the human-part ↔ link groups, and the
style links. A new body model needs only a `HumanSurfaceMotion`: posed vertices, faces, skinning
weights and joint frames.

## Structure

`OTRetargeter` (`retargeter.py`) is the method of the paper and nothing else: probes, transport plans,
channels, the interaction and style terms, non-penetration and self-collision, and the trust-region SQP
(`solver.py`). It knows nothing of feet or hands. What one kind of embodiment needs on top is an add-on,
named in the `addons` option with its own options:

| Add-on | For | Does |
|---|---|---|
| `floor_stance` (`stance.py`) | legged robots on a floor | footprint offset, floor support, no-slip stance |
| `hand_grasp` (`grasp.py`) | dexterous hands | grasp closure, palm cold start, side re-approach |

The presets give every rig `floor_stance`, and a rig with palms `hand_grasp`:

```python
OTRetargeter.configure(
    **options,  # the method's options
    addons={"floor_stance": {}, "hand_grasp": {"palms": {...}, "fingertips": {...}, "closure": 0.06}},
)
```

A hand on its own (a floating wrist, no floor) takes `ground=False` and `addons={"hand_grasp": {...}}`. An
add-on subclasses `retargeter.Addon` and overrides the hooks it needs (per-demonstration setup, cold
starts, per-frame targets, terms on a probe set or the whole rig, a stage after each frame's solve).

## Notes

* **Grounding.** Some OMOMO subjects' body fits float above the floor that the object rests on, by up to
  ~5 cm for a whole clip. The method treats the floor as a contact surface at native scale, so it would
  copy that gap onto the robot. The loaders therefore move the human, never the objects, so the feet
  stand on z = 0 (`ground_human=True`).
* **Hands.** The paper retargets to a G1 with fixed hands. Here the Wuji fingers are probed at
  40 000 pts/m², on both the human and the robot side, so every phalanx carries enough samples for its
  own transport plan, and the finger links track the human finger frames in the style term.
* **Thin geometry.** Linearised non-penetration rows on features thinner than the 1 cm distance grid
  (a lamp pole, coat-stand hooks) can conflict. When an iterate's QP has no solution, the solver relaxes
  the smallest set of conflicting rows (an l1 elastic LP) and solves again. `info["qp_relaxed"]` counts
  these iterates.

## Departures from the paper

The paper's formulation is kept; each addition below fixes a failure observed on OMOMO with Wuji hands or
on LimX Oli, and each is an option of `OTRetargeter` or of an add-on (paper behaviour in brackets).

| Option (default) | What it fixes |
|---|---|
| witness residual as the vector `(rho / ‖c‖) c` (always) | The scalar geodesic distance is a cone at its target, so Gauss-Newton overshoots and flips back every iterate. Frames ended wherever the iteration budget cut that 2-cycle, which showed as jitter (root 1.4 cm/frame² against 0.2 for the human pelvis). The cost is the same; frames now converge in ~4 iterates. Its Jacobian has a lateral factor `rho / ‖c‖`, capped at `witness_lateral_cap` (4): across a 1 cm chair back from its target it reached 40, the QP's condition number reached 1e9, ProxQP stopped short and the frame froze. The gradient does not depend on the cap. |
| cold start of the first frame (always) [not specified] | The joints are fitted to the first frame's style orientations, then the robot is moved vertically so that its lowest point is at the demonstrator's lowest point. The constrained solve starts from there. Before, the robot started with straight legs at the pelvis. In a clip that opens in a squat ("pick up the fallen tripod") the legs were 30 cm in the floor, and the first frame ended with the robot kneeling or lying (body orientation 70-100° off). No later frame left that posture. |
| limbs drawn back at the cold start (always) | A larger robot in the demonstrated posture can start with a link inside the object the human held: a G1 forearm between the slats of a chair carried against the chest, a foot in the bottom plate of a kicked box, a hand round a pole. A link inside a thin part cannot get out by local steps: the rows on its two faces point in opposite directions, so it stayed inside for the whole clip, every QP was relaxed and the object moved up to 20 cm off its path. Each limb (a leg, an arm up to the palm, a finger: a chain between two branchings of the link tree) with a probe more than 2 mm inside a free object is blended back toward the T-pose until it is clear, and the cold iterations bring it to the contact from outside. The other limbs keep the demonstrated posture, and the object does not move: moving the object off the robot instead lost which side of the object each hand belonged on. |
| `floor_stance`: `footprint_offset=True` [off], `footprint_contact=0.02` | Robot floor witnesses are shifted sideways by the T-pose probe offset, in the demonstrated heading frame. The floor is translation invariant, and Oli's hips are twice as wide as a human's; on the human footprints its knees knocked in and its pelvis swung over each stance foot. The shift is lateral only (the fore-aft part was 10 cm on the SOMA rig, whose foot joint sits at the heel). A probe in floor contact keeps the shift it landed with: turned with the pelvis every frame, it slid the stance foot of a spin round a circle (19–21 cm/s skate on a chaînés turn, now 1 cm/s). |
| `floor_stance`: `support_height=0.02` [none] | Floor support is a constraint: while the human's sole is within 2 cm of the floor, the robot unit's lowest downward-facing point stays within 5 mm of the human sole's height. In the cost, normalised per link, two soles weigh as much as two of the ~40 hand links, so a G1 holding a lamp where the human held it floated 3.7 cm through every stance. A 1 cm band missed the many OMOMO stances whose fitted foot rests 1–2 cm up, and the G1 foot hovered 3–5 cm above the floor in them. |
| `floor_stance`: `stick=True` [none], `stick_slack`, `stick_height`, `stick_release`, `stick_anchors` | No-slip stance. While a foot's human sources touch the floor on two consecutive frames, the human foot's planar rigid motion is applied to the robot foot, about the robot's own counterpart of the human contact. Three anchors on the part of the robot foot that is down (within 5 mm of its lowest point) must follow that motion (a constraint), and the foot's floor witnesses move with it. After lift-off the footprint targets return over the first 10 cm of the human foot's travel. The floor witnesses alone are soft: a planted foot slid up to 7–12 cm per stance phase on OMOMO. Each design choice fixes a failure we saw: a "pin while slow" rule released the foot at every heel-off and the foot then snapped up to 4 cm; rotating about the human's contact made the robot foot orbit it in spins; and anchoring a heel that was already rising stopped the foot from rolling (20° ankle error at a jump's push-off). |
| Wuji presets: `joint_limits` on finger and thumb flexion, -0.25 rad backwards [the hand's 60°] | The contact terms pressed fingers back onto objects through the hand's whole backward range: on 40% of OMOMO runs (both robots), a finger was bent back more than 20° in over a fifth of the frames. The demonstrators' fingers bend back 14° at most (1st percentile; the extreme is 28°). The retargeted motion now keeps to that range. `joint_limits` narrows any joint's range, never widens it. |
| Wuji presets: the human palm matches the Wuji palm link only [one group: forearm-end wrist link, adapter mount, palm] | Each transport plan scales its two clouds to unit size. With the forearm-end wrist link and the adapter mount in the palm's group, the 10 cm human palm stretched over 17 cm of robot: the mount took its targets from the palm centre, and the solver held poles on the mount with the fingers closed on air (Oli: 13% of grasp frames a quarter palm nearer the wrist than the human's). Those links now join the forearm group. On 61 OMOMO clips per robot: object at the wrist 5.5% -> 3.2% (Oli), 2.5% -> 0.0% (G1). |
| `hand_grasp`: `closure=0.06`, `closure_weight=4000` (Wuji presets) [none] | The OMOMO hands hold poles loosely, with the pole sunk 1-1.4 cm into the palm and the finger pads 2-4 cm off, and the Wuji thumb cannot copy the human thumb's base rotation. The copied posture left the robot fingers open and the thumb beside them on a pole. After each frame's solve, a stage that moves only the hand joints pulls the object-facing probes of each finger whose human pad is within 6 cm of the object onto the object points nearest their human sources, while the hand grasps (human palm, or two pads, within 1 cm) and its palm faces the object. The next frame starts from the open-hand solution: closed in the main solve, the fingers pushed the arm and the object (a pole slid along the palm). Contacts reaching half way round a thin part: 26-30% -> 59% of frames (human 100%); robot thumb on the part where the human thumb is opposite the palm: 53-57% -> 75%. |
| `hand_grasp`: `cold_start_palm_weight=1000` (Wuji presets) [one cold start] | Fitted to joint orientations alone, a robot hand of another reach started a few centimetres off the human's place relative to the object; on a 2-3 cm pole that is the far side, and the hand closed there for the whole clip (G1 poles: 119 of 141 wrong-side episodes start at frame 0). The option adds a second cold start with each touching palm pulled to the human palm's place. Frame 0 is solved from both and keeps the one whose palms hold the object on the human palms' side (the object's material near the palm, along the palm normal, `rest_palm_normal`), then the one without relaxed QPs, then the lower cost. The cost alone cannot choose: the human palms sink into poles, and a lamp pole behind the hand cost less (7.3) than in front (10.9). Used alone, the pull put an arm through a chair (body error 9.5 -> 24.7°). Object behind the hand: 12.1% -> 5.5% (G1), 6.4% -> 3.9% (Oli). |
| `hand_grasp`: `side_reapproach=True` (Wuji presets) [none] | A pole that met the curled robot fingertips slid round them onto the back of the hand (the source fingers sink into the pole, so the human's slides into the palm), and the robot held it there for the rest of the grasp: a local step cannot carry a hand round a part. When the human palm holds one thin part clearly on one side (palm centre outside it, the part's material at least 5 mm in front of or behind the palm on two frames in a row) and the robot palm holds it more than 1 cm on the other, that arm and hand alone are re-solved from a start at the human palm's place relative to the object, with a palm goal across the palm, then the earlier frames backwards until the hand is 5 cm clear, joined to the first solution at most 0.08 rad per frame per joint; the goal then fades over 15 frames. A correction is dropped if the side is still wrong, the hand never comes clear in 45 frames, the join passes through the object, or a frame goes deeper than 2.5 mm. The side reads the part's material near the palm, not its nearest point: a palm sunk into a pole read the pole as behind it and a correct hand was sent round. Human hands that pass through the part (a third of these episodes) are left as they are. On 65 OMOMO clips with such hands: wrong-side contact frames 0.28 -> 0.26 (G1), 0.31 -> 0.26 (Oli); 13 corrections, all checked in renders; the 61 validation clips do not change beyond run-to-run noise. |
| palm-frame fingertip and pinch terms, sunken-witness facing [removed] | Tested on the thumb problem and removed. Palm-frame fingertip and DexPilot-style pinch terms put the tips within 1 cm of the human tips but did not change the thumb's abduction. Reading a sunken source's target on the face its skin looks at (as video-to-data pipelines push keypoints out of objects) raised the hands behind objects on Oli (5.8% -> 7.3%). |
| `ot_normal_weight=0.5` [0] | Transport cost `‖x−y‖² + 0.5 ‖n_x−n_y‖²`. With position alone, a quarter of a Wuji phalanx's probes took their targets from the back of the human finger and held the finger off the object. |
| area-weighted transport marginals (always) | Groups that mix probe densities (wrist links and palm) stay area-balanced. |
| `style_link_weights`: Wuji phalanges ×0.3 (preset) | An angle error costs the same on a 3 cm phalanx as on a thigh. At full weight the demonstrated finger curl outweighed palm contact, and the fingertips stopped on the pole the human held in the palm. Much lower (×0.1) and released fingers stayed curled round a chair the human had let go of. |
| Wuji presets: `interaction_weights=(512, 625, 2500)` [(256, 625, 2500)] | A Wuji hand is larger than a human hand, so round a thin object the human palm's placement and contact-plane targets cannot all be met. At the paper's balance they held the hand 2–4 cm off a 1.7 cm tripod leg. Over 36 OMOMO runs, twice the depth weight raised hand Jaccard from 0.74 to 0.80 (G1) and from 0.71 to 0.78 (Oli), and cut the hand depth error from 6.6 to 5.1 mm and from 7.8 to 5.6 mm. Placement error, penetration and posture did not change. |
| `collision_activation=0.15` [d̄ = 15 cm] | Non-penetration rows on every probe-channel pair within d̄, as in the paper. At 5 cm, one 5 cm trust step could carry a probe with no row through a surface. |
| object solids, `object_solid_voxel=0.004` (only where the scan's sign is wrong) | The sign comes from the generalised winding number. On the OMOMO trashcan (an open bin) it read the cavity as solid and flipped across the opening (a 10 cm jump between neighbouring points), and the plastic box's walls are zero-thickness sheets with no inside. Such a scan is closed into a solid before its field is baked; the solid's surface lies one voxel outside the scan. A scan is kept as it is when, with duplicate vertices merged, it is closed and the winding number reads inside and outside correctly at 98% of its surface: 9 of the 13 OMOMO objects. The OMOMO files store each triangle with its own vertices, so `is_watertight` reads every scan as open; when they were all closed into solids, a robot hand in contact stayed 4 mm off the tripod, the boxes and the tables. |
| step acceptance (always) | A step is rejected if it drives a probe deeper than it was (per probe, not the global maximum), with a cutting plane per such probe; a rejected step is re-solved in a box half its own size, DoF by DoF; a frame whose steps were all rejected takes the least deep one, unless its last re-solve converged (the cutting planes showed that staying put is optimal). The paper takes every step; its 6-iterate budget then cut a cycle wherever it was, and a box shrunk as a whole left a step inside it unchanged, so the robot froze for 30-100 frames. |
| `witness_lateral_cap=4` [1, the Euclidean value] | Curvature model of the geodesic witness residual. At 1 the model under-states the curvature where the geodesic runs round an edge, steps overshoot and the iterates cycle (finger steps flip sign). 4 fixed a chair clip's jitter (8.3× -> 1.9× the human's) with no clip worse. |
| face-aware geodesic table (always) | The table connects each sample to its nearest samples of the same facing, and each end of a distance query interpolates the samples of its own face only (the caller passes the outward normals). On parts thinner than the sample spacing, a point on a chair back blended both faces: the gradient of the distance reached 6e4 instead of ~1, and paths across a face ran up to 10× the straight line. The end normals that decide between the straight segment and the graph distance are softened (`soften=0.05`): at the narrow end of a thin slat the interpolated normal of its two opposite sides nearly cancels, and dividing by its length gave gradients up to 8e5. |
| `exact_band=0.01`, `geodesic_samples=4000`, finer rows on small links | Inside a 1.7 cm tripod leg the 1 cm grid reads a probe 4 mm deep as touching. At 1000 samples/m², the geodesic table cannot tell a finger on the near side of a thin leg from one on the far side. |

## Citation

```bibtex
@article{besset2026otretarget,
  title   = {OTRetarget: Joint Robot and Object Motion Retargeting via Optimal Transport},
  author  = {Besset, Guillaume and Carn, Erwann and Carecchio, Timoth{\'e}e and Tordjman--Levavasseur, Valentin
             and Schramm, Fabian and de Mont-Marin, Yann and Carpentier, Justin and Sathya, Ajay Suresha},
  journal = {arXiv preprint arXiv:2609.36602},
  year    = {2026}
}
```
