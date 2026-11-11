# OmniRetarget

Retarget human motion from the OMOMO dataset to a Unitree G1 humanoid using Eden's `OmniRetargeter`.

Two variants are provided:

- **`retarget_robot_only.py`** — Retarget body motion only, ignoring interaction objects.
- **`retarget_object_interaction.py`** — Retarget body motion together with object poses (non-penetration constraints enabled).

## Prerequisites

Download the [OMOMO](https://github.com/leolyliu/OMOMO-Benchmark) dataset (Holosoma format). The data directory should contain `.pt` files:

```
/path/to/omomo_holosoma/
  sub3_largebox_003.pt
  sub3_smallbox_001.pt
  ...
```

## Usage

### Robot-only retargeting

```bash
python examples/recipes/retargeting/omniretarget/retarget_robot_only.py \
    --data-path /path/to/OMOMO_new \
    --task-name sub3_largebox_003 \
    --num-frames 10
```

### Object interaction retargeting

```bash
python examples/recipes/retargeting/omniretarget/retarget_object_interaction.py \
    --data-path /path/to/OMOMO_new \
    --task-name sub3_largebox_003 \
    --num-frames 10
```

### Options

| Flag | Description |
|------|-------------|
| `--data-path` | **(required)** Path to OMOMO directory containing `.pt` files |
| `--task-name` | Sequence name (default: `sub3_largebox_003`) |
| `--num-frames` | Truncate to N frames for quick testing (default: all) |
| `--output-dir` | Output directory (default: `results/omniretarget`) |
| `--record` | Record video of the retargeted motion |

## How it works

1. Load human motion (SMPL-H 52-joint skeleton) from the OMOMO Holosoma dataset
2. Scale the human skeleton to match the G1 robot height
3. Solve per-frame IK via `OmniRetargeter` with:
   - Laplacian mesh matching (SMPL-H joints to G1 links)
   - Joint limit constraints (with manual overrides for wrist/waist)
   - Foot sticking constraints (velocity-based contact detection)
   - Ground contact constraints
   - Object non-penetration (object interaction variant only)
4. Output retargeted `qpos` sequence for the G1 robot

## Citation

```bibtex
@ARTICLE{2025arXiv250926633Y,
       author = {{Yang}, Lujie and {Huang}, Xiaoyu and {Wu}, Zhen and {Kanazawa}, Angjoo and {Abbeel}, Pieter and {Sferrazza}, Carmelo and {Liu}, C. Karen and {Duan}, Rocky and {Shi}, Guanya},
        title = "{OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction}",
         year = 2025,
          doi = {10.48550/arXiv.2509.26633},
archivePrefix = {arXiv},
 primaryClass = {cs.RO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250926633Y},
}
```
