# MetaMotivo

Run a pretrained [MetaMotivo](https://github.com/facebookresearch/metamotivo) (FBcpr) policy on Eden's SMPL humanoid. The policy produces diverse whole-body behaviors from a learned latent context vector `z`, with no task-specific retraining required.

## Prerequisites

Install the MetaMotivo package:

```bash
pip install metamotivo[huggingface]@git+https://github.com/facebookresearch/metamotivo.git
```

The pretrained model is downloaded automatically from HuggingFace on first run.

## Inference

```bash
python examples/policies/metamotivo/main.py --vis
```

### Goal-directed mode

Infer a context vector from a target pose (crouching) via the model's backward map:

```bash
python examples/policies/metamotivo/main.py --vis --goal
```

### Options

| Flag | Description |
|------|-------------|
| `--vis`, `-v` | Open the viewer |
| `--model-name` | Model variant: `facebook/metamotivo-S-1` (24.5M, default) or `facebook/metamotivo-M-1` (288M) |
| `--goal` | Use goal inference instead of random context sampling |
| `--seed` | Random seed for reproducible context sampling |
| `--num-envs` | Number of parallel environments (default: 1) |
| `--steps` | Simulation steps (default: 300) |
| `--cpu` | Use CPU backend instead of GPU |

## How it works

MetaMotivo uses a Forward-Backward (FBcpr) model that encodes behaviors as latent context vectors. The policy supports three modes:

- **Random context** (`sample_random_context`): sample `z` from the prior to generate diverse motions
- **Goal inference** (`compute_goal_context`): infer `z` from a target proprioceptive observation
- **Tracking** (`compute_tracking_context`): compute per-timestep `z` from a sequence of target observations

Custom terms replicate the original HumEnv training environment in Eden:

- **`MujocoSMPLActuator`** — emulates MuJoCo's affine general actuators with gain, bias, and force clamping
- **`humenv_proprio`** — computes heading-invariant egocentric proprioceptive observations matching HumEnv's encoding

> [!NOTE]
> The SMPL humanoid XML uses a Y-up coordinate frame (`up=(0, 1, 0)`) and PD gains are set to zero because `MujocoSMPLActuator` handles all control forces directly.

## Citation

```bibtex
@article{tirinzoni2024zeroshoot,
  title={Zero-Shot Whole-Body Humanoid Control via Behavioral Foundation Models},
  author={Tirinzoni, Andrea and Touati, Ahmed and Farebrother, Jesse and Guzek, Mateusz and Kanervisto, Anssi and Xu, Yingchen and Lazaric, Alessandro and Pirotta, Matteo},
  journal={arXiv},
  year={2024},
}
```
