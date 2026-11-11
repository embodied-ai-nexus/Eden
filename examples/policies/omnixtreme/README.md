# OmniXtreme: OmniXtreme: Breaking the Generality Barrier in High-Dynamic Humanoid Control

This is an inference code of OmniXtreme — a dual-policy (flow-matching base + residual correction) system for whole-body motion tracking on the Unitree G1 robot (29 DOFs).

## Policy Inference

```bash
python main.py --motion-file /path/to/motion.npz --vis
```

Motion NPZ files can be generated using the BeyondMimic `csv_to_npz.py` script, or a sample is available at `examples/projects/mimic/beyondmimic/sample/G1_gangnam_style_V01_bvh_60hz.npz`.

### Options

| Flag | Description |
|------|-------------|
| `--motion-file` | Path to NPZ motion file (required) |
| `-v`, `--vis` | Show viewer |
| `-d`, `--debug-vis` | Show ghost robot overlay with reference motion |
| `-r`, `--record` | Record video |
| `--cpu` | Use CPU backend instead of GPU |

### Example

```bash
# Using BeyondMimic sample motion
python main.py --motion-file sample/motion1.npz --vis

# With ghost visualization
python main.py --motion-file sample/motion1.npz --vis --debug-vis

# Record video
python main.py --motion-file sample/motion1.npz --record
```

## Citation

```bibtex
@article{he2025omnixtreme,
    title={OmniXtreme: Breaking the Generality Barrier in High-Dynamic Humanoid Control},
    author = {{Wang}, Yunshen and {Zhu}, Shaohang and {Zhi}, Peiyuan and {Li}, Yuhan and {Li}, Jiaxin and {Li}, Yong-Lu and {Xiao}, Yuchen and {Wang}, Xingxing and {Jia}, Baoxiong and {Huang}, Siyuan},
    year={2026}
}
```
