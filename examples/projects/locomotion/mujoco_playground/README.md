# MuJoCo Playground

> **Note:** These tasks originate from [MuJoCo Playground](https://playground.mujoco.org/) but are reimplemented with Genesis/Eden — no MuJoCo required.

## Get Up

Fall recovery task — the Go1 quadruped is dropped from a random height and orientation and must stand up.

```bash
# Train (automatically runs inference after training)
python examples/projects/locomotion/mujoco_playground/getup/main.py train

# Train with options
python examples/projects/locomotion/mujoco_playground/getup/main.py train --logger wandb --max-iterations 5000

# Run inference on a trained model
python examples/projects/locomotion/mujoco_playground/getup/main.py inference --model <path/to/model.pt>
```

## Handstand

The Go1 quadruped learns to perform a handstand, balancing on its front feet with the rear legs raised.

```bash
# Train (automatically runs inference after training)
python examples/projects/locomotion/mujoco_playground/handstand/main.py train

# Train with options
python examples/projects/locomotion/mujoco_playground/handstand/main.py train --logger wandb --max-iterations 5000

# Run inference on a trained model
python examples/projects/locomotion/mujoco_playground/handstand/main.py inference --model <path/to/model.pt>
```

## Citation

```bibtex
@misc{zakka2025mujocoplayground,
      title={MuJoCo Playground},
      author={Kevin Zakka and Baruch Tabanpour and Qiayuan Liao and Mustafa Haiderbhai and Samuel Holt and Jing Yuan Luo and Arthur Allshire and Erik Frey and Koushil Sreenath and Lueder A. Kahrs and Carmelo Sferrazza and Yuval Tassa and Pieter Abbeel},
      year={2025},
      eprint={2502.08844},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2502.08844},
}
```
