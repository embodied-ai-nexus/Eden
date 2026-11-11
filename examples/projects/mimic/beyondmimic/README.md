# BeyondMimic Motion Tracking

## Data Preparation

1. Convert CSV file into `.npz` motion file:
  ```bash
  python scripts/csv_to_npz.py --input-file sample/G1_gangnam_style_V01_bvh_60hz.csv --input-fps 60 --frame-range 2 1942
  ```

  > [!IMPORTANT]
  > `--frame-range 2 1942` is not cosmetic: row 1 of this CSV is a retargeting
  > artifact (the root jumps 11.6 cm between rows 1 and 2). Velocities are
  > differences of positions, so keeping it yields a 5.8 m/s root velocity at
  > frame 0 — the largest in the whole 32 s clip. `MotionCommand` resets the
  > robot onto that frame *including its velocity* (RSI), which launches the
  > robot and makes every motion tracker fall immediately, with no error raised.
  > `csv_to_npz.py` warns (`High linear velocity is detected at ... frame`) when
  > this happens — do not ignore it. Frame ranges are 1-indexed and inclusive.

2. Verify your `.npz` motion file:
  ```bash
  python scripts/replay_npz.py --file sample/G1_gangnam_style_V01_bvh_60hz.npz --record
  ```

## Policy Training

> [!NOTE]
> Make sure to check your npz file with `replay_npz.py` before you train a model.

```bash
python main.py train --motion-file sample/G1_gangnam_style_V01_bvh_60hz.npz
```

## Policy Inference

```
python main.py inference --model model_14999.pt --vis
```

## Policy Deploy

> [!NOTE]
> Make sure to add `--deployable` arg in training and inference to obtain a deployable policy, i.e., without using linear velocity. 

> [!CAUTION]
> Running these models on real robots is dangerous and entirely at your own risk. They are provided for research only, and we accept no responsibility for any harm, damage, or malfunction.

1. Start the robot in suspended state and wait until it enters zero-torque mode.
2. While in zero-torque mode, press `L2 + R2` on the controller. The robot will enter debug mode with joint damping enabled.
3. Connect your PC to the robot via Ethernet and verify with `ping 192.168.123.164` response.
4. Run following and hit enter when prompted.

```
python main.py deploy --model model_14999.pt
```

## Citation

```bibtex
@article{liao2025beyondmimic,
       author = {{Liao}, Qiayuan and {Truong}, Takara E. and {Huang}, Xiaoyu and {Gao}, Yuman and {Tevet}, Guy and {Sreenath}, Koushil and {Liu}, C. Karen},
        title = "{BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion}",
      journal = {arXiv e-prints},
         year = 2025,
}
```