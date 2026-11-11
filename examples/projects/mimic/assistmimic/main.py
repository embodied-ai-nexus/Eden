"""AssistMimic in Eden: a PHC-initialized caregiver helps a weakened recipient up (SMPL humanoids).

---
tier: project
capabilities: [task.mimic, dataset.inter_x, dataset.hhi_assist]
hardware: gpu
optional_deps: []
run: uv run python examples/projects/mimic/assistmimic/main.py train
---

``--dataset`` picks the task, and every stage takes it: ``inter-x`` (the default) helps the recipient up off
the floor on Inter-X, ``hhi-assist`` helps them out of a bed on HHI-Assist. It selects the reference clips,
the furniture, how weak the recipient is and whether the caregiver is paid for unloading them, so a run's
logs live under ``logs/assistmimic_<dataset>/<stage>/`` and its checkpoints are not interchangeable.

Both training stages run rl_games' PPO (``eden.envs.wrappers.rl_games_env``, two agents per env) with
AssistMimic's network (``assist_training/network.py``) and per-network learning rates (``assist_training/agent.py``);
checkpoints and ``progress.csv`` land in the run's log directory.

Stages (see README.md)::

    # 1. adapt PHC (phc_3, Isaac Gym) to Genesis on the same tracks, humanoids apart
    python main.py adapt -b 1024 -s 700
    # 2. AssistMimic: both roles initialized from the adapted PHC, trained together
    python main.py train --phc-caregiver <log>/phc_genesis_caregiver.pth --phc-recipient <log>/phc_genesis_recipient.pth
    # 3. roll out from the clip start (no PSI) and record
    python main.py play --checkpoint <log>/nn/last_assistmimic_ep_2000_rew_<r>.pth -r --clip 0
    # PHC-loading check (AssistMimic's test): the untrained pair should track 80+ steps
    python main.py play
    # the same, on the bed task
    python main.py play --dataset hhi-assist
"""

import argparse
import os

import eden as en  # isort: skip  (must precede genesis / torch: GPU isolation)
import genesis as gs  # isort: skip
import torch  # isort: skip

#: PHC's ``phc_3`` checkpoint on the policy hub (``Kashu7100/eden_policies``), the default ``--phc``.
PHC_HUB_FILE = "phc/phc_3/Humanoid.pth"
DEFAULT_PHC = os.environ.get("PHC_CHECKPOINT", "")


def _phc_path(path: str) -> str:
    """``path``, or the hub's ``phc_3`` (downloaded on first use, 439 MB) when it is empty."""
    if path:
        return path
    from eden.extensions.policies.base import PolicyBase

    return PolicyBase.fetch_asset(PHC_HUB_FILE)


def _log_root(stage: str, args) -> str:
    """One log tree per stage *and* dataset — their checkpoints are not interchangeable (different obs size)."""
    return os.path.join("logs", f"assistmimic_{args.dataset.replace('-', '_')}", stage)


#: AssistMimic's rl_games settings (``im_simpleliftup_mlp.yaml``) as Eden runner options; ``_runner_options`` fills
#: in the per-run ones.
PPO = {
    "horizon_length": 32,
    "mini_epochs": 6,
    "lr_schedule": "None",
    "gamma": 0.99,
    "tau": 0.95,
    "e_clip": 0.2,
    "clip_value": False,
    "entropy_coef": 0.0,
    # rl_games weighs the squared value error by 0.5 * critic_coef: 10 is AssistMimic's 5 on the plain error.
    "critic_coef": 10.0,
    "grad_norm": 50.0,
    "truncate_grads": True,
    "normalize_input": False,  # the network normalizes with PHC's frozen statistics instead
    "normalize_value": True,
    "normalize_advantage": True,
    "value_bootstrap": True,
    "clip_actions": 1.0,
    "bounds_loss_coef": 10.0,
    "torch_compile": False,
}


def _runner_options(args, num_envs: int, dt: float):
    from assist_training import ROLES

    from eden.options.learning.rl_games.rl import RlGamesNetworkOptions, RlGamesPpoRunnerOptions

    pair_rows = len(ROLES) * PPO["horizon_length"]  # one env's caregiver and recipient trajectories
    batch = num_envs * pair_rows
    if en.is_smoke():
        args.minibatch = min(args.minibatch, batch)
    if args.minibatch % pair_rows or batch % args.minibatch:
        raise ValueError(
            f"--minibatch {args.minibatch} must be a multiple of 2 x horizon ({pair_rows}), so each minibatch holds "
            "whole caregiver/recipient pairs (the network splits it in half by role), and divide the rollout batch "
            f"2 x num_envs x horizon = {batch}."
        )
    return RlGamesPpoRunnerOptions(
        experiment_name="assistmimic",
        max_epochs=1 if en.is_smoke() else args.steps,
        save_frequency=args.save_every,
        keep_last_checkpoints=args.keep_last or None,
        minibatch_size=args.minibatch,
        learning_rate=args.lr,
        # Eden's per-agent reward is the step's reward times dt; AssistMimic trains on the step's reward.
        reward_scale=1.0 / dt,
        agents=list(ROLES),
        network=RlGamesNetworkOptions(name="assistmimic"),
        **PPO,
    )


def _build_env(args, adapt: bool, num_envs: int, eval_mode: bool = False, runner: bool = False):
    """The task's env; with ``runner``, built behind rl_games' two-agent wrapper (which also seeds the run)."""
    from assist_training.env import QuarantineVecEnv

    from eden.envs.base import RLEnvBase
    from eden.tasks import TASK_REGISTRY
    from eden.tasks.assistmimic.config import (
        TASK_FOR_DATASET,
        phc_adaptation_config,
        remove_caregiver,
        separate_pair,
        strong_recipient,
    )

    cfg = TASK_REGISTRY.build(TASK_FOR_DATASET[args.dataset])
    if adapt:
        cfg = phc_adaptation_config(cfg)
    cfg.env_options.num_envs = num_envs
    if args.max_collision_pairs:
        cfg.env_options.max_collision_pairs = args.max_collision_pairs
    if args.motion_file:
        cfg.command_options.motion.motion_file = args.motion_file
    if args.strong_recipient:
        cfg.scene_options.recipient = strong_recipient(cfg)
    if args.apart:
        separate_pair(cfg)
    if args.termination_distance is not None:
        # Before remove_caregiver, which then excuses the caregiver from this termination entirely.
        fail = cfg.termination_options.tracking_failure
        fail.caregiver_threshold = fail.recipient_threshold = args.termination_distance
    if args.no_caregiver:
        remove_caregiver(cfg)
    if args.clip is not None:
        cfg.command_options.motion.fixed_clip = args.clip
    if args.no_psi:
        cfg.command_options.motion.psi_ratio = 0.0
    if args.psi_threshold is not None:
        cfg.command_options.motion.psi_return_threshold = args.psi_threshold
    if args.record:
        from eden.options.camera import CameraOptions

        cfg.cameras_options.rec = CameraOptions(
            cam_pos=(3.5, -3.5, 2.0), cam_lookat=(0.0, 0.0, 0.8), cam_resolution=(960, 720), cam_fov=50.0
        )
    if runner:
        env_opts = cfg.env_options
        cfg.runner_options = _runner_options(args, num_envs, env_opts.decimation * env_opts.sim_dt)
        return QuarantineVecEnv.from_config(cfg, eval_mode=eval_mode)
    env = RLEnvBase.from_config(cfg, eval_mode=eval_mode)
    env.build()
    return env


def _role_column(env) -> int:
    """Column of the per-agent observation holding the ``role`` flag."""
    import math

    om = env.observation_manager
    names, dims = om.group_term_names["caregiver"], om.group_obs_term_dim["caregiver"]
    return sum(math.prod(d) for d in dims[: names.index("role")])


def _learn(args, wrapper, phc, input_dim, extra_log=None):
    """Train ``wrapper``'s env with rl_games, both role actors initialized from ``phc`` (caregiver, recipient).

    Checkpoints go to ``<log_dir>/nn/``: ``last_assistmimic_ep_<epoch>_rew_<r>.pth`` every ``--save-every`` epochs
    and at the end, of which the newest ``--keep-last`` stay on disk. rl_games' best-reward checkpoint is off — at
    344 MB a save, it would rewrite one on most epochs.
    """
    from assist_training.agent import ProgressLog, register_agent
    from assist_training.network import register
    from rl_games.torch_runner import Runner

    from eden.tasks.assistmimic.config import PHC_OBS_DIM

    env = wrapper.unwrapped
    # A resumed run takes its weights from the checkpoint: PHC is loaded only into a fresh one.
    register(_role_column(env), input_dim, None if args.resume else phc, PHC_OBS_DIM)
    params = wrapper.runner_dict
    params["params"]["config"].update(
        critic_learning_rate=args.critic_lr, train_dir=en.log_dir, full_experiment_name=".", save_best_after=10**9
    )
    runner = Runner(algo_observer=ProgressLog(os.path.join(en.log_dir, "progress.csv"), env, extra_log))
    register_agent(runner)
    runner.load(params)
    wrapper.configure_runner(runner)
    runner.run({"train": True, "checkpoint": args.resume or ""})


def adapt(args):
    en.init(backend=gs.gpu, log_root_path=_log_root("adapt", args), performance_mode=True)
    from assist_training import ROLES
    from assist_training.network import export_phc, load_policy

    from eden.tasks.assistmimic.config import PHC_OBS_DIM, agent_obs_dim

    wrapper = _build_env(args, adapt=True, num_envs=8 if en.is_smoke() else args.num_envs, runner=True)
    phc = _phc_path(args.phc)
    _learn(args, wrapper, (phc, phc), PHC_OBS_DIM)
    env = wrapper.unwrapped
    policy = load_policy(_last_checkpoint(), agent_obs_dim(env), _role_column(env), PHC_OBS_DIM, env.device)
    for role, name in enumerate(ROLES):
        path = os.path.join(en.log_dir, f"phc_genesis_{name}.pth")
        export_phc(policy, role, path, PHC_OBS_DIM)
        print("exported", path)


def _last_checkpoint() -> str:
    """The run's final rl_games checkpoint (``nn/last_assistmimic_ep_<max_epochs>_rew_<r>.pth``)."""
    import glob

    paths = glob.glob(os.path.join(en.log_dir, "nn", "last_assistmimic_ep_*.pth"))
    if not paths:
        raise FileNotFoundError(f"no rl_games checkpoint under {en.log_dir}/nn")
    return max(paths, key=lambda p: int(os.path.basename(p).split("_ep_")[1].split("_")[0]))


def _phc_pair(args) -> tuple[str, str]:
    """The caregiver's and the recipient's PHC: the adapted exports, or the original ``phc_3`` for each one missing."""
    phc = None if args.phc_caregiver and args.phc_recipient else _phc_path(args.phc)
    return args.phc_caregiver or phc, args.phc_recipient or phc


def train(args):
    en.init(backend=gs.gpu, log_root_path=_log_root("train", args), performance_mode=True)
    wrapper = _build_env(args, adapt=False, num_envs=8 if en.is_smoke() else args.num_envs, runner=True)

    def head_log(env):
        return {"head_height": float(env.metric_manager.get_term("head_height").compute().mean())}

    _learn(args, wrapper, _phc_pair(args), None, extra_log=head_log)


class EpisodeStats:
    """Per-episode length, completion, help-up success and caregiver hand force, grouped by clip."""

    def __init__(self, env, success_height: float | None, low_start: float, success_frac: float):
        from eden.managers.terms.utils import resolve_link_idx
        from eden.tasks.assistmimic.custom_terms.observations import WRIST_LINKS

        self.env = env
        self.motion = env.command_manager.get_term("motion")
        self.head = env.metric_manager.get_term("head_height")
        self.hand_idx = resolve_link_idx(
            env.entities["caregiver"], [link for pair in WRIST_LINKS for link in pair], env.device
        )
        self.is_helpup = self.motion.start_head_height("recipient") < low_start
        peak = self.motion.peak_head_height("recipient")
        self.success_height = success_height
        self.success_frac = success_frac
        self.threshold = torch.full_like(peak, success_height) if success_height else success_frac * peak
        self.max_head = torch.zeros(env.num_envs, device=env.device)
        self.force_sum = torch.zeros(env.num_envs, device=env.device)
        self.rows = {k: [] for k in ("clip", "length", "completed", "success", "force")}

    def before_step(self):
        self.max_head = torch.maximum(self.max_head, self.head.compute())
        self.length = self.env.episode_length_buf + 1
        self.clip = self.motion.clip_id.clone()

    def after_step(self, term, trunc):
        force = self.env.entities["caregiver"].get_links_net_contact_force()[:, self.hand_idx].norm(dim=-1).sum(-1)
        self.force_sum += force
        done = term | trunc
        if not done.any():
            return
        completed = trunc & ~term
        success = completed & (self.max_head > self.threshold[self.clip]) & self.is_helpup[self.clip]
        for k, v in (
            ("clip", self.clip),
            ("length", self.length.float()),
            ("completed", completed.float()),
            ("success", success.float()),
            ("force", self.force_sum / self.length.float()),
        ):
            self.rows[k].append(v[done])
        self.max_head[done] = 0.0
        self.force_sum[done] = 0.0

    def report(self) -> None:
        if not self.rows["clip"]:
            print("no episode finished")
            return
        r = {k: torch.cat(v) for k, v in self.rows.items()}
        helpup = self.is_helpup[r["clip"]]
        rate = r["success"][helpup].mean() if helpup.any() else 0.0
        if self.success_height:
            reached = f"{self.success_height} m"
        else:
            lo, hi = self.threshold.min(), self.threshold.max()
            reached = f"{self.success_frac:.0%} of its own clip's peak, {lo:.2f}-{hi:.2f} m"
        print(f"episodes {r['clip'].numel()}, mean length {r['length'].mean():.1f}")
        print(
            f"help-up success rate {rate:.3f} over the {int(self.is_helpup.sum())} help-up clips "
            f"(completed, and the recipient's head reached {reached})"
        )
        clip_steps = (self.motion.clip_lengths / self.env.dt).round()
        for c, name in enumerate(self.motion.clip_names):
            sel = r["clip"] == c
            if sel.any():
                kind = "help-up" if self.is_helpup[c] else "standing"
                print(
                    f"  {name} [{kind}]: length {r['length'][sel].mean():.0f}/{clip_steps[c]:.0f}, completed "
                    f"{r['completed'][sel].mean():.2f}, success {r['success'][sel].mean():.2f}, caregiver hand "
                    f"contact force {r['force'][sel].mean():.0f} N ({int(sel.sum())} eps)"
                )


def play(args):
    en.init(backend=gs.gpu, log_root_path=_log_root("play", args))
    from assist_training import deinterleave, interleave
    from assist_training.network import AssistActorCritic, load_policy

    from eden.tasks.assistmimic.config import PHC_OBS_DIM, agent_obs_dim

    env = _build_env(args, adapt=args.adapt, num_envs=5 if en.is_smoke() else args.num_envs, eval_mode=True)
    input_dim = PHC_OBS_DIM if args.adapt else None
    if args.checkpoint:
        policy = load_policy(args.checkpoint, agent_obs_dim(env), _role_column(env), input_dim, env.device)
    else:  # AssistMimic's PHC-loading check: the untrained pair should track ~80+ steps from the first step
        action_dim = env.action_manager.total_action_dim // 2
        policy = AssistActorCritic(agent_obs_dim(env), action_dim, _role_column(env), input_dim).to(env.device)
        policy.load_phc_pair(_phc_pair(args), PHC_OBS_DIM)
    stats = EpisodeStats(env, args.success_height, args.low_start, args.success_frac)
    obs, _ = env.reset()
    cam = env.cameras["rec"] if args.record else None
    if cam is not None:
        clip = stats.motion.clip_names[int(stats.motion.clip_id[0])]
        cam.start_recording(os.path.join(en.log_dir, f"assistmimic_{clip}.mp4"), fps=30)
    with torch.inference_mode():
        for _ in range(5 if en.is_smoke() else args.steps):
            mu = policy.act_mean(interleave(obs))
            stats.before_step()
            obs, _, term, trunc, _ = env.step(deinterleave(mu.clamp(-1, 1)))
            stats.after_step(term, trunc)
            if cam is not None:
                root = env.entities["recipient"].get_links_pos()[0, 0].cpu().numpy()
                cam.set_pose(pos=(root[0] + 3.0, root[1] - 3.0, 1.8), lookat=(root[0], root[1], 0.8))
                cam.render_rgb()
    if cam is not None:
        cam.stop_recording()
    stats.report()


def main():
    parser = argparse.ArgumentParser(description="AssistMimic help-up in Eden.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("adapt", "train", "play"):
        p = sub.add_parser(name)
        if name != "play":  # play-only diagnostics, read by the shared _build_env
            p.set_defaults(
                strong_recipient=False,
                apart=False,
                no_caregiver=False,
                record=False,
                termination_distance=None,
                clip=None,
            )
        if name != "train":  # PSI: adapt always starts at the clip start, play's eval env never restarts
            p.set_defaults(no_psi=False, psi_threshold=None)
        p.add_argument(
            "-b", "--num-envs", type=int, default={"adapt": 1024, "train": 8192, "play": 128}[name], help="Envs"
        )
        p.add_argument(
            "-s",
            "--steps",
            type=int,
            default=2000 if name != "play" else 600,
            help="Epochs (a resumed run's count includes the checkpoint's) / play steps",
        )
        p.add_argument(
            "--dataset",
            type=str,
            default="inter-x",
            choices=("inter-x", "hhi-assist"),
            help="inter-x: help-up off the floor. hhi-assist: helping the recipient out of a bed",
        )
        p.add_argument("--motion-file", type=str, default=None, help="Converted .npz (default: the dataset's)")
        p.add_argument(
            "--max-collision-pairs",
            type=int,
            default=60 if name == "train" else None,
            help="Per-env collision pair budget (None: the config's 100, which overflows at 8192 envs)",
        )
        p.add_argument(
            "--phc",
            type=str,
            default=DEFAULT_PHC,
            help="PHC SMPL checkpoint (phc_3); default $PHC_CHECKPOINT, else phc_3 from Kashu7100/eden_policies",
        )
        if name != "play":
            p.add_argument("--lr", type=float, default=2e-5, help="Actor learning rate")
            p.add_argument("--critic-lr", type=float, default=1e-4, help="Critic learning rate")
            p.add_argument("--minibatch", type=int, default=16384 if name == "adapt" else 65536, help="PPO minibatch")
            p.add_argument("--save-every", type=int, default=500, help="Save a checkpoint every this many epochs")
            p.add_argument(
                "--keep-last", type=int, default=1, help="Checkpoints kept on disk, newest first (0: every one saved)"
            )
            p.add_argument(
                "--resume",
                type=str,
                default=None,
                help="rl_games checkpoint to resume from (PSI's stored trajectories are not in it, so its buffer refills)",
            )
        if name != "adapt":
            p.add_argument("--phc-caregiver", type=str, default=None, help="Adapted PHC for the caregiver")
            p.add_argument("--phc-recipient", type=str, default=None, help="Adapted PHC for the recipient")
        if name == "train":
            p.add_argument("--no-psi", action="store_true", help="Disable PSI (restarts from stored rollouts)")
            p.add_argument(
                "--psi-threshold", type=float, default=None, help="Min episode return to store a rollout (default 300)"
            )
        if name == "play":
            p.add_argument("--checkpoint", type=str, default=None, help="rl_games checkpoint; omitted: PHC-initialized")
            p.add_argument("--adapt", action="store_true", help="Checkpoint is from the adapt stage")
            p.add_argument("--clip", type=int, default=None, help="Play only this clip index")
            p.add_argument("-r", "--record", action="store_true", help="Record a video (env 0)")
            p.add_argument("--apart", action="store_true", help="Diagnostic: humanoids 3 m apart, no contact possible")
            p.add_argument(
                "--no-caregiver",
                action="store_true",
                help="Diagnostic: recipient and bed alone — can it get up unaided?",
            )
            p.add_argument(
                "--termination-distance",
                type=float,
                default=None,
                help="Tracking error [m] that ends an episode (default: the config's 0.25); raise it to let a "
                "failing rollout play out far enough to see how it falls",
            )
            p.add_argument("--strong-recipient", action="store_true", help="Diagnostic: full-strength recipient")
            p.add_argument(
                "--low-start", type=float, default=1.0, help="Recipient head [m] below which a clip starts down"
            )
            p.add_argument(
                "--success-height",
                type=float,
                default=None,
                help="Absolute recipient head [m] for success; default: --success-frac of the clip's own peak",
            )
            p.add_argument(
                "--success-frac",
                type=float,
                default=0.9,
                help="Fraction of a clip's own peak recipient head height that counts as helped up",
            )
    args = parser.parse_args()
    {"adapt": adapt, "train": train, "play": play}[args.command](args)


if __name__ == "__main__":
    main()
