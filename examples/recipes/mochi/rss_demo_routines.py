"""RoboSimStudio's three demo routines, played through ``env.step`` on Eden's Mochi versions of their tasks.

---
tier: recipe
capabilities: [solver.mochi, robot.yam, task.manipulation]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/mochi/rss_demo_routines.py --task mochi_wipe_the_bar
---

Each routine is the demo's own (``examples/08_demos/<demo>/`` in RoboSimStudio), driving the arms the way its
studio's gizmos do: the tool centre along straight lines and arcs at the demo's speeds, the jaws turned about the
vertical, and Open / Close. Every move goes through the task's policy actions (differential IK on the tool centre,
the binary grippers), so a routine that meets the task's goal shows the goal is reachable from its action space.

* ``mochi_wipe_the_bar`` — pick the sponge off the mat, carry it over the near block, press to 4 N and stroke
  the stripe one way at 60 mm/s, up to four strokes, until it is 96 % clean.
* ``mochi_fold_the_towel`` — the right hand presses the near half; the left takes the far edge with its fingertips
  on the bench, carries it over in a 5 s arc 6 cm high, lays it 12.5 mm short of its mirror and lets go.
* ``mochi_route_the_cable`` — the right hand lays the cable beside the first spool's far side, back across the
  line and round the second spool's near side; the left takes it 35 mm from its end, turns it down the port's channel and drops
  it in.

Usage:
    uv run python examples/recipes/mochi/rss_demo_routines.py --task mochi_fold_the_towel
    uv run python examples/recipes/mochi/rss_demo_routines.py --task mochi_route_the_cable -b 4
"""

import argparse
import math

import genesis as gs
import torch
from genesis.utils.geom import transform_by_quat

import eden as en
from eden.envs.base import RLEnvBase
from eden.tasks import TASK_REGISTRY
from eden.tasks.mochi.common import YAM_TCP_OFFSET, link_point

TASKS = ("mochi_wipe_the_bar", "mochi_fold_the_towel", "mochi_route_the_cable")


class Arms:
    """Both YAMs' tool centres, jaw turns and grips, stepped through the task's own actions."""

    def __init__(self, env: RLEnvBase):
        self.env = env
        self.n = env.num_envs
        self.dt = env.dt
        self.offset = torch.tensor(YAM_TCP_OFFSET, device=gs.device)
        # Where each action term sits in the action vector, and which terms move which arm: a task whose only acting
        # arm is the right one names its terms "hand" and "gripper", a bimanual one "<side>_hand", "<side>_gripper".
        manager = env.action_manager
        self.slots, start = {}, 0
        for name in manager.active_terms:
            self.slots[name] = (start, manager.get_term(name).action_dim)
            start += manager.get_term(name).action_dim
        self.dim = start
        self.hand, self.grip = {}, {}
        for side in ("right", "left"):
            for hand, grip in ((f"{side}_hand", f"{side}_gripper"), ("hand", "gripper")):
                if hand in self.slots and side not in self.hand and (hand != "hand" or side == "right"):
                    self.hand[side] = hand
                    if grip in self.slots:
                        self.grip[side] = grip
        self.sides = list(self.hand)
        self.links = {s: env.entities[s].get_link("link_6") for s in self.sides}
        self.target = dict.fromkeys(self.sides)
        self.yaw = dict.fromkeys(self.sides)
        self.full_turn = dict.fromkeys(self.sides, False)
        self.close = dict.fromkeys(self.sides, False)
        self.steps = 0

    def tcp(self, side: str) -> torch.Tensor:
        return link_point(self.links[side], self.offset)

    def jaw_angle(self, side: str) -> torch.Tensor:
        """Heading of the jaws' closing axis (the hand's x) in the bench plane."""
        x = transform_by_quat(
            torch.tensor([1.0, 0.0, 0.0], device=gs.device).expand(self.n, 3), self.links[side].get_quat()
        )
        return torch.atan2(x[:, 1], x[:, 0])

    def step(self) -> None:
        action = torch.zeros(self.n, self.dim, device=gs.device)
        for side in self.sides:
            start, _ = self.slots[self.hand[side]]
            scale = self.env.action_manager.get_term(self.hand[side]).effective_scale
            target = self.target[side] if self.target[side] is not None else self.tcp(side)
            action[:, start : start + 3] = torch.atanh(((target - self.tcp(side)) / scale[:3]).clamp(-0.97, 0.97))
            if self.yaw[side] is not None and float(scale[5]) > 0.0:
                period = 2 * math.pi if self.full_turn[side] else math.pi  # the jaws look the same turned half round
                error = torch.remainder(self.yaw[side] - self.jaw_angle(side) + period / 2, period) - period / 2
                action[:, start + 5] = torch.atanh((error / scale[5]).clamp(-0.97, 0.97))
            if side in self.grip:
                action[:, self.slots[self.grip[side]][0]] = 1.0 if self.close[side] else -1.0
        self.env.step(action)
        self.steps += 1

    def wait(self, seconds: float) -> None:
        for _ in range(round(seconds / self.dt)):
            self.step()

    def glide(self, side: str, path, seconds: float, during=None) -> None:
        """Move the tool centre along ``path(f)``, f from 0 to 1, eased; ``during(f)`` runs alongside."""
        n = max(1, round(seconds / self.dt))
        for k in range(1, n + 1):
            s = k / n
            self.target[side] = path(s * s * (3.0 - 2.0 * s))
            if during is not None:
                during(k / n)
            self.step()

    def reach(self, side: str, target: torch.Tensor, speed: float, settle: float = 0.4, eased: bool = False) -> None:
        """A straight line to ``target`` at ``speed`` [m/s] (eased at both ends, or not), then ``settle`` s there."""
        start = self.target[side] if self.target[side] is not None else self.tcp(side).clone()
        seconds = float((target - start).norm(dim=-1).max()) / speed
        if eased:
            self.glide(side, lambda f: start + (target - start) * f, max(0.2, seconds))
        else:
            n = max(1, round(seconds / self.dt))
            for k in range(1, n + 1):
                self.target[side] = start + (target - start) * k / n
                self.step()
        self.wait(settle)

    def point(self, x, y, z) -> torch.Tensor:
        return torch.stack([torch.as_tensor(v, device=gs.device).float().expand(self.n) for v in (x, y, z)], dim=-1)


def wipe_the_bar(env: RLEnvBase, arms: Arms) -> None:
    from eden.tasks.mochi.wipe_the_bar import config as C

    metrics, sponge = env.metric_manager, env.entities["sponge"]
    up = torch.tensor([0.0, 0.0, 0.10], device=gs.device)
    # Pick: top-down onto the sponge's middle, Close, wait for the latch (it engages once the grip settles).
    pick = sponge.get_vertex_positions().mean(dim=1)
    arms.reach("right", pick + up, 0.20)
    arms.reach("right", pick, 0.05)
    arms.close["right"] = True
    arms.wait(1.0)
    arms.reach("right", pick + up, 0.10)
    # Carry: over the near block at the crossing height, then over the near end of the stripe.
    drop = (arms.tcp("right")[:, 2] - sponge.get_vertex_positions()[..., 2].amin(dim=-1)).mean()
    clear_z = C.BAR_Z + C.TUBE_RADIUS + drop + 0.040
    half = 0.5 * C.SPONGE_SIZE[1]
    u0 = max(-C.STRIPE_HALF + half - 0.025, -C.STRAIGHT_HALF + half + 0.005)
    u1 = min(C.STRIPE_HALF - half + 0.045, C.STRAIGHT_HALF - half - 0.025)
    gate_y = C.BAR_Y - (C.STRAIGHT_HALF + C.BEND_RADIUS + C.BLOCK_INSET[1] + C.BLOCK_HALF[1] + 0.020)
    arms.reach("right", arms.point(0.5 * (pick[:, 0] + C.BAR_X), gate_y, clear_z), 0.10)
    arms.reach("right", arms.point(C.BAR_X, C.BAR_Y + u0, clear_z), 0.10)
    # Wipe: down a millimetre at a time until the stripe reads 4 N, then a one-way stroke at 60 mm/s with the
    # hand following the press (0.04 mm a step per newton short, at most 4 mm/s); up to four strokes.
    for _ in range(4):
        z, seat = clear_z, None
        while seat is None and z > clear_z - 0.08:
            z = z - 0.001
            for _ in range(3):
                arms.target["right"] = arms.point(C.BAR_X, C.BAR_Y + u0, z)
                arms.step()
            if float(metrics.get_value("press").mean()) >= 4.0:
                seat = z
        if seat is None:
            break
        height = torch.full((arms.n,), float(seat), device=gs.device)
        n = round((u1 - u0) / 0.06 / arms.dt)
        rate = 0.004 * arms.dt
        for k in range(1, n + 1):
            short = 4.0 - metrics.get_value("press")
            height = (height - (4e-5 * short).clamp(-rate, rate)).clamp(seat - 0.015, seat + 0.008)
            arms.target["right"] = arms.point(C.BAR_X, C.BAR_Y + u0 + (u1 - u0) * k / n, height)
            arms.step()
        arms.reach("right", arms.point(arms.tcp("right")[:, 0], arms.tcp("right")[:, 1], clear_z), 0.06)
        print(f"  stroke: {metrics.get_value('cleaned').tolist()} of the stripe clean")
        if float(metrics.get_value("cleaned").min()) >= 0.96:
            break
        arms.reach("right", arms.point(C.BAR_X, C.BAR_Y + u0, clear_z), 0.15)


def fold_the_towel(env: RLEnvBase, arms: Arms) -> None:
    from eden.tasks.mochi.fold_the_towel import config as C

    on_bench, up = 0.0155, torch.tensor([0.0, 0.0, 1.0], device=gs.device)
    edge = env.entities["towel"].get_vertex_positions()[..., 1].amax(dim=-1)
    grasp = arms.point(C.PINCH_X, edge - 0.006, on_bench)  # fingertips on the bench 6 mm in from the edge
    press = arms.point(*C.PRESS, on_bench)
    # Both hands to their marks; the right, shut, presses the near half.
    arms.reach("right", press + 0.08 * up, 0.25, settle=0.5, eased=True)
    arms.reach("left", grasp + 0.10 * up, 0.25, settle=0.5, eased=True)
    arms.reach("right", press, 0.10, settle=0.5, eased=True)
    arms.reach("left", grasp, 0.10, settle=0.5, eased=True)
    # Squeeze and rise 3 mm.
    arms.close["left"] = True
    arms.wait(0.2)
    start = arms.target["left"].clone()
    arms.glide("left", lambda f: start + 0.003 * f * up, 0.3)
    arms.wait(1.0)
    # Carry over the crease in a low arc, 12.5 mm short of the edge's mirror; the press lifts at 90 % of it.
    held = arms.tcp("left").clone()
    place = arms.point(held[:, 0], 2.0 * C.CENTER[1] - held[:, 1] + 0.0125, 0.03)
    apex = (held + place) / 2.0
    apex[:, 2] = torch.maximum(held[:, 2], place[:, 2]) + 0.06
    bend = 2.0 * apex - (held + place) / 2.0
    pressed = arms.tcp("right").clone()

    def lift_press(f):
        if f >= 0.9:
            arms.target["right"] = pressed + torch.tensor([0.0, -0.10, 0.10], device=gs.device) * min(
                1.0, (f - 0.9) / 0.1
            )

    arms.glide("left", lambda f: (1 - f) ** 2 * held + 2 * f * (1 - f) * bend + f**2 * place, 5.0, during=lift_press)
    # Let go, back off and lie still.
    arms.close["left"] = False
    arms.wait(1.0)
    arms.reach("left", place + torch.tensor([0.0, 0.08, 0.10], device=gs.device), 0.10, settle=0.5, eased=True)
    arms.wait(5.0)


def route_the_cable(env: RLEnvBase, arms: Arms) -> None:
    from eden.tasks.mochi.route_the_cable import config as C

    cable = env.entities["cable"]
    low = 0.0165  # tool centre over the bench that puts the fingertips on it, the cable between the pads

    def tangent(k):
        c = cable.get_vertex_positions()
        t = c[:, min(k + 1, C.SEGMENTS), :2] - c[:, max(k - 1, 0), :2]
        return torch.atan2(t[:, 1], t[:, 0])

    def take(side, k):
        """Over node k with the jaws across the cable's tangent, down onto it, Close."""
        arms.full_turn[side] = False
        arms.yaw[side] = tangent(k) + math.pi / 2
        p = cable.get_vertex_positions()[:, k]
        arms.reach(side, arms.point(p[:, 0], p[:, 1], 0.06), 0.12)
        p = cable.get_vertex_positions()[:, k]
        arms.reach(side, arms.point(p[:, 0], p[:, 1], low), 0.05)
        arms.close[side] = True
        arms.wait(0.8)

    def place(side, s, to, carry=0.03):
        """Pinch the cable ``s`` m from its welded end, carry that point to ``to``, set it down, let go."""
        take(side, round(s / C.LENGTH * C.SEGMENTS))
        here = arms.tcp(side)
        for target in (arms.point(here[:, 0], here[:, 1], carry), arms.point(*to, carry), arms.point(*to, low + 0.003)):
            arms.reach(side, target, 0.05)
        arms.close[side] = False
        arms.wait(0.6)
        arms.reach(side, arms.point(*to, 0.07), 0.08)

    def plug(side, s):
        """Take the cable ``s`` m along, turn it down the port's channel (+y), and drop its end in."""
        k = round(s / C.LENGTH * C.SEGMENTS)
        channel_x, mouth = C.PORT[0], C.PORT[1] - C.TERMINAL
        take(side, k)
        # Carried high (as the demo carries round the second spool) so the cable behind the hand hangs rather than
        # sweeping the bench as the hand turns, and turned so the cable runs from the hand toward its end along +y.
        here = arms.tcp(side)
        arms.reach(side, arms.point(here[:, 0], here[:, 1], 0.065), 0.05)
        arms.full_turn[side] = True
        arms.yaw[side] = (
            arms.jaw_angle(side) + torch.remainder(math.pi / 2 - tangent(k) + math.pi, 2 * math.pi) - math.pi
        )
        arms.reach(side, arms.point(channel_x, mouth - 0.044, 0.065), 0.05)
        arms.reach(side, arms.point(channel_x, mouth - 0.044, 0.04), 0.05)
        for _ in range(2):  # turn until the cable ahead points down the channel, then slide its line onto the mouth
            c = cable.get_vertex_positions()
            ahead = c[:, -1, :2] - c[:, k, :2]
            arms.yaw[side] = arms.jaw_angle(side) + torch.atan2(ahead[:, 0], ahead[:, 1])
            arms.wait(1.0)
            c = cable.get_vertex_positions()
            ahead = c[:, -1, :2] - c[:, k, :2]
            meets = c[:, k, 0] + ahead[:, 0] / ahead[:, 1].clamp_min(1e-6) * (mouth - c[:, k, 1])
            here = arms.tcp(side)
            arms.reach(side, arms.point(here[:, 0] + channel_x - meets, mouth - 0.044, 0.04), 0.03)
        here = arms.tcp(side)
        arms.reach(side, arms.point(here[:, 0], mouth - 0.014, 0.04), 0.025)
        arms.reach(side, arms.point(here[:, 0], mouth - 0.014, low + 0.003), 0.025)
        arms.close[side] = False
        arms.wait(0.6)
        arms.reach(side, arms.point(here[:, 0], mouth - 0.034, 0.07), 0.08)

    # The task's marks: beside the first spool, across the spools' line, round the second. The first and the last are
    # carried high, over a spool's rim, so the cable hanging from the hand clears it.
    (s0, to0), (s1, to1), (s2, to2), _ = C.MARKS
    place("right", s0, to0, carry=0.065)
    place("right", s1, to1)
    place("right", s2, to2, carry=0.065)
    arms.reach("right", arms.point(-0.10, -0.20, 0.15), 0.15)
    plug("left", 0.565)
    arms.reach("left", arms.point(-0.10, 0.20, 0.15), 0.15)
    arms.wait(2.0)


ROUTINES = {
    "mochi_wipe_the_bar": wipe_the_bar,
    "mochi_fold_the_towel": fold_the_towel,
    "mochi_route_the_cable": route_the_cable,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=TASKS, default="mochi_wipe_the_bar", help="The task whose routine to play")
    parser.add_argument("-b", "--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on the CPU backend instead of the GPU")
    args = parser.parse_args()
    en.init(backend=gs.cpu if args.cpu else gs.gpu)

    config = TASK_REGISTRY.build(args.task)
    config.env_options.num_envs = args.num_envs
    # The routine runs to its own end: no time limit, and no reset on success before the measures are read.
    for name in config.termination_options.term_keys():
        config.termination_options.disable_term(name)
    env = RLEnvBase.from_config(config)
    env.build()
    env.reset()
    arms = Arms(env)
    if en.is_smoke():
        arms.wait(5 * arms.dt)
        return
    ROUTINES[args.task](env, arms)
    metrics = env.metric_manager
    readings = {name: [round(v, 3) for v in metrics.get_value(name).tolist()] for name in metrics.active_terms}
    print(f"{args.task} after {arms.steps * arms.dt:.1f} s: {readings}")
    print(f"goal met: {metrics.success_buf.tolist()}")


if __name__ == "__main__":
    main()
