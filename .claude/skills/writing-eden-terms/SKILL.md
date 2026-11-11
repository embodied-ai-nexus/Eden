---
name: writing-eden-terms
description: How to write good Eden manager terms (observation / reward / termination / metric / command / event / curriculum). Covers the function-vs-class decision, the build()/compute() lifecycle, never stashing state on env, which per-step hook a piece of state belongs in, and sharing computation across terms via a MetricTerm. Use when writing, reviewing, or refactoring any Eden term.
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
---

**Read `docs/guides/terms.md` now** — it is the authoritative contract for terms and covers the
function-vs-class decision, the `build()`/`compute()`/`reset()`/`post_reset()`/`post_physics_refresh()` lifecycle, why neither state nor the
entity *handle* may be cached, which per-step hook a piece of state belongs in (grade the step, or
prepare the next one), the shared `eden/managers/terms/utils.py` helpers, sharing a
computation through a `MetricTerm`, what validation raises when, the registries and their naming, and
the runtime gotchas (DOF order, wxyz, options-only observation modifiers, sync-averse hot paths).
Then use the checklist below, which is what to *verify* before calling a term finished.

The built-in terms are written to that guide; match them rather than inventing a house style.

## Checklist before finishing a term

1. **Nothing static on `env`.** No `setattr(env, ...)` / attribute assignment to cache indices,
   tensors, or thresholds. If you find that pattern in a function term, convert it to a class term
   and resolve in `build()`. (Reading a wrapper-owned buffer the env wrapper maintains — e.g.
   `getattr(env, "_prev_dofs_vel", ...)` — is a different, accepted case.)
2. **No cached entity handle.** `build()` *touches* each entity so a bad name fails at env
   construction, then keeps only what it derived; `compute()` goes through
   `self.resolve_entity(name)`. A handle held across the deployment hot-swap silently feeds the
   policy simulated numbers on hardware.
3. **Nothing constant rebuilt in `compute()`.** No name resolution, `hasattr` checks, Python-list →
   tensor rebuilds, or `torch.tensor(...)` of a constant on the hot path — all of it belongs in
   `build()`, via the shared `eden/managers/terms/utils.py` helpers (`resolve_link_idx`,
   `gravity_vec`, `resolve_per_dof_std`, …) rather than open-coded.
4. **Per-step caches are coherent after a reset.** If the term caches anything refreshed once per
   step, a freshly reset env must not observe the dead episode's values. Zero what is unknowable
   until the next physics step (contact forces, latches, air time) in `reset()`; rebuild what the
   post-reset scene determines (poses, targets) in `post_reset(envs_idx)`, which the env dispatches
   only after *every* manager has reset. Buffers must be mutated in place there — the fan-out is
   inference-moded.
5. **Per-step state is in the hook that matches its epoch.** The step has two, split by the
   auto-reset: phases 6–9 *grade* the step that just happened (fed by `post_physics_refresh` at 5),
   16–20 *prepare* the next one. So for a command term, a measurement of what the step produced —
   hold counters, contact latches, tracking errors — is computed in `post_physics_refresh()` (phase
   5, above the graded layer), and `_update_command()` (phase 17, below the reward and after the
   reset) publishes the command. The test is what the state *is*, not who reads it: a tracking
   reward reads the command too, and the command still belongs at 17. Written in `_update_command`,
   a measurement reaches the reward a whole control step late, and anything that same call *clears*
   is never read at all — a reward that silently pays zero on the step that earned the most. A
   shadow buffer carrying a value backwards across that boundary is the smell; move the update
   instead. **If the term will be deployed**, remember `DeploymentBase.step` runs a command term's
   `pre_physics_advance` and `compute()` but never its `post_physics_refresh`, so whatever the
   policy's observations read has to be produced from one of those two.
   **An observation term's per-step state follows the same rule, with `compute()` as the wrong
   place.** `ObservationManager.compute()` is a *read* that runs several times a step — at build for
   the shape, in `reset()`, at phase 19, at phase 12 for the final-observation snapshot, from
   `get_observations()` — and the cross-group cache may skip an instance entirely, so an integral,
   counter or latch advanced in `compute()` is a wrong number with no exception. Allocate in
   `build()`, advance in `post_physics_refresh()` (phase 7 — after command caches and metrics, and
   once per control tick on hardware), zero rows in `reset()`, return the buffer from `compute()`.
   `VelocityTrackingErrorIntegral` (`eden/managers/terms/observations/locomotion.py`) is the shape to
   copy. Function terms have **no** hooks — state in a closure or module global is never reset.
6. **No duplicated computation.** If another term computes the same quantity, make it a `MetricTerm`
   and pull it with `rewards.MetricReward` / `observations.MetricObservation`.
7. **Class term shape.** Every config knob is a class-attribute field with a sane default, and call
   sites use `MyTerm.configure(...)` — not `func=` / `params=`. A term bound to one entity opens its
   configuration errors with `term_subject(self)` (`eden.utils.string`) — `"<Class> on entity '<name>'"`
   — not a hand-written class name.
8. **Registered under its own name.** `@REGISTRY.register()` is present, and takes no `name=` —
   a term's key is always `to_snake_case(__name__)`, the name `configure()` stamps into the options
   and the manager resolves the term by. Converting a function term to a class therefore means
   naming the class after the function (`foo_bar` → `FooBar`); any other class name changes the
   key, and configs naming the old one stop resolving. If the function took `params`, also list
   those keys in `_function_era_params`, or every saved config that set them fails to load.
9. **Output contract.** Shape and dtype right for the manager (observation `(num_envs, dim)`; reward
   `(num_envs,)` float; termination `(num_envs,)` bool; metric `(num_envs,)` float), quaternions
   wxyz, and the index space matches the API you called.
10. **It runs — and isn't secretly slow.** Build the task, take a few `env.step()`s, confirm finite
    outputs, and run any test that covers the term (`uv run pytest tests/test_X.py` — one file at a
    time). For a term doing nontrivial tensor math, also profile it: `eden smoke --task <name>
    --profile` ranks per-term wall cost and writes `<log_dir>/term_cost.json`. A
    correct-but-expensive term (usually the item-3 anti-patterns — per-call `torch.tensor`
    constants, host syncs) is invisible otherwise; the profiler surfaces it before it costs a
    training run. See the profiling section of `docs/guides/terms.md`.
