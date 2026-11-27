---
paths:
  - "eden/extensions/deployment/**"
  - "examples/recipes/deployment/**"
  - "examples/policies/**"
  - "docs/guides/deployment*.md"
---

<!--
Adapted from google-deepmind/robotics-antigravity-plugin, plugins/robotics/rules/physical_robot_safety.md
(Copyright 2026 Google LLC, Apache License 2.0). Modified for Eden: the hardware rules themselves now live in
docs/guides/deployment.md ("Safety"); this file keeps only what an agent does differently from a person.
-->

# Physical robot safety

Applies whenever you run, write or change code that commands a real robot, gripper, hand or motor bus. It does not
apply to `eden deploy --mujoco` or to simulation-only work.

1. **Read `docs/guides/deployment.md`, "Safety", first**, including "Before a first hardware run" and "Scripts that
   command hardware outside `eden deploy`". Code you write follows those rules.
2. **Ask before anything moves.** Before you start a command that moves hardware, tell the user what will move and
   how, and wait for them to confirm that someone is at the robot, the workspace is clear and the e-stop is in
   reach. Approval for one command does not extend to the next.
3. **Never move hardware unattended.** Do not run such a command in the background, from a loop, or from a
   subagent, and do not retry a failed hardware command without asking again.
4. **Rehearse first.** Run the exact command with `--mujoco` before proposing the hardware run, and report what it
   showed.
