# Notes

## Reset Pose and Contact Geometry

The MJX helper named `side_lying_qpos` does apply a 90 degree roll to the free
root, but that does not guarantee the dinosaur is dropped onto its body side.

The checked `+90 deg` pose has:

- root euler approximately `[90, 0, 0]`
- gravity in the IMU frame approximately `[0, -1, 0]`
- lowest collision geoms on the toes/feet, especially toe 03 and toe 04

So the root is rolled, but the initial ground contact is foot/toe dominated. The
rendered videos looking like a foot/toe drop are consistent with the actual
collision geometry. Calling this reset "side lying" is misleading unless the
ground clearance/orientation is adjusted so the torso or side-body collision
geometry is the support contact.

The old PyBullet environment also appears closer to a foot-drop/stand task than
a true side-getup task:

- base position: `[0, 0, 3]`
- base orientation: `[0, 0, 0]`
- starting leg pose:
  - femur left/right: `-0.6`
  - tibia left/right: `0.4`
  - tarsometatarsus left/right: `-1.2`

## Old Objective

The old objective in `old/trex_env.py` is:

```python
reward = -lifting_com_penalty - station_keeping_penalty - energy_penalty
```

With the default training weights from `old/trex_train.py`:

```text
lifting_com_penalty = 200 * (2.5 - head_z)^2
station_keeping_penalty = 1.0 * (head_x^2 + head_y^2)
energy_penalty = 1e-6 * total_joint_power
```

This means the old task primarily rewarded lifting the head toward 2.5 m,
staying near the initial xy location, and using less joint power.

## Current MJX Objective

The current MJX `TrexGetup` objective is different:

- orientation reward
- torso/IMU height reward gated by orientation
- stand-still reward
- action-rate penalty
- actuator-force penalty
- excess joint-velocity penalty
- total reward clipped to be nonnegative

Important differences from the old objective:

- Old uses head height with a target of 2.5 m; MJX uses torso/IMU height with a
  target of 1.0 m.
- Old has xy station-keeping; MJX currently does not.
- Old penalizes joint power; MJX currently penalizes actuator force and excess
  joint velocity.
- MJX directly rewards orientation; old does not.
- MJX clips negative rewards to zero; old rewards can be negative.

## Decision Point

There are two distinct tasks:

1. Reproduce the old foot-drop stand task.
   - Use an upright/flexed reset similar to the old PyBullet setup.
   - Use head-height target, xy drift penalty, and power/energy penalty.
   - Consider removing orientation and stand-still rewards until the old task is
     matched.

2. Build a true side-getup task.
   - Rename or replace the current `side_lying_qpos` behavior.
   - Place torso/neck/side-body collision geometry near the ground, not toes.
   - Recompute clearance per side orientation and verify the first contacts are
     body-side contacts.
   - Use a separate objective appropriate for side recovery.

The safer next step is to reproduce the old task first, then introduce true
side-getup as a separate follow-on task once the standing/drop objective is
well understood.

## MuJoCo Playground Humanoid Joystick Comparison

Reviewed the installed MuJoCo Playground joystick tasks for Apollo, G1, and
Berkeley Humanoid:

- `.venv/lib/python3.12/site-packages/mujoco_playground/_src/locomotion/apollo/joystick.py`
- `.venv/lib/python3.12/site-packages/mujoco_playground/_src/locomotion/g1/joystick.py`
- `.venv/lib/python3.12/site-packages/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py`

The successful humanoid joystick tasks are simpler than the current T-Rex
joystick task:

- Reset starts from a nominal standing keyframe, with small xy/yaw/joint/qvel
  randomization.
- Actions are direct residuals around the default pose:
  `motor_targets = default_pose + action * action_scale`.
- Gait phase is an observation and a reward target, not an action prior.
- Feet are initialized with opposite phases `[0, pi]`.
- Foot phase reward tracks a desired swing-foot height from
  `mujoco_playground._src.gait.get_rz`.
- Feet air-time, swing peak, foot slip, and foot-floor contacts are tracked
  directly.
- Fall termination is always active for joystick locomotion.
- Velocity tracking, orientation/base-height costs, pose costs, and foot rewards
  are mostly direct reward terms rather than being hidden behind an
  already-running gate.

This differs from the current T-Rex joystick task in important ways:

- T-Rex currently mixes getup, standing, and locomotion behavior in one
  environment.
- The running task has a separate standing/moving action branch.
- Several gait rewards are gated by already-upright or already-running state,
  which removes useful shaping when the policy first needs to learn a step.
- Earlier T-Rex runs used an action-space gait prior; the humanoid examples use
  phase as a policy input and reward target instead.

Immediate implication:

- Treat the first locomotion curriculum as a standing-start joystick task, not a
  combined getup task.
- Prefer humanoid-style phase-height and air-time rewards over action-space gait
  priors.
- Keep fall termination enabled during locomotion training.
- Once standing-start locomotion works, connect it back to getup as a follow-on
  phase.

## T-Rex Joystick Locomotion Debugging

Latest committed environment fixes:

- Phase-height reward tolerance now scales with `gait_swing_height`; the old
  fixed denominator made zero clearance look acceptable for low swing heights.
- Foot contact rewards now use foot clearance instead of summed constraint
  force. Warp was reporting high force-contact duty even when clearance showed a
  foot was airborne.
- Added an explicit normalized `phase_clearance_error` cost so dragging the
  commanded swing foot gives a direct penalty.
- Moving-command resets can randomize the initial gait phase to avoid always
  starting on the same requested swing side.
- Added `tools/analyze_open_loop_gait.py` to test scripted gait actions without
  PPO.

Remote run findings:

- Best pre-fix stable policy:
  `/workspace/runs/TrexJoystick-20260514-071303-gaitprior-grounded-5m-n512-lr1e4-ent4e3/checkpoints/000005734400`
  survived 10 s at about 0.2 m/s with low lateral drift, but was a double-support
  shuffle.
- Stronger sagittal gait prior and hip-free runs lifted a foot but caused
  lateral fall; they did not produce a usable gait.
- `Kp=400` gave better open-loop foot response than `Kp=200`; `Kp=800` was too
  unstable.
- Best Kp=400 continuation so far:
  `/workspace/runs/TrexJoystick-20260514-092451-kp400-clearance-20m-cont-n512-lr1e4-ent4e3/checkpoints/000013926400`
  survived 10 s at about 0.17 m/s with low lateral drift, but still kept the left
  foot planted nearly all the time.
- Later Kp=400 checkpoints and random-phase/slow-phase continuations improved
  some reward terms but did not solve left/right alternation.

Current failure mode:

- The model can stand and make slow forward progress.
- It has not learned an alternating bipedal gait.
- Policies repeatedly converge to left-foot-planted/right-foot-swing behavior or
  double-support shuffling.
- Open-loop tests show the actuators can lift either foot, but simple scripted
  single-support motion causes large lateral velocity and falls.

Recommended next step:

- Stop broad PPO sweeps until adding a balance objective or curriculum that
  explicitly teaches stable single support.
- Candidate directions are stance-foot/COM support rewards, staged one-foot
  balance resets, or a separate slow marching task before forward locomotion.

May 14 follow-up:

- Confirmed with a remote Warp open-loop probe that the actuator pattern can lift
  both feet. Large scripted gait amplitudes lifted both feet but fell laterally;
  smaller stable amplitudes produced the same double-support shuffle as PPO.
- Added separate phase reward terms for swing-foot clearance, swing-foot release,
  and stance-foot contact. These are intentionally simpler than the combined
  gait rewards so rollout diagnostics can distinguish "requested swing foot is
  still planted" from "stance foot is not supporting".
- Added phase-binned output to the open-loop gait analyzer. The remote Warp
  sweep showed that the phase-offset-0 action pattern lifts the left foot, while
  offsets around `pi/2` or `pi` reproduce the right-foot-only swing mode.
- Added an explicit gait-prior tracking reward so PPO cannot satisfy stability
  by canceling the alternating action prior back toward the stand pose.
- Added `curriculum_task="march"` mode to `TrexJoystick`. This keeps the same
  action and observation shape as joystick locomotion but replaces the reward
  with an in-place marching objective: low base velocity, upright posture,
  alternating swing-foot release/clearance, stance contact, and single-support
  balance. The intent is to train controlled single support before returning to
  forward velocity tracking.
- Added `fixed_gait_phase` so we can train left- or right-single-support
  subtasks by holding the requested swing phase constant. This is the next
  attempt to break the left-foot-planted local optimum.
- Added sharper swing-foot terms for the fixed-phase curriculum:
  `phase_swing_lift` is linear and exactly zero when the requested swing foot
  stays on the ground, while `phase_swing_contact` directly penalizes contact on
  the requested swing foot.
- Fixed the march curriculum gait prior so it is not disabled by the low
  in-place march command. Previously `_gait_prior_action` was gated by running
  speed, so fixed-support runs with tiny commands had no scripted lift prior.

## Joystick Reset Toward G1/Berkeley Pattern

Reason for reset:

- The custom walking/marching curricula repeatedly found two local optima:
  either a stable stand-still/double-support shuffle, or a lateral fall once
  the rewards pushed hard enough for foot lift.
- Adding stronger gait priors, phase-specific swing rewards, and constrained
  residual actions helped diagnostics but did not produce robust alternating
  gait learning.
- The G1 and Berkeley joystick tasks are much simpler: actions are residuals
  around the default pose, gait phase is an observation/reward target, and the
  reward uses direct velocity, orientation, height, feet phase, air-time, slip,
  pose, and control costs.

What changed locally:

- `TrexJoystick.step` now maps the policy action directly to
  `stand_pose_action + action * action_residual_scale`, clipped to `[-1, 1]`.
- Removed the active standing/moving action branch, stand-pose override, and
  action-space gait-prior injection from the control path.
- Replaced the active reward scale set with humanoid-style terms:
  `tracking_lin_vel`, `tracking_ang_vel`, `orientation`, `base_height`,
  `non_foot_clearance`, `lin_vel_z`, `ang_vel_xy`, `feet_phase`,
  `feet_air_time`, `feet_slip`, `stand_still`, `pose`,
  `hip_adduction_neutral`, `termination`, `action_rate`, `torques`, and
  `dof_vel`.
- Kept the old phase/gait helper methods available for diagnostic scripts, but
  they are no longer part of the active PPO reward.

Validation:

- `uv run python -m py_compile mjx_gym/trex_joystick.py tools/mjx_gym_test.py`
  passed.
- Focused joystick unittest subset passed: reset/step, opt-in fall termination,
  default-pose residual actions, active reward term set, tracking axes,
  anti-phase gait targets, walk command sampling, and environment registration.
- Local JAX one-step smoke passed with joystick observation shapes `(88,)` and
  `(174,)`, finite reward `0.030260900035500526`, `done=0`, and the simplified
  17 reward metrics above.

Next check:

- Do not spend remote GPU credits until approved.
- When compute is available, run a short Warp training smoke from this reset
  and inspect early rollout frames before launching a longer sweep.

## RunPod L40S Setup And Reset Smoke

May 14 L40S setup:

- New L40S pod: `root@103.196.86.48 -p 37586`.
- Repo code was cloned from GitHub at commit `22f7540` instead of copied by
  hand.
- Installing `.venv` under `/workspace/trex-gym` failed with `Stale file
  handle` from the network filesystem. The working layout is to keep the
  executable checkout and `.venv` under `/root/trex-gym`, while reserving
  `/workspace/runs` for training outputs.
- The latest JAX environment pulls CUDA 12.9/13 Python wheels. The L40S host
  driver reported CUDA 12.8, so JAX failed to initialize cuDNN until
  `cuda-compat-13-2` was installed and
  `LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat:$LD_LIBRARY_PATH` was set.
- `/root/trex_env.sh` on the L40S now sets `PATH`, `MUJOCO_GL=egl`, and the
  CUDA 13.2 compat library path.
- Validation passed after sourcing `/root/trex_env.sh`: JAX sees
  `CudaDevice(id=0)`, simple JAX array operations run, and a Warp-backed
  `TrexJoystick` reset/step returns obs shapes `(88,)` and `(174,)` with finite
  reward.

May 14 A5000 reset smoke:

- A5000 pod: `root@69.30.85.239 -p 22081`.
- The pod repo was switched from `git@github.com` to HTTPS and fast-forwarded
  to `origin/mjx` at commit `22f7540`.
- The first 5M-step Warp `TrexJoystick` smoke was stopped during startup
  because Warp repeatedly reported broadphase overflow and requested
  `naconmax` around 16.6k, just above the default 16,384 budget.
- Restarted the smoke in tmux session `train-a5000-reset-smoke` with
  `--playground_config_overrides='{"naconmax":32768,"njmax":1024}'`.
- Run directory:
  `/workspace/runs/TrexJoystick-20260514-154640-humanoid-reset-a5000-5m-nacon32768`.
- Log file: `/workspace/train-a5000-reset-smoke.log`.
- Purpose: test whether the G1/Berkeley-style joystick reset has reward terms
  that move in the right direction before spending L40S time on broader sweeps.

May 14 parallel GPU experiments:

- A5000 baseline reset smoke completed quickly:
  `/workspace/runs/TrexJoystick-20260514-154640-humanoid-reset-a5000-5m-nacon32768`.
  It used default mixed side/standing resets and default joystick command
  sampling with `naconmax=32768,njmax=1024`.
- A5000 baseline rewards: `-67.603`, `-27.341`, `-2.638`, `-20.948`,
  `-22.637`. Runtime reported compile `36.5 s`, train `179.9 s`. This is
  encouraging enough to inspect videos/diagnostics, but the later evals fell
  back from the best reward.
- Started L40S baseline comparison in tmux `train-l40s-reset-baseline`:
  `/workspace/run_l40s_reset_baseline.sh`, log
  `/workspace/train-l40s-reset-baseline.log`, same training/config overrides as
  the A5000 baseline. Purpose: compare throughput/reproducibility on the L40S.
- Started A5000 low-speed standing-start variant in tmux
  `train-a5000-walk-standing`: `/workspace/run_a5000_walk_standing.sh`, log
  `/workspace/train-a5000-walk-standing.log`. Overrides add
  `curriculum_task="walk"`, `reset_standing_prob=1.0`, `terminate_on_fall=true`,
  walk command range `0.15-0.8 m/s`, turn range `+/-0.25 rad/s`, and
  `walk_command_zero_prob=0.1`. Purpose: test a more G1/Berkeley-like
  standing-start low-speed joystick task against the mixed getup/joystick
  baseline.
- A5000 low-speed standing-start seed 0 completed:
  `/workspace/runs/TrexJoystick-20260514-155438-humanoid-reset-a5000-walk-standing-5m`.
  Rewards were `-1.587`, `4.689`, `5.625`, `9.805`, `13.689`; compile
  `40.7 s`, train `187.1 s`. This is the clearest positive learning trend from
  the reset so far.
- Started A5000 low-speed standing-start seed 1 in tmux
  `train-a5000-walk-standing-seed1`, log
  `/workspace/train-a5000-walk-standing-seed1.log`, to check reproducibility.
- L40S baseline comparison started at
  `/workspace/runs/TrexJoystick-20260514-155707-humanoid-reset-l40s-5m-nacon32768`.
  Initial launch failed because `/root/trex_env.sh` referenced unset
  `LD_LIBRARY_PATH` under `set -u`; fixed it to use `${LD_LIBRARY_PATH:-}` and
  restarted successfully.
- L40S baseline completed with rewards `-68.974`, `-25.624`, `-0.096`,
  `-20.885`, `-8.580`; compile `29.0 s`, train `118.7 s`. This reproduces the
  mixed-reset baseline shape and shows the L40S is materially faster than the
  A5000 for this setup.
- Started L40S low-speed standing-start seed 2 in tmux
  `train-l40s-walk-standing-seed2`, log
  `/workspace/train-l40s-walk-standing-seed2.log`, to add a second
  reproducibility check for the promising standing-start walk setup.
