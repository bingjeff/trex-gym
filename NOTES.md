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
