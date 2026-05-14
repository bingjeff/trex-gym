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
