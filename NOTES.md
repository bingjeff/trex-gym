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
- A5000 low-speed standing-start seed 1 completed:
  `/workspace/runs/TrexJoystick-20260514-155953-humanoid-reset-a5000-walk-standing-5m-seed1`.
  Rewards were `-1.589`, `4.809`, `5.849`, `8.495`, `14.652`; compile
  `39.2 s`, train `184.9 s`. This independently reproduces the positive trend.
- Started A5000 low-speed standing-start seed 3 in tmux
  `train-a5000-walk-standing-seed3`, log
  `/workspace/train-a5000-walk-standing-seed3.log`, to keep the A5000 loaded
  while L40S throughput is tuned.
- L40S low-speed standing-start seed 2 completed:
  `/workspace/runs/TrexJoystick-20260514-160306-humanoid-reset-l40s-walk-standing-5m-seed2`.
  Rewards were `-1.161`, `6.036`, `6.667`, `8.649`, `10.943`; compile
  `28.2 s`, train `118.4 s`. This is the third positive standing-start result.
- Started a more aggressive L40S load test in tmux
  `train-l40s-walk-standing-n4096-seed4`: 20M steps, `num_envs=4096`,
  `batch_size=1024`, `num_eval_envs=32`, and contact budget
  `naconmax=131072,njmax=4096`. Purpose: determine whether the L40S should run
  larger batches/env counts rather than the A5000-style 1024-env setup.
- A5000 low-speed standing-start seed 3 completed:
  `/workspace/runs/TrexJoystick-20260514-160555-humanoid-reset-a5000-walk-standing-5m-seed3`.
  Rewards were `-2.110`, `4.221`, `9.204`, `7.599`, `9.328`; compile
  `40.3 s`, train `191.6 s`. This is still positive, though noisier than seeds
  0 and 1.
- The L40S 4096-env run reached 100% GPU utilization, about 17.5 GB VRAM, and
  about 270 W, which is a much better use of the card than the 1024-env runs.
- Started A5000 20M low-speed standing-start seed 5 in tmux
  `train-a5000-walk-standing-20m-seed5`, log
  `/workspace/train-a5000-walk-standing-20m-seed5.log`, to compare a longer
  known-good 1024-env run against the larger L40S run.
- L40S 4096-env seed 4 completed:
  `/workspace/runs/TrexJoystick-20260514-160711-humanoid-reset-l40s-walk-standing-20m-n4096-seed4`.
  Rewards were `-1.194`, `9.401`, `7.660`, `11.934`, `13.184`; compile
  `30.0 s`, train `317.3 s`. Because the runner rounded to 26.2M effective
  steps, this is about 82.6k steps/s, better than the 1024-env L40S runs
  (~55k steps/s) and far better than A5000 1024-env runs (~35k steps/s), but
  still not a 3x A5000 speedup.
- Started L40S 8192-env load test in tmux
  `train-l40s-walk-standing-n8192-seed6`: 10M requested steps,
  `num_envs=8192`, `batch_size=2048`, `naconmax=262144,njmax=8192`. Purpose:
  see whether the L40S throughput continues improving with larger batched
  rollouts or whether 4096 envs is the practical knee.
- L40S 8192-env load test failed after the initial eval with
  `RESOURCE_EXHAUSTED` while allocating 3.25 GiB, despite the larger contact
  budget. That suggests 8192 envs is beyond the practical memory limit for this
  current network/config.
- Restarted L40S at an intermediate load in tmux
  `train-l40s-walk-standing-n6144-seed7`: 20M requested steps,
  `num_envs=6144`, `batch_size=1536`, `naconmax=196608,njmax=6144`. Purpose:
  see whether 6144 envs avoids OOM while improving over the 4096-env throughput.
- A5000 20M low-speed standing-start seed 5 completed:
  `/workspace/runs/TrexJoystick-20260514-161028-humanoid-reset-a5000-walk-standing-20m-seed5`.
  Rewards were `-1.303`, `21.460`, `21.943`, `21.853`, `25.319`; compile
  `37.0 s`, train `385.1 s`. This is the strongest reward trend so far and
  suggests the standing-start curriculum benefits from longer than 5M steps.
- Started A5000 20M turn-range variant in tmux
  `train-a5000-walk-standing-turn05-seed8`, log
  `/workspace/train-a5000-walk-standing-turn05-seed8.log`. It keeps the same
  standing-start low-speed setup but increases `walk_command_turn_max` from
  `0.25` to `0.5 rad/s` to start testing whether the policy can learn a broader
  joystick turn distribution.
- L40S 6144-env seed 7 completed without OOM:
  `/workspace/runs/TrexJoystick-20260514-161737-humanoid-reset-l40s-walk-standing-20m-n6144-seed7`.
  Rewards were `-1.039`, `3.044`, `7.223`, `8.386`; compile `29.2 s`, train
  `330.8 s` for 22.1M effective steps. Throughput was about 66.9k steps/s,
  worse than the 4096-env run (~82.6k steps/s), so 4096 envs is currently the
  better L40S setting.
- Started a longer L40S 4096-env run in tmux
  `train-l40s-walk-standing-50m-seed9`: 50M requested steps, `num_envs=4096`,
  `batch_size=1024`, `naconmax=131072,njmax=4096`. Purpose: see whether the
  standing-start curriculum continues improving with a longer run at the best
  observed L40S throughput setting.
- A5000 turn-range variant completed:
  `/workspace/runs/TrexJoystick-20260514-162022-humanoid-reset-a5000-walk-standing-20m-turn05-seed8`.
  Rewards were `-9.088`, `19.678`, `18.636`, `21.904`, `23.344`; compile
  `42.6 s`, train `380.7 s`. Despite the worse initial eval from broader turn
  commands, it learned to similar final reward as the 0.25 rad/s turn setup.
- Started A5000 speed-range variant in tmux
  `train-a5000-walk-standing-speed12-seed10`, log
  `/workspace/train-a5000-walk-standing-speed12-seed10.log`. It keeps the
  standing-start setup but raises `walk_command_forward_max` from `0.8` to
  `1.2 m/s`.
- L40S 50M 4096-env seed 9 completed:
  `/workspace/runs/TrexJoystick-20260514-162436-humanoid-reset-l40s-walk-standing-50m-n4096-seed9`.
  Rewards were `-1.614`, `8.258`, `17.174`, `13.902`, `13.205`, `15.442`,
  `16.798`, `16.892`; compile `28.0 s`, train `627.7 s` for 57.3M effective
  steps. Throughput stayed high (~91.4k steps/s), but reward plateaued lower
  than the A5000 1024-env 20M seed 5 result.
- Started L40S 2048-env 50M comparison in tmux
  `train-l40s-walk-standing-50m-n2048-seed11`, log
  `/workspace/train-l40s-walk-standing-50m-n2048-seed11.log`. Purpose: test
  whether a smaller L40S batch improves optimization quality while still being
  faster than A5000.
- A5000 speed-range variant completed:
  `/workspace/runs/TrexJoystick-20260514-162849-humanoid-reset-a5000-walk-standing-20m-speed12-seed10`.
  Rewards were `-11.189`, `12.236`, `19.146`, `21.164`, `19.484`; compile
  `33.9 s`, train `374.2 s`. Raising `walk_command_forward_max` to `1.2 m/s`
  is learnable but not better than the current `0.8 m/s` max.
- Started rendering the best current 20M checkpoint from A5000 seed 5:
  `/workspace/runs/TrexJoystick-20260514-161028-humanoid-reset-a5000-walk-standing-20m-seed5/checkpoints/000021299200`.
  Render tmux: `render-a5000-seed5-best`, log
  `/workspace/render-a5000-seed5-best.log`. It will produce a standing
  forward/turn rollout plus a side-reset recovery probe.
- L40S 2048-env 50M seed 11 completed:
  `/workspace/runs/TrexJoystick-20260514-163734-humanoid-reset-l40s-walk-standing-50m-n2048-seed11`.
  Rewards were `-0.649`, `9.561`, `19.044`, `19.112`, `20.373`, `22.342`,
  `18.215`, `20.319`; compile `29.1 s`, train `468.4 s` for 51.6M effective
  steps. This is better reward quality than the L40S 4096-env seed 9 run but
  slower (~110k effective steps/s from train time alone because the effective
  step count rounded differently; wall-clock including eval/JIT remains close
  enough that 4096 is still preferred for raw throughput).
- Rendered the best A5000 standing-start seed 5 checkpoint. The side-reset
  probe frame is just lying on the ground, which is expected because this policy
  was trained from standing starts only. The standing forward/turn frames show
  reward improvement but not a clean gait: the body is low/crouched and one
  frame appears partly collapsed while still moving. This should be treated as
  useful locomotion signal, not a solved walking policy.
- First A5000 mixed-reset warm-start attempt failed before training because
  `--load_checkpoint_path` was pointed at one numbered checkpoint directory.
  Brax's training loader expects the parent `checkpoints` directory and sorts
  its numeric children, so the direct checkpoint path caused
  `ValueError: invalid literal for int() with base 10: 'd'`.
- Restarted A5000 mixed-reset warm-start in tmux
  `train-a5000-walk-mixed-warm-seed13`, log
  `/workspace/train-a5000-walk-mixed-warm-seed13.log`, using the parent
  checkpoint directory and `reset_standing_prob=0.5`.
- Restarted L40S on the best raw-throughput setting in tmux
  `train-l40s-walk-standing-50m-n4096-seed13`, log
  `/workspace/train-l40s-walk-standing-50m-n4096-seed13.log`, to keep the L40S
  fully loaded while the A5000 tests mixed-reset warm-starting.
- A5000 mixed-reset warm-start seed 13 completed:
  `/workspace/runs/TrexJoystick-20260514-164919-humanoid-reset-a5000-walk-mixed-warm-20m-seed13`.
  Rewards were `12.620`, `12.887`, `11.516`, `9.486`, `16.065`; compile
  `38.9 s`, train `344.4 s`. This did not clearly extend the standing-start
  policy into a good mixed side/standing recovery policy.
- Started A5000 lower-frequency gait-shaping run in tmux
  `train-a5000-walk-standing-gaitlow-seed14`, log
  `/workspace/train-a5000-walk-standing-gaitlow-seed14.log`. It keeps the
  standing-start walking setup but sets `gait_frequency_min=0.6`,
  `gait_frequency_max=1.0`, `gait_frequency_per_mps=0.1`, and raises
  `reward_config.scales.feet_phase` to `3.0` plus
  `reward_config.scales.feet_air_time` to `2.0`. Purpose: test whether slower
  phase dynamics and stronger alternating foot clearance produce a more
  interpretable gait.
- L40S 4096-env 50M seed 13 completed:
  `/workspace/runs/TrexJoystick-20260514-164917-humanoid-reset-l40s-walk-standing-50m-n4096-seed13`.
  Rewards were `-1.254`, `8.304`, `9.178`, `14.646`, `10.454`, `10.212`,
  `12.467`, `14.260`; compile `28.7 s`, train `620.1 s`. This repeats the
  pattern that 4096 envs is excellent for throughput but not currently giving
  the best policy quality.
- Started L40S stricter gait/posture shaping run in tmux
  `train-l40s-walk-standing-gaitstrict-seed15`, log
  `/workspace/train-l40s-walk-standing-gaitstrict-seed15.log`. It uses 4096
  envs for a fast 20M requested-step probe, lower gait frequency, stronger
  `feet_phase`/`feet_air_time`, and stronger base height/orientation/pose/slip
  weights.
- A5000 lower-frequency gait-shaping seed 14 completed:
  `/workspace/runs/TrexJoystick-20260514-165706-humanoid-reset-a5000-walk-standing-20m-gaitlow-seed14`.
  Rewards were `0.990`, `41.665`, `51.315`, `59.616`, `66.898`; compile
  `34.3 s`, train `383.8 s`. This is by far the highest scalar reward so far,
  but it is not yet validated visually and may partly reflect the larger
  positive gait reward weights.
- Started A5000 diagnostic render/rollout analysis in tmux
  `render-a5000-gaitlow-seed14` for checkpoint
  `/workspace/runs/TrexJoystick-20260514-165706-humanoid-reset-a5000-walk-standing-20m-gaitlow-seed14/checkpoints/000021299200`.
  It writes analysis to `videos/analyze_forward05.txt` and a forward 0.5 m/s
  video plus frames under that run's `videos/` directory.
- L40S stricter gait/posture shaping seed 15 completed:
  `/workspace/runs/TrexJoystick-20260514-170112-humanoid-reset-l40s-walk-standing-20m-gaitstrict-seed15`.
  Rewards were `-0.696`, `14.180`, `45.577`, `82.131`, `71.389`; compile
  `29.8 s`, train `317.7 s`. The peak checkpoint is currently
  `checkpoints/000019660800`.
- Started L40S diagnostic render/rollout analysis in tmux
  `render-l40s-gaitstrict-seed15` for checkpoint
  `/workspace/runs/TrexJoystick-20260514-170112-humanoid-reset-l40s-walk-standing-20m-gaitstrict-seed15/checkpoints/000019660800`.
  It writes analysis to `videos/analyze_forward05.txt` and a forward 0.5 m/s
  video plus frames under that run's `videos/` directory.
- A5000 gaitlow forward 0.5 m/s diagnostic on checkpoint `000021299200`:
  no termination; mean forward velocity `0.484 m/s`; mean lateral velocity
  `-0.236 m/s`; mean turn `-0.007 rad/s`; torso height range `2.434-2.764`;
  orientation reward range `0.712-0.997`; left/right contact duty
  `0.615/0.590`; phase bins show alternating contact/clearance, but the visual
  frame is still low and crouched with the body pitched forward.
- L40S gaitstrict forward 0.5 m/s diagnostic on checkpoint `000019660800`:
  no termination; mean forward velocity `0.516 m/s`; mean lateral velocity
  `-0.018 m/s`; mean turn `0.003 rad/s`; torso height range `2.289-2.402`;
  orientation reward range `0.939-0.997`; left/right contact duty
  `0.598/0.565`; phase bins show alternating contact/clearance. This is the
  best numeric checkpoint so far, but the frame is still a very low crouched
  gait rather than the final physical posture we want.
- Added posture gating for joystick locomotion rewards in commit `d6f3662`.
  The gate only pays velocity tracking and gait rewards when the torso is near
  target height and upright, plus adds a `low_torso_height` cost. Local tests
  passed (`33 tests OK`).
- L40S anti-crouch warm-start seed 16 from gaitstrict checkpoint:
  `/workspace/runs/TrexJoystick-20260514-172425-anticrouch-l40s-warm-20m-seed16`.
  Rewards were `4.787`, `4.751`, `15.501`, `53.065`, `79.661`; compile
  `28.1 s`, train `318.4 s`. Forward 0.5 m/s diagnostic improved posture
  (`torso_height_range=2.576-2.706`, orientation `0.995-1.000`) but slowed to
  `0.257 m/s` mean forward velocity.
- The anti-crouch checkpoint exposed a zero-command bug: with command `[0, 0]`,
  the policy still received large positive `feet_phase` reward and drifted
  `5.884 m` while "standing". Commit `a7792e9` disables gait/air-time rewards
  for stand commands.
- L40S standfix warm-start seed 17:
  `/workspace/runs/TrexJoystick-20260514-174320-standfix-l40s-warm-20m-seed17`.
  Rewards were `33.930`, `4.300`, `8.938`, `25.422`, `43.124`; compile
  `29.2 s`, train `317.3 s`. Stand drift improved to `3.153 m`, but was still
  not quiet; forward 0.5 m/s remained slow at `0.265 m/s`.
- Added `commanded_stand_still` reward in commit `49ac054`: zero-command base
  linear/angular velocity is rewarded directly, and the reward is zero for
  moving commands.
- L40S quietstand warm-start seed 18:
  `/workspace/runs/TrexJoystick-20260514-180117-quietstand-l40s-warm-20m-seed18`.
  Rewards were `41.523`, `12.764`, `46.437`, `55.726`, `66.965`; compile
  `29.7 s`, train `320.4 s`. Stand diagnostic is now good: no termination,
  mean forward/lateral/vertical velocities `-0.001/0.004/-0.000`, mean turn
  `0.004`, both feet contact duty `1.000/1.000`, and drift `0.047 m`.
  Forward and turning are still underpowered: forward 0.5 m/s gives
  `0.242 m/s`; forward 0.5 plus turn 0.25 gives `0.259 m/s` forward and
  `-0.043 rad/s` turn.
- Started L40S track-rebalance warm-start seed 19 in tmux
  `train-l40s-trackrebalance-warm-seed19`, log
  `/workspace/train-l40s-trackrebalance-warm-seed19.log`. It starts from the
  quietstand checkpoint, keeps the quiet-stand reward, increases velocity and
  turn tracking weights, lowers gait reward dominance, and increases leg action
  residual range. Purpose: recover command tracking without losing quiet stand.
- L40S track-rebalance seed 19 completed:
  `/workspace/runs/TrexJoystick-20260514-182121-trackrebalance-l40s-warm-20m-seed19`.
  Rewards were `75.454`, `76.067`, `76.453`, `112.164`, `134.588`; compile
  `28.4 s`, train `320.7 s`. Diagnostics: stand is no longer as quiet as
  quietstand seed 18 but remains acceptable for a moving-policy checkpoint
  (`0.513 m` drift, both feet contact, no fall). Forward 0.5 m/s tracks at
  `0.413 m/s`; forward 0.5 plus turn 0.25 tracks at `0.413 m/s` and
  `0.251 rad/s`. Forward 0.8 reaches `0.646 m/s`; forward 0.8 plus turn 0.25
  reaches `0.626 m/s` and `0.204 rad/s`. Render frame looks upright with an
  alternating gait posture and no visible collapse. Side reset still fails
  immediately under `terminate_on_fall=true`, so the get-up portion remains
  unsolved.
- L40S mixed/side get-up attempts from the standing-start joystick policy have
  not yet solved recovery. Mixed-reset seed 20 stayed down and damaged standing
  behavior; mixed-reset seed 21 showed partial height recovery after gating
  `commanded_stand_still` by posture but did not become upright.
- L40S side-only warm-start seed 22:
  `/workspace/runs/TrexJoystick-20260514-192334-sidegetup-l40s-warm-20m-seed22`.
  It used only zero commands, `reset_standing_prob=0.0`,
  `terminate_on_fall=false`, and stronger orientation/height/stand-still costs.
  Final side diagnostic still failed: episode reward `-421.604`,
  orientation reward range `0.002-0.069`, torso height range `0.443-2.168`,
  mean vertical velocity `0.853 m/s`, and base displacement `7.612 m`. The
  policy learned energetic motion but not a controlled get-up.
- Added zero-default explicit get-up shaping terms to `TrexJoystick` so the
  side-getup curriculum can reuse the proven `TrexGetup` style terms without
  changing joystick observations or actions: `getup_torso_height`,
  `getup_foot_support`, `getup_foot_balance`, `getup_foot_placement`, and
  `getup_standing_pose`. Focused joystick reward tests passed.
- L40S side-getup-shaped seed 23 turned on those terms from mixed-reset seed 21:
  `/workspace/runs/TrexJoystick-20260514-194000-sidegetup-shaped-l40s-warm-20m-seed23`.
  Reward improved from `-276.094` to `-208.860`, and the side diagnostic
  improved to `episode_reward_sum=-141.274` with torso height reaching
  `2.875 m`. It still failed to get upright: orientation reward only reached
  `0.364`, base displacement was `4.976 m`, and rendered frames showed a
  rolling/flinging maneuver rather than a controlled stand-up.
- L40S dedicated `TrexGetup` seed 24:
  `/workspace/runs/TrexGetup-20260514-195323-getup-l40s-20m-seed24`.
  Reward improved from `1.276` to `11.594`, but diagnostics showed the same
  failure mode: orientation reward stayed `0.008-0.092`, torso height only
  reached `2.068 m`, and base displacement was `6.220 m`.
- Root cause found: `non_foot_clearance` was paid even when the body was not
  upright. In the dedicated get-up run this dominated the objective
  (`reward/non_foot_clearance=443.924`) and rewarded flinging the body/head/tail
  away from the ground without standing. Fixed both `TrexGetup` and
  `TrexJoystick` so non-foot clearance is multiplied by the upright orientation
  reward. Added a focused test proving side-lying clearance no longer pays.
- L40S corrected `TrexGetup` seed 25:
  `/workspace/runs/TrexGetup-20260514-200431-getup-gatedclearance-l40s-20m-seed25`.
  With the clearance gate fixed, scalar reward was no longer inflated
  (`-0.087` to `3.211`). The rollout still failed: orientation reward range
  `0.006-0.233`, torso height range `0.661-1.532`, and base displacement
  `6.977 m`.
- The earlier local successful joystick checkpoints from `CHECKPOINTS.md` were
  copied to the L40S for warm-start reuse. They were trained with 86 state
  observation entries and 172 privileged entries, while current `TrexJoystick`
  emits 88 and 174. Added `tools/upgrade_joystick_checkpoint.py` to pad running
  statistics and first-layer kernels with neutral values for newly appended
  observation features. Local check-load passed on the upgraded
  `TrexJoystick-20260513-044426-joystick-warm-10m` checkpoint.
- Direct-action compatibility is required for these older joystick checkpoints:
  current `TrexJoystick.step` interprets policy output as residual around
  `stand_pose_action`, while the old checkpoints produced direct action
  targets. With `stand_pose_action=[0]*10` and `action_residual_scale=[1]*10`,
  the upgraded joystick-warm checkpoint still gets upright from side reset, but
  it drifts and tracks poorly.
- L40S compat warm-start seed 27:
  `/workspace/runs/TrexJoystick-20260514-204301-compat-warm-l40s-20m-seed27`.
  This fine-tuned the upgraded joystick-warm checkpoint with direct-action
  compatibility. Final standing stop is useful: no termination, forward
  velocity `-0.005 m/s`, turn `0.023 rad/s`, base displacement `0.221 m`,
  height `2.574-2.637`, and orientation `0.977-0.999`. Side reset also gets
  upright, but drifts too far (`5.324 m`) and orientation is only
  `0.883-0.936`. Locomotion remains too weak: forward `0.5 m/s` command reaches
  only `0.137 m/s`.
- The upgraded speed-only checkpoint still runs fast under direct-action
  compatibility (`11.711 m/s` for a `10 m/s` command) but fails stand/low-speed
  behavior and uses mostly one foot, so it is not a good base for the current
  objective.
- L40S compat locomotion seed 28:
  `/workspace/runs/TrexJoystick-20260514-210346-compat-locomotion-l40s-20m-seed28`.
  This tried to restore locomotion from seed 27 with standing-only resets and
  stronger tracking/gait terms. It failed: final checkpoint terminated on a
  standing stop (`first_done_step=324`), with orientation reward only
  `0.001-0.018` and torso height `0.407-1.103`. Do not use seed 28.
- L40S compat balanced seed 29:
  `/workspace/runs/TrexJoystick-20260514-211706-compat-balanced-l40s-20m-seed29`.
  This tried a more conservative locomotion fine-tune from seed 27. It also
  regressed: standing-stop diagnostic had large drift (`3.898 m`), height range
  `0.394-2.845`, and orientation range `0.000-1.000`, indicating intermittent
  collapse. Do not use seed 29.
- Rendered seed 27 side-stop, standing-stop, and forward-0.5 videos/frames:
  `/workspace/runs/TrexJoystick-20260514-204301-compat-warm-l40s-20m-seed27/videos/`.
  Visual inspection matches the metrics: seed 27 is the best current branch, but
  it is still a crouched, mostly static posture with weak leg motion and does
  not satisfy the final gait/locomotion goal.
- L40S compat balanced low-LR seed 30:
  `/workspace/runs/TrexJoystick-20260514-213601-compat-balanced-lowlr-l40s-20m-seed30`.
  This repeated the conservative direct-action fine-tune from seed 27 with
  lower learning rate (`5e-5`) and entropy (`0.005`). Scalar reward regressed
  from `15.056` to `-382.468` by the final checkpoint, with intermediate values
  `-283.672`, `-440.876`, and `-243.812`. Do not use seed 30.
- L40S residual side-getup fixed-clearance seed 31:
  `/workspace/runs/TrexJoystick-20260514-214426-residual-sidegetup-fixed-l40s-20m-seed31`.
  This returned to the current residual-action semantics and warm-started from
  the standing-start locomotion checkpoint
  `/workspace/runs/TrexJoystick-20260514-182121-trackrebalance-l40s-warm-20m-seed19/checkpoints`.
  It used side-only reset, zero command, `terminate_on_fall=false`, corrected
  non-foot clearance, and explicit get-up terms. Rewards improved from
  `-576.366` to `-323.112`, but the final side diagnostic was not upright:
  episode reward `-181.824`, torso height `0.508-1.829`, orientation
  `0.007-0.129`, and base displacement `2.508 m`.
- L40S residual side-getup scale-1 seed 32:
  `/workspace/runs/TrexJoystick-20260514-215420-residual-sidegetup-scale1-l40s-20m-seed32`.
  This continued seed 31 with full residual action scale (`[1]*10`). Scalar
  rewards were `-516.380`, `-313.647`, `-239.015`, `-378.772`, and `-256.498`.
  The `000013107200` checkpoint was the best scalar checkpoint and reached much
  higher recovery metrics than seed 31: episode reward `-77.759`, torso height
  `0.512-2.999`, orientation `0.009-0.928`, base displacement `2.265 m`, and
  left/right contact duty `0.440/0.120`. The final `000026214400` checkpoint had
  episode reward `-50.718`, torso height `0.466-2.573`, orientation
  `0.002-0.959`, base displacement `3.732 m`, and left/right contact duty
  `0.595/0.448`. Seed 32 is the best corrected get-up branch so far, but it is
  not yet a successful checkpoint because it only reaches upright
  intermittently and still has too much drift/instability.

May 14 reset summary:

- Current status: no May 14 checkpoint satisfies the full target. The combined
  objective remains unsolved: clean get-up from a fall, quiet stand, full-speed
  running with moderate turning, and bipedal alternating footfalls in one policy.
  Do not promote any of the May 14 runs to `CHECKPOINTS.md`.
- L40S cleanup status at reset: no Trex training/render/analyze processes, no
  tmux sessions, and GPU idle (`0%`, `1 MiB` memory used).
- Preserved committed code changes since the last logged experiments:
  `e04ed7b` added zero-default `getup_base_lin_vel` and `getup_base_ang_vel`
  reward terms to `TrexJoystick`; `038edb9` added zero-default
  `getup_non_foot_clearance_deficit`. Focused reward-term tests passed for both.
  These terms only affect behavior when explicitly enabled in run config.
- Seed 32 visual diagnosis:
  `/workspace/runs/TrexJoystick-20260514-215420-residual-sidegetup-scale1-l40s-20m-seed32/videos/`.
  The 13.1M checkpoint was a fling: it reached height/orientation briefly but
  with high vertical velocity and poor control. The final checkpoint was quieter
  but remained a low crouch, propped by tail/body geometry rather than standing
  on the feet. This explained why scalar metrics looked partially promising but
  the rollout was not physically acceptable.
- Seed 33:
  `/workspace/runs/TrexJoystick-20260514-221845-residual-sidegetup-damped-l40s-10m-seed33`.
  It continued from seed 32 final with lower LR, earlier stillness gate, and
  enabled base linear/angular velocity costs. Scalar reward started at
  `-204.481`, then degraded to `-314.712` final. The 3.27M checkpoint was the
  only useful one: side diagnostic episode reward `-5.512`, near-zero base
  velocity, `0.261 m` displacement, torso height `1.969-1.982`, orientation
  `0.823-0.838`, and both feet in contact. Visual inspection showed it was a
  quiet, tail/body-propped crouch, not an upright stand.
- Seed 34 failed startup:
  `/workspace/runs/TrexJoystick-20260514-223516-residual-sidegetup-clearance-l40s-10m-seed34`.
  The run stopped before restore/training because the load path pointed at a
  specific numeric checkpoint instead of a parent checkpoint directory.
- Seed 34b:
  `/workspace/runs/TrexJoystick-20260514-223732-residual-sidegetup-clearance-l40s-10m-seed34b`.
  It restarted from a one-checkpoint warm-start directory pointing at seed 33's
  3.27M checkpoint, enabled strong non-foot-clearance deficit cost, and increased
  height/orientation shaping. Rewards were `-415.131`, `-397.516`, `-368.256`,
  `-181.799`, then `-628.442`. The best scalar checkpoint (`000009830400`) had
  episode reward `-49.698`, near-zero velocities, `1.205 m` displacement, height
  `1.883-2.283`, and orientation `0.485-0.843`, but non-foot clearance remained
  zero and visual inspection still showed a body/tail-propped posture.
- Practical conclusion from seeds 32-34b: added damping/stillness can suppress
  the fling, and stronger height/clearance shaping can improve height, but PPO
  keeps finding a local optimum where the body or tail props the dinosaur up.
  The reward-only approach is not reliably discovering a clean transition from
  side-lying to foot-supported standing.
- Last uncommitted local direction before the reset: a reset-curriculum hook was
  being added to `TrexGetup`/`TrexJoystick` to sample partially rolled-upright
  side starts and optionally blend side-start joints toward the standing pose.
  The idea was to train intermediate recovery states before full side-zero
  get-up. This work is dirty in the local tree and should be reviewed, kept, or
  discarded deliberately before more training.
- Recommended reset decision points:
  use seed 19 only as the best known standing-start residual locomotion base;
  do not continue seed 28-30; treat seed 33 3.27M and seed 34b 9.83M as
  diagnostics only; strongly consider changing the task formulation before more
  GPU time, for example a staged reset curriculum, explicit contact-state
  objective for "only feet touching", or splitting get-up and locomotion into
  separate skills before attempting a unified joystick policy.

May 14 policy reset implementation:

- Implemented the first infrastructure step of the reset plan: `TrexGetup`,
  `TrexBalance`, `TrexWalk`, `TrexJoystick`, and `TrexRun` are now separate
  trainable environment names rather than one overloaded joystick task.
- `TrexBalance` starts from standing, samples only zero commands, enables
  modest randomized push perturbations, and emphasizes quiet standing,
  non-foot clearance, orientation, and height.
- `TrexWalk` starts from standing and samples straight-line 0.5-1.5 m/s
  commands with the gait-phase rewards turned up and yaw commands disabled.
- `TrexJoystick` is now the moderate velocity-steered task, capped at 3 m/s
  forward speed with yaw-rate commands and mild push perturbations.
- `TrexRun` is the separate high-speed task, sampling 3-10 m/s forward commands
  with only small yaw commands.
- PPO defaults were reset toward MuJoCo Playground humanoid practice:
  `(512, 256, 128)` actor/critic networks, 4096 envs, 1024 batch size, 32
  minibatches, clipping epsilon 0.2, and longer task-specific budgets.
- Rollout and render diagnostics now accept `--task` for the joystick-derived
  tasks, so policy checks can be run against the same task class that was
  trained.
- Local baseline checkpoint verification was rerun with CPU/JAX and an absolute
  checkpoint path. The saved `TrexGetup-20260513-033812-still2-warp-10m`
  checkpoint still reaches upright/feet-supported standing under current code:
  over steps 500-750 for seed 0, torso height was `2.454-2.749`, orientation
  reward `0.942-1.000`, foot-floor contact occurred on 244/250 sampled steps,
  non-foot floor contact was zero, and minimum non-foot clearance was `0.074 m`.
  It is still not a quiet stand by the stricter new gate: base displacement was
  `5.407 m`, mean base linear speed `1.273 m/s`, and mean angular speed
  `0.481 rad/s`.

May 15 single-policy reset results:

- Committed `167b40f2eaab9493b5218489bac9647b1c574a7c` to apply the walking
  gait prior as an action center for moving `TrexWalk` commands. The previous
  code only used `_gait_prior_action()` as a reward target, so the walking
  policy had to discover leg cycling from residual noise around a static stand
  pose. A focused regression test now verifies that a zero residual action in
  `TrexWalk` produces `stand_pose_action + gait_prior_action` for a moving
  command, while the zero-command default stand behavior remains unchanged.
- Lowered the first `TrexWalk` curriculum to `0.25-0.8 m/s`, reduced the
  forward-speed deficit cost, and strengthened orientation/height penalties.
  This intentionally targets a reliable low-speed velocity-guided walk before
  expanding to faster commands.
- Remote training ran on the L40S with Warp from the verified balance checkpoint:
  `/workspace/runs/TrexWalk-20260515-011532-walk-gaitprior-20m-from-balance`.
  Checkpoints and logs were written under `/workspace/runs` per the storage
  directive. The eval reward sequence was `-710.969`, `-657.894`, `-540.841`,
  `-9.190`, `21.782`; final checkpoint `000026214400` was selected.
- Final walking checkpoint diagnostics, standing reset, seed 0, final 500 steps:
  command `0.25 m/s` tracked at `0.255 m/s`, command `0.5 m/s` tracked at
  `0.494 m/s`, and command `0.8 m/s` moved forward at `0.611 m/s`. The first two
  are acceptable low-speed velocity-guided walking checks; the 0.8 m/s rollout
  is forward but degraded and should not be treated as high-speed success.
- Rendered videos with `MUJOCO_GL=egl` under:
  `/workspace/runs/TrexWalk-20260515-011532-walk-gaitprior-20m-from-balance/videos/`.
  The local copies are in
  `checkpoints/TrexWalk-20260515-011532-walk-gaitprior-20m-from-balance/videos/`.
  Visual frame inspection of the 0.5 m/s rollout showed upright posture with
  feet under the body rather than the prior fall/spin failure.
- Copied the verified balance checkpoint locally from
  `/workspace/runs/TrexBalance-20260514-235829-balance-10m-buf-best/checkpoints/000009830400`
  to `checkpoints/TrexBalance-20260514-235829-balance-10m-buf-best/000009830400`.
  A local JAX smoke test loaded it and ran 50 standing steps with no termination,
  `0.023 m` XY displacement, torso height `2.772-2.852 m`, and orientation
  reward `0.997-1.000`.
- Copied the verified walking checkpoint locally to
  `checkpoints/TrexWalk-20260515-011532-walk-gaitprior-20m-from-balance/000026214400`.
  A local JAX smoke test loaded it and ran 50 standing-start steps at command
  `0.5 m/s` with no termination, mean forward velocity `0.447 m/s`, mean
  forward error `0.053 m/s`, torso height `2.639-2.853 m`, and orientation
  reward `0.966-1.000`.
- Current status against the reset plan: get-up from ground has a verified older
  checkpoint, balance has a verified new single-policy checkpoint, and low-speed
  velocity-guided walking now has a verified single-policy checkpoint. Remaining
  work is to expand beyond low-speed walking into turning and running, and later
  combine get-up, balance, and walk only after each single skill is stronger.

May 15 expanded walking phase:

- Started a straight-line `TrexWalk` continuation from the verified gait-prior
  walking checkpoint. Remote run:
  `/workspace/runs/TrexWalk-20260515-024930-walk-expand-30m-from-gaitprior`.
  Overrides widened the speed curriculum to `0.25-1.2 m/s`, kept turn commands
  at zero, reduced zero-command sampling to `0.05`, and raised gait frequency
  scaling/max slightly.
- The final checkpoint `000032768000` had the best scalar eval reward:
  `-48.491`, `-222.306`, `-255.569`, `-99.695`, `11.538`, `95.853`.
- Remote Warp diagnostics, standing reset, seed 0, final 500 steps:
  `0.25 m/s` command tracked at `0.329 m/s`, `0.5 m/s` at `0.506 m/s`,
  `0.8 m/s` at `0.755 m/s`, and `1.0 m/s` at `0.943 m/s`, all with no
  termination and upright orientation ranges above `0.937` except the lower
  commands, which were stronger. The `1.2 m/s` command failed to track
  (`0.457 m/s`, orientation `0.595-0.784`, one foot effectively stuck).
- Rendered EGL videos for `0.8 m/s` and `1.0 m/s`. Visual frame inspection of
  the `1.0 m/s` sample showed the model upright with feet under the body, not a
  fall/spin exploit.
- Copied the checkpoint and videos locally to
  `checkpoints/TrexWalk-20260515-024930-walk-expand-30m-from-gaitprior/`.
  A local JAX smoke test loaded the checkpoint and ran 50 steps at `1.0 m/s`
  with no termination. Short-horizon local mean speed was `0.791 m/s`; the
  longer remote Warp diagnostic remains the primary settled-speed evidence.
- This completes the straight-line walking expansion phase through about
  `1.0 m/s`. Next phase should introduce turning/joystick commands from this
  checkpoint, not push speed further yet.

May 15 signed joystick turning phase:

- First joystick-turn attempt:
  `/workspace/runs/TrexJoystick-20260515-032050-joystick-turn-30m-from-walk-expand`.
  It improved scalar rewards from `-631.991` to `134.404` and preserved walking,
  but fixed-command diagnostics showed a yaw sign failure: both positive and
  negative turn commands produced positive yaw rates. This was not promoted.
- Tightened `joystick_config()` in commit `a01ce05`: reduced command range to
  `0.25-0.8 m/s`, required nonzero turn samples, reduced turn max to
  `0.25 rad/s`, narrowed turn tracking sigma to `0.08`, and increased
  `tracking_ang_vel` to `6.0`.
- Second joystick-turn run:
  `/workspace/runs/TrexJoystick-20260515-034759-joystick-turn2-30m-from-turn1`.
  Eval rewards were `217.593`, `223.973`, `237.344`, `220.135`, `244.018`,
  `240.361`; the best scalar checkpoint was `000026214400`.
- Remote fixed-command diagnostics on `000026214400` verified signed yaw:
  `(0.5, +0.25)` tracked at forward `0.514 m/s`, yaw `0.195 rad/s`;
  `(0.5, -0.25)` tracked at forward `0.511 m/s`, yaw `-0.219 rad/s`;
  `(0.8, +0.25)` tracked at forward `0.816 m/s`, yaw `0.195 rad/s`;
  `(0.8, -0.25)` tracked at forward `0.786 m/s`, yaw `-0.219 rad/s`.
  All had no termination and orientation reward ranges above `0.992` except
  straight standing, which was also stable.
- Rendered left/right videos under
  `/workspace/runs/TrexJoystick-20260515-034759-joystick-turn2-30m-from-turn1/videos/`
  and copied the checkpoint/videos locally to
  `checkpoints/TrexJoystick-20260515-034759-joystick-turn2-30m-from-turn1/`.
- Local joystick entrypoint smoke test passed:
  `tools/drive_joystick_policy.py ... --check-load --impl jax` loaded the
  checkpoint, produced action size `10`, and stepped once. A short local rollout
  at `(0.5 m/s, +0.25 rad/s)` also tracked signed yaw with no termination.
- This completes the first standing-start joystickable policy. It does not get
  up from the ground; the remaining combined-policy phase must add recovery or
  orchestration between the get-up/balance policy and the joystick policy.

May 15 combined recovery and joystick phase:

- Added `tools/drive_combined_policy.py` as an orchestrated local fallback that
  can load get-up, optional balance, and joystick checkpoints. The first checks
  showed why a learned combined policy was still needed: direct handoff from the
  older get-up checkpoint destabilized, and the balance policy was not robust to
  the recovered get-up state.
- Combined PPO attempt 1:
  `/workspace/runs/TrexJoystick-20260515-043020-combined1-side-joystick-40m`.
  It learned side-start moving-command recovery and preserved signed joystick
  behavior, but side-start zero command failed to stand.
- Combined PPO attempt 2:
  `/workspace/runs/TrexJoystick-20260515-045113-combined2-zero-recovery-30m`.
  It increased zero-command sampling and get-up/stand rewards. Standing-start
  behavior and side-start moving commands remained good, but side-start zero
  command still failed.
- Added recovery-only residual scaling in commit `1c19c49`: while posture is
  not upright, the joystick task uses `recovery_action_residual_scale`
  (`0.75` for legs, `1.0` for tail) so neutral-command side recovery is not
  constrained to tiny residuals around the static standing pose. Once upright,
  the normal command-dependent residual scale is restored.
- Combined PPO attempt 3:
  `/workspace/runs/TrexJoystick-20260515-051704-combined3-recovery-scale-30m`.
  Eval rewards improved from `-25.925` to `360.719`.
- Final checkpoint `000032768000` passed all required gates in remote Warp
  diagnostics, seed 0, final 500 steps of 1000-step rollouts:
  standing zero stayed quiet; standing `(0.5,+/-0.25)` tracked signed turns;
  side zero recovered to quiet standing; side `(0.5,+/-0.25)` recovered and
  tracked signed turns. Side-zero final metrics were forward `-0.001 m/s`,
  turn `-0.013 rad/s`, base displacement `0.160 m`, torso height
  `2.506-2.589 m`, and orientation reward `0.992-1.000`.
- Copied the combined checkpoint and videos locally to
  `checkpoints/TrexJoystick-20260515-051704-combined3-recovery-scale-30m/`.
  `tools/drive_joystick_policy.py` now uses `joystick_config()` so local
  interactive driving matches the newer joystick/combined checkpoints. Local
  `--check-load --start side` succeeds with the final combined checkpoint.
- Current combined limitations: the verified speed/turn envelope is moderate
  (`0.5 m/s`, `+/-0.25 rad/s`). This is a successful joystickable combined
  baseline, not the final high-speed run policy.

May 15 high-speed TrexRun phase:

- Added stricter `TrexRun`-only reward shaping after the first 1-4 m/s run
  learned a bounding/floating solution: fall termination is enabled for run
  training, non-foot clearance no longer gives a positive run reward, and
  vertical velocity, tilt angular velocity, excess running height, no-foot
  contact, and contact-duty error are penalized more strongly.
- Added run-only contact-duty reward terms and a running-height-excess cost to
  the joystick reward dictionary. These default to zero outside the run config.
- Increased run-only leg action authority and gait prior for the high-speed
  curriculum after diagnostics showed the policy was touching action limits:
  `running_action_residual_scale` is now `0.75` for the eight leg actions,
  `gait_prior_scale` is `0.65`, `gait_frequency_per_mps` is `0.12`, and
  `gait_frequency_max` is `2.4`.
- The useful training sequence was:
  - `run4-strict-1to3`: consolidated clean tracking through `3.0 m/s`.
  - `run5-strict-1p5to3p5`: passed the `3.5 m/s` gate.
  - `run6-strict-2to4p5`: reached about `4.0 m/s` reliably.
  - `run9-speed-4to7`: reached about `5.6 m/s`.
  - `run10-capacity-5to8`: reached about `6.1 m/s`.
  - `run11-speed-6to10`: reached the high-speed regime, tracking `8.0 m/s`
    almost exactly but plateauing near `8.0 m/s` for a `10.0 m/s` command.
  - `run12-speed-8to10`: improved the high-speed mode but still did not solve
    `10.0 m/s` command tracking.
- Best current high-speed checkpoint copied locally:
  `checkpoints/TrexRun-20260515-093556-run12-speed-8to10-60m-from-run11/000117964800`.
- Final run12 fixed-command diagnostics, Warp, standing reset, seed 0, final
  500 steps:
  - command `8.0 m/s`: mean forward `9.583 m/s`, no termination, torso height
    `2.052-3.086 m`, orientation reward `0.868-1.000`.
  - command `9.0 m/s`: mean forward `8.595 m/s`, no termination, torso height
    `2.098-2.933 m`, orientation reward `0.848-1.000`.
  - command `10.0 m/s`: mean forward `8.622 m/s`, no termination, torso height
    `2.043-2.930 m`, orientation reward `0.836-1.000`.
- Additional seed checks at `10.0 m/s` on run12 gave `8.406 m/s` and
  `8.342 m/s`, confirming the undertracking is not just one rollout seed.
- Videos and frames for `8.0` and `10.0 m/s` were rendered and copied under
  `checkpoints/TrexRun-20260515-093556-run12-speed-8to10-60m-from-run11/`.
  The high-speed policy is stable, but visually it still looks more like a
  high-speed bounding mode than a fully satisfactory alternating run.
- Added `--task run` to `tools/drive_joystick_policy.py` so a `TrexRun`
  checkpoint can be loaded locally with the same run action scaling used during
  training. Local JAX `--check-load --task run` passed for the run12 checkpoint.
- Extended `tools/analyze_joystick_rollout.py` to print per-actuator action
  saturation, applied target saturation, control ranges, actuator force ranges,
  and leg joint qpos/qvel ranges. On the run12 `10.0 m/s` command, the key
  bottleneck signs were:
  - hip flexion applied targets saturated for roughly `35-45%` of sampled
    steps, with max actuator forces around `2.2e6`.
  - ankle applied targets saturated for roughly `45-47%` of sampled steps.
  - raw actions touched `+/-1` on most leg channels, but average action was only
    about `0.76`, so this is not a pure policy-output saturation problem.
  - gait anti-phase remained low (`0.15`) and phase/contact mismatch remained
    high.
- Tried one bounded gait-discipline continuation from run12:
  `/workspace/runs/TrexRun-20260515-102103-run13-gait-8to10-40m-from-run12`.
  It increased `gait_prior_tracking`, `gait_anti_phase`, `gait_symmetry`,
  `leg_action_alternation`, `phase_contact`, `phase_contact_error`, and
  `feet_phase` weights. This failed: eval reward degraded from `11.551` to
  `-54.790`, and a final `10.0 m/s` gate terminated at step `635`, averaged
  only `2.346 m/s`, and dropped torso height to `0.419 m`.
- Inspected the generated MuJoCo model. Action actuators are not force-limited,
  hip/femur position actuator gains are around `0.9-1.1e6`, knee gains around
  `0.53e6`, and ankle gains around `0.119e6`. Foot contact friction is already
  high at `[3.0, 0.1, 0.1]`, so ordinary sliding friction is not the obvious
  limiting parameter.
- Replayed the best run12 policy with higher global `Kp` values as a model
  sensitivity check. This failed immediately: `Kp=300` terminated at step `274`,
  `Kp=400` at step `88`, and `Kp=600` at step `24`. A global position-gain
  increase is not compatible with the learned policy and should not be treated
  as a simple fix.
- Current blocker: the high-speed policy has not met the original `10 m/s`
  target. It is a useful experimental checkpoint for roughly `8-9 m/s`
  straight-line high-speed locomotion, but the next step should be a
  model/control investigation rather than another blind PPO continuation.

May 15 ankle-gain adaptation:

- Added actuator-specific position-gain scaling so individual leg actuators can
  be tested without globally changing every joint. Focused tests verify that
  only named actuator gains are scaled.
- Trained run14 from run12 with ankle actuators at `1.5x` `Kp`:
  `/workspace/runs/TrexRun-20260515-110344-run14-ankle1p5-8to10-60m-from-run12`.
- The training scalar improved from `-58.824` to a peak of `16.489` at
  checkpoint `000104857600`, then fell to `9.314` at the final checkpoint.
- Fixed-command gate for the best scalar checkpoint:
  - `8.0 m/s` command: `9.694 m/s`, no termination.
  - `9.0 m/s` command: `9.127 m/s`, no termination.
  - `10.0 m/s` command: about `8.9 m/s`, no termination.
- Fixed-command gate for the final checkpoint:
  - `8.0 m/s` command: `10.217 m/s`, no termination.
  - `9.0 m/s` command: `9.212 m/s`, no termination.
  - `10.0 m/s` command: `9.089 m/s`, no termination.
- The best `10.0 m/s` diagnostic still has low gait anti-phase (`0.131`),
  substantial phase/contact mismatch, high no-foot-contact penalty, and long
  flight phases in rendered frames. Hip flexion targets are still often at
  applied limits and the ankles are near applied saturation about half the
  sampled steps.
- Conclusion: targeted ankle authority is a useful diagnostic direction, but
  run14 is not a solved high-speed running policy. The next high-speed step
  should change the model/control formulation or the gait template, not merely
  continue PPO from run14.

May 15 anti-flight reward rebalance:

- Trained run15 from run14 final with ankle `1.5x` still enabled, lower
  velocity/progress reward, and much stronger no-foot-contact, vertical
  velocity, excess-height, phase-contact-error, contact-duty-error, and
  foot-slip penalties:
  `/workspace/runs/TrexRun-20260515-114027-run15-antiflight-ankle1p5-30m-from-run14`.
- This did not recover. Eval reward started at `-289.565`, improved only to
  `-264.810`, and degraded to `-303.267` by the final checkpoint.
- The least-bad checkpoint still averaged `9.097 m/s` on an `8.0 m/s` command
  and `9.190 m/s` on a `10.0 m/s` command. Gait anti-phase stayed low
  (`0.127-0.132`) and torso height/contact behavior remained similar to run14.
- Conclusion: this is not just a scalar weight issue. The existing high-speed
  solution family remains a long-flight bounding mode even when made expensive.
  Further progress likely needs a better gait generator/template, different
  action parameterization, or model/contact changes rather than more PPO
  continuations from run12/run14.

May 15 forward-reward support gate:

- Added `gate_forward_rewards_by_support` and enabled it only for `TrexRun`.
  When enabled, positive `tracking_lin_vel`, `tracking_forward_vel`, and
  `forward_progress` rewards are multiplied by foot-support and running-height
  gates. This prevents unsupported/too-high flight from earning the main
  forward reward.
- Added phase-binned raw and applied action means to
  `tools/analyze_joystick_rollout.py`. On run14, the learned policy has
  phase-dependent actions, but they remain asymmetric and frequently hit
  applied hip/ankle limits; it is not simply ignoring the gait clock.
- A small open-loop gait-center probe showed the current sinusoidal gait prior
  is not itself a high-speed controller. Most tested variants fell within about
  70-113 steps and moved under `0.5 m/s`; the only stable variant was a
  double-support shuffle around `0.05 m/s`.
- Trained run16 with the new support/height gate from run14:
  `/workspace/runs/TrexRun-20260515-122319-run16-gatedforward-ankle1p5-30m-from-run14`.
  Reward improved only from `-60.833` to `-50.749`, then stayed negative.
- The least-bad run16 checkpoint averaged `9.073 m/s` for an `8.0 m/s` command
  and `8.477 m/s` for a `10.0 m/s` command; gait anti-phase stayed low
  (`0.130-0.154`).
- Conclusion: support/height gating is probably a correct objective fix, but it
  does not salvage the current high-speed checkpoint family. The next attempt
  should develop a better high-speed gait template or action parameterization
  before spending more PPO time.

May 15 phase-template replay:

- Extended `tools/analyze_open_loop_gait.py` with:
  - `--task` so open-loop probes can instantiate `TrexRun` directly.
  - `--phase-template` for JSON phase-binned action tables.
  - `--template-space raw|applied` so both raw policy-action means and final
    applied-action means can be replayed.
  - `--initial-forward-velocity` for sustain tests.
- Replayed run14 phase-binned raw action means and applied action means from
  the successful-looking high-speed rollout.
- From rest:
  - raw template: no termination over 500 steps, but only `0.274 m/s`.
  - applied template: termination at step `168`, only `0.336 m/s`.
- With initial forward velocity:
  - `5.0 m/s`: raw fell at step `30`; applied fell at step `27`.
  - `9.0 m/s`: raw fell at step `20`; applied fell at step `18`.
- Conclusion: the high-speed policy is not reducible to a phase-only open-loop
  template. It depends on closed-loop state feedback and likely exploits a
  narrow dynamic mode. A better prior probably needs a different structure
  rather than averaging the existing policy's actions by gait phase.

May 15 phase-action-center control hook:

- Added `phase_action_center` to the joystick config. It is disabled by
  default. When populated with a phase-binned table of applied actions, moving
  commands use the interpolated table as the action center, then apply the
  policy residual action exactly as before.
- This keeps the action size at 10, so existing training/drive/render plumbing
  still works. It also means a future policy can learn closed-loop corrections
  around an explicit gait template without needing a new policy architecture.
- Focused tests verify that:
  - zero residual action follows the phase action center exactly.
  - malformed phase-action center tables are rejected.
- No training run has been promoted with this hook yet. The next useful
  experiment is to design a better template, or start a new curriculum from a
  lower-speed walking checkpoint using this hook, rather than continuing from
  the run12/run14 high-speed bounding family.

May 15 phase-action-center extraction:

- Added `tools/extract_phase_action_center.py`, which rolls out a checkpoint,
  bins raw or applied actions by gait phase, and writes a JSON table compatible
  with `phase_action_center`.
- Extracted an 8-bin applied-action center from the stable expanded walking
  checkpoint at `1.0 m/s`:
  `/workspace/runs/phase_centers/walk_expand_f1p0_applied8.json`.
- Replaying that table open-loop was not dynamically viable:
  - `TrexWalk`, `1.0 m/s` command: no termination because walk does not use
    fall termination, but orientation fell as low as `0.001` and mean forward
    speed was only `0.303 m/s`.
  - `TrexRun`, `1.0 m/s` command: termination at step `83`, mean forward
    speed `0.288 m/s`.
  - `TrexRun`, `3.0 m/s` command: termination at step `90`, mean forward
    speed `0.292 m/s`.
- Conclusion: stable walking policies also depend on closed-loop feedback; a
  phase-average action table from a trained policy is not enough to produce a
  useful open-loop gait. The `phase_action_center` hook remains useful, but the
  table needs to be designed as a stabilizable gait center, not extracted by
  averaging a feedback policy.

May 15 phase-center residual bridge:

- Added `phase_action_center_path` so training configs can reference a saved
  phase-action-center JSON file without embedding a large table in command-line
  overrides.
- Trained run17 as a short bridge experiment:
  `/workspace/runs/TrexRun-20260515-132926-run17-phasecenter-walk1to3-20m-from-walk`.
  It warm-started from the stable expanded `TrexWalk` checkpoint, used the
  extracted walk phase center, and trained `TrexRun` commands from `1-3 m/s`.
- Eval scalar improved from `-71.928` to `-11.771`, but the fixed-command gate
  failed badly. The final checkpoint terminated at step `369` for `1.0 m/s`,
  step `77` for `2.0 m/s`, and step `65` for `3.0 m/s`; all ended collapsed
  with torso height under about `1.0 m`.
- Conclusion: scalar reward can be misleading for these bridge attempts. A
  walking-policy-derived phase center plus residual PPO did not preserve the
  stable walk behavior when moved into the stricter `TrexRun` task.

May 15 black-box phase-center search:

- Added `tools/search_phase_action_center.py`, a small CEM-style black-box
  search over a low-dimensional symmetric sine phase-action-center table.
- Remote search target:
  `/workspace/runs/phase_centers/search_run_trexrun_f1p0_symmetric8.json`.
- The best searched table is the first useful open-loop center found so far.
  It did not terminate over 250-step search rollouts and reached mean forward
  `0.606 m/s` for a `1.0 m/s` command, with mean orientation `0.897`.
- Longer 750-step TrexRun replay:
  - `0.5 m/s` command: no termination, mean forward `0.759 m/s`.
  - `1.0 m/s` command: no termination, mean forward `0.590 m/s`.
  - `1.5 m/s` command: no termination, mean forward `0.704 m/s`.
- This is slow and not yet a policy, but it is a stabilizable phase center.
  Unlike the policy-averaged tables, it is worth trying as the center for a
  residual PPO curriculum.

May 15 searched-center residual PPO:

- Trained run18 from scratch with the searched stabilizable center and
  `TrexRun` commands from `0.5-1.5 m/s`:
  `/workspace/runs/TrexRun-20260515-140048-run18-searchedcenter-0p5to1p5-20m-scratch`.
- Eval scalar improved from `-29.399` to `5.736`, but fixed-command gates
  failed. The deterministic final checkpoint terminated at step `318` for
  `0.5 m/s`, step `198` for `1.0 m/s`, and step `150` for `1.5 m/s`; all
  ended collapsed with near-zero forward velocity.
- Conclusion: even with a stabilizable searched center, the current residual
  PPO setup can produce misleading scalar improvement without a usable
  deterministic policy. Further work should inspect the evaluation/reset logic
  or simplify the objective before spending more training time.

May 15 termination-cost mismatch:

- Root cause for several misleading scalar improvements: `TrexJoystick.step`
  multiplies the full reward sum by `dt=0.02`. The old run termination scale of
  `-100` was therefore only about `-2` once at the falling step.
- Training/eval can reset after termination, while the fixed gates correctly
  treat any termination as failure. This made policies that briefly moved or
  scored reward before falling look much better in scalar eval than in the
  deterministic fixed-command gates.
- Increased the TrexRun termination scale to `-1000`, making the terminal
  penalty about `-20` before clipping. This should make PPO care more about
  survival and align scalar eval better with the gates.

May 15 searched-center residual PPO after termination-cost fix:

- Trained run19 from scratch with the searched stabilizable center, stricter
  `TrexRun` termination cost, and commands from `0.5-1.5 m/s`:
  `/workspace/runs/TrexRun-20260515-142215-run19-termcost-searchedcenter-0p5to1p5-20m-scratch`.
- This was the first searched-center residual run where scalar reward and
  deterministic fixed-command gates agreed on useful progress. Eval reward
  improved from `-47.382` to `80.181`.
- Fixed-command diagnostics on the final checkpoint:
  - `0.5 m/s`: no termination, mean forward `0.241 m/s`, but large lateral
    drift around `0.649 m/s`.
  - `1.0 m/s`: no termination, mean forward `0.990 m/s`, lateral drift
    `0.163 m/s`.
  - `1.5 m/s`: no termination, mean forward `1.365 m/s`, lateral drift near
    zero.
- Rendered frames show an upright but crouched/tilted gait. This is not a final
  running policy, but it is a usable low-speed `TrexRun` foothold.

May 15 run20 expansion in progress:

- Started run20 from the run19 final checkpoint:
  `/workspace/runs/TrexRun-20260515-143719-run20-termcost-searchedcenter-1to2p5-20m-from-run19`.
- The curriculum expands fixed straight-line commands to `1.0-2.5 m/s`, keeps
  the searched phase-action center, and uses Warp on the L40S.
- Eval rewards improved from `-70.148` to `158.549`.
- The final checkpoint passed deterministic fixed-command gates with no
  terminations:
  - `1.0 m/s`: mean forward `0.979 m/s`, lateral `0.007 m/s`, orientation
    reward `0.988-1.000`.
  - `1.5 m/s`: mean forward `1.471 m/s`, lateral `-0.187 m/s`, orientation
    reward `0.987-1.000`.
  - `2.0 m/s`: mean forward `1.986 m/s`, lateral `-0.301 m/s`, orientation
    reward `0.965-1.000`.
  - `2.5 m/s`: mean forward `2.437 m/s`, lateral `0.236 m/s`, orientation
    reward `0.785-1.000`.
- Rendered frames confirm this is useful but still not clean: the body is
  crouched/tilted, the policy uses hopping/bounding phases, and gait anti-phase
  is still low (`0.21-0.30`). This should not be expanded blindly to higher
  speed before gait/contact discipline improves.

May 15 lateral velocity and gait-discipline continuation:

- Added a `lateral_vel` reward term to the joystick reward dictionary. It is
  disabled by default and enabled for `TrexRun` with scale `-2.0`.
- Local and remote focused tests passed:
  `test_trex_joystick_humanoid_style_reward_terms`,
  `test_trex_single_skill_task_configs`, and
  `test_trex_run_gates_positive_forward_rewards_by_support_and_height`.
- Started run21 from run20:
  `/workspace/runs/TrexRun-20260515-151406-run21-termcost-gaitdiscipline-1to2p5-20m-from-run20`.
- Run21 keeps the `1.0-2.5 m/s` range and uses the same searched phase center,
  but tightens lateral velocity, phase contact, contact duty, foot contact
  balance, leg-action alternation, gait anti-phase, and gait symmetry. The goal
  is to improve the current low-speed running mode before any further speed
  expansion.
- Run21 improved scalar reward and deterministic gates but did not solve gait
  alternation. Fixed gates all survived and tracking remained good:
  `0.972`, `1.477`, `1.986`, and `2.527 m/s` for `1.0`, `1.5`, `2.0`, and
  `2.5 m/s` commands. Lateral velocity improved versus run20, especially at
  `2.0-2.5 m/s`, and orientation stayed near perfect.
- The failure is gait quality: gait anti-phase dropped to `0.13-0.24`, and
  rendered frames still show a low hopping/bounding mode. The next experiment
  should use stronger contact/phase/anti-phase terms at the same speed band,
  rather than expanding speed.

May 15 stronger gait-discipline continuation:

- Trained run22 from run21 with much stronger contact/phase/anti-phase terms:
  `/workspace/runs/TrexRun-20260515-153148-run22-termcost-gaitstrong-1to2p5-20m-from-run21`.
- This improved the fixed-command gates without losing tracking:
  - `1.0 m/s`: mean forward `0.991`, lateral `0.011`, anti-phase `0.250`.
  - `1.5 m/s`: mean forward `1.514`, lateral `0.006`, anti-phase `0.252`.
  - `2.0 m/s`: mean forward `2.007`, lateral `-0.038`, anti-phase `0.266`.
  - `2.5 m/s`: mean forward `2.508`, lateral `-0.066`, anti-phase `0.297`.
- Rendered frames are still not a fully natural alternating gait, but the body
  is more level and the lateral drift is much better controlled. Run22 is the
  best current low-speed `TrexRun` base for the next staged expansion.

May 15 staged expansion to 4 m/s:

- Trained run23 from run22 over `2.0-4.0 m/s`:
  `/workspace/runs/TrexRun-20260515-155011-run23-termcost-gaitstrong-2to4-20m-from-run22`.
- Fixed gates passed:
  - `2.0 m/s`: mean forward `1.994`, lateral `-0.023`, anti-phase `0.316`.
  - `3.0 m/s`: mean forward `3.052`, lateral `-0.008`, anti-phase `0.320`.
  - `4.0 m/s`: mean forward `4.431`, lateral `0.057`, anti-phase `0.300`.
- This is a successful staged expansion for survival and lateral control, but
  the `4.0 m/s` command overspeeds and rendered frames still show a simplified
  airborne/bounding gait. It is a bridge checkpoint, not final behavior.

May 15 failed 4-6 m/s expansions:

- Tried run24 from run23 over `4.0-6.0 m/s` with a tighter high-speed tracking
  sigma. Scalar reward stayed negative and fixed gates showed severe overspeed:
  `4.0 -> 4.631 m/s`, `5.0 -> 6.400 m/s`, `6.0 -> 6.873 m/s`; anti-phase
  dropped at higher commands.
- Added a default-disabled `forward_speed_error` cost so overspeed can be
  penalized explicitly. Local focused tests passed.
- Tried run25 from run23 over `4.0-6.0 m/s` with `forward_speed_error=-12`
  and no forward-progress reward. It still oversped badly:
  `4.0 -> 5.006 m/s`, `5.0 -> 6.728 m/s`, `6.0 -> 6.701 m/s`.
- Conclusion: jumping directly from the run23 `2-4 m/s` bridge to `4-6 m/s`
  is too aggressive for the current curriculum. The next attempt should be a
  smaller `3-5 m/s` expansion from run23, using the explicit speed-error cost.

May 15 smaller 3-5 m/s expansion:

- Trained run26 from run23 over `3.0-5.0 m/s`, using
  `forward_speed_error=-10` and no forward-progress reward:
  `/workspace/runs/TrexRun-20260515-164126-run26-speederr-gaitstrong-3to5-20m-from-run23`.
- This fixed the lower part of the range but failed at `5 m/s`:
  - `3.0 m/s`: mean forward `3.084`, lateral `-0.055`, anti-phase `0.356`.
  - `4.0 m/s`: mean forward `4.340`, lateral `0.089`, anti-phase `0.313`.
  - `5.0 m/s`: mean forward `7.189`, lateral `0.009`, anti-phase `0.293`.
- Conclusion: the current policy can bridge to about `4 m/s`, but commands
  around `5 m/s` trigger the old overspeed/bounding mode. The next attempt
  should directly target `4-5 m/s` with a much stronger symmetric speed-error
  cost instead of broadening the command distribution.

May 15 targeted 5 m/s blocker:

- Trained run27 from run26 over `4.0-5.0 m/s` with
  `forward_speed_error=-50`, no forward-progress reward, and stronger tracking:
  `/workspace/runs/TrexRun-20260515-165506-run27-target5-speederr-4to5-20m-from-run26`.
- The targeted run reduced but did not eliminate overspeed:
  - `4.0 m/s`: mean forward `4.238`, lateral `0.075`, anti-phase `0.297`.
  - `4.5 m/s`: mean forward `5.128`, lateral `0.114`, anti-phase `0.288`.
  - `5.0 m/s`: mean forward `6.116`, lateral `0.117`, anti-phase `0.274`.
- Conclusion: the current PPO/reward/curriculum branch is blocked at the
  transition to `5 m/s`. Stronger symmetric speed-error penalties and narrower
  command ranges still produce overspeed/bounding. Further progress probably
  needs a control/model change, such as a different gait/action parameterization,
  actuator/limit analysis, or an explicit running template that can sustain
  controlled `5+ m/s` without falling into the overspeed mode.

May 15 strict tracking at 5 m/s:

- Detailed diagnostics showed the overspeeding `5 m/s` policy was still earning
  a very large positive `tracking_forward_vel` reward because
  `high_speed_tracking_sigma_scale` widened the tracking reward at high command
  speeds. The speed-error penalty was present but not dominant enough.
- Trained run28 from run27 with `high_speed_tracking_sigma_scale=0.0` and
  strict `tracking_sigma=0.25`. This improved but did not solve the issue:
  `5.0 m/s` commanded `5.749 m/s`.
- Trained run29 from run28 over a narrower `4.5-5.0 m/s` band with
  `tracking_sigma=0.2` and `forward_speed_error=-100`:
  `/workspace/runs/TrexRun-20260515-173053-run29-target5-stricter-4p5to5-20m-from-run28`.
- Run29 is the first acceptable `5 m/s` bridge:
  - `4.5 m/s`: mean forward `4.748`, lateral `0.104`, anti-phase `0.290`.
  - `5.0 m/s`: mean forward `5.441`, lateral `0.098`, anti-phase `0.287`.
- Rendered frames still show a simplified bounding gait, but the posture is
  stable and the old `6-7+ m/s` overspeed mode is largely suppressed. Use run29
  as the next staged-expansion base.

May 15 failed strict 5-7 m/s expansion:

- Trained run30 from run29 over `5.0-7.0 m/s`, keeping strict tracking sigma
  and `forward_speed_error=-100`:
  `/workspace/runs/TrexRun-20260515-174521-run30-stricttrack-5to7-20m-from-run29`.
- Scalar reward degraded from `-222.224` to `-303.814`.
- Fixed gates:
  - `5.0 m/s`: mean forward `5.357`, lateral `0.101`, anti-phase `0.281`.
  - `6.0 m/s`: mean forward `7.181`, lateral `0.210`, anti-phase `0.234`.
  - `7.0 m/s`: mean forward `9.061`, lateral `0.290`, anti-phase `0.225`.
- Conclusion: run29 is a usable strict `5 m/s` bridge, but the current approach
  is blocked again above `5 m/s`. Strict tracking suppresses the first overspeed
  transition, but `6-7 m/s` still falls into the old fast bounding mode. Further
  progress likely needs a model/control change rather than another direct speed
  expansion.

May 15 TrexRun default reward update:

- Updated `TrexRun` defaults to preserve the useful strict-tracking diagnosis:
  `high_speed_tracking_sigma_scale=0.0`, `tracking_forward_vel=10.0`,
  `forward_progress=0.0`, and `forward_speed_error=-100.0`.
- This prevents future default `TrexRun` experiments from using the permissive
  high-speed tracking reward that let overspeeding policies score well.
- Focused local tests passed:
  `test_trex_single_skill_task_configs`,
  `test_trex_run_gates_positive_forward_rewards_by_support_and_height`, and
  `test_trex_joystick_humanoid_style_reward_terms`.
