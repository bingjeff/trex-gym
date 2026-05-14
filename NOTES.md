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
