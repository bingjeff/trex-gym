# T-Rex Policy Reset Plan

Status: current as of the reset after the unsuccessful all-in-one joystick
experiments.

## Current Direction

Train separate single-skill policies before attempting composition:

1. `TrexGetup`: get up from the ground.
2. `TrexBalance`: stand from the nominal pose and reject perturbations.
3. `TrexWalk`: walk straight at low speeds with an alternating gait.
4. `TrexJoystick`: velocity-steered forward/yaw locomotion.
5. `TrexRun`: high-speed forward running up to 10 m/s.

Only after the get-up, balance, and walk policies are individually reliable
should they be combined into one reset-mixture policy.

## Implemented Reset Infrastructure

- Added explicit joystick-derived task presets/classes for `TrexBalance`,
  `TrexWalk`, `TrexJoystick`, and `TrexRun`.
- Registered all five T-Rex tasks in `mjx_gym.train`.
- Switched the serious PPO defaults to humanoid-scale networks:
  `(512, 256, 128)` actor and critic, `4096` envs, `1024` batch size,
  `32` minibatches, and task-specific timestep budgets.
- Added optional push perturbations to joystick-derived tasks; they are enabled
  for balance and moderate joystick training, and disabled by default.
- Extended joystick rollout/render diagnostics so they can evaluate the
  balance, walk, joystick, and run task classes.

## Phase Gates

Every phase must produce metrics, rendered samples, a checkpoint note, and a
commit before moving to the next phase.

- Get-up gate: side-lying reset succeeds over many seeds, final support is on
  feet only, non-foot body parts clear the ground, and standing is quiet.
- Balance gate: starts standing, survives repeated pushes, does not spin, and
  does not use tail/body ground support.
- Walk gate: tracks 0.5, 1.0, and 1.5 m/s straight-line commands with
  alternating footfalls and bounded yaw/lateral drift.
- Joystick gate: tracks forward speed and yaw-rate commands, including zero
  command standing.
- Run gate: tracks up to 10 m/s with a visible gait, no skating, no spinning,
  and no non-foot propulsion.

## Research Defaults

The reset follows the MuJoCo Playground humanoid pattern: standing-start
locomotion tasks, gait phase observations, privileged critic state, push
perturbations, action-rate/smoothness costs, foot-contact/gait rewards, and
larger actor/critic networks. Get-up remains a separate two-stage problem:
first rediscover a robust get-up, then refine it for stillness and contact
quality.

# MJX Getup Training Plan

The immediate goal is a first MuJoCo Playground/MJX task where the T-Rex gets
up from its side and stands. The URDF remains the source of truth, but training
uses a simplified generated MJCF.

## Constraints

- Keep `assets/trex.urdf` as the source model.
- Put new training environment code under `mjx_gym`.
- Start with a getup-and-stand task only.
- Initial state is side-lying with zero joint configuration.
- Use position targets with PD gains for controlled joints.
- The policy action space includes the leg position targets and the two tail
  tendon targets.
- Do not load visual meshes in the MJX training model.

## Phase Workflow

Each phase should end with a stable checkpoint before starting the next phase.

For every phase:

1. Implement the smallest coherent set of changes for that phase.
2. Test thoroughly for that phase's scope.
   - Prefer focused unit tests for deterministic model transforms.
   - Add MuJoCo load checks for every generated MJCF.
   - Add MJX creation and short-step checks whenever the model or environment
     changes.
   - Use tiny PPO smoke runs only after reset/step checks pass.
3. Update this plan with what was completed, what was learned, and any changed
   assumptions or next steps.
4. Commit the code, tests, and plan update together.
5. Move to the next phase only after the working tree is clean except for
   intentionally untracked generated artifacts.

## Phase 1: Simplified MJX Model

Status: completed in the first implementation pass.

Build a simplification step that converts the full generated MJCF into a
training-focused MJCF.

1. Generate full MJCF from the URDF.
2. Remove all visual mesh assets and visual geoms from the MJX training model.
3. Keep collision geoms needed for floor interaction and body contacts.
4. Fuse all bodies connected only by fixed joints.
   - Preserve the kinematic transforms of child geoms, inertials, sites, and
     sensors when moving them into the fused parent body.
   - Preserve revolute, floating, and tendon-actuated DOFs.
   - Preserve the root free joint.
5. Keep the existing contact capsule geometry, but verify capsule ownership
   after fixed-body fusion.
6. Keep passive joint stiffness, damping, and friction.
7. Keep the two tail tendons and their actuators.
8. Convert leg actuators to position-control semantics with PD gains.
9. Add scene elements after simplification:
   - floor geom
   - cameras
   - root or torso IMU site
   - foot/contact sites
   - sensors needed by the getup environment
   - keyframes for zero/side-lying starting state and any nominal stand target

Phase 1 results:

- Added `tools/mjx_model_simplification.py`.
- Added `tools/urdf_to_mjx_mujoco.py`.
- The simplifier removes visual meshes and visual geoms from the MJX training
  MJCF.
- Fixed-joint links are fused in URDF space before MJCF generation.
- Fused child collision capsules are transformed into the retained parent link
  frame.
- Fused child inertials are merged into the retained parent link using the
  parallel-axis theorem.
- Existing MuJoCo metadata is preserved.
- Emitted training actuators are converted from motor shortcuts to position
  actuator shortcuts with an initial `kp=35`.
- Current simplified T-Rex training model complexity:
  - bodies: 31
  - joints: 32
  - qpos: 38
  - qvel: 37
  - actuators: 10
  - tendons: 2
  - geoms: 45 contact, 0 visual
  - mesh assets: 0
- Tests verify deterministic fixed-link fusion, visual mesh removal, MuJoCo
  load, MJX model/data creation, mass preservation, and expected model counts.
- Manual end-to-end validation generated `/tmp/trex.mjx.xml`, loaded it in
  MuJoCo, copied it to MJX, and stepped it for three zero-control simulation
  steps.

Deferred to later phases:

- Scene additions: floor, cameras, sites, sensors, and keyframes.
- PPO smoke tests for the final environment.

## Phase 2: Complexity Checks

Status: completed with a repeatable comparison tool.

After simplification, compare the full and simplified models.

Measure:

- body count
- joint count
- qpos/qvel dimensions
- actuator count
- tendon count
- geom count, split by visual/contact
- sensor count
- total mass
- approximate center of mass at zero configuration

Pass criteria:

- simplified MJCF loads in MuJoCo
- simplified MJCF can be copied with `mjx.put_model`
- zero-action MJX stepping runs for a few steps
- no visual mesh assets or visual geoms remain in the simplified model
- all intended DOFs and actuators remain present

Phase 2 results:

- Added `tools/compare_mjx_model_complexity.py`.
- Added reusable comparison helpers to `tools/mjx_model_simplification.py`.
- Added tests that load the full model with mesh assets, load the simplified
  model, compare the measured complexity, and verify the formatted report.
- Current full-to-simplified comparison:
  - bodies: 134 -> 31
  - joints: 32 -> 32
  - qpos: 38 -> 38
  - qvel: 37 -> 37
  - actuators: 10 -> 10
  - tendons: 2 -> 2
  - geoms: 297 -> 45
  - visual geoms: 252 -> 0
  - contact geoms: 45 -> 45
  - mesh assets: 252 -> 0
  - sensors: 0 -> 0
  - total mass preserved at 5180.28
  - zero-configuration center of mass preserved at
    `1.10481 -0.648769 0.00176733`

Conclusion:

- The simplifier removes visual and fixed-body overhead without changing the
  current articulated DOFs or contact capsule count.
- The remaining complexity is dominated by movable DOFs, not collision geoms or
  visual assets.

## Phase 3: `mjx_gym` Environment

Status: completed for the first getup-and-stand environment skeleton.

Create a small local package for the T-Rex task.

Files:

- `mjx_gym/__init__.py`
- `mjx_gym/trex_constants.py`
- `mjx_gym/trex_getup.py`
- `mjx_gym/train.py`

The environment should follow the shape of MuJoCo Playground's `Go1Getup`:

- `default_config()`
- `TrexGetup`
- `reset(rng)`
- `step(state, action)`
- `action_size`
- `mj_model`
- `mjx_model`
- `xml_path`

Observation proposal:

- torso gravity vector
- torso gyro
- controlled action history
- selected joint positions and velocities
- optional full joint state in `privileged_state`

Reward proposal:

- torso upright orientation
- torso/root height
- stand-still term once upright
- posture term near zero or nominal stand target
- action-rate cost
- torque or energy cost
- joint-limit cost

Phase 3 results:

- Added the `mjx_gym` package to the project package list.
- Added `mjx_gym/trex_constants.py` for generated T-Rex getup scene MJCF.
- Added `mjx_gym/trex_getup.py` with a Playground-compatible `TrexGetup`
  environment.
- Added `mjx_gym/train.py` to register `TrexGetup` and patch a local PPO
  config before delegating to MuJoCo Playground's `train_jax_ppo` runner.
- The generated getup scene adds:
  - floor geom
  - tracking camera
  - torso IMU site
  - gyro, accelerometer, upvector, global linear velocity, and global angular
    velocity sensors
  - zero-upright and side-lying-zero keyframes
- The first environment version starts side-lying with zero joint
  configuration.
- The policy action space is ten-dimensional and maps to all generated
  position actuators: eight leg joint targets plus the sagittal and
  medial-lateral tail tendon targets.
- Current environment model size:
  - bodies: 31
  - joints: 32
  - qpos: 38
  - qvel: 37

## Phase 4: Joystick Locomotion

Status: in progress; not yet successful.

Goal:

- Keep the getup/stand behavior.
- Add joystick-style forward and turn commands.
- Run with physical ground contact rather than simulator-skating.
- Produce an alternating bipedal gait before expanding to faster running and
  turning.

Completed implementation work:

- Added `TrexJoystick` with forward/turn commands, gait phase in the policy
  observation, standing-command handling, and low-speed curriculum overrides.
- Increased ground contact friction and switched contact defaults to `condim=4`.
- Changed joystick foot contact scoring from capsule height proximity to
  actual floor-contact constraint force.
- Raised foot-contact force normalization so incidental contact is not treated
  as full support.
- Added rewards/costs for phase-timed swing clearance, actual feet air time,
  phase-contact error, and one-sided hip-adduction use.
- Rendered diagnostic frames for every promising checkpoint before judging it.

Best current checkpoint:

- Remote path:
  `/workspace/runs/TrexJoystick-20260513-221241-gate-low-10m-lr1e4-ent1e2/checkpoints/000011059200`
- It can track roughly 2 m/s in some fixed-command rollouts, but the rendered
  motion is a hopping/bracing behavior, not an acceptable alternating bipedal
  gait.

Important negative results:

- Stronger double-support, swing-clearance, phase-contact, and feet-air-time
  reward weights consistently push the policy into slow bracing rather than a
  cleaner gait.
- A dense phase-contact error cost reduces the incentive to load the wrong foot
  but also collapses forward speed.
- Penalizing the observed left hip-adduction lean reduces one exploit but does
  not produce a gait.
- Open-loop tests of the built-in sinusoidal gait prior show that the prior is
  not dynamically viable by itself: nonzero prior amplitudes collapse the model,
  while stable sinusoidal hand-search candidates barely move forward.

Current conclusion:

- The locomotion failure is no longer just an insufficient reward-weight sweep.
  The current gait prior and reward representation do not provide a stable
  stepping template for PPO to refine.
- The next phase should change the gait representation or action
  parameterization before more long PPO runs. Candidate directions:
  - learn residuals around a validated low-level stepping controller,
  - expose a smaller phase/amplitude action space for leg cycling,
  - add a reference trajectory term only after a dynamically stable open-loop
    stepping template is found,
  - or simplify the model/contact geometry further for a first bipedal walking
    curriculum.

Phase workflow reminder:

- Every locomotion change must be followed by focused tests, a short remote
  training/diagnostic run, rendered frames or video for visual inspection, a
  `PLAN.md` update, and a commit.
  - actuators: 10
  - geoms: 46, including the floor
  - sensors: 5
- Observation shapes:
  - `state`: 78
  - `privileged_state`: 164
- Tests verify model construction, environment reset, one zero-action step,
  observation shapes, sensor names, and local Playground registration.
- Manual validation confirmed `python -m mjx_gym.train --env_name=TrexGetup
  --only_check_args=true` succeeds.

Deferred to Phase 4:

- JIT reset/step validation.
- Tiny PPO smoke run.
- Reward tuning for a useful first getup policy.

## Phase 4: Local Smoke Tests

Status: completed for the first local training smoke path.

Before PPO, add focused tests/scripts that verify:

1. simplified MJCF generation
2. model complexity report
3. MuJoCo load
4. MJX model creation
5. environment reset
6. several zero-action steps
7. JIT reset/step
8. tiny Playground PPO run with low env count and timestep count

Phase 4 results:

- Existing tests cover simplified MJCF generation, model complexity reporting,
  MuJoCo load, MJX creation, environment reset, and one zero-action step.
- Manual JIT validation succeeded for `TrexGetup` reset and one step:
  - command shape: `jax.jit(env.reset)` and `jax.jit(env.step)`
  - output observation shapes remained `state=(78,)` and
    `privileged_state=(164,)`
- Manual tiny PPO smoke test succeeded with:
  - `env_name=TrexGetup`
  - `num_timesteps=128`
  - `num_envs=2`
  - `num_eval_envs=1`
  - `episode_length=20`
  - `num_minibatches=1`
  - `num_updates_per_batch=1`
  - `batch_size=2`
  - `unroll_length=2`
  - `run_evals=false`
  - `num_videos=0`
- The smoke run completed training, produced a checkpoint directory under
  `/tmp/trex-getup-ppo-smoke`, and reached inference/rendering.
- On local CPU, PPO compile time was about 60 seconds for this tiny run.

Next direction:

- Start reward and initialization tuning for actual getup behavior.
- Add a more useful nominal standing pose and consider rendering short rollout
  clips for policy debugging.
- Once getup/stand is learning, split out a walking task rather than overloading
  this recovery task.

## Phase 5: Collision Filtering and Gain Tuning

Status: completed for the first stable tuned model.

Goals:

1. Restrict collision pairs so every valid contact includes the ground.
2. Use MuJoCo's zero-configuration mass matrix to compute per-DOF gain ratios.
3. Use mass-matrix row-sum magnitudes, not only diagonal entries, as the DOF
   scale.
4. Tune passive stiffness, passive damping, armature, and position actuator
   gains through centralized factors so solver timestep changes can be handled
   quickly.

Implementation:

- Robot contact capsules are configured with `contype=0`, `conaffinity=1`.
- The floor is configured with `contype=1`, `conaffinity=0`.
- This permits floor-vs-capsule contacts and excludes capsule-vs-capsule
  self-collisions for the current run.
- Mass matrix scales are computed from `sum(abs(M[dof, :]))` at zero
  configuration.
- Tendon actuator scales are coefficient-weighted sums of their coupled joint
  scales.
- Tuned factors at `sim_dt=0.004`:
  - passive stiffness per row sum: `1000`
  - passive damping per row sum: `80`
  - armature per row sum: `0.2`
  - actuator `kp` per row sum: `1000`

Validation:

- Tests verify that all active MuJoCo contacts include the floor.
- Tests verify mass-scaled tuning uses mass-matrix row sums.
- Tests verify minimum tuned passive stiffness, damping, and actuator gains are
  above the first target thresholds.
- A side-lying MuJoCo drop rollout ran for 4 seconds without warnings, with
  only floor contacts.
- Representative 0.1 rad perturbations to caudal, toe, femur, and cervical
  joints decayed below `2e-4` rad without warnings.
- JIT reset/step still succeeds after tuning.
- Tiny PPO smoke test still completes after tuning:
  - `env_name=TrexGetup`
  - `num_timesteps=128`
  - `num_envs=2`
  - compile time around 50 seconds on local CPU.

Follow-up debugging:

- A RunPod PPO smoke reached the end of the first training attempt but failed
  Brax's final `pmap.assert_is_replicated(training_state)` check.
- On one device that check can fail when the replicated training-state
  fingerprint is NaN, so local debugging focused on finding non-finite or
  explosive rollout values.
- Reset and zero-action steps remained finite.
- Random-action rollouts with the original active `Kp=1000` produced actuator
  forces in the millions and joint velocities in the thousands, which is large
  enough to destabilize PPO value/optimizer updates.
- Passive gains remain mass-scaled and stiff, but the default active position
  actuator gain factor was reduced to `Kp=0.2`.
- A regression test now checks that short random-action rollouts remain finite,
  keep bounded joint velocities and actuator forces, and produce nonzero
  reward.

## Open Decisions

- Exact nominal standing posture for fixed leg PD targets.
- Exact sensor set for the first version: only torso IMU and foot/contact
  sites, or additional body pose sensors for debugging.

## Phase 6: Single-Policy Skill Reset

Status: completed for the first successful single-skill checkpoints.

Goal:

1. Keep get-up, standing balance, and velocity-guided walking as separate
   policies before attempting a combined joystick policy.
2. Verify each policy with rollout metrics, local checkpoint loading, and
   rendered samples when useful.
3. Keep persistent remote checkpoints under `/workspace/runs`.

Results:

- Get-up from ground remains covered by the existing verified checkpoint
  `checkpoints/TrexGetup-20260513-033812-still2-warp-10m/000011468800`.
  Current-code diagnostics show it reaches upright, foot-supported standing
  from the side reset, though it is not a quiet balance policy.
- Standing balance is covered by
  `checkpoints/TrexBalance-20260514-235829-balance-10m-buf-best/000009830400`,
  sourced from
  `/workspace/runs/TrexBalance-20260514-235829-balance-10m-buf-best/checkpoints/000009830400`.
  It holds a zero command from standing with near-zero drift and no termination.
- Low-speed velocity-guided walking is covered by
  `checkpoints/TrexWalk-20260515-011532-walk-gaitprior-20m-from-balance/000026214400`,
  sourced from
  `/workspace/runs/TrexWalk-20260515-011532-walk-gaitprior-20m-from-balance/checkpoints/000026214400`.
  It tracks `0.25 m/s` and `0.5 m/s` standing-start forward commands with low
  error and no termination. The `0.8 m/s` rollout moves forward but is not yet
  clean enough to count as high-speed walking.
- The `TrexWalk` task now applies the gait prior as the moving-command action
  center, instead of using it only as a reward target. Focused regression tests
  cover this behavior and preserve the default zero-command standing action.
- Walking verification videos are stored under
  `checkpoints/TrexWalk-20260515-011532-walk-gaitprior-20m-from-balance/videos/`.

Validation:

- Focused local tests passed:
  - `test_trex_walk_zero_residual_applies_gait_prior_for_moving_command`
  - `test_trex_joystick_actions_are_default_pose_residuals`
  - `test_trex_single_skill_task_configs`
- Local JAX checkpoint-load smoke tests passed for the balance and walk
  checkpoints.
- Remote Warp diagnostics passed for walking commands `0.25`, `0.5`, and
  `0.8 m/s`; the first two are accepted as the first low-speed walking success.

Next direction:

- Improve the walk policy beyond the current low-speed envelope before adding
  turns.
- After walking is stronger, train a separate velocity-steered joystick policy.
- Leave the combined get-up/balance/walk policy until the individual policies
  are more robust.

## Phase 7: Expanded Straight-Line Walking

Status: completed through approximately `1.0 m/s`.

Goal:

1. Continue from the first low-speed `TrexWalk` checkpoint.
2. Expand the straight-line command range before introducing turn commands.
3. Verify fixed-speed rollouts at several commands and reject scalar-only
   success.

Results:

- Continued from
  `/workspace/runs/TrexWalk-20260515-011532-walk-gaitprior-20m-from-balance/checkpoints/000026214400`.
- Remote run:
  `/workspace/runs/TrexWalk-20260515-024930-walk-expand-30m-from-gaitprior`.
- Final checkpoint:
  `checkpoints/TrexWalk-20260515-024930-walk-expand-30m-from-gaitprior/000032768000`.
- Verified fixed-command walking up to about `1.0 m/s`:
  - `0.25 m/s` command: `0.329 m/s` mean forward velocity.
  - `0.5 m/s` command: `0.506 m/s` mean forward velocity.
  - `0.8 m/s` command: `0.755 m/s` mean forward velocity.
  - `1.0 m/s` command: `0.943 m/s` mean forward velocity.
  - `1.2 m/s` command: failed, only `0.457 m/s` mean forward velocity with
    degraded orientation.
- Rendered videos for `0.8 m/s` and `1.0 m/s` and inspected frames for posture.
- Local JAX checkpoint-load smoke test passed at `1.0 m/s`.

Next direction:

- Start a separate turning/joystick phase from the expanded walking checkpoint.
- Use moderate forward speeds and low yaw-rate commands first.
- Keep `TrexRun` and combined get-up/balance/walk policy as later phases.

## Phase 8: Signed Joystick Turning

Status: completed for a standing-start joystick policy.

Goal:

1. Warm-start from expanded straight-line walking.
2. Add moderate signed yaw-rate commands.
3. Reject policies that only turn one direction or merely preserve forward
   walking.
4. Verify local loading through the interactive joystick entrypoint.

Results:

- First run preserved walking but failed signed yaw: both positive and negative
  commands produced positive yaw. It was kept as a warm start only.
- Tightened the joystick curriculum to `0.25-0.8 m/s` and `+/-0.25 rad/s`,
  forced nonzero turn samples, and increased the yaw-rate tracking weight.
- Successful checkpoint:
  `checkpoints/TrexJoystick-20260515-034759-joystick-turn2-30m-from-turn1/000026214400`.
- Verified remote fixed-command tracking:
  - `(0.5, +0.25)`: forward `0.514 m/s`, yaw `0.195 rad/s`.
  - `(0.5, -0.25)`: forward `0.511 m/s`, yaw `-0.219 rad/s`.
  - `(0.8, +0.25)`: forward `0.816 m/s`, yaw `0.195 rad/s`.
  - `(0.8, -0.25)`: forward `0.786 m/s`, yaw `-0.219 rad/s`.
- Rendered left/right turn videos and copied them locally.
- Local `tools/drive_joystick_policy.py --check-load --impl jax` passed, so the
  checkpoint can be loaded by the local joystick entrypoint.

Next direction:

- Decide how to combine get-up/recovery with the standing-start joystick policy.
- The low-risk route is an explicit local controller/orchestrator that uses the
  get-up policy until upright and then switches to the joystick policy.
- A single monolithic get-up-plus-joystick PPO policy remains possible but has
  been much harder historically and should be attempted only after the
  orchestrated baseline works.

## Phase 9: Combined Recovery And Joystick

Status: completed for the first single combined checkpoint.

Goal:

1. Produce a policy that can start from the side, recover, stand under a neutral
   command, and track moderate signed joystick commands.
2. Verify both side-start and standing-start behavior.
3. Make sure the final checkpoint can be loaded by a local joystick entrypoint.

Results:

- An orchestrated local runner was added as a fallback, but direct handoff from
  the older get-up policy was not robust enough to count as the combined
  solution.
- A monolithic `TrexJoystick` combined policy was trained through three
  attempts. The third attempt added recovery-only residual scaling so the
  neutral-command fallen state has enough action authority to get up.
- Successful checkpoint:
  `checkpoints/TrexJoystick-20260515-051704-combined3-recovery-scale-30m/000032768000`.
- Remote source:
  `/workspace/runs/TrexJoystick-20260515-051704-combined3-recovery-scale-30m/checkpoints/000032768000`.
- Verified remote Warp gates:
  - standing reset, zero command: quiet stand.
  - standing reset, `(0.5,+0.25)`: forward `0.547 m/s`, yaw `0.223 rad/s`.
  - standing reset, `(0.5,-0.25)`: forward `0.492 m/s`, yaw `-0.209 rad/s`.
  - side reset, zero command: get-up and quiet stand.
  - side reset, `(0.5,+0.25)`: get-up, forward `0.547 m/s`, yaw `0.224 rad/s`.
  - side reset, `(0.5,-0.25)`: get-up, forward `0.492 m/s`, yaw `-0.209 rad/s`.
- Rendered side-start videos for neutral, left-turn, and right-turn commands.
- Local `tools/drive_joystick_policy.py --check-load --start side --impl jax`
  loads the final combined checkpoint and steps the environment.

Remaining future work:

- Expand the combined policy beyond the current moderate command envelope.
- Train or fine-tune a separate high-speed `TrexRun` policy up to the original
  `10 m/s` goal.
- Consider adding perturbation robustness after the combined baseline is less
  speed-limited.

## Phase 10: High-Speed TrexRun

Status: partially completed; blocked before the original `10 m/s` target.

Goal:

1. Train a separate straight-line `TrexRun` policy after the combined joystick
   baseline.
2. Preserve physical posture/contact constraints while increasing speed.
3. Verify fixed-command tracking up to `10 m/s`.

Results:

- The initial 1-4 m/s continuation learned a floating/bounding exploit, so
  `TrexRun` was tightened before further expansion: fall termination is enabled,
  excess running height, vertical velocity, tilt angular velocity, missing foot
  contact, contact-duty error, and phase-contact error are penalized more
  strongly, and positive non-foot-clearance reward is removed for run training.
- The stricter curriculum was expanded in stages from `1-3 m/s` through
  `8-10 m/s`. The best current checkpoint is:
  `checkpoints/TrexRun-20260515-093556-run12-speed-8to10-60m-from-run11/000117964800`.
- Verified run12 diagnostics, Warp, standing reset, seed 0:
  - command `8.0 m/s`: mean forward `9.583 m/s`, no termination.
  - command `9.0 m/s`: mean forward `8.595 m/s`, no termination.
  - command `10.0 m/s`: mean forward `8.622 m/s`, no termination.
- Additional `10.0 m/s` seed checks gave `8.406 m/s` and `8.342 m/s`.
- Videos for `8.0` and `10.0 m/s` were rendered and copied locally.
- Local `tools/drive_joystick_policy.py --check-load --task run --impl jax`
  passed after adding a task switch so the run checkpoint uses `TrexRun`
  dynamics and action scaling.

Conclusion:

- The current model can train a stable high-speed straight-line mode around
  `8-9 m/s`, but it does not yet command-track to `10 m/s`.
- A follow-up gait-discipline run from the best high-speed checkpoint failed:
  increasing gait-prior, anti-phase, symmetry, leg-action alternation, and
  phase-contact terms caused the final policy to terminate and average only
  `2.346 m/s` on a `10.0 m/s` command.
- Enhanced rollout diagnostics show that the current high-speed policy has
  frequent hip-flexion and ankle target saturation, high actuator forces, low
  gait anti-phase, and substantial phase/contact mismatch.
- Directly replaying the best checkpoint with higher global `Kp` destabilized
  the policy, so a global position-servo gain increase is not a safe shortcut.
  Any actuator-authority change should be targeted and retrained.
- Added actuator-specific `Kp` scaling so individual leg actuators can be
  strengthened without changing the whole model. A targeted ankle `1.5x`
  continuation from run12 was trained as
  `/workspace/runs/TrexRun-20260515-110344-run14-ankle1p5-8to10-60m-from-run12`.
  The run was stable but did not pass the `10 m/s` gate: the best scalar
  checkpoint averaged `9.127 m/s` for a `9.0 m/s` command but only
  `8.838-8.969 m/s` for a `10.0 m/s` command, and it oversped the `8.0 m/s`
  command at `9.694 m/s`. Rendered frames show a long-flight bounding mode
  rather than a clean physical running gait.
- A follow-up anti-flight reward rebalance from run14 made no useful progress:
  the least-bad run15 checkpoint still averaged about `9.1 m/s`, had low gait
  anti-phase around `0.13`, and kept similar contact/height behavior, but the
  scalar reward stayed strongly negative once flight and vertical-motion costs
  were made dominant.
- Added a structural TrexRun gate so positive forward tracking/progress rewards
  are only paid when the body has foot support and is not above the running
  height band. A 30M continuation from run14 did not solve the mode: the
  least-bad run16 checkpoint averaged `8.477 m/s` on a `10.0 m/s` command and
  still had low gait anti-phase around `0.15`.
- Extended the open-loop gait diagnostic to replay explicit phase-binned action
  templates. Replaying the run14 raw/applied phase-action means did not produce
  a viable gait: from rest the raw template moved only `0.274 m/s`, and with
  `5-9 m/s` initial velocity both raw and applied templates fell within
  `18-30` steps. The learned high-speed behavior is therefore closed-loop and
  not a reusable phase-only gait template.
- Added a `phase_action_center` control hook to the joystick environment. When
  configured, moving commands use an interpolated phase-binned applied-action
  table as the action center, and the policy learns residual feedback around
  that table. This preserves the existing 10-action interface while allowing
  future curricula to test explicit gait centers instead of the fixed sine
  prior.
- Added a phase-action-center extractor and tested a table extracted from the
  stable expanded walking checkpoint. The extracted walking table also failed
  open-loop: in `TrexRun` it terminated within about `83-90` steps and moved
  only around `0.29 m/s`. Averaging a feedback policy by phase is therefore not
  enough; the explicit center needs to be designed as a stabilizable gait.
- A short residual-policy bridge from the stable walking checkpoint into
  `TrexRun` with that extracted phase center improved training scalar reward
  but failed fixed-command gates: the final checkpoint terminated at steps
  `369`, `77`, and `65` for `1`, `2`, and `3 m/s` commands respectively.
  Do not treat scalar improvement alone as progress for this phase.
- Added a black-box phase-center search over a compact symmetric sine gait
  table. The first searched table is slow but stabilizable in `TrexRun`,
  surviving 750-step open-loop probes at `0.5-1.5 m/s` commands and moving
  about `0.6-0.75 m/s`. This is the best current candidate for a residual PPO
  center.
- The next step should be model/control debugging, not another blind PPO run:
  inspect actuator force/position limits, whether the leg/tail action space can
  generate the required stride impulse, whether the simplified collision feet
  are giving enough traction/contact fidelity, and whether the high-speed gait
  prior should be replaced with a better open-loop running template.
