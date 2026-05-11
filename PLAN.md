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

## Open Decisions

- Exact nominal standing posture for fixed leg PD targets.
- Exact sensor set for the first version: only torso IMU and foot/contact
  sites, or additional body pose sensors for debugging.
