# Checkpoints

## TrexGetup minclear Warp 10M

- Local path: `checkpoints/TrexGetup-20260513-020641-minclear-warp-10m/`
- Checkpoint: `checkpoints/TrexGetup-20260513-020641-minclear-warp-10m/000011468800/`
- Source run on pod: `/workspace/runs/TrexGetup-20260513-020641-minclear-warp-10m`
- Repo commit: `e44a200e4f8211fc5d3c386611d3546a09ae8430`
- Training backend: MuJoCo MJX Warp
- Training length: 10M requested steps, final saved checkpoint at `000011468800`
- Final eval reward: `97.650`

This checkpoint is the first successful getup-and-stand policy for the simplified
T-Rex MJX model. The policy starts from the side-lying reset distribution, gets
upright, and settles into a two-foot balance. The strict non-foot clearance
reward used for this run prevents head, tail, torso, or other non-foot contacts
from being hidden by averaging across all collision geometries.

Steady-state diagnostics on the final checkpoint used the final 250 steps of a
750-step rollout:

- Seed 0: foot-floor contact on 248/250 steps, non-foot floor contact 0, self/non-floor contact 0, min non-foot clearance 1.14 m.
- Seed 1: foot-floor contact on 243/250 steps, non-foot floor contact 0, self/non-floor contact 0, min non-foot clearance 1.14 m.
- Seed 2: foot-floor contact on 240/250 steps, non-foot floor contact 0, self/non-floor contact 0, min non-foot clearance 1.15 m.
- Seed 3: foot-floor contact on 243/250 steps, non-foot floor contact 0, self/non-floor contact 0, min non-foot clearance 1.00 m.

Representative rollout videos copied locally:

- `checkpoints/TrexGetup-20260513-020641-minclear-warp-10m/videos/rollout0.mp4`
- `checkpoints/TrexGetup-20260513-020641-minclear-warp-10m/videos/rollout1.mp4`
- `checkpoints/TrexGetup-20260513-020641-minclear-warp-10m/videos/rollout3.mp4`
- `checkpoints/TrexGetup-20260513-020641-minclear-warp-10m/videos/rollout7.mp4`

Re-run the steady-state diagnostic with:

```bash
uv run python tools/analyze_policy_rollout.py \
  checkpoints/TrexGetup-20260513-020641-minclear-warp-10m/000011468800 \
  --impl jax \
  --seed 0 \
  --steps 750 \
  --skip-steps 500
```

## TrexGetup gated-stillness Warp 10M

- Local path: `checkpoints/TrexGetup-20260513-033812-still2-warp-10m/`
- Checkpoint: `checkpoints/TrexGetup-20260513-033812-still2-warp-10m/000011468800/`
- Source run on pod: `/workspace/runs/TrexGetup-20260513-033812-still2-warp-10m`
- Training source: `34ffa07866aab44537d1cc0b2002f5132bc5038e` plus the gated-stillness reward changes committed with this checkpoint record.
- Training backend: MuJoCo MJX Warp
- Training length: 10M requested steps, final saved checkpoint at `000011468800`
- Final eval reward: `99.944`

This checkpoint fine-tunes the first getup policy with an explicit stillness
objective. Base linear and angular velocity costs are gated by the upright
height/orientation reward, so the policy is free to move while recovering from
its side but is pushed to stop once it reaches the standing pose.

Final TensorBoard eval terms showed the intended trend:

- Episode reward improved from `48.762` to `99.944`.
- Base linear velocity penalty improved from `-2477.495` to `-96.241`.
- Base angular velocity penalty improved from `-157.559` to `-43.056`.

Steady-state diagnostics on the final checkpoint used the final 250 steps of a
750-step rollout with Warp:

- Seed 0: mean base linear velocity 0.136 m/s, max 0.259 m/s; mean base angular velocity 0.103 rad/s, max 0.205 rad/s; base XY displacement 0.514 m; min non-foot clearance 1.109 m.
- Seed 1: mean base linear velocity 0.142 m/s, max 0.275 m/s; mean base angular velocity 0.080 rad/s, max 0.190 rad/s; base XY displacement 0.599 m; min non-foot clearance 1.100 m.
- Seed 2: mean base linear velocity 0.148 m/s, max 0.288 m/s; mean base angular velocity 0.104 rad/s, max 0.243 rad/s; base XY displacement 0.578 m; min non-foot clearance 1.098 m.
- Seed 3: mean base linear velocity 0.162 m/s, max 0.295 m/s; mean base angular velocity 0.084 rad/s, max 0.211 rad/s; base XY displacement 0.692 m; min non-foot clearance 1.105 m.

Representative rollout videos copied locally:

- `checkpoints/TrexGetup-20260513-033812-still2-warp-10m/videos/rollout0.mp4`
- `checkpoints/TrexGetup-20260513-033812-still2-warp-10m/videos/rollout1.mp4`
- `checkpoints/TrexGetup-20260513-033812-still2-warp-10m/videos/rollout3.mp4`
- `checkpoints/TrexGetup-20260513-033812-still2-warp-10m/videos/rollout7.mp4`

Re-run the steady-state diagnostic with:

```bash
uv run python tools/analyze_policy_rollout.py \
  checkpoints/TrexGetup-20260513-033812-still2-warp-10m/000011468800 \
  --impl warp \
  --seed 0 \
  --steps 750 \
  --skip-steps 500
```

## TrexJoystick warm-start Warp 10M

- Local path: `checkpoints/TrexJoystick-20260513-044426-joystick-warm-10m/`
- Checkpoint: `checkpoints/TrexJoystick-20260513-044426-joystick-warm-10m/000011468800/`
- Source run on pod: `/workspace/runs/TrexJoystick-20260513-044426-joystick-warm-10m`
- Warm start: `/workspace/runs/TrexJoystick-warmstart-getup/checkpoints/000000000000`
- Training source: `1f514f4aaf9428893fa90030d8051d0d02722bb4`
- Training backend: MuJoCo MJX Warp
- Training length: 10M requested steps, final saved checkpoint at `000011468800`
- Final eval reward: `123.566`

This is the first successful joystick policy. It was initialized by padding the
`TrexGetup` policy checkpoint into the larger `TrexJoystick` observation space:
the original 78 policy observation entries stay in the same order and the new
local velocity / command entries are appended with zero input weights. The run
then fine-tuned command tracking for forward speed and turn rate.

Final TensorBoard eval terms showed the intended trend:

- Episode reward improved from `107.023` to `123.566`.
- Forward tracking reward improved from `819.784` to `1666.087`.
- Turn tracking reward improved from `660.320` to `771.518`.
- Orientation stayed high, ending at `963.327`.
- Non-foot clearance stayed high, ending at `985.632`.

Fixed-command diagnostics on the final checkpoint used the final 500 steps of a
1000-step rollout with Warp:

- Standing stop command `(0.0 m/s, 0.0 rad/s)`: mean forward velocity -0.027 m/s, mean turn velocity 0.050 rad/s, XY displacement 0.278 m, orientation reward 0.999-1.000.
- Standing forward command `(0.5 m/s, 0.0 rad/s)`: mean forward velocity 0.451 m/s, mean turn velocity -0.042 rad/s, XY displacement 4.546 m, orientation reward 0.999-1.000.
- Standing forward command `(1.0 m/s, 0.0 rad/s)`: mean forward velocity 0.933 m/s, mean turn velocity -0.081 rad/s, XY displacement 9.175 m, orientation reward 0.999-1.000.
- Standing forward command `(1.5 m/s, 0.0 rad/s)`: mean forward velocity 1.418 m/s, mean turn velocity -0.098 rad/s, XY displacement 13.838 m, orientation reward 0.999-1.000.
- Standing turn command `(0.0 m/s, 0.5 rad/s)`: mean forward velocity -0.014 m/s, mean turn velocity 0.457 rad/s, XY displacement 0.218 m, orientation reward 0.995-0.999.
- Standing turn command `(0.0 m/s, -0.5 rad/s)`: mean forward velocity 0.036 m/s, mean turn velocity -0.529 rad/s, XY displacement 0.223 m, orientation reward 0.991-0.997.
- Side reset stop command `(0.0 m/s, 0.0 rad/s)`: mean forward velocity -0.025 m/s, mean turn velocity 0.040 rad/s, XY displacement 0.345 m, orientation reward 0.999-1.000.
- Side reset forward command `(1.0 m/s, 0.0 rad/s)`: mean forward velocity 0.935 m/s, mean turn velocity -0.079 rad/s, XY displacement 9.229 m, orientation reward 0.999-1.000.
- Side reset turn command `(0.0 m/s, -0.5 rad/s)`: mean forward velocity 0.035 m/s, mean turn velocity -0.558 rad/s, XY displacement 0.106 m, orientation reward 0.991-0.996.

Representative rollout videos copied locally:

- `checkpoints/TrexJoystick-20260513-044426-joystick-warm-10m/videos/rollout0.mp4`
- `checkpoints/TrexJoystick-20260513-044426-joystick-warm-10m/videos/rollout1.mp4`
- `checkpoints/TrexJoystick-20260513-044426-joystick-warm-10m/videos/rollout2.mp4`
- `checkpoints/TrexJoystick-20260513-044426-joystick-warm-10m/videos/rollout3.mp4`

Re-run the fixed-command diagnostic with:

```bash
uv run python tools/analyze_joystick_rollout.py \
  checkpoints/TrexJoystick-20260513-044426-joystick-warm-10m/000011468800 \
  --impl warp \
  --reset-pose standing \
  --forward 1.0 \
  --turn 0.0 \
  --steps 1000 \
  --skip-steps 500
```

## TrexJoystick speed-only Warp 10M

- Local path: `checkpoints/TrexJoystick-20260513-101726-speedonly-10m/`
- Checkpoint: `checkpoints/TrexJoystick-20260513-101726-speedonly-10m/000011468800/`
- Source run on pod: `/workspace/runs/TrexJoystick-20260513-101726-speedonly-10m`
- Warm start: `/workspace/runs/TrexJoystick-20260513-063049-fast10-upright-20m/checkpoints/000022118400`
- Training source: `75ba4d485e2e432b519ec04f3f12b2bedf2e1735`
- Training backend: MuJoCo MJX Warp
- Training length: 10M requested steps, final saved checkpoint at `000011468800`

This checkpoint is the first joystick policy that combines the stable
zero-command stand hold with a high-speed bounding run. It was fine-tuned from
the prior fast joystick policy using standing resets only, no zero-command
samples, and forward commands concentrated in the 8-10 m/s range. The guarded
zero-command stand hold comes from the environment code at the source hash
above, so the checkpoint should be used with that code or newer.

Fixed-command diagnostics on the final checkpoint used the final 500 steps of a
1000-step rollout with Warp:

- Standing stop command `(0.0 m/s, 0.0 rad/s)`: mean foot speed 0.000 m/s, XY displacement 0.001 m, torso height 2.374 m, orientation reward 1.000-1.000, non-foot clearance full.
- Standing forward command `(10.0 m/s, 0.0 rad/s)`: mean forward velocity 10.460 m/s, mean turn velocity 0.012 rad/s, mean stride extent 2.020 m, max stride extent 2.651 m, orientation reward 0.978-1.000, non-foot clearance full.

Representative rollout videos copied locally:

- `checkpoints/TrexJoystick-20260513-101726-speedonly-10m/videos/stand_zero_5s.mp4`
- `checkpoints/TrexJoystick-20260513-101726-speedonly-10m/videos/run_10ms_10s.mp4`

Verify local policy loading without a gamepad or viewer:

```bash
uv run python tools/drive_joystick_policy.py \
  /home/bingjeff/projects/trex-gym/checkpoints/TrexJoystick-20260513-101726-speedonly-10m/000011468800 \
  --check-load \
  --impl jax
```

Drive the policy interactively with a connected gamepad:

```bash
uv run python tools/drive_joystick_policy.py \
  /home/bingjeff/projects/trex-gym/checkpoints/TrexJoystick-20260513-101726-speedonly-10m/000011468800 \
  --start standing \
  --impl jax
```

Re-render the videos with:

```bash
MUJOCO_GL=egl uv run python tools/render_joystick_policy.py \
  /home/bingjeff/projects/trex-gym/checkpoints/TrexJoystick-20260513-101726-speedonly-10m/000011468800 \
  checkpoints/TrexJoystick-20260513-101726-speedonly-10m/videos/run_10ms_10s.mp4 \
  --impl warp \
  --steps 500 \
  --forward 10.0 \
  --turn 0.0 \
  --reset-pose standing \
  --width 960 \
  --height 540 \
  --camera-distance 12
```

## TrexBalance Warp 10M

- Local path: `checkpoints/TrexBalance-20260514-235829-balance-10m-buf-best/`
- Checkpoint: `checkpoints/TrexBalance-20260514-235829-balance-10m-buf-best/000009830400/`
- Source run on pod: `/workspace/runs/TrexBalance-20260514-235829-balance-10m-buf-best`
- Training source: `96a9bb5f508ca6f8a2b98daafc8c8fb7d6618b44` plus the later task-specific diagnostic tooling.
- Training backend: MuJoCo MJX Warp
- Training length: 10M requested steps; best scalar checkpoint copied from step `000009830400`

This is the first successful single-policy standing-balance checkpoint from the
task split. It starts from nominal standing, holds a zero command, and rejects
small randomized pushes during training.

Standing diagnostic evidence:

- Remote long diagnostic, seed 0, final 500 sampled steps: mean forward velocity
  `-0.003 m/s`, mean lateral velocity `0.001 m/s`, base XY displacement
  `0.072 m`, torso height `2.620-2.642 m`, orientation reward `0.927-0.944`,
  and no termination.
- Remote long diagnostic, seed 1, final 500 sampled steps: base XY displacement
  `0.014 m`, torso height `2.611-2.676 m`, orientation reward `0.930-0.987`,
  and no termination.
- Local load/run smoke test with JAX, seed 0, 50 steps: no termination, mean
  forward velocity `0.016 m/s`, mean lateral velocity `-0.016 m/s`, base XY
  displacement `0.023 m`, torso height `2.772-2.852 m`, and orientation reward
  `0.997-1.000`.

Verify local policy loading:

```bash
uv run python tools/analyze_joystick_rollout.py \
  /home/bingjeff/projects/trex-gym/checkpoints/TrexBalance-20260514-235829-balance-10m-buf-best/000009830400 \
  --task TrexBalance \
  --impl jax \
  --seed 0 \
  --steps 50 \
  --skip-steps 0 \
  --forward 0.0 \
  --turn 0.0 \
  --reset-pose standing
```

## TrexWalk gait-prior Warp 20M

- Local path: `checkpoints/TrexWalk-20260515-011532-walk-gaitprior-20m-from-balance/`
- Checkpoint: `checkpoints/TrexWalk-20260515-011532-walk-gaitprior-20m-from-balance/000026214400/`
- Source run on pod: `/workspace/runs/TrexWalk-20260515-011532-walk-gaitprior-20m-from-balance`
- Warm start: `/workspace/runs/TrexBalance-20260514-235829-balance-10m-buf-best/checkpoints/000009830400`
- Training source: `167b40f2eaab9493b5218489bac9647b1c574a7c`
- Training backend: MuJoCo MJX Warp
- Training length: 20M requested steps; final saved checkpoint at `000026214400`

This is the first successful low-speed velocity-guided walking checkpoint from
the reset plan. The task uses a gait-prior action center for moving commands and
learns residual actions around that center. The verified command range is
currently low-speed walking; it is not a run policy.

Training eval rewards moved in the intended direction:

- `0`: `-710.969`
- `6553600`: `-657.894`
- `13107200`: `-540.841`
- `19660800`: `-9.190`
- `26214400`: `21.782`

Fixed-command diagnostics on the final checkpoint used Warp with standing
resets, seed 0, and the final 500 steps of a 1000-step rollout:

- Command `0.25 m/s`: mean forward velocity `0.255 m/s`, mean forward error
  `0.005 m/s`, base XY displacement `2.586 m`, torso height `2.575-2.740 m`,
  orientation reward `0.913-0.981`, left/right contact duty `0.672/0.622`, and
  no termination.
- Command `0.5 m/s`: mean forward velocity `0.494 m/s`, mean forward error
  `0.006 m/s`, base XY displacement `5.005 m`, torso height `2.660-2.804 m`,
  orientation reward `0.923-0.948`, left/right contact duty `0.578/0.752`, and
  no termination.
- Command `0.8 m/s`: mean forward velocity `0.611 m/s`, mean forward error
  `0.189 m/s`, base XY displacement `6.049 m`, orientation reward
  `0.590-0.803`, and no termination. This is forward motion but not yet a clean
  high-speed walk.
- Local load/run smoke test with JAX at command `0.5 m/s`, seed 0, 50 steps:
  no termination, mean forward velocity `0.447 m/s`, mean forward error
  `0.053 m/s`, base XY displacement `0.574 m`, torso height `2.639-2.853 m`,
  and orientation reward `0.966-1.000`.

Representative rollout videos copied locally:

- `checkpoints/TrexWalk-20260515-011532-walk-gaitprior-20m-from-balance/videos/walk_0p25ms.mp4`
- `checkpoints/TrexWalk-20260515-011532-walk-gaitprior-20m-from-balance/videos/walk_0p5ms.mp4`
- `checkpoints/TrexWalk-20260515-011532-walk-gaitprior-20m-from-balance/videos/walk_0p8ms.mp4`

Verify local policy loading:

```bash
uv run python tools/analyze_joystick_rollout.py \
  /home/bingjeff/projects/trex-gym/checkpoints/TrexWalk-20260515-011532-walk-gaitprior-20m-from-balance/000026214400 \
  --task TrexWalk \
  --impl jax \
  --seed 0 \
  --steps 50 \
  --skip-steps 0 \
  --forward 0.5 \
  --turn 0.0 \
  --reset-pose standing
```

## TrexWalk expanded straight-line Warp 30M

- Local path: `checkpoints/TrexWalk-20260515-024930-walk-expand-30m-from-gaitprior/`
- Checkpoint: `checkpoints/TrexWalk-20260515-024930-walk-expand-30m-from-gaitprior/000032768000/`
- Source run on pod: `/workspace/runs/TrexWalk-20260515-024930-walk-expand-30m-from-gaitprior`
- Warm start: `/workspace/runs/TrexWalk-20260515-011532-walk-gaitprior-20m-from-balance/checkpoints/000026214400`
- Training source: `6225cc4f7c0d4be94620a9f2d90ee0b75061b80e`
- Training backend: MuJoCo MJX Warp
- Training length: 30M requested steps; final saved checkpoint at `000032768000`

This checkpoint expands the straight-line walking envelope. It uses the same
gait-prior action-center formulation as the first `TrexWalk` checkpoint, with a
wider training command range of `0.25-1.2 m/s`. The useful verified range is
currently up to about `1.0 m/s`; `1.2 m/s` still degrades.

Training eval rewards:

- `0`: `-48.491`
- `6553600`: `-222.306`
- `13107200`: `-255.569`
- `19660800`: `-99.695`
- `26214400`: `11.538`
- `32768000`: `95.853`

Fixed-command diagnostics on the final checkpoint used Warp with standing
resets, seed 0, and the final 500 steps of a 1000-step rollout:

- Command `0.25 m/s`: mean forward velocity `0.329 m/s`, mean forward error
  `0.079 m/s`, torso height `2.703-2.798 m`, orientation reward `0.992-1.000`,
  left/right contact duty `0.740/0.660`, and no termination.
- Command `0.5 m/s`: mean forward velocity `0.506 m/s`, mean forward error
  `0.006 m/s`, torso height `2.639-2.735 m`, orientation reward `0.973-0.992`,
  left/right contact duty `0.746/0.612`, and no termination.
- Command `0.8 m/s`: mean forward velocity `0.755 m/s`, mean forward error
  `0.045 m/s`, torso height `2.642-2.723 m`, orientation reward `0.944-0.966`,
  left/right contact duty `0.682/0.668`, and no termination.
- Command `1.0 m/s`: mean forward velocity `0.943 m/s`, mean forward error
  `0.057 m/s`, torso height `2.659-2.739 m`, orientation reward `0.937-0.956`,
  left/right contact duty `0.654/0.696`, and no termination.
- Command `1.2 m/s`: mean forward velocity `0.457 m/s`, mean forward error
  `0.743 m/s`, orientation reward `0.595-0.784`, right contact duty `1.000`.
  This is not counted as success.
- Local load/run smoke test with JAX at command `1.0 m/s`, seed 0, 50 steps:
  no termination, mean forward velocity `0.791 m/s`, base XY displacement
  `0.875 m`, torso height `2.762-2.932 m`, and orientation reward
  `0.907-1.000`.

Representative rollout videos copied locally:

- `checkpoints/TrexWalk-20260515-024930-walk-expand-30m-from-gaitprior/videos/walk_0p8ms.mp4`
- `checkpoints/TrexWalk-20260515-024930-walk-expand-30m-from-gaitprior/videos/walk_1p0ms.mp4`

Verify local policy loading:

```bash
uv run python tools/analyze_joystick_rollout.py \
  /home/bingjeff/projects/trex-gym/checkpoints/TrexWalk-20260515-024930-walk-expand-30m-from-gaitprior/000032768000 \
  --task TrexWalk \
  --impl jax \
  --seed 0 \
  --steps 50 \
  --skip-steps 0 \
  --forward 1.0 \
  --turn 0.0 \
  --reset-pose standing \
  --config-overrides '{"walk_command_forward_min":0.25,"walk_command_forward_max":1.2,"walk_command_zero_prob":0.05,"gait_frequency_max":1.6,"gait_frequency_per_mps":0.25}'
```

## TrexJoystick signed-turn Warp 30M

- Local path: `checkpoints/TrexJoystick-20260515-034759-joystick-turn2-30m-from-turn1/`
- Checkpoint: `checkpoints/TrexJoystick-20260515-034759-joystick-turn2-30m-from-turn1/000026214400/`
- Source run on pod: `/workspace/runs/TrexJoystick-20260515-034759-joystick-turn2-30m-from-turn1`
- Warm start: `/workspace/runs/TrexJoystick-20260515-032050-joystick-turn-30m-from-walk-expand/checkpoints/000032768000`
- Training source: `a01ce05b2e8a14bb5981d3e74f04d977551f88fa`
- Training backend: MuJoCo MJX Warp
- Training length: 30M requested steps; best scalar checkpoint at `000026214400`

This is the first successful standing-start joystick policy with signed yaw-rate
control. It is not a get-up policy. The verified command envelope is moderate:
about `0.5-0.8 m/s` forward speed and `+/-0.25 rad/s` yaw rate.

Training eval rewards:

- `0`: `217.593`
- `6553600`: `223.973`
- `13107200`: `237.344`
- `19660800`: `220.135`
- `26214400`: `244.018`
- `32768000`: `240.361`

Fixed-command diagnostics on checkpoint `000026214400` used Warp with standing
resets, seed 0, and the final 500 steps of a 1000-step rollout:

- Command `(0.0 m/s, 0.0 rad/s)`: mean forward velocity `0.006 m/s`,
  mean turn velocity `0.009 rad/s`, base XY displacement `0.036 m`, orientation
  reward `0.969-0.999`, and no termination.
- Command `(0.5 m/s, 0.0 rad/s)`: mean forward velocity `0.518 m/s`,
  mean turn velocity `-0.020 rad/s`, orientation reward `0.998-1.000`, and no
  termination.
- Command `(0.5 m/s, +0.25 rad/s)`: mean forward velocity `0.514 m/s`,
  mean turn velocity `0.195 rad/s`, mean turn error `0.055 rad/s`, orientation
  reward `0.995-0.999`, and no termination.
- Command `(0.5 m/s, -0.25 rad/s)`: mean forward velocity `0.511 m/s`,
  mean turn velocity `-0.219 rad/s`, mean turn error `0.031 rad/s`,
  orientation reward `0.996-1.000`, and no termination.
- Command `(0.8 m/s, +0.25 rad/s)`: mean forward velocity `0.816 m/s`,
  mean turn velocity `0.195 rad/s`, mean turn error `0.055 rad/s`, orientation
  reward `0.994-0.999`, and no termination.
- Command `(0.8 m/s, -0.25 rad/s)`: mean forward velocity `0.786 m/s`,
  mean turn velocity `-0.219 rad/s`, mean turn error `0.031 rad/s`,
  orientation reward `0.992-0.999`, and no termination.
- Local drive-entrypoint smoke test loaded the policy with
  `tools/drive_joystick_policy.py --check-load --impl jax`, produced action size
  `10`, and completed one env step.
- Local JAX rollout smoke at command `(0.5 m/s, +0.25 rad/s)` ran 50 steps with
  no termination, mean forward velocity `0.466 m/s`, mean turn velocity
  `0.219 rad/s`, torso height `2.581-2.855 m`, and orientation reward
  `0.997-1.000`.

Representative rollout videos copied locally:

- `checkpoints/TrexJoystick-20260515-034759-joystick-turn2-30m-from-turn1/videos/joystick_f0p8_t0p25.mp4`
- `checkpoints/TrexJoystick-20260515-034759-joystick-turn2-30m-from-turn1/videos/joystick_f0p8_tm0p25.mp4`

Check local load without opening the viewer:

```bash
uv run python tools/drive_joystick_policy.py \
  /home/bingjeff/projects/trex-gym/checkpoints/TrexJoystick-20260515-034759-joystick-turn2-30m-from-turn1/000026214400 \
  --check-load \
  --impl jax
```

Run interactively with a connected gamepad:

```bash
uv run python tools/drive_joystick_policy.py \
  /home/bingjeff/projects/trex-gym/checkpoints/TrexJoystick-20260515-034759-joystick-turn2-30m-from-turn1/000026214400 \
  --start standing \
  --impl jax
```

## TrexJoystick combined recovery and joystick Warp 30M

- Local path: `checkpoints/TrexJoystick-20260515-051704-combined3-recovery-scale-30m/`
- Checkpoint: `checkpoints/TrexJoystick-20260515-051704-combined3-recovery-scale-30m/000032768000/`
- Source run on pod: `/workspace/runs/TrexJoystick-20260515-051704-combined3-recovery-scale-30m`
- Warm start: `/workspace/runs/TrexJoystick-20260515-045113-combined2-zero-recovery-30m/checkpoints/000032768000`
- Training source: `1c19c49f8ea9342f91b650c17dd4713e2abbcf7c`
- Training backend: MuJoCo MJX Warp
- Training length: 30M requested steps; final saved checkpoint at `000032768000`

This is the first successful single combined policy. From a side reset, it can
recover to quiet standing under a zero command and can also recover into signed
moderate joystick locomotion. The verified joystick envelope remains moderate:
about `0.5 m/s` forward and `+/-0.25 rad/s` yaw.

The key implementation change for this run was a recovery-only residual action
scale. When the posture gate says the model is not upright, `TrexJoystick`
temporarily gives the policy a larger action envelope; once upright, it returns
to the existing standing/joystick residual scale. This fixed the earlier
zero-command side-recovery failure.

Training eval rewards:

- `0`: `-25.925`
- `6553600`: `45.616`
- `13107200`: `214.927`
- `19660800`: `301.984`
- `26214400`: `329.282`
- `32768000`: `360.719`

Fixed-command diagnostics on the final checkpoint used Warp, seed 0, and the
final 500 steps of a 1000-step rollout:

- Standing reset, command `(0.0, 0.0)`: mean forward velocity `-0.006 m/s`,
  mean turn velocity `-0.013 rad/s`, base XY displacement `0.209 m`, torso
  height `2.496-2.584 m`, orientation reward `0.990-1.000`, and no termination.
- Standing reset, command `(0.5, +0.25)`: mean forward velocity `0.547 m/s`,
  mean turn velocity `0.223 rad/s`, orientation reward `0.993-0.999`, and no
  termination.
- Standing reset, command `(0.5, -0.25)`: mean forward velocity `0.492 m/s`,
  mean turn velocity `-0.209 rad/s`, orientation reward `0.993-1.000`, and no
  termination.
- Side reset, command `(0.0, 0.0)`: mean forward velocity `-0.001 m/s`, mean
  turn velocity `-0.013 rad/s`, base XY displacement `0.160 m`, torso height
  `2.506-2.589 m`, orientation reward `0.992-1.000`, and no termination.
- Side reset, command `(0.5, +0.25)`: mean forward velocity `0.547 m/s`, mean
  turn velocity `0.224 rad/s`, orientation reward `0.993-0.999`, and no
  termination.
- Side reset, command `(0.5, -0.25)`: mean forward velocity `0.492 m/s`, mean
  turn velocity `-0.209 rad/s`, orientation reward `0.993-1.000`, and no
  termination.
- Local joystick-entrypoint smoke test loaded the policy with
  `tools/drive_joystick_policy.py --check-load --impl jax --start side` and
  completed one env step.

Representative side-start rollout videos copied locally:

- `checkpoints/TrexJoystick-20260515-051704-combined3-recovery-scale-30m/videos/combined_side_f0p0_t0p0.mp4`
- `checkpoints/TrexJoystick-20260515-051704-combined3-recovery-scale-30m/videos/combined_side_f0p5_t0p25.mp4`
- `checkpoints/TrexJoystick-20260515-051704-combined3-recovery-scale-30m/videos/combined_side_f0p5_tm0p25.mp4`

Check local load without opening the viewer:

```bash
uv run python tools/drive_joystick_policy.py \
  /home/bingjeff/projects/trex-gym/checkpoints/TrexJoystick-20260515-051704-combined3-recovery-scale-30m/000032768000 \
  --check-load \
  --impl jax \
  --start side \
  --max-forward 0.8 \
  --max-turn 0.25
```

Run interactively with a connected gamepad:

```bash
uv run python tools/drive_joystick_policy.py \
  /home/bingjeff/projects/trex-gym/checkpoints/TrexJoystick-20260515-051704-combined3-recovery-scale-30m/000032768000 \
  --start side \
  --impl jax \
  --max-forward 0.8 \
  --max-turn 0.25
```

## TrexRun high-speed experimental Warp 60M

- Local path: `checkpoints/TrexRun-20260515-093556-run12-speed-8to10-60m-from-run11/`
- Checkpoint: `checkpoints/TrexRun-20260515-093556-run12-speed-8to10-60m-from-run11/000117964800/`
- Source run on pod: `/workspace/runs/TrexRun-20260515-093556-run12-speed-8to10-60m-from-run11`
- Warm start: `/workspace/runs/TrexRun-20260515-091029-run11-speed-6to10-60m-from-run10/checkpoints/000117964800`
- Training backend: MuJoCo MJX Warp
- Training length: 60M requested steps; final saved checkpoint at `000117964800`

This is the best current high-speed straight-line `TrexRun` checkpoint. It is
stable in the high-speed regime but is not a successful `10 m/s` tracking
checkpoint. Treat it as experimental evidence for the current model/control
limit.

Fixed-command diagnostics, Warp, standing reset, seed 0, final 500 steps:

- Command `8.0 m/s`: mean forward velocity `9.583 m/s`, no termination, torso
  height `2.052-3.086 m`, orientation reward `0.868-1.000`.
- Command `9.0 m/s`: mean forward velocity `8.595 m/s`, no termination, torso
  height `2.098-2.933 m`, orientation reward `0.848-1.000`.
- Command `10.0 m/s`: mean forward velocity `8.622 m/s`, no termination,
  torso height `2.043-2.930 m`, orientation reward `0.836-1.000`.

Additional `10.0 m/s` seed checks gave `8.406 m/s` and `8.342 m/s`, so the
undertracking appears persistent rather than seed-specific.

Representative local videos:

- `checkpoints/TrexRun-20260515-093556-run12-speed-8to10-60m-from-run11/videos/run_f8p0.mp4`
- `checkpoints/TrexRun-20260515-093556-run12-speed-8to10-60m-from-run11/videos/run_f10p0.mp4`

Local JAX load smoke passed with `--task run`; the loaded checkpoint produced
action size `10` and completed one environment step.

Check local load without opening the viewer:

```bash
uv run python tools/drive_joystick_policy.py \
  /home/bingjeff/projects/trex-gym/checkpoints/TrexRun-20260515-093556-run12-speed-8to10-60m-from-run11/000117964800 \
  --check-load \
  --task run \
  --impl jax \
  --start standing \
  --max-forward 10.0 \
  --max-turn 0.0
```
