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

## TrexRun ankle-gain adaptation experiment Warp 60M

- Remote path:
  `/workspace/runs/TrexRun-20260515-110344-run14-ankle1p5-8to10-60m-from-run12/`
- Warm start:
  `/workspace/runs/TrexRun-20260515-093556-run12-speed-8to10-60m-from-run11/checkpoints/000117964800`
- Training backend: MuJoCo MJX Warp
- Training length: 60M requested steps; final saved checkpoint at
  `000117964800`
- Code change: per-actuator `Kp` scaling with ankle actuators set to `1.5x`

This was an experiment, not a promoted checkpoint. It tested whether the run12
undertracking was caused by weak ankle position authority. Training scalar
reward peaked at checkpoint `000104857600` and then declined:

- `0`: `-58.824`
- `39321600`: `2.878`
- `65536000`: `10.083`
- `91750400`: `15.563`
- `104857600`: `16.489`
- `117964800`: `9.314`

Fixed-command diagnostics, Warp, standing reset, seed 0, final 500 steps:

- Best scalar checkpoint `000104857600`:
  - command `8.0 m/s`: mean forward `9.694 m/s`, no termination.
  - command `9.0 m/s`: mean forward `9.127 m/s`, no termination.
  - command `10.0 m/s`: mean forward `8.838-8.969 m/s`, no termination.
- Final checkpoint `000117964800`:
  - command `8.0 m/s`: mean forward `10.217 m/s`, no termination.
  - command `9.0 m/s`: mean forward `9.212 m/s`, no termination.
  - command `10.0 m/s`: mean forward `9.089 m/s`, no termination.

Conclusion:

- Targeted ankle `Kp` helped the policy reach the 9 m/s regime more reliably,
  but it did not solve the `10 m/s` command and made the lower command band
  overspeed.
- Rendered frames for the best `10 m/s` rollout show long airborne bounding,
  not a satisfactory alternating physical run.
- This run should be treated as evidence that actuator authority is part of the
  limit, but not as a checkpoint to use locally.

## TrexRun anti-flight reward rebalance experiment Warp 30M

- Remote path:
  `/workspace/runs/TrexRun-20260515-114027-run15-antiflight-ankle1p5-30m-from-run14/`
- Warm start:
  `/workspace/runs/TrexRun-20260515-110344-run14-ankle1p5-8to10-60m-from-run12/checkpoints/000117964800`
- Training backend: MuJoCo MJX Warp
- Training length: 30M requested steps; final saved checkpoint at
  `000032768000`
- Code/config change: kept ankle `1.5x` `Kp`, reduced forward-tracking reward,
  and sharply increased no-foot-contact, vertical velocity, excess-height,
  phase-contact-error, contact-duty-error, and foot-slip penalties.

This was not promoted. The stricter objective exposed the long-flight bounding
behavior but did not train it away.

Training eval rewards:

- `0`: `-289.565`
- `6553600`: `-267.191`
- `13107200`: `-264.810`
- `19660800`: `-284.959`
- `26214400`: `-291.173`
- `32768000`: `-303.267`

Least-bad checkpoint `000013107200`, Warp, standing reset, seed 0, final 500
steps:

- command `8.0 m/s`: mean forward `9.097 m/s`, no termination, gait anti-phase
  `0.132`, torso height `2.086-3.006 m`.
- command `10.0 m/s`: mean forward `9.190 m/s`, no termination, gait anti-phase
  `0.127`, torso height `2.027-2.874 m`.

Conclusion:

- Stronger anti-flight penalties did not fix the high-speed mode; they mainly
  made the same behavior score worse.
- Further work should change the control/model formulation or gait template
  instead of continuing scalar reward reweighting from the same checkpoint.

## TrexRun forward-reward support gate experiment Warp 30M

- Remote path:
  `/workspace/runs/TrexRun-20260515-122319-run16-gatedforward-ankle1p5-30m-from-run14/`
- Warm start:
  `/workspace/runs/TrexRun-20260515-110344-run14-ankle1p5-8to10-60m-from-run12/checkpoints/000117964800`
- Training backend: MuJoCo MJX Warp
- Training length: 30M requested steps; final saved checkpoint at
  `000032768000`
- Code change: TrexRun now gates positive forward velocity/progress rewards by
  foot support and running-height band. Ordinary walk/joystick tasks keep the
  old ungated behavior.

This was not promoted. It is a structural reward fix, but a short continuation
from the existing high-speed policy did not escape the same solution family.

Training eval rewards:

- `0`: `-60.833`
- `6553600`: `-50.749`
- `13107200`: `-59.109`
- `19660800`: `-55.833`
- `26214400`: `-56.959`
- `32768000`: `-57.682`

Least-bad checkpoint `000006553600`, Warp, standing reset, seed 0, final 500
steps:

- command `8.0 m/s`: mean forward `9.073 m/s`, no termination, gait anti-phase
  `0.130`, torso height `2.068-2.985 m`.
- command `10.0 m/s`: mean forward `8.477 m/s`, no termination, gait anti-phase
  `0.154`, torso height `2.035-2.869 m`.

Conclusion:

- Gating positive velocity rewards by support/height makes the objective less
  exploitable, but does not by itself recover a physical `10 m/s` gait when
  fine-tuned from run14.
- The next high-speed attempt should start from a different control template or
  curriculum, not from the same run12/run14 bounding family.

## TrexRun phase-center residual bridge experiment Warp 20M

- Remote path:
  `/workspace/runs/TrexRun-20260515-132926-run17-phasecenter-walk1to3-20m-from-walk/`
- Warm start:
  `/workspace/runs/TrexWalk-20260515-024930-walk-expand-30m-from-gaitprior/checkpoints/000032768000`
- Phase center:
  `/workspace/runs/phase_centers/walk_expand_f1p0_applied8.json`
- Training backend: MuJoCo MJX Warp
- Training length: 20M requested steps; final saved checkpoint at
  `000026214400`

This was not promoted. It tested whether a `TrexRun` policy could bridge from
the stable expanded walking checkpoint while using an extracted walking
phase-action center as the residual-action center.

Training eval reward improved monotonically:

- `0`: `-71.928`
- `6553600`: `-38.968`
- `13107200`: `-19.683`
- `19660800`: `-14.195`
- `26214400`: `-11.771`

Fixed-command gates on the final checkpoint failed:

- command `1.0 m/s`: terminated at step `369`, mean forward `0.063 m/s`, torso
  height `0.446-1.035 m`, orientation reward `0.000-0.021`.
- command `2.0 m/s`: terminated at step `77`, mean forward `-0.024 m/s`, torso
  height `0.496-0.693 m`, orientation reward `0.002`.
- command `3.0 m/s`: terminated at step `65`, mean forward `-0.003 m/s`, torso
  height `0.510-0.601 m`, orientation reward `0.002`.

Conclusion:

- Scalar reward was misleading for this bridge experiment. The policy collapsed
  and did not produce usable low-speed `TrexRun` locomotion.
- The phase-center hook is still useful infrastructure, but a phase center
  extracted from the walking feedback policy is not a good starting template for
  TrexRun.

## TrexRun searched-center residual experiment Warp 20M

- Remote path:
  `/workspace/runs/TrexRun-20260515-140048-run18-searchedcenter-0p5to1p5-20m-scratch/`
- Phase center:
  `/workspace/runs/phase_centers/search_run_trexrun_f1p0_symmetric8.json`
- Training backend: MuJoCo MJX Warp
- Training length: 20M requested steps; final saved checkpoint at
  `000026214400`

This was not promoted. It tested whether a residual policy trained from scratch
could improve on the first stabilizable searched open-loop phase center.

Training eval reward improved and crossed positive:

- `0`: `-29.399`
- `6553600`: `-14.219`
- `13107200`: `-6.165`
- `19660800`: `-1.063`
- `26214400`: `5.736`

Fixed-command gates on the final deterministic checkpoint failed:

- command `0.5 m/s`: terminated at step `318`, mean forward `0.000 m/s`,
  torso height `0.508-0.517 m`, orientation reward `0.001`.
- command `1.0 m/s`: terminated at step `198`, mean forward `-0.001 m/s`,
  torso height `0.732-0.822 m`, orientation reward `0.015-0.017`.
- command `1.5 m/s`: terminated at step `150`, mean forward `-0.002 m/s`,
  torso height `0.743-0.816 m`, orientation reward `0.017-0.019`.

Conclusion:

- Positive scalar reward is still not a reliable success signal for this setup.
- A searched stabilizable open-loop center was not enough to produce a
  deterministic residual PPO policy under the current TrexRun objective.

## TrexRun searched-center residual with termination-cost fix Warp 20M

- Remote path:
  `/workspace/runs/TrexRun-20260515-142215-run19-termcost-searchedcenter-0p5to1p5-20m-scratch/`
- Phase center:
  `/workspace/runs/phase_centers/search_run_trexrun_f1p0_symmetric8.json`
- Training backend: MuJoCo MJX Warp
- Training length: 20M requested steps; final saved checkpoint at
  `000026214400`
- Code change: `TrexRun` fall termination scale increased from `-100` to
  `-1000`, which is about `-20` after reward `dt` scaling.

This is the current low-speed `TrexRun` foothold, but it has not been promoted
as the final running checkpoint. It validates that making fall termination
visible to PPO changes the searched-center residual result from collapse to
survival.

Training eval rewards:

- `0`: `-47.382`
- `6553600`: `-76.741`
- `13107200`: `-76.352`
- `19660800`: `-53.079`
- `26214400`: `80.181`

Fixed-command diagnostics, Warp, standing reset, seed 0, final 500 steps:

- command `0.5 m/s`: no termination, mean forward `0.241 m/s`, mean lateral
  `0.649 m/s`, torso height `2.553-2.602 m`, orientation reward
  `0.792-0.857`.
- command `1.0 m/s`: no termination, mean forward `0.990 m/s`, mean lateral
  `0.163 m/s`, torso height `2.484-2.582 m`, orientation reward
  `0.932-1.000`.
- command `1.5 m/s`: no termination, mean forward `1.365 m/s`, mean lateral
  `0.003 m/s`, torso height `2.408-2.583 m`, orientation reward
  `0.749-0.999`.

Videos on the remote:

- `/workspace/runs/TrexRun-20260515-142215-run19-termcost-searchedcenter-0p5to1p5-20m-scratch/videos/run19_f1p0.mp4`
- `/workspace/runs/TrexRun-20260515-142215-run19-termcost-searchedcenter-0p5to1p5-20m-scratch/videos/run19_f1p5.mp4`

Conclusion:

- This is useful as a low-speed bridge into expanded straight-line running, not
  as a finished `TrexRun` policy. It still needs better lateral control at low
  command speeds and staged curriculum expansion.

## TrexRun searched-center expansion run20 Warp 20M

- Remote path:
  `/workspace/runs/TrexRun-20260515-143719-run20-termcost-searchedcenter-1to2p5-20m-from-run19/`
- Warm start:
  `/workspace/runs/TrexRun-20260515-142215-run19-termcost-searchedcenter-0p5to1p5-20m-scratch/checkpoints/000026214400`
- Phase center:
  `/workspace/runs/phase_centers/search_run_trexrun_f1p0_symmetric8.json`
- Training backend: MuJoCo MJX Warp
- Command range: `1.0-2.5 m/s`, straight-line only

Training eval rewards:

- `0`: `-70.148`
- `6553600`: `18.868`
- `13107200`: `100.613`
- `19660800`: `133.570`
- `26214400`: `158.549`

Fixed-command diagnostics, Warp, standing reset, seed 0, final 500 steps:

- command `1.0 m/s`: no termination, mean forward `0.979 m/s`, mean lateral
  `0.007 m/s`, gait anti-phase `0.211`, torso height `2.532-2.570 m`,
  orientation reward `0.988-1.000`.
- command `1.5 m/s`: no termination, mean forward `1.471 m/s`, mean lateral
  `-0.187 m/s`, gait anti-phase `0.216`, torso height `2.554-2.586 m`,
  orientation reward `0.987-1.000`.
- command `2.0 m/s`: no termination, mean forward `1.986 m/s`, mean lateral
  `-0.301 m/s`, gait anti-phase `0.234`, torso height `2.469-2.595 m`,
  orientation reward `0.965-1.000`.
- command `2.5 m/s`: no termination, mean forward `2.437 m/s`, mean lateral
  `0.236 m/s`, gait anti-phase `0.297`, torso height `2.340-2.629 m`,
  orientation reward `0.785-1.000`.

Videos on the remote:

- `/workspace/runs/TrexRun-20260515-143719-run20-termcost-searchedcenter-1to2p5-20m-from-run19/videos/run20_f2p0.mp4`
- `/workspace/runs/TrexRun-20260515-143719-run20-termcost-searchedcenter-1to2p5-20m-from-run19/videos/run20_f2p5.mp4`

Conclusion:

- This is the current best low-speed straight-line `TrexRun` bridge. It passes
  fixed-command gates up to `2.5 m/s`, but rendered frames show a crouched
  hopping/bounding mode and gait anti-phase is still weak. It should be refined
  at this speed range before expanding toward the original high-speed target.

## TrexRun gait-discipline continuation run21 Warp 20M

- Remote path:
  `/workspace/runs/TrexRun-20260515-151406-run21-termcost-gaitdiscipline-1to2p5-20m-from-run20/`
- Warm start:
  `/workspace/runs/TrexRun-20260515-143719-run20-termcost-searchedcenter-1to2p5-20m-from-run19/checkpoints/000026214400`
- Code change: added a default-disabled `lateral_vel` cost and enabled it for
  `TrexRun`.
- Config change: keeps `1.0-2.5 m/s` and the searched phase center, with
  stronger lateral velocity, phase contact, contact duty, foot contact balance,
  leg-action alternation, gait anti-phase, and gait symmetry terms.

Training eval rewards:

- `0`: `125.959`
- `6553600`: `149.498`
- `13107200`: `177.043`
- `19660800`: `218.248`
- `26214400`: `235.616`

Fixed-command diagnostics, Warp, standing reset, seed 0, final 500 steps:

- command `1.0 m/s`: no termination, mean forward `0.972 m/s`, mean lateral
  `0.038 m/s`, gait anti-phase `0.133`, torso height `2.494-2.548 m`,
  orientation reward `0.993-1.000`.
- command `1.5 m/s`: no termination, mean forward `1.477 m/s`, mean lateral
  `-0.085 m/s`, gait anti-phase `0.187`, torso height `2.511-2.549 m`,
  orientation reward `0.997-1.000`.
- command `2.0 m/s`: no termination, mean forward `1.986 m/s`, mean lateral
  `-0.165 m/s`, gait anti-phase `0.200`, torso height `2.511-2.558 m`,
  orientation reward `0.997-1.000`.
- command `2.5 m/s`: no termination, mean forward `2.527 m/s`, mean lateral
  `-0.117 m/s`, gait anti-phase `0.236`, torso height `2.535-2.584 m`,
  orientation reward `0.997-1.000`.

Videos on the remote:

- `/workspace/runs/TrexRun-20260515-151406-run21-termcost-gaitdiscipline-1to2p5-20m-from-run20/videos/run21_f2p0.mp4`
- `/workspace/runs/TrexRun-20260515-151406-run21-termcost-gaitdiscipline-1to2p5-20m-from-run20/videos/run21_f2p5.mp4`

Conclusion:

- Run21 is numerically better than run20 for lateral drift and upright posture,
  but it did not improve the alternating gait metric. Rendered frames still
  look like a low hopping/bounding mode. Do not expand speed from this yet;
  first try stronger contact/phase/anti-phase discipline at the same command
  range.

## TrexRun stronger gait-discipline continuation run22 Warp 20M

- Remote path:
  `/workspace/runs/TrexRun-20260515-153148-run22-termcost-gaitstrong-1to2p5-20m-from-run21/`
- Warm start:
  `/workspace/runs/TrexRun-20260515-151406-run21-termcost-gaitdiscipline-1to2p5-20m-from-run20/checkpoints/000026214400`
- Config change: same `1.0-2.5 m/s` range and phase center, with stronger
  contact/phase/anti-phase terms, lower forward-progress weight, and stronger
  no-foot-contact/vertical/tilt/excess-height costs.

Training eval rewards:

- `0`: `247.702`
- `6553600`: `250.772`
- `13107200`: `258.070`
- `19660800`: `264.230`
- `26214400`: `271.479`

Fixed-command diagnostics, Warp, standing reset, seed 0, final 500 steps:

- command `1.0 m/s`: no termination, mean forward `0.991 m/s`, mean lateral
  `0.011 m/s`, gait anti-phase `0.250`, torso height `2.510-2.545 m`,
  orientation reward `0.986-1.000`.
- command `1.5 m/s`: no termination, mean forward `1.514 m/s`, mean lateral
  `0.006 m/s`, gait anti-phase `0.252`, torso height `2.479-2.527 m`,
  orientation reward `0.978-1.000`.
- command `2.0 m/s`: no termination, mean forward `2.007 m/s`, mean lateral
  `-0.038 m/s`, gait anti-phase `0.266`, torso height `2.459-2.522 m`,
  orientation reward `0.982-1.000`.
- command `2.5 m/s`: no termination, mean forward `2.508 m/s`, mean lateral
  `-0.066 m/s`, gait anti-phase `0.297`, torso height `2.483-2.539 m`,
  orientation reward `0.987-1.000`.

Videos on the remote:

- `/workspace/runs/TrexRun-20260515-153148-run22-termcost-gaitstrong-1to2p5-20m-from-run21/videos/run22_f2p0.mp4`
- `/workspace/runs/TrexRun-20260515-153148-run22-termcost-gaitstrong-1to2p5-20m-from-run21/videos/run22_f2p5.mp4`

Conclusion:

- Run22 is the current best low-speed `TrexRun` base. It still does not look
  like a fully natural alternating gait, but it improves tracking, lateral
  drift, uprightness, and anti-phase relative to run20/run21. Use this for the
  next staged speed expansion.

## TrexRun staged expansion run23 Warp 20M

- Remote path:
  `/workspace/runs/TrexRun-20260515-155011-run23-termcost-gaitstrong-2to4-20m-from-run22/`
- Warm start:
  `/workspace/runs/TrexRun-20260515-153148-run22-termcost-gaitstrong-1to2p5-20m-from-run21/checkpoints/000026214400`
- Command range: `2.0-4.0 m/s`, straight-line only.

Training eval rewards:

- `0`: `-40.958`
- `6553600`: `77.598`
- `13107200`: `164.326`
- `19660800`: `199.615`
- `26214400`: `231.268`

Fixed-command diagnostics, Warp, standing reset, seed 0, final 500 steps:

- command `2.0 m/s`: no termination, mean forward `1.994 m/s`, mean lateral
  `-0.023 m/s`, gait anti-phase `0.316`, torso height `2.470-2.518 m`,
  orientation reward `0.982-1.000`.
- command `3.0 m/s`: no termination, mean forward `3.052 m/s`, mean lateral
  `-0.008 m/s`, gait anti-phase `0.320`, torso height `2.484-2.552 m`,
  orientation reward `0.986-1.000`.
- command `4.0 m/s`: no termination, mean forward `4.431 m/s`, mean lateral
  `0.057 m/s`, gait anti-phase `0.300`, torso height `2.539-2.647 m`,
  orientation reward `0.977-0.999`.

Videos on the remote:

- `/workspace/runs/TrexRun-20260515-155011-run23-termcost-gaitstrong-2to4-20m-from-run22/videos/run23_f3p0.mp4`
- `/workspace/runs/TrexRun-20260515-155011-run23-termcost-gaitstrong-2to4-20m-from-run22/videos/run23_f4p0.mp4`

Conclusion:

- Run23 successfully expands the controlled straight-line bridge to `4 m/s`.
  It overspeeds the `4 m/s` command and still looks like a simplified
  airborne/bounding gait, so it should not be promoted as final. It is suitable
  as the next staged-expansion starting point.

## TrexRun failed 4-6 m/s expansions run24/run25

Run24:

- Remote path:
  `/workspace/runs/TrexRun-20260515-160806-run24-termcost-gaitstrong-4to6-20m-from-run23/`
- Change from run23: command range `4.0-6.0 m/s`, tighter high-speed tracking
  sigma.
- Final eval reward: `-16.671`.
- Fixed gates survived but oversped badly:
  - `4.0 m/s`: mean forward `4.631 m/s`, lateral `0.186`, anti-phase `0.299`.
  - `5.0 m/s`: mean forward `6.400 m/s`, lateral `0.055`, anti-phase `0.215`.
  - `6.0 m/s`: mean forward `6.873 m/s`, lateral `0.270`, anti-phase `0.193`.

Run25:

- Remote path:
  `/workspace/runs/TrexRun-20260515-162638-run25-speederr-gaitstrong-4to6-20m-from-run23/`
- Code/config change: added `forward_speed_error` and trained with
  `forward_speed_error=-12`, no forward-progress reward, command range
  `4.0-6.0 m/s`.
- Final eval reward: `-37.717`.
- Fixed gates again survived but oversped:
  - `4.0 m/s`: mean forward `5.006 m/s`, lateral `0.157`, anti-phase `0.276`.
  - `5.0 m/s`: mean forward `6.728 m/s`, lateral `-0.023`, anti-phase `0.208`.
  - `6.0 m/s`: mean forward `6.701 m/s`, lateral `0.100`, anti-phase `0.226`.

Conclusion:

- Both `4-6 m/s` continuations failed as command-tracking policies. The next
  expansion should be smaller (`3-5 m/s`) and should continue to use the
  explicit speed-error term.

## TrexRun smaller expansion run26 Warp 20M

- Remote path:
  `/workspace/runs/TrexRun-20260515-164126-run26-speederr-gaitstrong-3to5-20m-from-run23/`
- Warm start:
  `/workspace/runs/TrexRun-20260515-155011-run23-termcost-gaitstrong-2to4-20m-from-run22/checkpoints/000026214400`
- Config change: command range `3.0-5.0 m/s`, `forward_speed_error=-10`,
  no forward-progress reward.

Training eval rewards:

- `0`: `20.620`
- `6553600`: `-0.970`
- `13107200`: `22.630`
- `19660800`: `71.805`
- `26214400`: `97.510`

Fixed-command diagnostics, Warp, standing reset, seed 0, final 500 steps:

- command `3.0 m/s`: no termination, mean forward `3.084 m/s`, mean lateral
  `-0.055 m/s`, gait anti-phase `0.356`, torso height `2.459-2.522 m`,
  orientation reward `0.985-0.999`.
- command `4.0 m/s`: no termination, mean forward `4.340 m/s`, mean lateral
  `0.089 m/s`, gait anti-phase `0.313`, torso height `2.492-2.591 m`,
  orientation reward `0.982-0.998`.
- command `5.0 m/s`: no termination, mean forward `7.189 m/s`, mean lateral
  `0.009 m/s`, gait anti-phase `0.293`, torso height `2.091-2.844 m`,
  orientation reward `0.789-1.000`.

Conclusion:

- Run26 is not a valid `5 m/s` bridge. It confirms the current curriculum is
  stable to about `4 m/s`, but `5 m/s` still falls into the old overspeed
  bounding mode. A targeted `4-5 m/s` run with stronger speed-error cost is the
  next diagnostic.

## TrexRun targeted 5 m/s diagnostic run27 Warp 20M

- Remote path:
  `/workspace/runs/TrexRun-20260515-165506-run27-target5-speederr-4to5-20m-from-run26/`
- Warm start:
  `/workspace/runs/TrexRun-20260515-164126-run26-speederr-gaitstrong-3to5-20m-from-run23/checkpoints/000026214400`
- Config change: command range `4.0-5.0 m/s`, `forward_speed_error=-50`,
  no forward-progress reward, and stronger tracking-forward reward.

Training eval rewards:

- `0`: `-35.295`
- `6553600`: `-136.210`
- `13107200`: `-151.527`
- `19660800`: `3.229`
- `26214400`: `125.754`

Fixed-command diagnostics, Warp, standing reset, seed 0, final 500 steps:

- command `4.0 m/s`: no termination, mean forward `4.238 m/s`, mean lateral
  `0.075 m/s`, gait anti-phase `0.297`, torso height `2.511-2.607 m`,
  orientation reward `0.987-0.999`.
- command `4.5 m/s`: no termination, mean forward `5.128 m/s`, mean lateral
  `0.114 m/s`, gait anti-phase `0.288`, torso height `2.504-2.634 m`,
  orientation reward `0.983-0.999`.
- command `5.0 m/s`: no termination, mean forward `6.116 m/s`, mean lateral
  `0.117 m/s`, gait anti-phase `0.274`, torso height `2.498-2.661 m`,
  orientation reward `0.981-1.000`.

Conclusion:

- Run27 reduced the overspeed compared with run26 but did not solve it. The
  current branch is blocked at the `5 m/s` transition: even a narrow target
  range and strong symmetric speed-error cost still produce an overspeeding
  bounding mode.

## TrexRun strict 5 m/s bridge run28/run29

Run28:

- Remote path:
  `/workspace/runs/TrexRun-20260515-171717-run28-target5-stricttrack-4to5-20m-from-run27/`
- Warm start:
  `/workspace/runs/TrexRun-20260515-165506-run27-target5-speederr-4to5-20m-from-run26/checkpoints/000026214400`
- Config change: strict speed tracking with
  `reward_config.high_speed_tracking_sigma_scale=0.0`, `tracking_sigma=0.25`,
  `forward_speed_error=-50`.
- Fixed gates:
  - `4.0 m/s`: mean forward `4.138 m/s`, lateral `0.051`, anti-phase `0.310`.
  - `4.5 m/s`: mean forward `4.849 m/s`, lateral `0.118`, anti-phase `0.288`.
  - `5.0 m/s`: mean forward `5.749 m/s`, lateral `0.080`, anti-phase `0.286`.

Run29:

- Remote path:
  `/workspace/runs/TrexRun-20260515-173053-run29-target5-stricter-4p5to5-20m-from-run28/`
- Warm start:
  `/workspace/runs/TrexRun-20260515-171717-run28-target5-stricttrack-4to5-20m-from-run27/checkpoints/000026214400`
- Config change: narrower `4.5-5.0 m/s` range, `tracking_sigma=0.2`,
  `forward_speed_error=-100`.
- Training eval rewards:
  - `0`: `52.562`
  - `6553600`: `-38.372`
  - `13107200`: `25.587`
  - `19660800`: `67.152`
  - `26214400`: `115.568`
- Fixed gates:
  - `4.5 m/s`: no termination, mean forward `4.748 m/s`, mean lateral
    `0.104 m/s`, gait anti-phase `0.290`, torso height `2.506-2.625 m`,
    orientation reward `0.979-0.995`.
  - `5.0 m/s`: no termination, mean forward `5.441 m/s`, mean lateral
    `0.098 m/s`, gait anti-phase `0.287`, torso height `2.490-2.628 m`,
    orientation reward `0.976-0.993`.
- Video on remote:
  `/workspace/runs/TrexRun-20260515-173053-run29-target5-stricter-4p5to5-20m-from-run28/videos/run29_f5p0.mp4`

Conclusion:

- Run29 is the current best `5 m/s` bridge. It still overspeeds by about
  `0.44 m/s` and visually remains a simplified bounding gait, but it suppresses
  the previous `6-7+ m/s` overspeed mode enough to use as the next staged
  expansion base.

## TrexRun failed strict 5-7 m/s expansion run30

- Remote path:
  `/workspace/runs/TrexRun-20260515-174521-run30-stricttrack-5to7-20m-from-run29/`
- Warm start:
  `/workspace/runs/TrexRun-20260515-173053-run29-target5-stricter-4p5to5-20m-from-run28/checkpoints/000026214400`
- Config change: command range `5.0-7.0 m/s`, strict tracking sigma, and
  `forward_speed_error=-100`.

Training eval rewards:

- `0`: `-222.224`
- `6553600`: `-224.501`
- `13107200`: `-238.143`
- `19660800`: `-289.806`
- `26214400`: `-303.814`

Fixed-command diagnostics, Warp, standing reset, seed 0, final 500 steps:

- command `5.0 m/s`: no termination, mean forward `5.357 m/s`, mean lateral
  `0.101 m/s`, gait anti-phase `0.281`, torso height `2.492-2.628 m`,
  orientation reward `0.976-0.993`.
- command `6.0 m/s`: no termination, mean forward `7.181 m/s`, mean lateral
  `0.210 m/s`, gait anti-phase `0.234`, torso height `2.493-2.678 m`,
  orientation reward `0.983-1.000`.
- command `7.0 m/s`: no termination, mean forward `9.061 m/s`, mean lateral
  `0.290 m/s`, gait anti-phase `0.225`, torso height `2.518-2.754 m`,
  orientation reward `0.974-0.998`.

Conclusion:

- Run30 failed as a `5-7 m/s` expansion. The strict tracking setup preserves
  the `5 m/s` bridge but does not prevent the overspeed/bounding mode at
  `6-7 m/s`.

## TrexRun failed speed-gated 5-6 m/s expansion run31

- Remote path:
  `/workspace/runs/TrexRun-20260515-181312-run31-gaitgate-5to6-20m-from-run29/`
- Warm start:
  `/workspace/runs/TrexRun-20260515-173053-run29-target5-stricter-4p5to5-20m-from-run28/checkpoints/000026214400`
- Config change: command range `5.0-6.0 m/s`, strict tracking sigma,
  `forward_speed_error=-100`, and positive gait/contact rewards gated by
  forward speed tracking.

Training eval rewards:

- `0`: `-268.308`
- `6553600`: `-304.364`
- `13107200`: `-320.068`
- `19660800`: `-301.575`
- `26214400`: `-159.296`

Fixed-command diagnostics, Warp, standing reset, seed 0, final 500 steps:

- command `5.0 m/s`: no termination, mean forward `5.304 m/s`, mean lateral
  `0.083 m/s`, gait anti-phase `0.283`, torso height `2.503-2.638 m`,
  orientation reward `0.969-0.989`.
- command `6.0 m/s`: no termination, mean forward `6.810 m/s`, mean lateral
  `0.012 m/s`, gait anti-phase `0.260`, torso height `2.476-2.647 m`,
  orientation reward `0.968-0.993`.

Conclusion:

- Run31 failed as a clean `5-6 m/s` expansion. Gating positive gait rewards by
  speed tracking improved the final scalar but did not eliminate the
  overspeed/bounding mode. Run29 remains the best current strict `5 m/s`
  bridge.

## TrexRun absolute-speed diagnostics run32/run33/run34

Run32:

- Remote path:
  `/workspace/runs/TrexRun-20260515-184136-run32-absspeed-5to6-20m-from-run29/`
- Warm start:
  `/workspace/runs/TrexRun-20260515-173053-run29-target5-stricter-4p5to5-20m-from-run28/checkpoints/000026214400`
- Config change: `forward_speed_abs_error=-4`, command range `5.0-6.0 m/s`.
- Fixed gates:
  - `5.0 m/s`: mean forward `5.334 m/s`, lateral `0.139`, vertical `0.403`,
    anti-phase `0.281`.
  - `6.0 m/s`: mean forward `6.742 m/s`, lateral `0.065`, vertical `0.498`,
    anti-phase `0.251`.

Run33:

- Remote path:
  `/workspace/runs/TrexRun-20260515-185327-run33-absspeed12-5to6-20m-from-run32/`
- Warm start:
  `/workspace/runs/TrexRun-20260515-184136-run32-absspeed-5to6-20m-from-run29/checkpoints/000026214400`
- Config change: `forward_speed_abs_error=-12`.
- Fixed gates:
  - `5.0 m/s`: mean forward `5.227 m/s`, lateral `0.103`, vertical `0.464`,
    anti-phase `0.292`.
  - `6.0 m/s`: mean forward `6.418 m/s`, lateral `-0.034`, vertical `0.694`,
    anti-phase `0.273`.
- Rendered sample:
  `/workspace/runs/TrexRun-20260515-185327-run33-absspeed12-5to6-20m-from-run32/videos/run33_f6p0.mp4`

Run34:

- Remote path:
  `/workspace/runs/TrexRun-20260515-190803-run34-grounded-5to6-20m-from-run33/`
- Warm start:
  `/workspace/runs/TrexRun-20260515-185327-run33-absspeed12-5to6-20m-from-run32/checkpoints/000026214400`
- Config change: kept `forward_speed_abs_error=-12` and strengthened vertical
  velocity, height excess, missing-foot-contact, foot slip, phase-contact, and
  contact-duty penalties.
- Fixed gates:
  - `5.0 m/s`: mean forward `5.238 m/s`, lateral `0.073`, vertical `0.368`,
    anti-phase `0.297`.
  - `6.0 m/s`: mean forward `6.277 m/s`, lateral `0.083`, vertical `0.517`,
    anti-phase `0.275`.
- Rendered sample:
  `/workspace/runs/TrexRun-20260515-190803-run34-grounded-5to6-20m-from-run33/videos/run34_f6p0.mp4`

Conclusion:

- The absolute speed term improves command tracking, and `-12` is better than
  `-4`, but the rendered `6 m/s` samples still show airborne bounding rather
  than a grounded alternating gait. Run34 is numerically better than run33 but
  should not be promoted as a successful high-speed checkpoint.
