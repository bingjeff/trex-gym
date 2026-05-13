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
