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
