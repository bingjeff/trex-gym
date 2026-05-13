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
