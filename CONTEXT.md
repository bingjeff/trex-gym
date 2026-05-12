# T-Rex MJX Training Context

This document is a handoff note for resuming work if the chat/session is
interrupted.

## Current Goal

Train the T-Rex model with MuJoCo MJX and MuJoCo Playground PPO. The first task
is `TrexGetup`: start from a side-lying zero configuration, get up, and stand.
Walking is a follow-on task.

## Branch and Repo State

- Active branch: `mjx`
- URDF remains the source of truth: `assets/trex.urdf`
- Generated full visual MJCF may exist locally as `assets/trex.xml`; it is
  untracked and should not be committed unless intentionally regenerated.
- Main MJX training package: `mjx_gym`
- Old pre-MJX gym code has been moved to `old/`

Recent important commits:

- `42aab35 Add MJX model simplification`
- `6c3c254 Add MJX model complexity comparison`
- `acc8ab4 Add Trex MJX getup environment`
- `45120ab Record TrexGetup smoke validation`
- `f097559 Tune Trex contacts and gains`

## Python Environment

The project uses `uv`.

```bash
uv sync
```

Current dependency notes:

- Python: `>=3.13`
- MuJoCo + MJX are required.
- MuJoCo Playground is used for PPO.
- `jax` and `jaxlib` are pinned below `0.10` because the current Playground/Brax
  stack still uses APIs removed in JAX `0.10`.

## MJX Model Generation

The tuned MJX training model used by `TrexGetup` is produced in memory by:

```python
from mjx_gym import trex_constants
xml = trex_constants.trex_getup_xml()
```

To write that exact tuned model to disk:

```bash
uv run python -c 'from pathlib import Path; from mjx_gym import trex_constants; Path("/tmp/trex_getup_tuned.xml").write_text(trex_constants.trex_getup_xml())'
```

This model has:

- no visual meshes
- fixed-body fusion for the MJX training model
- a free root
- ground-only robot contacts
- passive and active gains scaled by MuJoCo mass-matrix row sums at zero config
- side-lying and zero-upright initialization helpers

## Current Collision and Gain Policy

For the current training run:

- Valid collision pairs should always include the ground.
- Robot capsule-vs-capsule self contacts are disabled.
- Floor geoms use `contype=1`, `conaffinity=0`.
- Robot contact geoms use `contype=0`, `conaffinity=1`.

Gain scaling:

- Build a MuJoCo mass matrix at zero configuration.
- For each DOF, use `sum(abs(M[dof, :]))` as the gain scale.
- Current tuned factors at `sim_dt=0.004`:
  - passive stiffness: `1000 * row_sum`
  - passive damping: `80 * row_sum`
  - armature: `0.2 * row_sum`
  - position actuator `kp`: `200 * row_sum`
- Actuated joints use `actuated_joint_passive_stiffness_scale=0.0` in
  `TrexGetup`, so the driven leg joints and tail tendon joints are not pinned by
  passive springs that overpower the position actuators.
- Tendon actuator scales are coefficient-weighted sums of their coupled joint
  scales.
- The torso-height reward is gated by upright orientation so the side-lying
  initial pose cannot receive full height reward while doing nothing.

## Validation Commands

Run all tool tests:

```bash
uv run python -m unittest discover -s tools -p '*test.py'
```

Run the most relevant MJX/gain/contact tests:

```bash
uv run python -m unittest \
  tools.mjx_gym_test.TestMjxGym.test_trex_getup_contacts_are_ground_only \
  tools.mjx_gym_test.TestMjxGym.test_trex_getup_mass_scaled_passive_gains_settle_joint_perturbations \
  tools.mjx_gym_test.TestMjxGym.test_trex_getup_actions_move_driven_joints
```

Quick JAX/MuJoCo visibility check:

```bash
uv run python - <<'PY'
import jax
import mujoco
print(jax.devices())
print(mujoco.__version__)
PY
```

Tiny PPO smoke:

```bash
uv run python -m mjx_gym.train \
  --env_name=TrexGetup \
  --num_timesteps=128 \
  --num_envs=2 \
  --num_eval_envs=1 \
  --episode_length=20 \
  --num_evals=1 \
  --num_minibatches=1 \
  --num_updates_per_batch=1 \
  --batch_size=2 \
  --unroll_length=2 \
  --run_evals=false \
  --num_videos=0 \
  --logdir=/tmp/trex-getup-ppo-smoke
```

## Training Entry Point

Local training entry point:

```bash
uv run python -m mjx_gym.train --env_name=TrexGetup
```

Current default PPO config in `mjx_gym/train.py`:

- `num_timesteps=50_000_000`
- `num_evals=5`
- `num_envs=1024`
- `batch_size=256`
- `unroll_length=20`
- `num_minibatches=8`
- `num_updates_per_batch=4`
- policy hidden layers: `(128, 128, 128)`
- value hidden layers: `(256, 256, 256)`

Current `TrexGetup` env config:

- `ctrl_dt=0.02`
- `sim_dt=0.004`
- `Kp=200`
- `passive_stiffness=1000`
- `actuated_joint_passive_stiffness_scale=0.0`
- action space: 10
  - 8 leg position target actions
  - 2 tail tendon actions
- policy observation shape: `(78,)`
- privileged observation shape: `(164,)`

## Checkpoints

Checkpoints are handled by MuJoCo Playground/Brax PPO.

Example launch:

```bash
uv run python -m mjx_gym.train \
  --env_name=TrexGetup \
  --logdir=/workspace/runs \
  --suffix=getup-v1
```

This creates:

```text
/workspace/runs/TrexGetup-YYYYMMDD-HHMMSS-getup-v1/
  checkpoints/
    config.json
    00000012500992/
      ppo_network_config.json
      ...
```

Saved checkpoint contents:

- observation normalizer state
- policy parameters
- value parameters
- network reconstruction config

Not saved:

- optimizer state
- rollout state
- exact PPO loop state

Restoring is therefore best treated as a warm start from a trained policy/value,
not a bit-for-bit continuation of the optimizer.

Resume from latest checkpoint in a run:

```bash
uv run python -m mjx_gym.train \
  --env_name=TrexGetup \
  --logdir=/workspace/runs \
  --suffix=getup-v1-resume \
  --load_checkpoint_path=/workspace/runs/TrexGetup-YYYYMMDD-HHMMSS-getup-v1/checkpoints
```

## RunPod Plan

RunPod is the recommended initial GPU path because it is likely easier than
Vast.ai for the first validated cloud training run.

Initial setup:

1. Create a RunPod account and add a small budget.
2. Add an SSH public key.
3. Launch a single-GPU pod.
4. Start with a regular volume disk; a network volume is not necessary for the
   first smoke test.
5. Suggested storage:
   - volume disk: `50-100 GB`
   - container disk: `20-40 GB`
6. Suggested GPU ladder:
   - L40S or RTX 4090 for first setup/smoke
   - A100 for longer runs
   - H100 only after A100 throughput is clearly limiting

SSH key creation if needed:

```bash
ssh-keygen -t ed25519 -C "runpod-trex-gym"
cat ~/.ssh/runpod_trex_gym.pub
```

Paste only the `.pub` key into RunPod.

SSH connection shape:

```bash
ssh -i ~/.ssh/runpod_trex_gym root@<runpod-host> -p <runpod-port>
```

Bootstrap on the pod:

```bash
cd /workspace
git clone <repo-url> trex-gym
cd trex-gym
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv sync
```

GPU validation:

```bash
uv run python - <<'PY'
import jax
import mujoco
from mujoco import mjx
print(jax.devices())
print(mujoco.__version__)
PY
```

First GPU PPO smoke:

```bash
uv run python -m mjx_gym.train \
  --env_name=TrexGetup \
  --logdir=/workspace/runs \
  --suffix=runpod-smoke \
  --num_timesteps=1000000 \
  --num_evals=2 \
  --num_videos=0
```

First longer run:

```bash
uv run python -m mjx_gym.train \
  --env_name=TrexGetup \
  --logdir=/workspace/runs \
  --suffix=getup-v1 \
  --num_timesteps=50000000 \
  --num_evals=5 \
  --num_videos=0
```

Important: do not rely on the pod or RunPod volume as the only checkpoint copy.
Before expensive runs, configure a checkpoint backup path, such as `rclone`,
`aws s3 sync`, or `rsync` back to local storage.

## Next Likely Work

1. Launch RunPod GPU pod.
2. Clone repo and run `uv sync`.
3. Verify JAX sees the GPU.
4. Run tests or at least the MJX-specific smoke tests.
5. Run a 1M-step GPU PPO smoke.
6. Confirm checkpoints are produced.
7. Add checkpoint backup automation.
8. Start a 50M-step `getup-v1` run.
