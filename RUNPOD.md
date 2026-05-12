# RunPod Image Notes

This documents the first working RunPod environment used for MJX training and
the plan for recreating it.

## Current Template

- RunPod template: `runpod-torch-v240`
- Base image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Observed OS: Ubuntu 22.04.5 LTS
- Observed GPU: NVIDIA RTX A5000, driver `550.127.05`
- CUDA install: CUDA 12.4 under `/usr/local/cuda-12.4`
- Startup: RunPod `/start.sh` starts nginx, sshd, and Jupyter Lab.

The pod did not have Docker installed inside the container, so image history was
not available from inside the pod.

## Runtime State We Added

- Installed `uv` at `/root/.local/bin/uv`.
- Synced the project with Python 3.12:

  ```bash
  uv sync --locked --group gpu --group dev
  ```

- Installed/used system media packages for video inspection, especially
  `ffmpeg`.
- Used TensorBoard from the project environment:

  ```bash
  uv run tensorboard --logdir=/workspace/runs --host=0.0.0.0 --port=6006
  ```

Large transient directories observed on the pod:

- `/tmp/uv-cache`: about 11 GB
- `/root/trex-gym/.venv`: about 12 GB

Do not bake run outputs, checkpoints, videos, TensorBoard logs, or uv caches into
a clean reusable image unless intentionally creating a prewarmed image.

## Verified Python Stack

The working pod environment reported:

- Python `3.12.13`
- uv `0.11.13`
- JAX `0.9.2`
- jaxlib `0.9.2`
- CUDA device visible to JAX
- MuJoCo `3.8.0`
- `mujoco-mjx 3.8.0`
- `playground 0.2.0`

Validation command:

```bash
uv run python - <<'PY'
import jax
import mujoco
import sys

print(sys.version)
print(jax.__version__)
print(jax.devices())
print(mujoco.__version__)
PY
```

## Rebuild Plan

Start from the same RunPod base image:

```dockerfile
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
```

Install the small set of system tools we need:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libosmesa6 \
    openssh-server \
    tmux \
    && rm -rf /var/lib/apt/lists/*
```

Install uv:

```dockerfile
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/opt/trex-gym/.venv/bin:/usr/local/cuda/bin:${PATH}"
ENV UV_CACHE_DIR=/tmp/uv-cache
ENV MUJOCO_GL=egl
```

Build the project environment:

```dockerfile
WORKDIR /opt/trex-gym
COPY pyproject.toml uv.lock ./
COPY mjx_gym ./mjx_gym
COPY tools ./tools
COPY assets ./assets
COPY old ./old
RUN uv sync --locked --group gpu --group dev
```

For a development image, copy the whole repo instead of only package/runtime
files. For a training image, prefer mounting the repo and writing all outputs to
`/workspace/runs`.

## Validation Checklist

Run these after building a new image:

```bash
uv run python -c "import jax; print(jax.devices())"
uv run python -c "import mujoco; print(mujoco.__version__)"
uv run python -m unittest tools.mjx_gym_test
uv run python -m tools.debug_trex_mujoco_controls
```

Optional tiny PPO smoke:

```bash
uv run python -m mjx_gym.train \
  --env_name=TrexGetup \
  --logdir=/workspace/runs \
  --suffix=image-smoke \
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
  --num_videos=0
```

## Suggested Tags

Tag images by base stack and project commit:

```text
trex-gym:mjx-<commit>-py312-jax092-mujoco380-cuda124
```

For the first reproducible image based on commit `9a01c40`:

```text
trex-gym:mjx-9a01c40-py312-jax092-mujoco380-cuda124
```

