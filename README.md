# trex-gym
OpenAI Gym environment using pybullet for a Tyrannosaur.

## Installation
This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

Install the locked environment:

```
uv sync
```

Run commands inside the environment:

```
uv run python trex_gym/trex_train.py
```

The biggest runtime dependencies are:

* [pyBullet](https://github.com/bulletphysics/bullet3) - Used for physics and rendering.
* [OpenAI Gym](https://github.com/openai/gym) - Provides the basis for the model "environment".
* [OpenAI Baselines](https://github.com/openai/baselines) - Used for the RL agents.
* [Tensorflow](https://github.com/tensorflow/tensorflow) - Used as a dependency for the ML infrastructure.

## MuJoCo XML
The URDF remains the source of truth. Regenerate MuJoCo XML from it with:

```
uv run python tools/urdf_to_mujoco.py assets/trex.urdf assets/trex.xml
```
