# trex-gym
Tyrannosaur model tooling.

## Installation
This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

Install the locked Python 3.13 environment:

```
uv sync
```

Run tests inside the environment:

```
uv run python -m unittest discover -s tools -p '*test.py'
```

The active runtime dependencies are intentionally small:

* [MuJoCo](https://mujoco.org/) - Physics model loading and viewer.
* [NumPy](https://numpy.org/) - Numeric arrays.
* [SciPy](https://scipy.org/) - Rotation math for kinematics conversion.

## MuJoCo XML
The URDF remains the source of truth. Regenerate MuJoCo XML from it with:

```
uv run python tools/urdf_to_mujoco.py assets/trex.urdf assets/trex.xml
```

Launch the generated model in the MuJoCo viewer:

```
uv run python -m mujoco.viewer --mjcf=assets/trex.xml
```
