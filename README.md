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

## Collision Capsules
The URDF stores generated collision geometry as custom capsule elements under
`<collision>` blocks. Visual mesh geoms remain visual-only in MuJoCo, while
collision geoms are generated as MuJoCo capsules.

Regenerate capsule collisions from the visual OBJ meshes:

```
uv run python tools/generate_collision_capsules.py assets/trex.urdf --report
```

The generator fits capsules in each owning link frame. It uses the visual mesh
vertices, URDF forward kinematics at the neutral pose, PCA-oriented axes from the
mesh primitive tools, and a volume-minimizing capsule fit along the dominant
axis. The checked-in set covers the foot bones with each tarsometatarsus split
into three adjacent capsules and redundant hidden toe capsules omitted, plus
composite capsules for the pelvis, rear/mid/front torso, neck, head, and four
tail sections.

The `--report` output includes fit quality metrics for every generated capsule:
`max_outside_distance`, `mean_abs_surface_error`, and
`p95_abs_surface_error`. The tests assert that all generated capsules enclose
their source mesh vertices within tolerance and that no collision meshes are
emitted into MuJoCo.
