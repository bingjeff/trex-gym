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

The active runtime dependencies are:

* [MuJoCo](https://mujoco.org/) - Physics model loading and viewer.
* [MuJoCo MJX](https://mujoco.readthedocs.io/en/latest/mjx.html) - JAX-backed
  MuJoCo simulation.
* [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) -
  PPO training tools built on MJX.
* [NumPy](https://numpy.org/) - Numeric arrays.
* [SciPy](https://scipy.org/) - Rotation math for kinematics conversion.

The Playground training stack is currently pinned to `jax<0.10` because
`brax==0.14.2` still calls a JAX API removed in JAX 0.10.

## MuJoCo XML
The URDF remains the source of truth. Regenerate MuJoCo XML from it with:

```
uv run python tools/urdf_to_mujoco.py assets/trex.urdf assets/trex.xml
```

Launch the generated model in the MuJoCo viewer:

```
uv run python -m mujoco.viewer --mjcf=assets/trex.xml
```

Validate that the generated model can be copied to MJX:

```
uv run python -c "import mujoco; from mujoco import mjx; m=mujoco.MjModel.from_xml_path('assets/trex.xml'); mjx.put_model(m)"
```

Run a small MuJoCo Playground PPO smoke test:

```
uv run train-jax-ppo --env_name=CartpoleBalance --num_timesteps=1024 --num_envs=16 --num_eval_envs=4 --episode_length=100 --num_evals=1 --num_minibatches=1 --num_updates_per_batch=1 --batch_size=16 --unroll_length=5 --run_evals=false --num_videos=0 --logdir=/tmp/trex-gym-playground-smoke
```

The URDF represents hip adduction with intermediate `link_hip_adduction_*`
links. Their adduction joints use `linked_dof_body="link_femur_*"` so the
MuJoCo converter collapses those marker links and emits the adduction hinge as
an additional DOF on each femur body.

The URDF also carries MuJoCo-style control metadata in a top-level `<mujoco>`
extension. The converter maps `<passive>` to stiffness, damping, and
friction-loss attributes on every emitted hinge joint. Tail coupling is
represented by two fixed tendons, `tail_sagittal` and `tail_mediolateral`, with
joint coefficients for each caudal DOF in that plane. The actuator set is
limited to hip adduction, hip flexion, knee, ankle, and the two tail tendons;
all other movable joints remain passive in the generated XML.

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
into three adjacent capsules and redundant hidden toe capsules omitted, femur
capsules for the upper legs, plus composite capsules for the rear/mid/front
torso, neck, head, and four tail sections. Tail-section capsules are attached
to the distal fixed vertebra in their section so they move with that section's
revolute joint. The caudal chain uses x-axis hinges for sagittal flexion and
the next distal z-axis hinge in each section for medio-lateral tail wag.

The `--report` output includes fit quality metrics for every generated capsule:
`max_outside_distance`, `mean_abs_surface_error`, and
`p95_abs_surface_error`. The tests assert that all generated capsules enclose
their source mesh vertices within tolerance and that no collision meshes are
emitted into MuJoCo.
