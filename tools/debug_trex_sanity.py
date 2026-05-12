"""Small TrexGetup sanity check for remote debugging."""

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from mjx_gym import trex_getup


def main() -> None:
    print("start", flush=True)
    env = trex_getup.TrexGetup()
    print(f"action_size {env.action_size}", flush=True)
    print(f"ctrl_ranges {np.asarray(env.mj_model.actuator_ctrlrange)}", flush=True)

    states = [env.reset(jax.random.PRNGKey(seed)) for seed in range(3)]
    for seed, state in enumerate(states):
        print(f"reset {seed} {np.asarray(state.data.qpos[:7])}", flush=True)
    reset_diff = not np.allclose(
        np.asarray(states[0].data.qpos), np.asarray(states[1].data.qpos)
    )
    print(f"reset_diff_0_1 {reset_diff}", flush=True)

    model = env.mj_model
    model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    leg_joint_names = (
        "joint_hip_adduction_right",
        "joint_hip_adduction_left",
        "joint_femur_right",
        "joint_femur_left",
        "joint_tibia_right",
        "joint_tibia_left",
        "joint_tarsometatarsus_right",
        "joint_tarsometatarsus_left",
    )
    start_pos = {
        name: data.qpos[model.jnt_qposadr[model.joint(name).id]]
        for name in leg_joint_names
    }
    action_ids = np.asarray(env._action_actuator_ids)
    data.ctrl[action_ids] = (
        np.asarray(env._action_ctrl_positive_scale) * env._config.action_scale
    )
    for _ in range(250):
        mujoco.mj_step(model, data)
    for name in leg_joint_names:
        joint_id = model.joint(name).id
        qpos_id = model.jnt_qposadr[joint_id]
        delta = data.qpos[qpos_id] - start_pos[name]
        print(f"delta {name} {float(delta):.6g}", flush=True)
    print(f"warnings {data.warning.number}", flush=True)

    state = env.reset(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    for index in range(3):
        action = jax.random.uniform(
            jax.random.PRNGKey(index + 10),
            (env.action_size,),
            minval=-1.0,
            maxval=1.0,
        )
        state = step(state, action)
        print(
            "jax_step"
            f" {index} reward {float(state.reward):.6g}"
            f" max_qvel {float(jp.max(jp.abs(state.data.qvel))):.6g}"
            f" max_force {float(jp.max(jp.abs(state.data.actuator_force))):.6g}",
            flush=True,
        )
    print("done", flush=True)


if __name__ == "__main__":
    main()
