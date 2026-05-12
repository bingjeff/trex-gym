"""Pure MuJoCo control-authority check for the T-Rex model."""

import mujoco
import numpy as np

from mjx_gym import trex_getup


def main() -> None:
    env = trex_getup.TrexGetup()
    model = env.mj_model
    print(f"action_size {env.action_size}", flush=True)
    print(f"ctrl_ranges {np.asarray(model.actuator_ctrlrange)}", flush=True)

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
        print(f"positive_delta {name} {float(delta):.6g}", flush=True)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    start_pos = {
        name: data.qpos[model.jnt_qposadr[model.joint(name).id]]
        for name in leg_joint_names
    }
    data.ctrl[action_ids] = (
        -np.asarray(env._action_ctrl_negative_scale) * env._config.action_scale
    )
    for _ in range(250):
        mujoco.mj_step(model, data)
    for name in leg_joint_names:
        joint_id = model.joint(name).id
        qpos_id = model.jnt_qposadr[joint_id]
        delta = data.qpos[qpos_id] - start_pos[name]
        print(f"negative_delta {name} {float(delta):.6g}", flush=True)

    print(f"warnings {data.warning.number}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
