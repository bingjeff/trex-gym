"""Drive get-up recovery followed by a joystick locomotion policy."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mjx_gym import trex_getup
from mjx_gym import trex_joystick
from tools.analyze_policy_rollout import _load_policy
from tools.drive_joystick_policy import Gamepad
from tools.drive_joystick_policy import _copy_to_mujoco_viewer


def _joystick_state_from_getup(
    joystick_env: trex_joystick.TrexJoystick,
    joystick_template,
    getup_state,
    command: jax.Array,
):
    info = dict(joystick_template.info)
    info["command"] = command
    info["steps_until_next_cmd"] = jp.array(10_000, dtype=jp.int32)
    info["last_act"] = jp.zeros(joystick_env.action_size)
    info["last_last_act"] = jp.zeros(joystick_env.action_size)
    info["stand_hold_act"] = jp.zeros(joystick_env.action_size)
    info["last_foot_centers"] = joystick_env._foot_centers_world(getup_state.data)
    info["contact_duty"] = jp.zeros(2)
    info["feet_air_time"] = jp.zeros(2)
    info["last_contact"] = jp.zeros(2, dtype=bool)
    info["gait_phase"] = jp.array(0.0)
    obs = joystick_env._get_obs(getup_state.data, info)
    return joystick_template.replace(
        data=getup_state.data,
        obs=obs,
        reward=jp.zeros(()),
        done=jp.zeros(()),
        info=info,
    )


def _switch_ready(
    joystick_env: trex_joystick.TrexJoystick,
    data,
    orientation_threshold: float,
    height_fraction: float,
) -> tuple[bool, float, float]:
    gravity = joystick_env.get_gravity(data)
    orientation = float(jax.device_get(joystick_env._reward_orientation(gravity)))
    torso_height = float(jax.device_get(data.site_xpos[joystick_env._imu_site_id][2]))
    height_ok = torso_height >= joystick_env._target_torso_height * height_fraction
    return orientation >= orientation_threshold and height_ok, orientation, torso_height


def drive(args: argparse.Namespace) -> None:
    getup_config = trex_getup.default_config()
    getup_config.impl = args.impl
    joystick_config = trex_joystick.joystick_config()
    joystick_config.impl = args.impl
    joystick_config.reset_standing_prob = 1.0

    getup_env = trex_getup.TrexGetup(getup_config)
    joystick_env = trex_joystick.TrexJoystick(joystick_config)
    getup_policy = jax.jit(_load_policy(args.getup_checkpoint))
    balance_policy = (
        jax.jit(_load_policy(args.balance_checkpoint))
        if args.balance_checkpoint
        else None
    )
    joystick_policy = jax.jit(_load_policy(args.joystick_checkpoint))
    getup_step = jax.jit(getup_env.step)
    joystick_step = jax.jit(joystick_env.step)

    rng = jax.random.PRNGKey(args.seed)
    state = getup_env.reset(rng)
    joystick_template = joystick_env.reset(rng)
    stage = "joystick" if args.start == "standing" else "getup"
    switch_step: int | None = 0 if stage == "joystick" else None
    balance_start_step: int | None = None
    joystick_start_step: int | None = 0 if stage == "joystick" else None
    ready_count = 0
    orientation = 0.0
    torso_height = 0.0

    if stage == "joystick":
        state = _joystick_state_from_getup(
            joystick_env, joystick_template, joystick_template, jp.zeros(2)
        )

    if args.check_load:
        max_steps = (
            args.max_recovery_steps
            + args.balance_steps
            + args.joystick_check_steps
        )
        for step_index in range(max_steps):
            rng, action_rng = jax.random.split(rng)
            if stage == "joystick":
                command = jp.array([args.check_forward, args.check_turn])
                state.info["command"] = command
                state.info["steps_until_next_cmd"] = args.hold_steps
                state = state.replace(obs=joystick_env._get_obs(state.data, state.info))
                action, _ = joystick_policy(state.obs, action_rng)
                state = joystick_step(state, action)
            elif stage == "balance":
                state.info["command"] = jp.zeros(2)
                state.info["steps_until_next_cmd"] = args.hold_steps
                state = state.replace(obs=joystick_env._get_obs(state.data, state.info))
                policy = balance_policy or joystick_policy
                action, _ = policy(state.obs, action_rng)
                state = joystick_step(state, action)
                ready, orientation, torso_height = _switch_ready(
                    joystick_env,
                    state.data,
                    args.switch_orientation,
                    args.switch_height_fraction,
                )
                ready_count = ready_count + 1 if ready else 0
                balance_elapsed = step_index - (balance_start_step or step_index)
                if (
                    balance_elapsed >= args.balance_steps
                    and ready_count >= args.switch_stable_steps
                ):
                    stage = "joystick"
                    joystick_start_step = step_index + 1
                    ready_count = 0
            else:
                action, _ = getup_policy(state.obs, action_rng)
                state = getup_step(state, action)
                ready, orientation, torso_height = _switch_ready(
                    joystick_env,
                    state.data,
                    args.switch_orientation,
                    args.switch_height_fraction,
                )
                ready_count = ready_count + 1 if ready else 0
                if (
                    step_index + 1 >= args.min_recovery_steps
                    and ready_count >= args.switch_stable_steps
                ):
                    switch_step = step_index + 1
                    state = _joystick_state_from_getup(
                        joystick_env,
                        joystick_template,
                        state,
                        jp.zeros(2),
                    )
                    ready_count = 0
                    if balance_policy or args.balance_steps > 0:
                        stage = "balance"
                        balance_start_step = step_index + 1
                    else:
                        stage = "joystick"
                        joystick_start_step = step_index + 1
            if (
                stage == "joystick"
                and joystick_start_step is not None
                and step_index >= joystick_start_step + args.joystick_check_steps
            ):
                break

        ready, orientation, torso_height = _switch_ready(
            joystick_env,
            state.data,
            args.switch_orientation,
            args.switch_height_fraction,
        )
        local_linvel = jax.device_get(joystick_env.get_local_linvel(state.data))
        local_angvel = jax.device_get(joystick_env.get_local_angvel(state.data))
        print(f"Loaded getup: {args.getup_checkpoint}")
        print(f"Loaded joystick: {args.joystick_checkpoint}")
        print(f"Stage: {stage}")
        print(f"Getup switch step: {switch_step}")
        print(f"Balance start step: {balance_start_step}")
        print(f"Joystick start step: {joystick_start_step}")
        print(f"Ready now: {ready}")
        print(f"Torso height: {torso_height:.3f}")
        print(f"Orientation reward: {orientation:.3f}")
        print(f"Local forward velocity: {float(local_linvel[0]):.3f}")
        print(f"Local turn velocity: {float(local_angvel[1]):.3f}")
        return

    gamepad = Gamepad(args.gamepad, args.deadzone)
    mj_data = mujoco.MjData(joystick_env.mj_model)
    _copy_to_mujoco_viewer(state.data, mj_data)
    mujoco.mj_forward(joystick_env.mj_model, mj_data)

    print(f"Loaded getup: {args.getup_checkpoint}")
    print(f"Loaded joystick: {args.joystick_checkpoint}")
    print(f"Using gamepad: {gamepad.name}")
    print("Recovery runs first; left stick drives after automatic switch.")

    with mujoco.viewer.launch_passive(joystick_env.mj_model, mj_data) as viewer:
        next_step = time.monotonic()
        step_index = 0
        while viewer.is_running():
            loop_start = time.monotonic()
            rng, action_rng = jax.random.split(rng)
            if stage == "joystick":
                command = gamepad.command(
                    args.max_forward, args.max_reverse, args.max_turn
                )
                state.info["command"] = command
                state.info["steps_until_next_cmd"] = args.hold_steps
                state = state.replace(obs=joystick_env._get_obs(state.data, state.info))
                action, _ = joystick_policy(state.obs, action_rng)
                state = joystick_step(state, action)
            elif stage == "balance":
                state.info["command"] = jp.zeros(2)
                state.info["steps_until_next_cmd"] = args.hold_steps
                state = state.replace(obs=joystick_env._get_obs(state.data, state.info))
                policy = balance_policy or joystick_policy
                action, _ = policy(state.obs, action_rng)
                state = joystick_step(state, action)
                ready, _, _ = _switch_ready(
                    joystick_env,
                    state.data,
                    args.switch_orientation,
                    args.switch_height_fraction,
                )
                ready_count = ready_count + 1 if ready else 0
                if (
                    ready_count >= args.switch_stable_steps
                    and balance_start_step is not None
                    and step_index - balance_start_step >= args.balance_steps
                ):
                    stage = "joystick"
                    ready_count = 0
            else:
                action, _ = getup_policy(state.obs, action_rng)
                state = getup_step(state, action)
                ready, _, _ = _switch_ready(
                    joystick_env,
                    state.data,
                    args.switch_orientation,
                    args.switch_height_fraction,
                )
                ready_count = ready_count + 1 if ready else 0
                if (
                    step_index + 1 >= args.min_recovery_steps
                    and ready_count >= args.switch_stable_steps
                ):
                    state = _joystick_state_from_getup(
                        joystick_env, joystick_template, state, jp.zeros(2)
                    )
                    ready_count = 0
                    if balance_policy or args.balance_steps > 0:
                        stage = "balance"
                        balance_start_step = step_index + 1
                    else:
                        stage = "joystick"

            _copy_to_mujoco_viewer(state.data, mj_data)
            mujoco.mj_forward(joystick_env.mj_model, mj_data)
            viewer.sync()

            step_index += 1
            next_step += joystick_env.dt
            sleep_time = next_step - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            elif time.monotonic() - loop_start > 2.0 * joystick_env.dt:
                next_step = time.monotonic()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("getup_checkpoint", type=Path)
    parser.add_argument("joystick_checkpoint", type=Path)
    parser.add_argument("--balance-checkpoint", type=Path)
    parser.add_argument("--impl", choices=("jax", "warp"), default="jax")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamepad", type=int, default=0)
    parser.add_argument("--deadzone", type=float, default=0.12)
    parser.add_argument("--max-forward", type=float, default=0.8)
    parser.add_argument("--max-reverse", type=float, default=0.0)
    parser.add_argument("--max-turn", type=float, default=0.25)
    parser.add_argument("--hold-steps", type=int, default=10_000)
    parser.add_argument("--start", choices=("side", "standing"), default="side")
    parser.add_argument("--switch-orientation", type=float, default=0.9)
    parser.add_argument("--switch-height-fraction", type=float, default=0.9)
    parser.add_argument("--switch-stable-steps", type=int, default=25)
    parser.add_argument("--min-recovery-steps", type=int, default=300)
    parser.add_argument("--max-recovery-steps", type=int, default=1000)
    parser.add_argument("--balance-steps", type=int, default=200)
    parser.add_argument("--joystick-check-steps", type=int, default=100)
    parser.add_argument("--check-forward", type=float, default=0.5)
    parser.add_argument("--check-turn", type=float, default=0.25)
    parser.add_argument(
        "--check-load",
        action="store_true",
        help="Run recovery and joystick phases without opening the viewer.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    drive(parse_args())
