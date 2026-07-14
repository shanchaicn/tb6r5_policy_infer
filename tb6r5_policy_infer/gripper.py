"""Gripper observation and command helpers for TB6-R5 policy inference."""

from __future__ import annotations

import numpy as np

from .constants import DEFAULT_GRIPPER_OBSERVATION_MM, RED, RESET


def clip_gripper_mm(distance_mm: float, min_distance: float, max_distance: float) -> float:
    lo = min(float(min_distance), float(max_distance))
    hi = max(float(min_distance), float(max_distance))
    return float(np.clip(float(distance_mm), lo, hi))


def rpc_strides(control_fps: float, arm_rpc_rate_hz: float, gripper_rpc_rate_hz: float) -> tuple[int, int]:
    control_fps = max(float(control_fps), 0.1)
    arm_rpc_rate_hz = max(float(arm_rpc_rate_hz), 0.1)
    gripper_rpc_rate_hz = max(float(gripper_rpc_rate_hz), 0.1)
    arm_stride = max(1, round(control_fps / arm_rpc_rate_hz))
    grip_stride = max(1, round(control_fps / gripper_rpc_rate_hz))
    return arm_stride, grip_stride


def on_rpc_tick(control_step: int, stride: int) -> bool:
    return control_step % max(int(stride), 1) == 0


def clamp_joint_step(q_target: np.ndarray, q_current: np.ndarray, max_step: float) -> np.ndarray:
    dq = np.clip(q_target - q_current, -max_step, max_step)
    return q_current + dq


def gripper_state_label_mm(distance_mm: float, min_distance: float, max_distance: float) -> str:
    lo = min(float(min_distance), float(max_distance))
    hi = max(float(min_distance), float(max_distance))
    if distance_mm <= lo + 0.05 * (hi - lo):
        return "闭合"
    if distance_mm >= hi - 0.05 * (hi - lo):
        return "张开"
    return "中间"


def gripper_mm_to_normalized(distance_mm: float, max_distance: float) -> float:
    return float(np.clip(distance_mm / max_distance, 0.0, 1.0))


def gripper_normalized_to_mm(norm: float, max_distance: float) -> float:
    return float(np.clip(norm, 0.0, 1.0) * max_distance)


def resolve_gripper_observation_mm(arm, constant_mm: float | None) -> float:
    if constant_mm is not None:
        return float(constant_mm)
    if arm is not None:
        feedback = arm.get_gripper_distance_mm()
        if feedback is not None:
            return float(feedback)
    return DEFAULT_GRIPPER_OBSERVATION_MM


def validate_gripper_hysteresis_mm(
    close_mm: float,
    open_mm: float,
    min_distance: float,
    max_distance: float,
) -> None:
    lo = min(float(min_distance), float(max_distance))
    hi = max(float(min_distance), float(max_distance))
    if not (lo <= close_mm <= hi):
        raise ValueError(f"--gripper-close-mm ({close_mm}) must be within [{lo}, {hi}]")
    if not (lo <= open_mm <= hi):
        raise ValueError(f"--gripper-open-mm ({open_mm}) must be within [{lo}, {hi}]")
    if close_mm >= open_mm:
        raise ValueError(
            f"--gripper-close-mm ({close_mm}) must be < --gripper-open-mm ({open_mm}) "
            "(deadband: mm <= close → closed, mm >= open → open)"
        )


def gripper_desired_open_mm(
    gripper_mm: float,
    held_open: bool | None,
    close_mm: float,
    open_mm: float,
) -> bool:
    """Asymmetric hysteresis on mm distance (min_distance=closed, max_distance=open).

    - mm >= open_mm → open intent
    - mm <= close_mm → closed intent
    - between close_mm and open_mm → hold previous latched state
    """
    if held_open is None:
        if gripper_mm >= open_mm:
            return True
        if gripper_mm <= close_mm:
            return False
        return gripper_mm >= (close_mm + open_mm) / 2.0

    if held_open:
        if gripper_mm <= close_mm:
            return False
        return True

    if gripper_mm >= open_mm:
        return True
    return False


def latched_gripper_mm(
    held_open: bool | None,
    *,
    min_distance: float,
    max_distance: float,
) -> float | None:
    if held_open is None:
        return None
    return float(max_distance) if held_open else float(min_distance)


def update_legacy_gripper_state(
    *,
    gripper_mm: float,
    held_open: bool | None,
    pending_mm: float | None,
    close_mm: float,
    open_mm: float,
    control_step: int,
    last_edge_step: int,
    edge_min_steps: int,
    min_distance: float,
    max_distance: float,
) -> tuple[bool | None, float | None, int, bool]:
    """Apply hysteresis edge; queue binary mm until a gripper RPC tick sends it."""
    desired_open = gripper_desired_open_mm(gripper_mm, held_open, close_mm, open_mm)
    edge = held_open is None or desired_open != held_open
    edge_accepted = False
    if edge and control_step - last_edge_step >= edge_min_steps:
        held_open = desired_open
        pending_mm = float(max_distance) if desired_open else float(min_distance)
        last_edge_step = control_step
        edge_accepted = True
    return held_open, pending_mm, last_edge_step, edge_accepted


def gripper_edge_min_steps(fps: float, min_interval_s: float) -> int:
    """Convert edge min interval (seconds) to control-loop steps using deployment fps."""
    return max(1, int(round(fps * min_interval_s)))


def should_send_gripper_mm(
    gripper_distance: float,
    last_sent_mm: float | None,
    cmd_delta: float,
) -> bool:
    """Return whether gripper distance changed enough to warrant a SubLoop1 gripper cmd."""
    if cmd_delta == float("inf"):
        return False
    if last_sent_mm is None:
        return True
    return abs(gripper_distance - last_sent_mm) >= cmd_delta


def print_gripper_status(
    *,
    gripper_raw: float,
    gripper_cmd_mm: float,
    gripper_obs: float,
    sent: bool,
    gripper_min_distance: float,
    gripper_max_distance: float,
    gripper_interval: float,
    chunk_step: int | None,
    chunk_size: int | None,
    legacy_mode: bool = False,
    desired_open: bool | None = None,
    send_gripper: bool = False,
    gripper_normalized: bool = False,
    gripper_subloop: str | None = None,
    pending_gripper_mm: float | None = None,
    edge_accepted: bool = False,
    log_prefix: str = "POLICY",
    dry_run: bool = False,
    on_arm_rpc_tick: bool = False,
) -> None:
    state = gripper_state_label_mm(gripper_cmd_mm, gripper_min_distance, gripper_max_distance)
    prefix = "dry-run: " if dry_run else ""
    if legacy_mode:
        if gripper_subloop == "NotRunExecute":
            pending_info = f" pending={pending_gripper_mm:.0f}mm" if pending_gripper_mm is not None else ""
            cmd_status = f"{prefix}legacy arm JogAnyJ, gripper=NotRunExecute" f"{pending_info} (edge={edge_accepted})"
        elif gripper_subloop:
            cmd_status = f"{prefix}legacy gripper {gripper_subloop} interval={gripper_interval:.1f}"
        else:
            cmd_status = f"{prefix}legacy hysteresis edge={edge_accepted}"
    elif not on_arm_rpc_tick:
        cmd_status = f"{prefix}skip (not arm RPC tick)"
    elif gripper_subloop == "NotRunExecute":
        cmd_status = f"{prefix}SubLoop1 arm JogAnyJ, gripper=NotRunExecute"
    elif gripper_subloop:
        cmd_status = f"{prefix}SubLoop1 gripper {gripper_subloop} interval={gripper_interval:.1f}"
    elif sent:
        cmd_status = f"{prefix}SubLoop1 sent"
    else:
        cmd_status = f"{prefix}SubLoop1 arm only"
    chunk_info = ""
    if chunk_step is not None and chunk_size is not None:
        chunk_info = f" chunk={chunk_step + 1}/{chunk_size}"
    latched_info = ""
    if legacy_mode and desired_open is not None:
        latched_info = f" latched={'open' if desired_open else 'closed'}"
    if gripper_normalized:
        action_str = f"action[6]={gripper_raw:.3f} (norm)"
        obs_str = f"obs={gripper_obs:.3f} (norm)"
    else:
        action_str = f"action[6]={gripper_raw:.2f}mm"
        obs_str = f"obs={gripper_obs:.2f}mm"
    line = (
        f"[{log_prefix}][GRIPPER] {action_str} latched_cmd={gripper_cmd_mm:.2f}mm "
        f"{obs_str} state={state}{latched_info} "
        f"{'would_send' if dry_run else 'sent'}={sent}{chunk_info} | {cmd_status}"
    )
    print(line)


def print_gripper_config(
    gripper_max_distance: float,
    gripper_min_distance: float,
    gripper_interval: float,
    gripper_cmd_delta: float,
    gripper_continuous: bool,
    chunk_size: int | None,
    n_action_steps: int | None,
    temporal_ensemble_coeff: float | None,
    refresh_policy_every_step: bool,
    gripper_normalized: bool = False,
    arm_rpc_rate_hz: float | None = None,
    gripper_rpc_rate_hz: float | None = None,
    control_fps: float | None = None,
    gripper_close_mm: float | None = None,
    gripper_open_mm: float | None = None,
    log_prefix: str = "POLICY",
) -> None:
    mode = (
        "continuous mm + SubLoop1"
        if gripper_continuous
        else "legacy hysteresis (arm SubLoop1 JogAnyJ + binary gripper)"
    )
    print(
        f"{RED}[{log_prefix}][GRIPPER] 配置: min_dist={gripper_min_distance:.1f}mm "
        f"max_dist={gripper_max_distance:.1f}mm "
        f"interval={gripper_interval:.1f} cmd_delta={gripper_cmd_delta:.2f}mm mode={mode}{RESET}"
    )
    if arm_rpc_rate_hz is not None and gripper_rpc_rate_hz is not None and control_fps is not None:
        arm_stride, grip_stride = rpc_strides(control_fps, arm_rpc_rate_hz, gripper_rpc_rate_hz)
        print(
            f"{RED}[{log_prefix}][GRIPPER] RPC 分频: control={control_fps:.0f}Hz "
            f"arm={arm_rpc_rate_hz:.0f}Hz (stride={arm_stride}) "
            f"gripper={gripper_rpc_rate_hz:.0f}Hz (stride={grip_stride}){RESET}"
        )
    if gripper_normalized:
        print(
            f"{RED}[{log_prefix}][GRIPPER] action[6]/state[6] 训练为 [0,1] 归一化；"
            f"推理 obs: mm/{gripper_max_distance:.0f}，action: norm×{gripper_max_distance:.0f}mm。{RESET}"
        )
    else:
        print(
            f"{RED}[{log_prefix}][GRIPPER] action[6]/state[6] 单位为 mm"
            f"（{gripper_min_distance:.0f}=闭合，{gripper_max_distance:.0f}=张开）；"
            f"与 record_tb6r5_lerobot.sh 一致。{RESET}"
        )
    if not gripper_continuous and gripper_close_mm is not None and gripper_open_mm is not None:
        print(
            f"{RED}[{log_prefix}][GRIPPER] 滞回阈值: mm<={gripper_close_mm:.1f}→{gripper_min_distance:.0f}mm(闭合), "
            f"mm>={gripper_open_mm:.1f}→{gripper_max_distance:.0f}mm(张开){RESET}"
        )
    if temporal_ensemble_coeff is not None:
        print(
            f"{RED}[{log_prefix}][ACTION] Temporal Ensemble coeff={temporal_ensemble_coeff:g}, "
            f"chunk_size={chunk_size}：每步推理并融合重叠 chunk 预测。{RESET}"
        )
    elif refresh_policy_every_step:
        print(
            f"{RED}[{log_prefix}][ACTION] 每步 policy.reset() + 重推理（action queue 不累积；"
            f"chunk_size={chunk_size}）。{RESET}"
        )
    elif n_action_steps is not None:
        print(
            f"{RED}[{log_prefix}][ACTION] Action queue: chunk_size={chunk_size}, n_action_steps={n_action_steps}，"
            f"每 {n_action_steps} 步重推理。可调 --n-action-steps 或 --temporal-ensemble-coeff 0.01。{RESET}"
        )
