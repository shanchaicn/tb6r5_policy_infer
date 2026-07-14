"""Hardware policy inference control loop for TB6-R5."""

from __future__ import annotations

import time

import numpy as np
import torch

from .lerobot_compat import predict_action

from .camera import CameraPreview, create_camera_stream, destroy_camera_windows, parse_camera_serials
from .constants import BOLD_GREEN, DEFAULT_REALSENSE_SERIAL_DICT, INFER_LOG_PREFIX, RESET
from .gripper import (
    clamp_joint_step,
    clip_gripper_mm,
    gripper_desired_open_mm,
    gripper_edge_min_steps,
    gripper_mm_to_normalized,
    gripper_normalized_to_mm,
    latched_gripper_mm,
    on_rpc_tick,
    print_gripper_config,
    print_gripper_status,
    resolve_gripper_observation_mm,
    rpc_strides,
    should_send_gripper_mm,
    update_legacy_gripper_state,
    validate_gripper_hysteresis_mm,
)

from .policy import load_policy_components, policy_action_queue_info, prepare_policy_for_hardware


def go_home(
    arm,
    home_joint_deg: tuple[float, ...],
    settle_time_s: float,
    *,
    gripper_interval: float,
    gripper_max_distance: float,
    gripper_min_distance: float,
) -> bool:
    home_q = np.deg2rad(np.asarray(home_joint_deg, dtype=float))
    print(f"[{INFER_LOG_PREFIX}] Homing to {tuple(home_joint_deg)} deg ...", flush=True)
    print(
        f"{BOLD_GREEN}[{INFER_LOG_PREFIX}][GRIPPER] 复位：SubLoop1 MoveAbsJ + "
        f"MoveTwoFingersGripper(distance={gripper_max_distance:.1f}mm, "
        f"interval={gripper_interval:.1f}){RESET}",
        flush=True,
    )
    ok = arm.go_home(
        home_q,
        gripper_distance=gripper_max_distance,
        interval=gripper_interval,
        max_distance=gripper_max_distance,
        min_distance=gripper_min_distance,
        settle_timeout_s=max(30.0, float(settle_time_s)),
    )
    if settle_time_s > 0:
        time.sleep(settle_time_s)
    if ok:
        print(f"[{INFER_LOG_PREFIX}] Homing done.", flush=True)
    else:
        print(f"[{INFER_LOG_PREFIX}] Homing FAILED (MoveAbsJ settle or exit).", flush=True)
    return bool(ok)


def run_inference(args) -> int:
    if args.gripper_max_distance <= 0:
        raise ValueError("--gripper-max-distance must be > 0")
    if args.gripper_min_distance < 0:
        raise ValueError("--gripper-min-distance must be >= 0")
    if args.gripper_min_distance > args.gripper_max_distance:
        raise ValueError("--gripper-min-distance must be <= --gripper-max-distance")
    if not args.gripper_continuous:
        validate_gripper_hysteresis_mm(
            args.gripper_close_mm,
            args.gripper_open_mm,
            args.gripper_min_distance,
            args.gripper_max_distance,
        )
        if args.gripper_edge_min_interval < 0:
            raise ValueError("--gripper-edge-min-interval must be >= 0")

    policy, preprocessor, postprocessor = load_policy_components(
        policy_path=args.policy_path,
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        device=args.device,
    )
    # Validate CLI against the policy type, fill per-policy fps/RPC defaults,
    # and apply type-specific inference overrides. This resolves args.fps /
    # args.arm_rpc_rate_hz / args.gripper_rpc_rate_hz (which default to None).
    prepare_policy_for_hardware(policy, args)

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.arm_rpc_rate_hz <= 0 or args.gripper_rpc_rate_hz <= 0:
        raise ValueError("--arm-rpc-rate-hz and --gripper-rpc-rate-hz must be > 0")

    cam_stream = None
    black = np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8)
    if not args.no_camera:
        cam_stream, camera_names = create_camera_stream(
            camera_urls=args.camera_urls,
            camera_devices=args.camera_devices,
            camera_serials=args.camera_serials,
            default_serial_dict=DEFAULT_REALSENSE_SERIAL_DICT,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        )
        cam_stream.start()
        cam_stream.wait_ready()
    else:
        camera_names = sorted(parse_camera_serials(args.camera_serials, DEFAULT_REALSENSE_SERIAL_DICT).keys())
        print(f"[{INFER_LOG_PREFIX}] --no-camera: feeding black frames (predictions will be meaningless)")

    arm = None
    home_joint_deg = tuple(args.home_joint_deg)
    arm_stride, grip_stride = rpc_strides(args.fps, args.arm_rpc_rate_hz, args.gripper_rpc_rate_hz)
    gripper_edge_min_steps_val = gripper_edge_min_steps(args.fps, args.gripper_edge_min_interval)
    if not args.dry_run:
        from .hardware.tb6r5 import TB6R5Interface

        arm = TB6R5Interface(
            ip=args.robot_ip,
            rpc_port=args.rpc_port,
            joint_count=6,
            rpc_cmd_rate_hz=max(args.fps, 20),
            zone_ratio=args.zone_ratio,
            joint_vel=args.joint_vel,
            joint_acc=args.joint_acc,
            joint_dec=args.joint_dec,
            print_rpc=args.print_rpc,
        )
        arm.connect()
        print(f"[{INFER_LOG_PREFIX}] Connected to TB6-R5 at {args.robot_ip}:{args.rpc_port}")
        if not args.no_home_on_start:
            go_home(
                arm,
                home_joint_deg,
                args.home_settle_time,
                gripper_interval=args.gripper_interval,
                gripper_max_distance=args.gripper_max_distance,
                gripper_min_distance=args.gripper_min_distance,
            )
    else:
        print(
            f"[{INFER_LOG_PREFIX}] Dry-run mode: inference only, no RPC "
            f"(gripper mode={'continuous' if args.gripper_continuous else 'hysteresis'})"
        )

    # Formatter-only interface for --print-rpc previews in dry-run (no connection / SDK load).
    rpc_formatter = None
    if args.dry_run and args.print_rpc:
        from .hardware.tb6r5 import TB6R5Interface

        rpc_formatter = TB6R5Interface(
            ip=args.robot_ip,
            rpc_port=args.rpc_port,
            joint_count=6,
            rpc_cmd_rate_hz=max(args.fps, 20),
            zone_ratio=args.zone_ratio,
            joint_vel=args.joint_vel,
            joint_acc=args.joint_acc,
            joint_dec=args.joint_dec,
        )
    rpc_stream_count = 0

    dt = 1.0 / args.fps
    last_print = 0.0
    last_images: dict[str, np.ndarray] = {name: black.copy() for name in camera_names}
    held_gripper_open: bool | None = True if (arm is not None and not args.no_home_on_start) else None
    pending_gripper_mm: float | None = None
    last_gripper_sent_mm: float | None = None

    control_step = 0
    last_gripper_rpc_step = 0 if (arm is not None and not args.no_home_on_start) else -gripper_edge_min_steps_val

    chunk_size = getattr(policy.config, "chunk_size", None)
    n_action_steps = getattr(policy.config, "n_action_steps", None)
    print_gripper_config(
        args.gripper_max_distance,
        args.gripper_min_distance,
        args.gripper_interval,
        args.gripper_cmd_delta,
        args.gripper_continuous,
        chunk_size,
        n_action_steps,
        args.temporal_ensemble_coeff,
        args.refresh_policy_every_step,
        args.gripper_normalized,
        arm_rpc_rate_hz=args.arm_rpc_rate_hz,
        gripper_rpc_rate_hz=args.gripper_rpc_rate_hz,
        control_fps=args.fps,
        gripper_close_mm=None if args.gripper_continuous else args.gripper_close_mm,
        gripper_open_mm=None if args.gripper_continuous else args.gripper_open_mm,
    )
    if args.show_camera and cam_stream is not None:
        print(
            f"[{INFER_LOG_PREFIX}] RGB preview enabled "
            f"(windows: {INFER_LOG_PREFIX} RGB - realsense_0/1). Use --no-show-camera to disable."
        )
    cam_preview = None
    if args.show_camera and cam_stream is not None:
        cam_preview = CameraPreview(cam_stream, fps=float(args.camera_preview_fps))
        cam_preview.start()
    print(f"[{INFER_LOG_PREFIX}] Inference loop started. Press Ctrl+C to stop.")
    print(
        f"[{INFER_LOG_PREFIX}] Control loop {args.fps:.0f} Hz | "
        f"arm RPC {args.arm_rpc_rate_hz:.0f} Hz | gripper RPC {args.gripper_rpc_rate_hz:.0f} Hz"
    )
    try:
        while True:
            start_t = time.time()

            if arm is not None:
                q_current = np.asarray(arm.get_joint_positions(), dtype=np.float32)[:6]
            else:
                q_current = np.zeros(6, dtype=np.float32)

            gripper_obs_mm = clip_gripper_mm(
                resolve_gripper_observation_mm(arm, args.gripper_observation_constant),
                args.gripper_min_distance,
                args.gripper_max_distance,
            )
            if args.gripper_normalized:
                gripper_obs = gripper_mm_to_normalized(gripper_obs_mm, args.gripper_max_distance)
            else:
                gripper_obs = gripper_obs_mm
            observation = {
                "observation.state": np.concatenate(
                    [q_current, np.array([gripper_obs], dtype=np.float32)],
                    axis=0,
                )
            }

            if cam_stream is not None:
                imgs = cam_stream.get_images()
                for name in camera_names:
                    if name in imgs:
                        last_images[name] = imgs[name]
            for name in camera_names:
                observation[f"observation.images.{name}"] = last_images[name]

            if args.refresh_policy_every_step:
                policy.reset()

            action_tensor = predict_action(
                observation=observation,
                policy=policy,
                device=torch.device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=False,
                task=args.task,
                robot_type="tb6r5",
            )
            action = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
            if action.shape[0] < 7:
                raise ValueError(f"Expected action dim >= 7, got {action.shape}")

            q_target = action[:6]
            q_cmd = clamp_joint_step(q_target, q_current, args.joint_step_max_rad)

            gripper_raw = float(action[6])
            now = time.time()
            sent = False
            if args.gripper_normalized:
                gripper_cmd_mm = clip_gripper_mm(
                    gripper_normalized_to_mm(gripper_raw, args.gripper_max_distance),
                    args.gripper_min_distance,
                    args.gripper_max_distance,
                )
            else:
                gripper_cmd_mm = clip_gripper_mm(
                    gripper_raw,
                    args.gripper_min_distance,
                    args.gripper_max_distance,
                )
            send_gripper = False
            edge_accepted = False

            if args.gripper_continuous:
                gripper_mm_for_rpc = gripper_cmd_mm
            else:
                held_gripper_open, pending_gripper_mm, last_gripper_rpc_step, edge_accepted = (
                    update_legacy_gripper_state(
                        gripper_mm=gripper_cmd_mm,
                        held_open=held_gripper_open,
                        pending_mm=pending_gripper_mm,
                        close_mm=args.gripper_close_mm,
                        open_mm=args.gripper_open_mm,
                        control_step=control_step,
                        last_edge_step=last_gripper_rpc_step,
                        edge_min_steps=gripper_edge_min_steps_val,
                        min_distance=args.gripper_min_distance,
                        max_distance=args.gripper_max_distance,
                    )
                )
                send_gripper = edge_accepted

            chunk_step, chunk_size = policy_action_queue_info(policy)

            gripper_subloop: str | None = None
            on_arm_rpc_tick = on_rpc_tick(control_step, arm_stride)
            if on_arm_rpc_tick and (arm is not None or args.dry_run):
                on_gripper_rpc_tick = on_rpc_tick(control_step, grip_stride)
                gripper_cmd_delta = args.gripper_cmd_delta if on_gripper_rpc_tick else float("inf")
                if args.gripper_continuous:
                    gripper_mm_for_rpc = gripper_cmd_mm
                    gripper_will_send = should_send_gripper_mm(
                        gripper_mm_for_rpc,
                        last_gripper_sent_mm,
                        gripper_cmd_delta,
                    )
                else:
                    latched_mm = latched_gripper_mm(
                        held_gripper_open,
                        min_distance=args.gripper_min_distance,
                        max_distance=args.gripper_max_distance,
                    )
                    gripper_mm_for_rpc = (
                        pending_gripper_mm
                        if pending_gripper_mm is not None
                        else (latched_mm if latched_mm is not None else float(args.gripper_max_distance))
                    )
                    legacy_gripper_cmd_delta = (
                        args.gripper_cmd_delta
                        if (on_gripper_rpc_tick and pending_gripper_mm is not None)
                        else float("inf")
                    )
                    gripper_will_send = should_send_gripper_mm(
                        gripper_mm_for_rpc,
                        last_gripper_sent_mm,
                        legacy_gripper_cmd_delta,
                    )
                    gripper_cmd_delta = legacy_gripper_cmd_delta

                if arm is not None:
                    sent = arm.set_joint_positions_with_gripper(
                        q_cmd,
                        gripper_mm_for_rpc,
                        interval=args.gripper_interval,
                        max_distance=args.gripper_max_distance,
                        min_distance=args.gripper_min_distance,
                        cmd_delta=gripper_cmd_delta,
                    )
                    if gripper_will_send:
                        last_gripper_sent_mm = gripper_mm_for_rpc
                    if not args.gripper_continuous and gripper_will_send and pending_gripper_mm is not None:
                        pending_gripper_mm = None
                else:
                    sent = True
                    if gripper_will_send:
                        last_gripper_sent_mm = gripper_mm_for_rpc
                    if rpc_formatter is not None:
                        preview_cmd = rpc_formatter.build_subloop1_stream_cmd(
                            q_cmd,
                            gripper_mm_for_rpc if gripper_will_send else None,
                            clear_buffer=0 if rpc_stream_count == 0 else 1,
                            interval=args.gripper_interval,
                            max_distance=args.gripper_max_distance,
                            min_distance=args.gripper_min_distance,
                        )
                        slot = "first" if rpc_stream_count == 0 else "stream"
                        print(f"[{INFER_LOG_PREFIX}][RPC] dry-run {slot} would send: {preview_cmd}")
                        rpc_stream_count += 1

                gripper_subloop = (
                    f"distance={gripper_mm_for_rpc:.2f}mm" if gripper_will_send else "NotRunExecute"
                )

            if now - last_print >= args.print_every:
                if args.gripper_normalized:
                    action6_str = f"action[6]={gripper_raw:.3f} (norm)"
                else:
                    action6_str = f"action[6]={gripper_raw:.2f}mm"
                print(
                    f"[{INFER_LOG_PREFIX}] "
                    f"q_cur={np.round(q_current, 3)} "
                    f"q_tgt={np.round(q_target, 3)} "
                    f"q_cmd={np.round(q_cmd, 3)} "
                    f"{action6_str}"
                )
                latched_display_mm = (
                    latched_gripper_mm(
                        held_gripper_open,
                        min_distance=args.gripper_min_distance,
                        max_distance=args.gripper_max_distance,
                    )
                    if not args.gripper_continuous
                    else None
                )
                print_gripper_status(
                    gripper_raw=gripper_raw,
                    gripper_cmd_mm=latched_display_mm if latched_display_mm is not None else gripper_cmd_mm,
                    gripper_obs=gripper_obs,
                    sent=sent,
                    gripper_min_distance=args.gripper_min_distance,
                    gripper_max_distance=args.gripper_max_distance,
                    gripper_interval=args.gripper_interval,
                    chunk_step=chunk_step,
                    chunk_size=chunk_size,
                    legacy_mode=not args.gripper_continuous,
                    desired_open=held_gripper_open if not args.gripper_continuous else None,
                    send_gripper=send_gripper,
                    gripper_normalized=args.gripper_normalized,
                    gripper_subloop=gripper_subloop,
                    pending_gripper_mm=pending_gripper_mm,
                    edge_accepted=edge_accepted,
                    dry_run=args.dry_run,
                    on_arm_rpc_tick=on_arm_rpc_tick,
                )
                last_print = now

            control_step += 1
            elapsed = time.time() - start_t
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print(f"\n[{INFER_LOG_PREFIX}] Stopped by user.")
    finally:
        if cam_preview is not None:
            cam_preview.stop()
        if args.show_camera and cam_stream is not None:
            destroy_camera_windows()
        if cam_stream is not None:
            cam_stream.stop()
        if arm is not None:
            try:
                if not args.no_home_on_exit:
                    go_home(
                        arm,
                        home_joint_deg,
                        args.home_settle_time,
                        gripper_interval=args.gripper_interval,
                        gripper_max_distance=args.gripper_max_distance,
                        gripper_min_distance=args.gripper_min_distance,
                    )
                arm.disable()
                print(f"[{INFER_LOG_PREFIX}] Robot disabled.", flush=True)
            except Exception as exc:
                print(f"[{INFER_LOG_PREFIX}] Exit cleanup error: {exc}", flush=True)
    return 0
