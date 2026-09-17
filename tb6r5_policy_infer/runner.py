"""Hardware policy inference control loop for TB6-R5."""

from __future__ import annotations

import time

import numpy as np
import torch

from .lerobot_compat import predict_action

from .camera import (
    CameraPreview,
    create_camera_stream,
    destroy_camera_windows,
    parse_camera_devices,
    parse_camera_serials,
    parse_camera_urls,
)
from .constants import BOLD_GREEN, DEFAULT_REALSENSE_SERIAL_DICT, INFER_LOG_PREFIX, RESET
from .ee_pose import (
    canonicalize_quat_xyzw,
    clamp_ee_step,
    clamp_ee_workspace,
    gripper_m_to_mm,
    gripper_mm_to_m,
    pack_ee_state_xyzw,
    parse_xyz_limit,
    unpack_ee_action_xyzw,
    validate_ee_workspace,
)
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
    resolve_gripper_cmd_delta,
    resolve_gripper_observation_mm,
    rpc_strides,
    should_send_gripper_mm,
    update_legacy_gripper_state,
    validate_gripper_hysteresis_mm,
)

from .policy import load_policy_components, policy_action_queue_info, prepare_policy_for_hardware


def _policy_image_keys(policy) -> list[str]:
    feats = getattr(getattr(policy, "config", None), "input_features", None) or {}
    return sorted(str(k) for k in feats if "images" in str(k) and "depth" not in str(k))


def _is_padding_camera_name(name: str) -> bool:
    """OpenPI/pi05 pads unused views as empty_camera_* — not a physical camera."""
    return name.startswith("empty_camera") or name.startswith("empty_")


def _assert_camera_keys_match_policy(policy, camera_names: list[str]) -> None:
    expected = _policy_image_keys(policy)
    expected_real = [k for k in expected if not _is_padding_camera_name(k.rsplit(".", 1)[-1])]
    fed = [f"observation.images.{n}" for n in camera_names]
    missing = [k for k in expected_real if k not in fed]
    extra = [k for k in fed if k not in expected]
    pad = [k for k in expected if k not in expected_real]
    print(f"[{INFER_LOG_PREFIX}][camera] policy expects: {expected}")
    print(f"[{INFER_LOG_PREFIX}][camera] runner will feed: {fed}")
    if missing:
        raise ValueError(
            "Camera names do not match the checkpoint. The policy will not see the trained images.\n"
            f"  missing in feed: {missing}\n"
            f"  extra in feed:   {extra}\n"
            "Fix --camera-serials / --camera-urls / --camera-devices left-hand names "
            "(must match observation.images.<name> in config.json)."
        )
    if extra:
        print(f"[{INFER_LOG_PREFIX}][camera] WARNING: extra image keys not in policy (ignored): {extra}")
    if pad:
        print(
            f"[{INFER_LOG_PREFIX}][camera] Leaving unused slots unfed {pad}; "
            "PI05/SmolVLA fill them with masked dummy tensors (ignored).",
            flush=True,
        )


def _image_stats_line(name: str, arr) -> str:
    a = np.asarray(arr)
    if a.size == 0:
        return f"{name} empty"
    return f"{name} shape={tuple(a.shape)} mean={float(a.mean()):.1f} min={int(a.min())} max={int(a.max())}"


def _probe_images_into_preprocessor(observation: dict, preprocessor, device, task: str | None) -> None:
    """One-shot check that image tensors survive prepare + preprocessor (what the net actually gets)."""
    from copy import copy

    try:
        from lerobot.policies.utils import prepare_observation_for_inference
    except ImportError:
        from lerobot.common.policies.utils import prepare_observation_for_inference

    obs = copy(observation)
    obs = prepare_observation_for_inference(obs, torch.device(device), task, "tb6r5")
    processed = preprocessor(obs)
    for key in sorted(k for k in processed if "images" in str(k)):
        t = processed[key]
        a = t.detach().float().cpu().numpy()
        print(
            f"[{INFER_LOG_PREFIX}][camera] into-model {key} tensor={tuple(a.shape)} "
            f"mean={float(a.mean()):.4f} min={float(a.min()):.3f} max={float(a.max()):.3f}",
            flush=True,
        )


def go_home(
    arm,
    home_joint_deg: tuple[float, ...],
    settle_time_s: float,
    *,
    gripper_interval: float,
    gripper_max_distance: float,
    gripper_min_distance: float,
    g_model: int = 2,
) -> bool:
    home_q = np.deg2rad(np.asarray(home_joint_deg, dtype=float))
    print(f"[{INFER_LOG_PREFIX}] Homing to {tuple(home_joint_deg)} deg ...", flush=True)
    if int(g_model) == 3:
        grip_desc = (
            f"JogAnyJ j1↔mm(distance={gripper_max_distance:.1f}mm, "
            f"vel/acc/dec=0.5)"
        )
    else:
        grip_desc = (
            f"MoveTwoFingersGripper(distance={gripper_max_distance:.1f}mm, "
            f"interval={gripper_interval:.1f})"
        )
    print(
        f"{BOLD_GREEN}[{INFER_LOG_PREFIX}][GRIPPER] 复位：exit SubLoop1 → "
        f"{{NotRunExecute||{grip_desc}}} → {{MoveAbsJ||NotRunExecute}} "
        f"(g_model={g_model}){RESET}",
        flush=True,
    )
    ok = arm.go_home(
        home_q,
        gripper_distance=gripper_max_distance,
        interval=gripper_interval,
        max_distance=gripper_max_distance,
        min_distance=gripper_min_distance,
        settle_timeout_s=max(5.0, float(settle_time_s)),
    )
    if settle_time_s > 0:
        time.sleep(settle_time_s)
    if ok:
        print(f"[{INFER_LOG_PREFIX}] Homing done.", flush=True)
    else:
        print(f"[{INFER_LOG_PREFIX}] Homing FAILED (gripper/MoveAbsJ dual-model or settle).", flush=True)
    return bool(ok)


def run_inference(args) -> int:
    if args.gripper_max_distance <= 0:
        raise ValueError("--gripper-max-distance must be > 0")
    if args.gripper_min_distance < 0:
        raise ValueError("--gripper-min-distance must be >= 0")
    if args.gripper_min_distance > args.gripper_max_distance:
        raise ValueError("--gripper-min-distance must be <= --gripper-max-distance")
    g_model = int(getattr(args, "g_model", 3))
    if g_model not in (2, 3):
        raise ValueError(f"--g-model must be 2 or 3, got {g_model}")
    if g_model == 3:
        from .hardware.tb6r5 import DEFAULT_GRIPPER_G3_MAX_D, DEFAULT_GRIPPER_MAX_D as _HW_G2_MAX

        if abs(float(args.gripper_max_distance) - float(_HW_G2_MAX)) < 1e-9:
            args.gripper_max_distance = float(DEFAULT_GRIPPER_G3_MAX_D)
    if not args.gripper_continuous:
        validate_gripper_hysteresis_mm(
            args.gripper_close_mm,
            args.gripper_open_mm,
            args.gripper_min_distance,
            args.gripper_max_distance,
        )
        if args.gripper_edge_min_interval < 0:
            raise ValueError("--gripper-edge-min-interval must be >= 0")

    action_space = str(getattr(args, "action_space", "joints")).strip().lower()
    if action_space not in ("joints", "ee_pose"):
        raise ValueError(f"--action-space must be 'joints' or 'ee_pose', got {action_space!r}")
    if action_space == "ee_pose" and args.gripper_normalized:
        raise ValueError("--gripper-normalized is not supported with --action-space ee_pose (use gripper_m)")

    ee_xyz_min = parse_xyz_limit(getattr(args, "ee_xyz_min", None), name="ee_xyz_min")
    ee_xyz_max = parse_xyz_limit(getattr(args, "ee_xyz_max", None), name="ee_xyz_max")
    validate_ee_workspace(ee_xyz_min, ee_xyz_max)
    ee_limit_mode = str(getattr(args, "ee_limit_mode", "clamp") or "clamp").strip().lower()
    if ee_limit_mode not in ("clamp", "abort"):
        raise ValueError(f"--ee-limit-mode must be 'clamp' or 'abort', got {ee_limit_mode!r}")
    if action_space == "ee_pose" and (ee_xyz_min is not None or ee_xyz_max is not None):
        lo = "off" if ee_xyz_min is None else [round(float(v), 4) for v in ee_xyz_min]
        hi = "off" if ee_xyz_max is None else [round(float(v), 4) for v in ee_xyz_max]
        print(
            f"[{INFER_LOG_PREFIX}] Cartesian workspace limit mode={ee_limit_mode} "
            f"xyz_min={lo} xyz_max={hi} (meters)"
        )

    dummy = bool(getattr(args, "dummy", False) or getattr(args, "replay_episode", None) is not None)
    dummy_policy = None
    policy = None
    preprocessor = None
    postprocessor = None
    if dummy:
        from .dummy import DummyPolicy, apply_dummy_loop_defaults

        if getattr(args, "replay_episode", None) is not None:
            from .replay import ReplayPolicy, dataset_fps, load_episode_vectors

            if getattr(args, "fps", None) is None:
                fps = dataset_fps(args.dataset_root)
                if fps is not None:
                    args.fps = fps
            apply_dummy_loop_defaults(args)
            vecs = load_episode_vectors(
                args.dataset_root,
                int(args.replay_episode),
                source=getattr(args, "replay_source", "action"),
            )
            dummy_policy = ReplayPolicy(
                vecs,
                loop=bool(getattr(args, "replay_loop", False)),
                episode_index=int(args.replay_episode),
                source=getattr(args, "replay_source", "action"),
            )
            print(
                f"[{INFER_LOG_PREFIX}] Replay → RPC ({action_space}). "
                f"Use --ee-step-max-m / --joint-step-max-rad to clamp jumps from the live pose.",
                flush=True,
            )
        else:
            apply_dummy_loop_defaults(args)
            dummy_policy = DummyPolicy(
                action_space=action_space,
                mode=getattr(args, "dummy_mode", "hold"),
                wiggle_amp_m=float(getattr(args, "dummy_wiggle_amp_m", 0.02)),
                wiggle_period_s=float(getattr(args, "dummy_wiggle_period_s", 6.0)),
            )
        policy = dummy_policy
    else:
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
        if args.camera_urls:
            camera_names = sorted(parse_camera_urls(args.camera_urls))
        elif args.camera_devices:
            camera_names = sorted(parse_camera_devices(args.camera_devices))
        else:
            camera_names = sorted(parse_camera_serials(args.camera_serials, DEFAULT_REALSENSE_SERIAL_DICT))
        print(f"[{INFER_LOG_PREFIX}] --no-camera: feeding black frames (predictions will be meaningless)")
    if dummy:
        print(f"[{INFER_LOG_PREFIX}][camera] dummy feed names: {list(camera_names)}")
    else:
        _assert_camera_keys_match_policy(policy, camera_names)

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
            g_model=g_model,
            cd_version=args.cd_version,
            subloop=int(getattr(args, "subloop", 1)),
        )
        arm.connect()
        print(
            f"[{INFER_LOG_PREFIX}] Connected to TB6-R5 at {args.robot_ip}:{args.rpc_port} "
            f"(g_model={g_model}, cd_version={args.cd_version})"
        )
        if not args.no_home_on_start:
            go_home(
                arm,
                home_joint_deg,
                args.home_settle_time,
                gripper_interval=args.gripper_interval,
                gripper_max_distance=args.gripper_max_distance,
                gripper_min_distance=args.gripper_min_distance,
                g_model=g_model,
            )
    else:
        print(
            f"[{INFER_LOG_PREFIX}] Dry-run mode: inference only, no RPC "
            f"(gripper mode={'continuous' if args.gripper_continuous else 'hysteresis'}, "
            f"g_model={g_model})"
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
            g_model=g_model,
            cd_version=args.cd_version,
            subloop=int(getattr(args, "subloop", 1)),
        )
    rpc_stream_count = 0

    dt = 1.0 / args.fps
    last_print = 0.0
    last_ee_limit_print = 0.0
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
        gripper_threshold=args.gripper_threshold,
    )
    grip_rpc_label = (
        "JogAnyJ j1(m)" if g_model == 3 else "MoveTwoFingersGripper"
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
        f"action_space={action_space} | "
        f"arm RPC {args.arm_rpc_rate_hz:.0f} Hz | gripper RPC {args.gripper_rpc_rate_hz:.0f} Hz | "
        f"g_model={g_model} ({grip_rpc_label})"
    )
    try:
        while True:
            start_t = time.time()

            if arm is not None:
                q_current = np.asarray(arm.get_joint_positions(), dtype=np.float32)[:6]
                xyz_current, quat_xyzw_current, tcp_ok = arm.get_robottarget()
                xyz_current = np.asarray(xyz_current, dtype=np.float32)
                quat_xyzw_current = canonicalize_quat_xyzw(quat_xyzw_current)
            else:
                q_current = np.zeros(6, dtype=np.float32)
                xyz_current = np.zeros(3, dtype=np.float32)
                quat_xyzw_current = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                tcp_ok = False

            gripper_obs_mm = clip_gripper_mm(
                resolve_gripper_observation_mm(arm, args.gripper_observation_constant),
                args.gripper_min_distance,
                args.gripper_max_distance,
            )
            if action_space == "ee_pose":
                if arm is not None and hasattr(arm, "get_gripper_distance_m"):
                    grip_m = arm.get_gripper_distance_m()
                    if grip_m is None:
                        grip_m = gripper_mm_to_m(gripper_obs_mm)
                else:
                    grip_m = gripper_mm_to_m(gripper_obs_mm)
                grip_m = float(
                    np.clip(
                        grip_m,
                        args.gripper_min_distance / 1000.0,
                        args.gripper_max_distance / 1000.0,
                    )
                )
                gripper_obs = grip_m
                if arm is not None and not tcp_ok:
                    # Keep last zeros only on first frames; still feed policy.
                    pass
                observation = {
                    "observation.state": pack_ee_state_xyzw(xyz_current, quat_xyzw_current, grip_m)
                }
            else:
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

            if control_step == 0:
                for name in camera_names:
                    key = f"observation.images.{name}"
                    print(f"[{INFER_LOG_PREFIX}][camera] raw {_image_stats_line(key, observation[key])}", flush=True)
                    mean = float(np.asarray(observation[key]).mean())
                    if mean < 1.0:
                        print(
                            f"[{INFER_LOG_PREFIX}][camera] WARNING: {key} looks black (mean={mean:.2f}). "
                            "Policy is not seeing the scene — check serials/URLs and preview.",
                            flush=True,
                        )
                if dummy:
                    from .dummy import save_camera_snapshots

                    save_camera_snapshots(observation, camera_names)
                elif preprocessor is not None:
                    _probe_images_into_preprocessor(
                        observation,
                        preprocessor,
                        policy.config.device,
                        args.task,
                    )

            if args.refresh_policy_every_step and not dummy:
                policy.reset()

            if dummy:
                action = dummy_policy.predict(observation, time.time())
            else:
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

            ee_xyz_cmd = None
            ee_quat_xyzw_cmd = None
            xyz_tgt = None
            ee_limit_hit = False
            q_target = q_current
            q_cmd = q_current
            if action_space == "ee_pose":
                xyz_tgt, quat_xyzw_tgt, gripper_raw_m = unpack_ee_action_xyzw(action)
                # Dry-run / missing Topic: current TCP is zeros. Do not clamp from origin
                # or replay/policy targets get squashed to ee_step_max_m.
                if tcp_ok:
                    ee_xyz_cmd = clamp_ee_step(
                        xyz_tgt,
                        xyz_current,
                        getattr(args, "ee_step_max_m", None),
                    )
                else:
                    ee_xyz_cmd = np.asarray(xyz_tgt, dtype=np.float32).ravel()[:3].copy()
                ee_xyz_cmd, ee_limit_hit, ee_limit_overflow = clamp_ee_workspace(
                    ee_xyz_cmd, ee_xyz_min, ee_xyz_max
                )
                if ee_limit_hit:
                    t_hit = time.time()
                    if ee_limit_mode == "abort" or (t_hit - last_ee_limit_print >= args.print_every):
                        print(
                            f"[{INFER_LOG_PREFIX}][ee-limit] TCP xyz outside workspace "
                            f"(overflow_m={np.round(ee_limit_overflow, 4)}, mode={ee_limit_mode}) "
                            f"cmd={np.round(ee_xyz_cmd, 4)}",
                            flush=True,
                        )
                        last_ee_limit_print = t_hit
                    if ee_limit_mode == "abort":
                        print(
                            f"[{INFER_LOG_PREFIX}][ee-limit] abort: skipping JogAnyC and stopping.",
                            flush=True,
                        )
                        break
                ee_quat_xyzw_cmd = canonicalize_quat_xyzw(quat_xyzw_tgt)
                gripper_raw = gripper_raw_m
                gripper_cmd_mm = clip_gripper_mm(
                    gripper_m_to_mm(gripper_raw_m),
                    args.gripper_min_distance,
                    args.gripper_max_distance,
                )
            else:
                if action.shape[0] < 7:
                    raise ValueError(f"Expected action dim >= 7, got {action.shape}")
                q_target = action[:6]
                q_cmd = clamp_joint_step(q_target, q_current, args.joint_step_max_rad)
                gripper_raw = float(action[6])
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

            now = time.time()
            sent = False
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

            chunk_step, chunk_size = (None, 1) if dummy else policy_action_queue_info(policy)

            gripper_subloop: str | None = None
            on_arm_rpc_tick = on_rpc_tick(control_step, arm_stride)
            if on_arm_rpc_tick and (arm is not None or args.dry_run):
                on_gripper_rpc_tick = on_rpc_tick(control_step, grip_stride)
                if args.gripper_continuous:
                    gripper_mm_for_rpc = gripper_cmd_mm
                    gripper_cmd_delta = resolve_gripper_cmd_delta(
                        on_gripper_rpc_tick=on_gripper_rpc_tick,
                        threshold_enabled=args.gripper_threshold,
                        cmd_delta=args.gripper_cmd_delta,
                    )
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
                    gripper_cmd_delta = resolve_gripper_cmd_delta(
                        on_gripper_rpc_tick=on_gripper_rpc_tick,
                        threshold_enabled=args.gripper_threshold,
                        cmd_delta=args.gripper_cmd_delta,
                        allow_send=pending_gripper_mm is not None,
                    )
                    gripper_will_send = should_send_gripper_mm(
                        gripper_mm_for_rpc,
                        last_gripper_sent_mm,
                        gripper_cmd_delta,
                    )

                if arm is not None:
                    if action_space == "ee_pose":
                        sent = arm.set_cartesian_target_with_gripper(
                            ee_xyz_cmd,
                            ee_quat_xyzw_cmd,
                            gripper_mm_for_rpc,
                            interval=args.gripper_interval,
                            max_distance=args.gripper_max_distance,
                            min_distance=args.gripper_min_distance,
                            cmd_delta=gripper_cmd_delta,
                        )
                    else:
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
                        if action_space == "ee_pose":
                            preview_cmd = rpc_formatter.build_cartesian_stream_cmd(
                                ee_xyz_cmd,
                                ee_quat_xyzw_cmd,
                                gripper_mm_for_rpc if gripper_will_send else None,
                                clear_buffer=0 if rpc_stream_count == 0 else 1,
                                interval=args.gripper_interval,
                                max_distance=args.gripper_max_distance,
                                min_distance=args.gripper_min_distance,
                            )
                        else:
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
                if gripper_will_send and g_model == 3 and arm is not None:
                    j1 = arm._gripper_mm_to_joint(gripper_mm_for_rpc)
                    gripper_subloop = f"JogAnyJ j1={j1:.4f}m ({gripper_mm_for_rpc:.2f}mm)"
                elif gripper_will_send and g_model == 3 and rpc_formatter is not None:
                    j1 = rpc_formatter._gripper_mm_to_joint(gripper_mm_for_rpc)
                    gripper_subloop = f"JogAnyJ j1={j1:.4f}m ({gripper_mm_for_rpc:.2f}mm)"

            if now - last_print >= args.print_every:
                cam_bits = []
                for name in camera_names:
                    img = last_images.get(name)
                    if img is None:
                        cam_bits.append(f"{name}=missing")
                    else:
                        cam_bits.append(f"{name} mean={float(np.asarray(img).mean()):.1f} max={int(np.asarray(img).max())}")
                print(f"[{INFER_LOG_PREFIX}][camera] {' | '.join(cam_bits)}", flush=True)
                if action_space == "ee_pose":
                    action7_str = f"action[7]={gripper_raw:.4f}m"
                    xyz_act = np.round(xyz_tgt, 4) if xyz_tgt is not None else "n/a"
                    print(
                        f"[{INFER_LOG_PREFIX}] "
                        f"xyz={np.round(xyz_current, 4)} "
                        f"quat_xyzw={np.round(quat_xyzw_current, 4)} "
                        f"xyz_act={xyz_act} "
                        f"xyz_cmd={np.round(ee_xyz_cmd, 4)} "
                        f"quat_cmd={np.round(ee_quat_xyzw_cmd, 4)} "
                        f"{action7_str}"
                        + (" ee_limit=HIT" if ee_limit_hit else "")
                    )
                else:
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
            if dummy and getattr(dummy_policy, "finished", False):
                print(
                    f"[{INFER_LOG_PREFIX}] Replay episode finished ({control_step} ticks). Stopping.",
                    flush=True,
                )
                break
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
                        g_model=g_model,
                    )
                arm.disable()
                print(f"[{INFER_LOG_PREFIX}] Robot disabled.", flush=True)
            except Exception as exc:
                print(f"[{INFER_LOG_PREFIX}] Exit cleanup error: {exc}", flush=True)
    return 0
