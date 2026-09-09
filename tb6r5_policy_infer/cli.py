"""CLI entry point for TB6-R5 LeRobot policy hardware inference."""

from __future__ import annotations

import argparse
import sys

from .constants import (
    DEFAULT_CD_VERSION,
    DEFAULT_GRIPPER_MAX_D,
    DEFAULT_GRIPPER_MIN_D,
    DEFAULT_HOME_JOINT_DEG,
    DEFAULT_HOME_SETTLE_TIME_S,
    DEFAULT_JOG_ANY_JOINT_ACC,
    DEFAULT_JOG_ANY_JOINT_DEC,
    DEFAULT_JOG_ANY_JOINT_VEL,
    DEFAULT_SUBLOOP,
    DEFAULT_TWO_FINGERS_GRIPPER_CMD_DELTA,
    DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL,
    DEFAULT_ZONE_RATIO,
)

_DOC = """
Run a trained LeRobot policy on TB6-R5 hardware.
Supported types (auto-detected from the checkpoint): ACT, Diffusion, SmolVLA, pi0, pi0.5 (pi05), pi0-FAST.

Inference follows the official LeRobot flow (lerobot-record policy path):
  - load policy + pre/post processors from a pretrained checkpoint
  - build observation dict (state + camera images [+ language task for VLA])
  - predict_action() -> policy.select_action()
  - apply robot-side safety limits and send via SubLoop1 / dual-model

Pass options via CLI and/or --config path/to.yaml (CLI overrides YAML).

Per-policy hardware defaults (fps / arm RPC Hz) are applied unless --no-policy-defaults:
  - act:       fps=30
  - diffusion: fps=15   (n_obs_steps history + denoising)
  - smolvla:   fps=10   (VLA, requires --task)
  - pi0_fast:  fps=10   (VLA, requires --task)
  - pi0/pi05:  fps=8    (VLA, requires --task)
Gripper RPC defaults to 2 Hz for all. Any explicit CLI value overrides the default.

ACT-only flags: --temporal-ensemble-coeff, --refresh-policy-every-step
All chunk policies: --n-action-steps (bounds differ by type; see --help)

Homing: start defaults ON (--no-home-on-start to skip); Ctrl+C/exit defaults OFF
(--home-on-exit to enable). See README.

Use --dry-run first to validate outputs before sending commands to the robot.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=_DOC)
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        metavar="PATH",
        help="YAML config file. Keys use argparse dest names (robot_ip, policy_path, ...). CLI flags override YAML.",
    )
    parser.add_argument(
        "--robot-ip",
        default=None,
        help="TB6-R5 robot IP (required via CLI or YAML)",
    )
    parser.add_argument("--rpc-port", type=int, default=5868, help="TB6 RPC port")
    parser.add_argument(
        "--policy-path",
        default=None,
        help="Path (or HF repo) of pretrained checkpoint (required via CLI or YAML)",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional LeRobot dataset root (only needed if loading stats from the dataset)",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Optional LeRobot repo_id (only used together with --dataset-root)",
    )
    parser.add_argument(
        "--task",
        default="tb6r5 teleoperation",
        help="Language task for SmolVLA / VLA policies (lerobot-record --dataset.single_task). Required for smolvla.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto (default), cuda, or cpu. Falls back to CPU if CUDA is unavailable.",
    )
    parser.add_argument(
        "--policy-type",
        default="auto",
        choices=("auto", "act", "diffusion", "smolvla", "pi0", "pi05", "pi0_fast"),
        help=(
            "Policy type. Default 'auto' reads it from the checkpoint config.json. "
            "Set explicitly to assert the checkpoint matches (errors on mismatch)."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "Control loop frequency (Hz). Default: per-policy (ACT 30, diffusion 15, "
            "smolvla/pi0_fast 10, pi0/pi05 8). Overridden by an explicit value; "
            "use --no-policy-defaults for the generic 30 Hz."
        ),
    )
    parser.add_argument("--joint-step-max-rad", type=float, default=0.03, help="Per-step joint delta clamp (rad)")
    parser.add_argument(
        "--action-space",
        default="joints",
        choices=("joints", "ee_pose"),
        help=(
            "Policy state/action layout. "
            "'joints': 7D [j0..j5, gripper_mm] + JogAnyJ (default). "
            "'ee_pose': 8D [x,y,z,qx,qy,qz,qw,gripper_m] from Topic robottarget + JogAnyC."
        ),
    )
    parser.add_argument(
        "--ee-step-max-m",
        type=float,
        default=0.03,
        help="Per-step TCP translation clamp in meters (ee_pose only; <=0 disables)",
    )
    parser.add_argument(
        "--joint-vel",
        type=float,
        default=DEFAULT_JOG_ANY_JOINT_VEL,
        help="JogAnyJ joint_vel in RPC (default: 1.0, conservative for hardware runs)",
    )
    parser.add_argument(
        "--joint-acc",
        type=float,
        default=DEFAULT_JOG_ANY_JOINT_ACC,
        help="JogAnyJ joint_acc in RPC (default: 1.0)",
    )
    parser.add_argument(
        "--joint-dec",
        type=float,
        default=DEFAULT_JOG_ANY_JOINT_DEC,
        help="JogAnyJ joint_dec in RPC (default: 1.0)",
    )
    parser.add_argument(
        "--zone-ratio",
        type=float,
        default=DEFAULT_ZONE_RATIO,
        help="JogAnyJ zone_ratio in RPC (default: 0.0; ignored when --cd-version 45)",
    )
    parser.add_argument(
        "--cd-version",
        type=int,
        default=DEFAULT_CD_VERSION,
        choices=(44, 45),
        dest="cd_version",
        help=(
            "JogAnyJ RPC dialect: 44=include --zone_ratio/--clear_buffer (default); "
            "45=omit zone_ratio and clear_buffer."
        ),
    )
    parser.add_argument(
        "--subloop",
        type=int,
        default=DEFAULT_SUBLOOP,
        choices=(0, 1),
        help=(
            "Motion RPC wrapping: 1=SubLoop1 --exec nesting (default); "
            "0=direct {arm||grip} dual-model (no SubLoop1 exit)."
        ),
    )
    parser.add_argument(
        "--arm-rpc-rate-hz",
        type=float,
        default=None,
        help="Arm SubLoop1 RPC rate in Hz. Default: per-policy (matches --fps).",
    )
    parser.add_argument(
        "--gripper-rpc-rate-hz",
        type=float,
        default=None,
        help=(
            "Gripper SubLoop1 update rate in Hz. Default: 2 (g_model=2), "
            "or matched to --arm-rpc-rate-hz when --g-model 3."
        ),
    )
    parser.add_argument(
        "--gripper-observation-constant",
        type=float,
        default=None,
        help="Force constant mm value for observation.state[6] (default: read actual_pos feedback).",
    )
    parser.add_argument(
        "--gripper-max-distance",
        type=float,
        default=DEFAULT_GRIPPER_MAX_D,
        help="Gripper max open distance in mm (record script GRIPPER_MAX_D, default 70)",
    )
    parser.add_argument(
        "--gripper-min-distance",
        type=float,
        default=DEFAULT_GRIPPER_MIN_D,
        help="Gripper min distance in mm (record script GRIPPER_MIN_D, default 30)",
    )
    parser.add_argument(
        "--gripper-normalized",
        action="store_true",
        help=(
            "Training used gripper in [0,1]: obs.state[6]=feedback_mm/max_distance, "
            "action[6]=policy_norm*max_distance before sending to robot."
        ),
    )
    parser.add_argument(
        "--g-model",
        type=int,
        default=2,
        choices=(2, 3),
        dest="g_model",
        help=(
            "Gripper hardware model: 2=MoveTwoFingersGripper (default, 0–70mm); "
            "3=JogAnyJ j1 meters (Enable||Enable, gripper RPC=arm rate, "
            "collect/control 0–80mm, state from models[1].joint[0])."
        ),
    )
    parser.add_argument(
        "--gripper-interval",
        type=float,
        default=DEFAULT_TWO_FINGERS_GRIPPER_INTERVAL,
        help="MoveTwoFingersGripper interval (g_model=2 only)",
    )
    parser.add_argument(
        "--gripper-cmd-delta",
        type=float,
        default=DEFAULT_TWO_FINGERS_GRIPPER_CMD_DELTA,
        help="Minimum gripper distance change (mm) before re-sending SubLoop1 RPC (requires --gripper-threshold).",
    )
    parser.add_argument(
        "--gripper-threshold",
        action="store_true",
        default=False,
        help=(
            "Enable mm delta gating via --gripper-cmd-delta. "
            "Default off: send gripper target on every gripper RPC tick."
        ),
    )
    parser.add_argument(
        "--gripper-continuous",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map action[6] directly to mm distance (default). Use --no-gripper-continuous for legacy hysteresis.",
    )
    parser.add_argument(
        "--gripper-close-mm",
        type=float,
        default=40.0,
        help="Legacy hysteresis: latched open→closed when action mm <= this (only with --no-gripper-continuous).",
    )
    parser.add_argument(
        "--gripper-open-mm",
        type=float,
        default=50.0,
        help="Legacy hysteresis: latched closed→open when action mm >= this (only with --no-gripper-continuous).",
    )
    parser.add_argument(
        "--gripper-edge-min-interval",
        type=float,
        default=2.0,
        help="Legacy hysteresis: min seconds between gripper RPC edges.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override n_action_steps at inference. "
            "ACT/SmolVLA/pi0/pi05/pi0_fast: 1..chunk_size. "
            "Diffusion: 1..(horizon - n_obs_steps + 1). "
            "Default: checkpoint value."
        ),
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        metavar="COEFF",
        help=(
            "Enable ACT temporal ensembling (original ACT paper uses 0.01). "
            "Every control step runs inference and fuses overlapping chunk predictions. "
            "Implies n_action_steps=1; do not combine with --refresh-policy-every-step."
        ),
    )
    parser.add_argument(
        "--refresh-policy-every-step",
        action="store_true",
        help="ACT only: Call policy.reset() every control step (disables action queue; more reactive, slower).",
    )
    parser.add_argument(
        "--camera-serials",
        default=None,
        help=(
            "RealSense mode: comma-separated name=serial pairs, e.g. "
            "'realsense_0=135522071053,realsense_1=327122073649'. "
            "Ignored when --camera-devices or --camera-urls is set. Defaults to the teleop serial dict."
        ),
    )
    parser.add_argument(
        "--camera-devices",
        default=None,
        help=(
            "V4L2 mode: comma-separated name=device pairs, e.g. "
            "'realsense_0=/dev/video0,realsense_1=/dev/video2' or 'realsense_0=0,realsense_1=2'. "
            "Ignored when --camera-urls is set. Uses OpenCV VideoCapture (no pyrealsense2)."
        ),
    )
    parser.add_argument(
        "--camera-urls",
        default=None,
        help=(
            "HTTP mode: comma-separated name=url pairs, e.g. "
            "'realsense_0=http://192.168.2.42:8888/RsCameraSensor/0/0/color,"
            "realsense_1=http://192.168.2.42:8888/RsCameraSensor/1/0/color'. "
            "Each URL is polled via GET; response must be JPEG/PNG image bytes."
        ),
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--camera-preview-fps",
        type=int,
        default=30,
        help="OpenCV preview refresh rate (independent of control loop; default: 30).",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Skip camera capture and feed black frames (pipeline test only; outputs are meaningless).",
    )
    parser.add_argument(
        "--show-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display RGB preview windows during inference (default: on).",
    )
    parser.add_argument(
        "--no-policy-defaults",
        action="store_true",
        help="Do not apply per-policy fps/RPC defaults (use CLI values as-is).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Predict and print actions without sending to robot")
    parser.add_argument(
        "--print-rpc",
        action="store_true",
        help=(
            "Also print high-rate SubLoop1 JogAnyJ stream commands on every send "
            "(init/home/exit/disable are always printed). Dry-run: print the command "
            "that would be sent. Not throttled by --print-every."
        ),
    )
    parser.add_argument(
        "--home-joint-deg",
        type=float,
        nargs=6,
        default=DEFAULT_HOME_JOINT_DEG,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Home joint angles in degrees; used on start and with --home-on-exit (default: teleop home pose)",
    )
    parser.add_argument(
        "--home-settle-time",
        type=float,
        default=DEFAULT_HOME_SETTLE_TIME_S,
        help="Seconds to wait after MoveAbsJ homing",
    )
    parser.add_argument(
        "--no-home-on-start",
        action="store_true",
        help="Skip MoveAbsJ home when the script starts (default: home on start)",
    )
    parser.add_argument(
        "--home-on-exit",
        action="store_true",
        help="MoveAbsJ to home on Ctrl+C / exit (default: stop in place then Disable)",
    )
    parser.add_argument(
        "--print-every",
        type=float,
        default=0.5,
        help="Minimum print interval (seconds) for action debug lines",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI, optionally loading YAML defaults from --config (CLI wins)."""
    from .config_loader import apply_config_defaults, load_yaml_config, validate_required_args

    argv = list(sys.argv[1:] if argv is None else argv)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", "-c", default=None)
    pre_args, _ = pre.parse_known_args(argv)

    parser = build_parser()
    if pre_args.config:
        cfg = load_yaml_config(pre_args.config)
        apply_config_defaults(parser, cfg)
        print(f"[INFER] Loaded config: {pre_args.config}")

    args = parser.parse_args(argv)
    validate_required_args(args)
    return args


def main(argv: list[str] | None = None) -> int:
    from .runner import run_inference

    args = parse_args(argv)
    return run_inference(args)


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
