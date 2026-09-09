"""Load tb6r5-policy-infer options from a YAML file and merge with CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

# Keys accepted in YAML (argparse dest names). Unknown keys are rejected.
_KNOWN_KEYS = frozenset(
    {
        "robot_ip",
        "rpc_port",
        "policy_path",
        "dataset_root",
        "repo_id",
        "task",
        "device",
        "policy_type",
        "fps",
        "joint_step_max_rad",
        "action_space",
        "ee_step_max_m",
        "joint_vel",
        "joint_acc",
        "joint_dec",
        "zone_ratio",
        "cd_version",
        "subloop",
        "arm_rpc_rate_hz",
        "gripper_rpc_rate_hz",
        "gripper_observation_constant",
        "gripper_max_distance",
        "gripper_min_distance",
        "gripper_normalized",
        "g_model",
        "gripper_interval",
        "gripper_cmd_delta",
        "gripper_threshold",
        "gripper_continuous",
        "gripper_close_mm",
        "gripper_open_mm",
        "gripper_edge_min_interval",
        "n_action_steps",
        "temporal_ensemble_coeff",
        "refresh_policy_every_step",
        "camera_serials",
        "camera_devices",
        "camera_urls",
        "camera_width",
        "camera_height",
        "camera_fps",
        "camera_preview_fps",
        "no_camera",
        "show_camera",
        "no_policy_defaults",
        "dry_run",
        "print_rpc",
        "home_joint_deg",
        "home_settle_time",
        "no_home_on_start",
        "home_on_exit",
        "print_every",
    }
)

# Convenience aliases in YAML (map to dest / invert boolean).
_ALIASES = {
    "home_on_start": ("no_home_on_start", True),  # home_on_start: false -> no_home_on_start=True
}


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for --config. Install with: pip install pyyaml"
        ) from exc

    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config root must be a mapping/object, got {type(raw).__name__}: {cfg_path}")

    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key in ("config",):  # ignore nested self-ref
            continue
        if key in _ALIASES:
            dest, invert = _ALIASES[key]
            if invert:
                out[dest] = not bool(value)
            else:
                out[dest] = value
            continue
        if key not in _KNOWN_KEYS:
            raise ValueError(
                f"Unknown config key {key!r} in {cfg_path}. "
                f"Use argparse dest names (underscores), e.g. robot_ip, policy_path, home_on_exit."
            )
        out[key] = value

    if "home_joint_deg" in out and out["home_joint_deg"] is not None:
        hj = list(out["home_joint_deg"])
        if len(hj) != 6:
            raise ValueError(f"home_joint_deg must have 6 values, got {len(hj)}")
        out["home_joint_deg"] = [float(x) for x in hj]

    return out


def apply_config_defaults(parser: argparse.ArgumentParser, config: Mapping[str, Any]) -> None:
    """Set argparse defaults from YAML (CLI values still win when passed explicitly)."""
    if not config:
        return
    parser.set_defaults(**dict(config))


def validate_required_args(args: argparse.Namespace) -> None:
    missing = []
    if not getattr(args, "robot_ip", None):
        missing.append("robot_ip (--robot-ip or YAML)")
    if not getattr(args, "policy_path", None):
        missing.append("policy_path (--policy-path or YAML)")
    if missing:
        raise SystemExit(f"Missing required options: {', '.join(missing)}")
