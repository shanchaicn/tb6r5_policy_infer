"""Per-policy real-robot deployment defaults and validation.

Different LeRobot policies (ACT / Diffusion / SmolVLA / pi0 / pi0.5 / pi0-FAST)
have different inference-time behavior, so the safe hardware defaults (control
fps, RPC rate, action-queue handling, language-task requirement) differ too.

The values below mirror the LeRobot policy configs and their inference cost:
  - ACT       : chunk_size=100, n_action_steps=100, temporal ensemble optional; fast.
  - Diffusion : n_obs_steps=2, horizon=16, n_action_steps=8; ~100-step denoising, slow.
  - SmolVLA   : chunk_size=50,  n_action_steps=50,  n_obs_steps=1; VLA, needs task.
  - pi0/pi05  : chunk_size=50,  n_action_steps=50,  n_obs_steps=1; VLA, needs task, slow.
  - pi0_fast  : autoregressive FAST decoding; VLA, needs task.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import (
    DEFAULT_ARM_RPC_RATE_HZ,
    DEFAULT_CONTROL_FPS,
    DEFAULT_GRIPPER_RPC_RATE_HZ,
    BOLD_GREEN,
    RESET,
)

# Policy type strings that require a natural-language task (VLA models).
LANGUAGE_TASK_POLICIES = frozenset({"smolvla", "pi0", "pi05", "pi0_fast"})

# ACT-only deployment knobs (invalid for other policies).
ACT_ONLY_POLICIES = frozenset({"act"})

_PLACEHOLDER_TASK = "tb6r5 teleoperation"


@dataclass(frozen=True)
class PolicyDeployDefaults:
    """Safe TB6-R5 hardware defaults per policy type."""

    fps: float
    arm_rpc_rate_hz: float
    gripper_rpc_rate_hz: float
    note: str


# Conservative defaults tuned to each policy's inference cost. Users can always
# override on the CLI; pass --no-policy-defaults to skip this table entirely.
POLICY_DEPLOY_DEFAULTS: dict[str, PolicyDeployDefaults] = {
    "act": PolicyDeployDefaults(30.0, 30.0, DEFAULT_GRIPPER_RPC_RATE_HZ, "fast; supports temporal ensemble"),
    "diffusion": PolicyDeployDefaults(15.0, 15.0, DEFAULT_GRIPPER_RPC_RATE_HZ, "n_obs_steps history + denoising, slower"),
    "smolvla": PolicyDeployDefaults(10.0, 10.0, DEFAULT_GRIPPER_RPC_RATE_HZ, "VLA; requires --task; GPU inference slow"),
    "pi0": PolicyDeployDefaults(8.0, 8.0, DEFAULT_GRIPPER_RPC_RATE_HZ, "VLA; requires --task; very slow"),
    "pi05": PolicyDeployDefaults(8.0, 8.0, DEFAULT_GRIPPER_RPC_RATE_HZ, "VLA; requires --task; very slow"),
    "pi0_fast": PolicyDeployDefaults(10.0, 10.0, DEFAULT_GRIPPER_RPC_RATE_HZ, "VLA (FAST decode); requires --task"),
}

_FALLBACK_DEFAULTS = PolicyDeployDefaults(
    DEFAULT_CONTROL_FPS, DEFAULT_ARM_RPC_RATE_HZ, DEFAULT_GRIPPER_RPC_RATE_HZ, "generic"
)

SUPPORTED_POLICY_TYPES = tuple(POLICY_DEPLOY_DEFAULTS.keys())


def policy_type_name(policy) -> str:
    """Return the LeRobot policy type string (config.type), lowercased."""
    ptype = getattr(policy.config, "type", None)
    if not ptype:
        ptype = getattr(policy, "name", None) or type(policy).__name__
    return str(ptype).lower()


def _deploy_defaults(ptype: str) -> PolicyDeployDefaults:
    return POLICY_DEPLOY_DEFAULTS.get(ptype, _FALLBACK_DEFAULTS)


def validate_inference_cli_args(policy, args) -> None:
    """Fail fast on policy/argument mismatches before touching the robot."""
    ptype = policy_type_name(policy)

    # Explicit --policy-type must match the checkpoint.
    requested = getattr(args, "policy_type", "auto")
    if requested and requested != "auto" and requested != ptype:
        raise ValueError(
            f"--policy-type {requested!r} does not match checkpoint type {ptype!r} "
            f"(config.json 'type'). Omit --policy-type to auto-detect."
        )

    # ACT-only knobs.
    if ptype not in ACT_ONLY_POLICIES:
        if getattr(args, "temporal_ensemble_coeff", None) is not None:
            raise ValueError(f"--temporal-ensemble-coeff is ACT-only (checkpoint type={ptype!r}).")
        if getattr(args, "refresh_policy_every_step", False):
            raise ValueError(f"--refresh-policy-every-step is ACT-only (checkpoint type={ptype!r}).")

    # VLA policies need a real language task.
    if ptype in LANGUAGE_TASK_POLICIES:
        task = getattr(args, "task", None)
        if not task or str(task).strip() == "" or str(task).strip() == _PLACEHOLDER_TASK:
            raise ValueError(
                f"Policy type {ptype!r} is a VLA and requires a language task. "
                f'Pass e.g. --task "Pick up the yellow ring and place it on the white plate."'
            )


def apply_policy_hardware_defaults(policy, args) -> None:
    """Fill fps / RPC rates from per-policy defaults when the user left them unset.

    CLI args default to None so we can distinguish "unset" from an explicit value.
    --no-policy-defaults uses the generic fallback instead of per-policy tuning.
    """
    ptype = policy_type_name(policy)
    use_policy_defaults = not getattr(args, "no_policy_defaults", False)
    defaults = _deploy_defaults(ptype) if use_policy_defaults else _FALLBACK_DEFAULTS

    if getattr(args, "fps", None) is None:
        args.fps = defaults.fps
    if getattr(args, "arm_rpc_rate_hz", None) is None:
        # Keep arm RPC aligned to control fps by default.
        args.arm_rpc_rate_hz = min(defaults.arm_rpc_rate_hz, args.fps)
    if getattr(args, "gripper_rpc_rate_hz", None) is None:
        args.gripper_rpc_rate_hz = defaults.gripper_rpc_rate_hz


def print_policy_deploy_summary(policy, args) -> None:
    ptype = policy_type_name(policy)
    defaults = _deploy_defaults(ptype)
    known = ptype in POLICY_DEPLOY_DEFAULTS
    cfg = policy.config

    chunk_size = getattr(cfg, "chunk_size", None)
    horizon = getattr(cfg, "horizon", None)
    n_obs = getattr(cfg, "n_obs_steps", None)
    n_action = getattr(cfg, "n_action_steps", None)

    print(f"{BOLD_GREEN}[POLICY] type={ptype} ({'known' if known else 'generic fallback'}) — {defaults.note}{RESET}")
    shape_bits = []
    if chunk_size is not None:
        shape_bits.append(f"chunk_size={chunk_size}")
    if horizon is not None:
        shape_bits.append(f"horizon={horizon}")
    if n_obs is not None:
        shape_bits.append(f"n_obs_steps={n_obs}")
    if n_action is not None:
        shape_bits.append(f"n_action_steps={n_action}")
    if shape_bits:
        print(f"{BOLD_GREEN}[POLICY] {' '.join(shape_bits)}{RESET}")
    print(
        f"{BOLD_GREEN}[POLICY] deploy: fps={args.fps:g} arm_rpc={args.arm_rpc_rate_hz:g}Hz "
        f"gripper_rpc={args.gripper_rpc_rate_hz:g}Hz{RESET}"
    )
    if ptype in LANGUAGE_TASK_POLICIES:
        print(f'{BOLD_GREEN}[POLICY] task="{args.task}"{RESET}')
