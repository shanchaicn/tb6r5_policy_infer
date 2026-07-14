"""LeRobot policy loading and deployment-time overrides.

Supports ACT / Diffusion / SmolVLA / pi0 / pi0.5 (pi05) / pi0-FAST.
"""

from __future__ import annotations

from collections import deque

from .deploy import (
    apply_policy_hardware_defaults,
    policy_type_name,
    print_policy_deploy_summary,
    validate_inference_cli_args,
)
from .lerobot_compat import import_policy_factory, load_pretrained_config, resolve_inference_device
from .constants import INFER_LOG_PREFIX


def load_policy_components(policy_path: str, dataset_root: str | None, repo_id: str | None, device: str):
    get_policy_class, make_policy, make_pre_post_processors = import_policy_factory()
    device = resolve_inference_device(device)
    cfg = load_pretrained_config(policy_path)
    cfg.pretrained_path = policy_path
    cfg.device = device

    dataset_stats = None
    if dataset_root and repo_id:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_root)
        dataset_stats = ds_meta.stats
        policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    else:
        policy_cls = get_policy_class(cfg.type)
        policy = policy_cls.from_pretrained(policy_path, config=cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    return policy, preprocessor, postprocessor


def prepare_policy_for_hardware(policy, args) -> None:
    """Validate CLI, apply per-policy hardware defaults, then inference overrides."""
    validate_inference_cli_args(policy, args)
    apply_policy_hardware_defaults(policy, args)
    apply_policy_inference_overrides(
        policy,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        refresh_policy_every_step=args.refresh_policy_every_step,
    )
    print_policy_deploy_summary(policy, args)


def apply_policy_inference_overrides(
    policy,
    *,
    n_action_steps: int | None,
    temporal_ensemble_coeff: float | None,
    refresh_policy_every_step: bool,
) -> None:
    ptype = policy_type_name(policy)
    if ptype == "act":
        apply_act_inference_overrides(
            policy,
            n_action_steps=n_action_steps,
            temporal_ensemble_coeff=temporal_ensemble_coeff,
            refresh_policy_every_step=refresh_policy_every_step,
        )
    elif ptype == "diffusion":
        apply_diffusion_inference_overrides(policy, n_action_steps=n_action_steps)
    elif ptype in ("smolvla", "pi0", "pi05", "pi0_fast"):
        # All chunk-queue VLAs: n_action_steps must be in [1, chunk_size].
        label = {"smolvla": "SmolVLA", "pi0": "pi0", "pi05": "pi0.5", "pi0_fast": "pi0-FAST"}[ptype]
        apply_chunk_queue_inference_overrides(policy, n_action_steps=n_action_steps, label=label)
    elif n_action_steps is not None:
        print(f"[{ptype}] Ignoring --n-action-steps (not supported for type={ptype!r}).")


def apply_act_inference_overrides(
    policy,
    *,
    n_action_steps: int | None,
    temporal_ensemble_coeff: float | None,
    refresh_policy_every_step: bool,
) -> None:
    """Apply deployment-time ACT inference settings (no retraining required)."""
    chunk_size = int(policy.config.chunk_size)
    ckpt_n_action = int(policy.config.n_action_steps)

    if temporal_ensemble_coeff is not None:
        if refresh_policy_every_step:
            raise ValueError(
                "--temporal-ensemble-coeff and --refresh-policy-every-step are incompatible "
                "(reset every step destroys the temporal ensemble buffer)."
            )
        if temporal_ensemble_coeff < 0:
            raise ValueError("--temporal-ensemble-coeff must be >= 0")
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        policy.config.temporal_ensemble_coeff = float(temporal_ensemble_coeff)
        policy.config.n_action_steps = 1
        policy.temporal_ensembler = ACTTemporalEnsembler(float(temporal_ensemble_coeff), chunk_size)
        policy.reset()
        print(
            f"[{INFER_LOG_PREFIX}] Temporal Ensemble ON: coeff={temporal_ensemble_coeff:g}, "
            f"chunk_size={chunk_size}, every-step inference"
        )
        return

    new_n_action = ckpt_n_action if n_action_steps is None else int(n_action_steps)
    if not 1 <= new_n_action <= chunk_size:
        raise ValueError(f"--n-action-steps must be in [1, {chunk_size}], got {new_n_action}")

    if new_n_action != ckpt_n_action:
        policy.config.n_action_steps = new_n_action
        if hasattr(policy, "_action_queue") and isinstance(policy._action_queue, deque):
            policy._action_queue = deque([], maxlen=new_n_action)
        policy.reset()
        print(
            f"[{INFER_LOG_PREFIX}] n_action_steps override: {ckpt_n_action} -> {new_n_action} "
            f"(chunk_size={chunk_size}, re-infer every {new_n_action} control steps)"
        )
    elif refresh_policy_every_step:
        print(f"[{INFER_LOG_PREFIX}] Action queue chunk_size={chunk_size}, n_action_steps={new_n_action}, refresh every step")


def apply_diffusion_inference_overrides(policy, *, n_action_steps: int | None) -> None:
    """Override Diffusion n_action_steps (LeRobot: must be <= horizon - n_obs_steps + 1)."""
    horizon = int(policy.config.horizon)
    n_obs = int(policy.config.n_obs_steps)
    max_n = horizon - n_obs + 1
    ckpt_n_action = int(policy.config.n_action_steps)
    new_n_action = ckpt_n_action if n_action_steps is None else int(n_action_steps)
    if not 1 <= new_n_action <= max_n:
        raise ValueError(
            f"--n-action-steps must be in [1, {max_n}] for diffusion "
            f"(horizon={horizon}, n_obs_steps={n_obs}), got {new_n_action}"
        )
    if new_n_action != ckpt_n_action:
        policy.config.n_action_steps = new_n_action
        policy.reset()
        print(
            f"[Diffusion] n_action_steps override: {ckpt_n_action} -> {new_n_action} "
            f"(horizon={horizon}, n_obs_steps={n_obs})"
        )
    else:
        print(
            f"[Diffusion] Using checkpoint queue: n_action_steps={ckpt_n_action}, "
            f"horizon={horizon}, n_obs_steps={n_obs}"
        )


def apply_chunk_queue_inference_overrides(policy, *, n_action_steps: int | None, label: str) -> None:
    """Override SmolVLA (and similar chunk-queue policies) n_action_steps."""
    chunk_size = int(policy.config.chunk_size)
    ckpt_n_action = int(policy.config.n_action_steps)
    new_n_action = ckpt_n_action if n_action_steps is None else int(n_action_steps)
    if not 1 <= new_n_action <= chunk_size:
        raise ValueError(f"--n-action-steps must be in [1, {chunk_size}], got {new_n_action}")
    if new_n_action != ckpt_n_action:
        policy.config.n_action_steps = new_n_action
        policy.reset()
        print(f"[{label}] n_action_steps override: {ckpt_n_action} -> {new_n_action} " f"(chunk_size={chunk_size})")
    else:
        print(f"[{label}] Using checkpoint queue: chunk_size={chunk_size}, n_action_steps={ckpt_n_action}")


def policy_action_queue_info(policy) -> tuple[int | None, int | None]:
    """Return (step_index_in_queue, queue_len) for policies with action queues."""
    ptype = policy_type_name(policy)

    if ptype == "act":
        if getattr(policy.config, "temporal_ensemble_coeff", None) is not None:
            return None, None
        queue_len = getattr(policy.config, "n_action_steps", None)
        queue = getattr(policy, "_action_queue", None)
        if queue_len is None or queue is None:
            return None, queue_len
        remaining = len(queue)
        step_index = max(int(queue_len) - remaining - 1, 0)
        return step_index, queue_len

    queue_len = getattr(policy.config, "n_action_steps", None)
    if queue_len is None:
        return None, None

    queue = getattr(policy, "_action_queue", None)
    if queue is None and hasattr(policy, "_queues"):
        from lerobot.utils.constants import ACTION

        queue = policy._queues.get(ACTION)

    if queue is None:
        return None, queue_len

    remaining = len(queue)
    step_index = max(int(queue_len) - remaining - 1, 0)
    return step_index, queue_len


# Backward-compatible alias
act_chunk_info = policy_action_queue_info
