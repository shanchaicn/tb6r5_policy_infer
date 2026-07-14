"""LeRobot version compatibility helpers."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import types
from pathlib import Path

import draccus
import torch
from lerobot.configs.policies import PreTrainedConfig

from .constants import INFER_LOG_PREFIX


def _install_groot_config_stub() -> None:
    """Register groot config only, skipping groot/__init__.py (and transformers)."""
    import lerobot

    groot_name = "lerobot.policies.groot"
    if groot_name in sys.modules:
        return

    groot_dir = Path(lerobot.__file__).resolve().parent / "policies" / "groot"
    config_name = f"{groot_name}.configuration_groot"
    if config_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(config_name, groot_dir / "configuration_groot.py")
        if spec is None or spec.loader is None:
            return
        config_mod = importlib.util.module_from_spec(spec)
        sys.modules[config_name] = config_mod
        spec.loader.exec_module(config_mod)

    config_mod = sys.modules[config_name]
    groot = types.ModuleType(groot_name)
    groot.__path__ = [str(groot_dir)]
    groot.configuration_groot = config_mod
    groot.GrootConfig = config_mod.GrootConfig
    sys.modules[groot_name] = groot


def _transformers_hub_mismatch_help() -> str | None:
    return (
        "transformers and huggingface-hub versions are incompatible.\n"
        "This project uses lerobot 0.5.x (SmolVLA / pi0 need transformers). Run:\n"
        '  pip install "transformers>=5.3.0,<6.0.0" "huggingface-hub>=1.16.0,<2.0.0"\n'
        "If you must stay on lerobot 0.4.x instead, pin the older pair:\n"
        '  pip install "lerobot==0.4.4" "transformers>=4.57.1,<5.0.0" '
        '"huggingface-hub>=0.34.2,<0.36.0"'
    )


def _import_predict_action():
    _install_groot_config_stub()
    try:
        from lerobot.utils.control_utils import predict_action

        return predict_action
    except ModuleNotFoundError:
        from lerobot.common.control_utils import predict_action

        return predict_action
    except ImportError as exc:
        msg = str(exc)
        if "huggingface-hub" in msg or "huggingface_hub" in msg:
            raise ImportError(_transformers_hub_mismatch_help()) from exc
        raise


predict_action = _import_predict_action()


def _dataclass_field_names(cls: type) -> set[str]:
    names: set[str] = set()
    for base in cls.__mro__:
        if dataclasses.is_dataclass(base):
            names.update(field.name for field in dataclasses.fields(base))
    return names


def load_pretrained_config(policy_path: str | Path) -> PreTrainedConfig:
    """Load policy config, stripping keys added by newer LeRobot versions."""
    policy_path = Path(policy_path)
    try:
        return PreTrainedConfig.from_pretrained(policy_path)
    except Exception as exc:
        if "are not valid for" not in str(exc):
            raise

    config_file = policy_path / "config.json"
    if not config_file.is_file():
        raise exc

    raw = json.loads(config_file.read_text())
    policy_type = raw.get("type")
    if not policy_type:
        raise ValueError(f"config.json missing 'type': {config_file}") from exc

    config_cls = PreTrainedConfig.get_known_choices()[policy_type]
    valid_names = _dataclass_field_names(config_cls)
    filtered = {key: value for key, value in raw.items() if key in valid_names}
    dropped = sorted(set(raw) - set(filtered))
    if dropped:
        print(f"[compat] Dropped unknown config keys for {config_cls.__name__}: {dropped}")

    with draccus.config_type("json"):
        return draccus.decode(config_cls, filtered)


def import_policy_factory():
    """Import policy factory helpers without loading Groot/transformers on lerobot 0.4.x."""
    _install_groot_config_stub()
    try:
        from lerobot.policies.factory import get_policy_class, make_policy, make_pre_post_processors

        return get_policy_class, make_policy, make_pre_post_processors
    except (ImportError, TypeError) as exc:
        msg = str(exc)
        if "is_offline_mode" in msg or "huggingface_hub" in msg:
            raise ImportError(
                "transformers and huggingface-hub are incompatible on this machine.\n"
                "ACT offline eval does not need transformers. On lerobot 0.4.x run:\n"
                "  pip uninstall transformers -y\n"
                "or align versions together:\n"
                '  pip install "transformers>=4.57.1,<5.0.0" "huggingface-hub>=0.34.2,<0.36.0"\n'
                "or upgrade the full stack:\n"
                '  pip install "lerobot>=0.5.1" "transformers>=5.3.0" "huggingface-hub>=1.16.0,<2.0.0"'
            ) from exc
        if "backbone_cfg" in msg or "non-default argument" in msg:
            raise ImportError(
                "lerobot Groot policy failed to import (transformers dataclass conflict).\n"
                "ACT eval does not need Groot. Sync latest tb6r5_policy_infer, or run:\n"
                "  pip uninstall transformers -y\n"
                'or upgrade: pip install "lerobot>=0.5.1" "transformers>=5.3.0" '
                '"huggingface-hub>=1.16.0,<2.0.0"'
            ) from exc
        raise


def _cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except RuntimeError:
        return False


def resolve_inference_device(device: str) -> str:
    """Map ``auto`` to cuda/cpu; fall back to CPU when CUDA is unavailable."""
    requested = (device or "auto").strip().lower()
    if requested == "auto":
        if _cuda_usable():
            return "cuda"
        print(f"[{INFER_LOG_PREFIX}] CUDA unavailable, using CPU (pass --device cuda to override when GPU is ready)")
        return "cpu"
    if requested.startswith("cuda") and not _cuda_usable():
        print(f"[{INFER_LOG_PREFIX}] WARNING: --device {device} requested but CUDA unavailable, using CPU")
        return "cpu"
    return device


__all__ = ["import_policy_factory", "load_pretrained_config", "predict_action", "resolve_inference_device"]
