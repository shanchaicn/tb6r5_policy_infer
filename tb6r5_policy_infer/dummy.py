"""Identity / wiggle dummy policy for camera + robot hardware bring-up.

Does not load LeRobot or a checkpoint. Action is built from the current
observation so you can verify Topic/RPC/cameras without a trained model.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from .constants import INFER_LOG_PREFIX
from .ee_pose import EE_POSE_DIM


class DummyPolicy:
    """Hold current pose, or add a small sinusoidal TCP/joint wiggle."""

    def __init__(
        self,
        *,
        action_space: str,
        mode: str = "hold",
        wiggle_amp_m: float = 0.02,
        wiggle_period_s: float = 6.0,
        wiggle_joint_rad: float = 0.03,
    ):
        mode = str(mode).strip().lower()
        if mode not in ("hold", "wiggle"):
            raise ValueError(f"--dummy-mode must be 'hold' or 'wiggle', got {mode!r}")
        self.action_space = action_space
        self.mode = mode
        self.wiggle_amp_m = float(wiggle_amp_m)
        self.wiggle_period_s = max(float(wiggle_period_s), 1e-3)
        self.wiggle_joint_rad = float(wiggle_joint_rad)
        self._t0: float | None = None
        self.config = SimpleNamespace(
            type="dummy",
            device="cpu",
            chunk_size=1,
            n_action_steps=1,
            input_features={},
        )

    def reset(self) -> None:
        self._t0 = None

    def predict(self, observation: dict, now_s: float) -> np.ndarray:
        state = np.asarray(observation["observation.state"], dtype=np.float32).ravel()
        action = state.copy()
        if self.mode != "wiggle":
            return action
        if self._t0 is None:
            self._t0 = now_s
        phase = 2.0 * np.pi * (now_s - self._t0) / self.wiggle_period_s
        delta = float(np.sin(phase))
        if self.action_space == "ee_pose":
            if action.shape[0] < EE_POSE_DIM:
                raise ValueError(f"dummy ee_pose state dim {action.shape}")
            # Wiggle world-Y (table plane), keep height / quat / gripper.
            action[1] = action[1] + self.wiggle_amp_m * delta
        else:
            if action.shape[0] < 7:
                raise ValueError(f"dummy joints state dim {action.shape}")
            action[0] = action[0] + self.wiggle_joint_rad * delta
        return action.astype(np.float32)


def dummy_action_tensor(action: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(action, dtype=np.float32)).unsqueeze(0)


def apply_dummy_loop_defaults(args) -> None:
    """Fill fps / RPC rates when no real policy defaults apply."""
    if getattr(args, "fps", None) is None:
        args.fps = 10.0
    if getattr(args, "arm_rpc_rate_hz", None) is None:
        args.arm_rpc_rate_hz = float(args.fps)
    if getattr(args, "gripper_rpc_rate_hz", None) is None:
        if int(getattr(args, "g_model", 3)) == 3:
            args.gripper_rpc_rate_hz = float(args.arm_rpc_rate_hz)
        else:
            args.gripper_rpc_rate_hz = 2.0
    print(
        f"[{INFER_LOG_PREFIX}] DUMMY policy: no checkpoint. "
        f"mode={getattr(args, 'dummy_mode', 'hold')} "
        f"action_space={args.action_space} fps={args.fps:.0f} "
        f"(hold=stay; wiggle=small sinusoid to verify tracking)",
        flush=True,
    )


def save_camera_snapshots(observation: dict, camera_names: list[str], out_dir: str = "/tmp/tb6r5_dummy") -> None:
    from pathlib import Path

    import cv2

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for name in camera_names:
        key = f"observation.images.{name}"
        rgb = observation.get(key)
        if rgb is None:
            continue
        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        path = dest / f"{name}.jpg"
        cv2.imwrite(str(path), bgr)
        print(f"[{INFER_LOG_PREFIX}][camera] snapshot {path} mean={float(np.asarray(rgb).mean()):.1f}", flush=True)
