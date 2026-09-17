"""End-effector pose helpers for pi05-style 8D observation/action.

Layout (policy / training convention):
  [x, y, z, qx, qy, qz, qw, gripper_m]
"""

from __future__ import annotations

import numpy as np

EE_POSE_DIM = 8


def canonicalize_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    """Force quaternion into the positive-w hemisphere (q and -q are the same rotation)."""
    q = np.asarray(quat_xyzw, dtype=np.float32).ravel()
    if len(q) < 4:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    q = q[:4].copy()
    if float(q[3]) < 0.0:
        q *= -1.0
    return q.astype(np.float32, copy=False)


def pack_ee_state_xyzw(
    xyz: np.ndarray,
    quat_xyzw: np.ndarray,
    gripper_m: float,
) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32).ravel()[:3]
    quat = canonicalize_quat_xyzw(quat_xyzw)
    return np.concatenate(
        [xyz, quat, np.array([float(gripper_m)], dtype=np.float32)],
        axis=0,
    ).astype(np.float32)


def unpack_ee_action_xyzw(action: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    a = np.asarray(action, dtype=np.float32).ravel()
    if a.shape[0] < EE_POSE_DIM:
        raise ValueError(f"Expected ee_pose action dim >= {EE_POSE_DIM}, got {a.shape}")
    xyz = a[:3].copy()
    quat_xyzw = canonicalize_quat_xyzw(a[3:7])
    gripper_m = float(a[7])
    return xyz, quat_xyzw, gripper_m


def clamp_ee_step(
    xyz_target: np.ndarray,
    xyz_current: np.ndarray,
    max_step_m: float | None,
) -> np.ndarray:
    """Clamp TCP translation step (meters). Orientation is passed through unchanged."""
    tgt = np.asarray(xyz_target, dtype=np.float32).ravel()[:3].copy()
    if max_step_m is None or max_step_m <= 0:
        return tgt
    cur = np.asarray(xyz_current, dtype=np.float32).ravel()[:3]
    delta = tgt - cur
    n = float(np.linalg.norm(delta))
    if n > float(max_step_m) and n > 1e-12:
        tgt = cur + delta * (float(max_step_m) / n)
    return tgt.astype(np.float32)


def parse_xyz_limit(value, *, name: str) -> np.ndarray | None:
    """Parse YAML/CLI xyz limit: None / empty → disabled; else 3 floats."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)) and not str(value).strip():
        return None
    arr = np.asarray(value, dtype=np.float32).ravel()
    if arr.size == 0:
        return None
    if arr.size != 3:
        raise ValueError(f"{name} must have 3 values [x, y, z], got {arr.size}: {value!r}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {arr}")
    return arr.astype(np.float32)


def validate_ee_workspace(xyz_min: np.ndarray | None, xyz_max: np.ndarray | None) -> None:
    if xyz_min is None or xyz_max is None:
        return
    if np.any(xyz_min > xyz_max):
        raise ValueError(
            f"ee workspace min must be <= max per axis, got min={xyz_min.tolist()} max={xyz_max.tolist()}"
        )


def clamp_ee_workspace(
    xyz: np.ndarray,
    xyz_min: np.ndarray | None,
    xyz_max: np.ndarray | None,
) -> tuple[np.ndarray, bool, np.ndarray]:
    """Clamp TCP xyz into an axis-aligned box.

    Returns ``(clamped_xyz, hit, overflow)`` where ``overflow = raw - clamped``
    (positive means above max / below min after clamp).
    """
    tgt = np.asarray(xyz, dtype=np.float32).ravel()[:3].copy()
    raw = tgt.copy()
    if xyz_min is None and xyz_max is None:
        return tgt, False, np.zeros(3, dtype=np.float32)
    if xyz_min is not None:
        tgt = np.maximum(tgt, np.asarray(xyz_min, dtype=np.float32).ravel()[:3])
    if xyz_max is not None:
        tgt = np.minimum(tgt, np.asarray(xyz_max, dtype=np.float32).ravel()[:3])
    overflow = raw - tgt
    hit = bool(np.any(np.abs(overflow) > 1e-9))
    return tgt.astype(np.float32), hit, overflow.astype(np.float32)


def gripper_m_to_mm(gripper_m: float) -> float:
    return float(gripper_m) * 1000.0


def gripper_mm_to_m(gripper_mm: float) -> float:
    return float(gripper_mm) / 1000.0
