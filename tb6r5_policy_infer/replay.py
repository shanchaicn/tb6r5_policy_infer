"""Replay one LeRobot episode's actions (or states) as hardware RPC targets."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .constants import INFER_LOG_PREFIX


def _dataset_info(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset (missing {info_path})")
    return json.loads(info_path.read_text(encoding="utf-8"))


def dataset_fps(root: Path) -> float | None:
    info = _dataset_info(root)
    fps = info.get("fps")
    return float(fps) if fps else None


def _row_vec(value) -> np.ndarray:
    if isinstance(value, dict):
        value = list(value.values())
    return np.asarray(value, dtype=np.float32).ravel()


def load_episode_vectors(
    dataset_root: str | Path,
    episode_index: int,
    *,
    source: str = "action",
) -> np.ndarray:
    """Load [T, D] vectors for one episode from parquet (no video decode)."""
    source = str(source).strip().lower()
    if source not in ("action", "state"):
        raise ValueError(f"--replay-source must be 'action' or 'state', got {source!r}")
    col = "action" if source == "action" else "observation.state"

    root = Path(dataset_root).expanduser().resolve()
    info = _dataset_info(root)
    total_ep = int(info.get("total_episodes", 0))
    if episode_index < 0 or (total_ep and episode_index >= total_ep):
        raise ValueError(f"episode_index {episode_index} out of range [0, {max(total_ep - 1, 0)}]")

    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet under {root / 'data'}. Copy the full dataset (data/ + meta/), not meta only."
        )

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to replay episodes (pip install pandas pyarrow)") from exc

    chunks: list = []
    extra_cols = [c for c in ("episode_index", col, "frame_index") if True]
    for path in files:
        try:
            df = pd.read_parquet(path, columns=extra_cols)
        except Exception:
            df = pd.read_parquet(path)
            missing = [c for c in (col, "episode_index") if c not in df.columns]
            if missing:
                raise ValueError(f"{path} missing columns {missing}") from None
        sub = df[df["episode_index"] == int(episode_index)]
        if len(sub):
            chunks.append(sub)
    if not chunks:
        raise ValueError(f"episode {episode_index} not found in {root}")

    ep = pd.concat(chunks, ignore_index=True)
    if "frame_index" in ep.columns:
        ep = ep.sort_values("frame_index")
    vecs = np.stack([_row_vec(v) for v in ep[col].tolist()], axis=0)
    if vecs.ndim != 2 or vecs.shape[0] == 0:
        raise ValueError(f"episode {episode_index} produced empty {col} array: {vecs.shape}")
    print(
        f"[{INFER_LOG_PREFIX}] Replay episode={episode_index} source={col} "
        f"frames={vecs.shape[0]} dim={vecs.shape[1]} fps={info.get('fps')}",
        flush=True,
    )
    return vecs.astype(np.float32)


class ReplayPolicy:
    """Step through recorded actions at the control-loop rate (one row per tick)."""

    def __init__(self, actions: np.ndarray, *, loop: bool = False, episode_index: int = 0, source: str = "action"):
        self.actions = np.asarray(actions, dtype=np.float32)
        if self.actions.ndim != 2:
            raise ValueError(f"replay actions must be [T, D], got {self.actions.shape}")
        self.loop = bool(loop)
        self.episode_index = int(episode_index)
        self.source = source
        self.i = 0
        self.finished = False
        self.config = SimpleNamespace(
            type="replay",
            device="cpu",
            chunk_size=1,
            n_action_steps=1,
            input_features={},
        )

    def reset(self) -> None:
        self.i = 0
        self.finished = False

    def predict(self, observation: dict, now_s: float) -> np.ndarray:
        del observation, now_s
        n = int(self.actions.shape[0])
        if n == 0:
            self.finished = True
            raise RuntimeError("replay episode has 0 frames")
        if self.i >= n:
            if self.loop:
                self.i = 0
            else:
                self.finished = True
                return self.actions[-1].copy()
        action = self.actions[self.i].copy()
        self.i += 1
        if self.i >= n and not self.loop:
            self.finished = True
        return action
