"""Offline evaluation for a trained ACT policy on an existing LeRobot TB6-R5 dataset."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from .lerobot_compat import import_policy_factory, load_pretrained_config, predict_action, resolve_inference_device


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def to_hwc_uint8(x: Any) -> np.ndarray:
    arr = to_numpy(x)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0:
                arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def plot_comparison_curves(
    indices: list[int],
    preds: np.ndarray,
    targets: np.ndarray,
    abs_err: np.ndarray,
    mae_per_dim: np.ndarray,
    output_dir: Path,
    stride: int,
    fps: float,
) -> None:
    """Plot predicted vs ground-truth action curves and per-dimension MAE bar chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_dims = preds.shape[1]
    x = np.arange(len(indices)) * stride / fps

    fig, axes = plt.subplots(n_dims, 1, figsize=(14, max(2.0 * n_dims, 6.0)), sharex=True)
    if n_dims == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(x, targets[:, i], label="ground truth", linewidth=1.5, color="C0")
        ax.plot(x, preds[:, i], label="prediction", linewidth=1.0, linestyle="--", color="C1")
        ax.fill_between(x, targets[:, i], preds[:, i], alpha=0.15, color="C3")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylabel(f"joint {i + 1}\n(MAE={mae_per_dim[i]:.4f})")

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"ACT offline eval: prediction vs ground truth (n={len(indices)}, stride={stride})")
    fig.tight_layout()
    curve_path = output_dir / "action_comparison.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    dims = np.arange(n_dims)
    ax2.bar(dims, mae_per_dim, color="C2", alpha=0.8)
    ax2.set_xlabel("action dimension")
    ax2.set_ylabel("MAE")
    ax2.set_title(f"Per-dimension MAE (overall={mae_per_dim.mean():.6f})")
    ax2.set_xticks(dims)
    ax2.set_xticklabels([f"joint {i + 1}" for i in dims], rotation=30, ha="right")
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    mae_path = output_dir / "mae_per_dim.png"
    fig2.savefig(mae_path, dpi=150)
    plt.close(fig2)

    err_mean = abs_err.mean(axis=1)
    fig3, ax3 = plt.subplots(figsize=(12, 3))
    ax3.plot(x, err_mean, linewidth=1.0, color="C3")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("mean abs error")
    ax3.set_title("Mean absolute error over action dimensions")
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    err_path = output_dir / "error_over_time.png"
    fig3.savefig(err_path, dpi=150)
    plt.close(fig3)

    print(f"Saved comparison curves: {curve_path}")
    print(f"Saved MAE bar chart:     {mae_path}")
    print(f"Saved error curve:       {err_path}")


def sync_inference_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def build_observation(sample: dict, image_keys: list[str], state_key: str) -> dict[str, np.ndarray]:
    obs = {state_key: to_numpy(sample[state_key]).astype(np.float32)}
    for k in image_keys:
        obs[k] = to_hwc_uint8(sample[k])
    return obs


def run_single_inference(
    sample: dict,
    *,
    image_keys: list[str],
    state_key: str,
    policy,
    device: torch.device,
    preprocessor,
    postprocessor,
    task: str,
) -> np.ndarray:
    obs = build_observation(sample, image_keys, state_key)
    policy.reset()
    sync_inference_device(device)
    t0 = time.perf_counter()
    pred = predict_action(
        observation=obs,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=False,
        task=task,
        robot_type="tb6r5",
    )
    sync_inference_device(device)
    latency_s = time.perf_counter() - t0
    return to_numpy(pred).squeeze(0).astype(np.float32), latency_s


def print_inference_benchmark(latencies_s: list[float], *, warmup_samples: int = 0) -> None:
    if not latencies_s:
        print("[benchmark] No timed samples recorded.")
        return
    arr = np.asarray(latencies_s, dtype=np.float64)
    mean_s = float(arr.mean())
    fps = 1.0 / mean_s if mean_s > 0 else float("inf")
    print("==== Inference Speed ====")
    if warmup_samples > 0:
        print(f"warmup:    {warmup_samples} (excluded from stats)")
    print(f"samples:   {len(arr)}")
    print(f"mean:      {mean_s * 1000:.2f} ms  ({fps:.2f} FPS)")
    print(f"min:       {float(arr.min()) * 1000:.2f} ms  ({1.0 / float(arr.min()):.2f} FPS)")
    print(f"max:       {float(arr.max()) * 1000:.2f} ms  ({1.0 / float(arr.max()):.2f} FPS)")
    print(f"p50:       {float(np.percentile(arr, 50)) * 1000:.2f} ms")
    print(f"p95:       {float(np.percentile(arr, 95)) * 1000:.2f} ms")
    print("=========================")


def run_inference_benchmark(args, *, policy, preprocessor, postprocessor, dataset, image_keys, state_key) -> int:
    if args.warmup_samples < 0:
        raise ValueError("--warmup-samples must be >= 0")
    if args.bench_samples <= 0:
        raise ValueError("--bench-samples must be > 0")

    device = torch.device(policy.config.device)
    n_total = len(dataset)
    if n_total == 0:
        raise ValueError("Dataset is empty.")

    warmup_indices = list(range(min(args.warmup_samples, n_total)))
    for idx in warmup_indices:
        run_single_inference(
            dataset[idx],
            image_keys=image_keys,
            state_key=state_key,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task=args.task,
        )

    bench_indices = [i % n_total for i in range(args.bench_samples)]
    latencies_s: list[float] = []
    for idx in bench_indices:
        _, latency_s = run_single_inference(
            dataset[idx],
            image_keys=image_keys,
            state_key=state_key,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task=args.task,
        )
        latencies_s.append(latency_s)

    print("==== ACT Inference Benchmark (TB6R5 / LeRobot) ====")
    print(f"dataset_root: {args.dataset_root}")
    print(f"repo_id:      {args.repo_id}")
    print(f"policy_path:  {args.policy_path}")
    print(f"device:       {policy.config.device}")
    print_inference_benchmark(latencies_s, warmup_samples=len(warmup_indices))
    return 0


def resolve_dataset_root(dataset_root: str | Path) -> Path:
    """Resolve dataset path and fail fast when offline metadata is missing."""
    root = Path(dataset_root).expanduser().resolve()
    info_json = root / "meta" / "info.json"
    if info_json.is_file():
        return root

    raise FileNotFoundError(
        "LeRobot dataset not found locally.\n"
        f"  dataset-root: {root}\n"
        f"  missing:      {info_json}\n"
        "Offline eval requires the full dataset copied to this machine "
        "(meta/, data/, videos/). Example from your dev PC:\n"
        "  rsync -avP ~/study/XRoboToolkit-Teleop-Sample-Python/data/lerobot/tb6r5_yellow_yogurt_47_v3/ \\\n"
        f"    root@TER30:/opt/lerobot/chaishan/data/lerobot/tb6r5_yellow_yogurt_47_v3/"
    )


def run_eval(args) -> int:
    if getattr(args, "benchmark_only", False):
        if args.warmup_samples < 0:
            raise ValueError("--warmup-samples must be >= 0")
        if args.bench_samples <= 0:
            raise ValueError("--bench-samples must be > 0")
    else:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be > 0")
        if args.stride <= 0:
            raise ValueError("--stride must be > 0")
        if args.warmup_samples < 0:
            raise ValueError("--warmup-samples must be >= 0")

    device = resolve_inference_device(args.device)
    cfg = load_pretrained_config(args.policy_path)
    cfg.pretrained_path = args.policy_path
    cfg.device = device

    dataset_root = resolve_dataset_root(args.dataset_root)
    _, make_policy, make_pre_post_processors = import_policy_factory()
    ds_meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=dataset_root)
    dataset = LeRobotDataset(repo_id=args.repo_id, root=dataset_root)

    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=args.policy_path,
        dataset_stats=ds_meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    image_keys = sorted(
        [k for k in ds_meta.features.keys() if k.startswith("observation.images.") and ".depth" not in k]
    )
    state_key = "observation.state"
    action_key = "action"

    if getattr(args, "benchmark_only", False):
        return run_inference_benchmark(
            args,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=dataset,
            image_keys=image_keys,
            state_key=state_key,
        )

    n_total = len(dataset)
    indices = list(range(0, n_total, args.stride))[: args.max_samples]
    if not indices:
        raise ValueError("No samples selected. Check --stride and dataset length.")

    device = torch.device(policy.config.device)
    warmup_indices = list(range(min(getattr(args, "warmup_samples", 0), n_total)))
    for idx in warmup_indices:
        run_single_inference(
            dataset[idx],
            image_keys=image_keys,
            state_key=state_key,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task=args.task,
        )

    abs_err_acc = []
    pred_acc = []
    tgt_acc = []
    latencies_s: list[float] = []
    for idx in indices:
        sample = dataset[idx]
        pred_np, latency_s = run_single_inference(
            sample,
            image_keys=image_keys,
            state_key=state_key,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task=args.task,
        )
        latencies_s.append(latency_s)
        tgt_np = to_numpy(sample[action_key]).astype(np.float32).reshape(-1)
        if pred_np.shape[0] != tgt_np.shape[0]:
            raise ValueError(f"Action dim mismatch: pred={pred_np.shape}, target={tgt_np.shape} at idx={idx}")

        abs_err_acc.append(np.abs(pred_np - tgt_np))
        pred_acc.append(pred_np)
        tgt_acc.append(tgt_np)

    abs_err = np.stack(abs_err_acc, axis=0)
    preds = np.stack(pred_acc, axis=0)
    targets = np.stack(tgt_acc, axis=0)
    mae_per_dim = abs_err.mean(axis=0)
    mae_all = float(mae_per_dim.mean())

    print("==== ACT Offline Evaluation (TB6R5 / LeRobot) ====")
    print(f"dataset_root: {args.dataset_root}")
    print(f"repo_id:      {args.repo_id}")
    print(f"policy_path:  {args.policy_path}")
    print(f"device:       {policy.config.device}")
    print(f"samples:      {len(indices)} / {len(dataset)} (stride={args.stride})")
    print(f"MAE(all):     {mae_all:.6f}")
    for i, v in enumerate(mae_per_dim):
        print(f"MAE(action[{i}]): {float(v):.6f}")
    if mae_per_dim.shape[0] >= 7:
        print(f"MAE(action[6]) gripper distance: {float(mae_per_dim[6]):.4f} mm")
    print("===============================================")
    print_inference_benchmark(latencies_s, warmup_samples=len(warmup_indices))

    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/eval_act") / Path(args.dataset_root).name
    fps = float(ds_meta.fps) if ds_meta.fps else 50.0
    plot_comparison_curves(indices, preds, targets, abs_err, mae_per_dim, output_dir, args.stride, fps)
    return 0
