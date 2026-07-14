"""CLI entry point for offline ACT policy evaluation."""

from __future__ import annotations

import argparse

from .eval import run_eval

_DOC = """
Offline evaluation for a trained ACT policy on an existing LeRobot TB6-R5 dataset.

Metric:
  - action MAE (overall + per-dimension) between policy prediction and dataset action.
  - inference latency / FPS (printed after eval; use --benchmark-only for a dedicated speed test).

This helps validate deployment readiness before hardware rollout.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=_DOC)
    parser.add_argument("--policy-path", required=True, help="Path (or HF repo) of ACT pretrained checkpoint")
    parser.add_argument("--dataset-root", required=True, help="LeRobot dataset root")
    parser.add_argument("--repo-id", required=True, help="LeRobot repo_id")
    parser.add_argument("--task", default="tb6r5 teleoperation", help="Task string passed to policy")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto (default), cuda, or cpu. Falls back to CPU if CUDA is unavailable.",
    )
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum number of samples to evaluate")
    parser.add_argument("--stride", type=int, default=5, help="Evaluate every N-th sample")
    parser.add_argument(
        "--warmup-samples",
        type=int,
        default=10,
        help="Warmup inferences excluded from FPS stats (default: 10)",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Only measure inference speed (skip MAE and plots)",
    )
    parser.add_argument(
        "--bench-samples",
        type=int,
        default=200,
        help="Number of timed inferences in --benchmark-only mode (default: 200)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save comparison plots (default: outputs/eval_act/<dataset_name>)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run_eval(args)


def cli() -> None:
    raise SystemExit(main())
