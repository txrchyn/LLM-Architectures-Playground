#!/usr/bin/env python
"""
Convenience entry point for evaluating local checkpoints with lm-evaluation-harness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from lm_eval import evaluator
except ImportError as exc:  # pragma: no cover - helper message
    raise SystemExit(
        "lm-evaluation-harness is not installed. Install it via "
        "`pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git@main`"
    ) from exc

# Ensure local adapter is registered with the harness registry.
import benchmark  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lm-evaluation-harness on local checkpoints.")
    parser.add_argument(
        "--model-name",
        default="modern",
        help="Model registry name used during training (e.g. modern, vanilla, gqa, linformer, ssm, moe).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=False,
        help="Path to a training checkpoint (.pth). Required unless --config is provided.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional path to a Python config (used when no checkpoint is supplied).",
    )
    parser.add_argument(
        "--tasks",
        required=True,
        help="Comma-separated list of lm-evaluation-harness tasks (e.g. hellaswag,arc_easy).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-forward batch size used by the local adapter.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Optional harness batch size override (defaults to --batch-size).",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Computation dtype for the model (float32, bfloat16, float16).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string, e.g. cuda:0 or cpu. Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Default generation length for tasks that request generate_until.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Optional limit on the number of evaluation samples per task (int or fraction).",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples to use (inherits harness defaults when omitted).",
    )
    parser.add_argument(
        "--use-cache",
        type=Path,
        default=None,
        help="Path to a sqlite cache file for re-using model outputs.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write raw harness results as JSON.",
    )
    parser.add_argument(
        "--verbosity",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Harness logging verbosity.",
    )
    parser.add_argument(
        "--gen-kwargs",
        default=None,
        help="Optional JSON string of generation kwargs forwarded to harness (e.g. '{\"temperature\":0.7}').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.checkpoint is None and args.config is None:
        raise SystemExit("Please provide either --checkpoint or --config to instantiate the model.")

    tasks: List[str] = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        raise SystemExit("No valid tasks specified.")

    model_args: Dict[str, Any] = {
        "model_name": args.model_name,
        "checkpoint_path": str(args.checkpoint) if args.checkpoint else None,
        "config_path": str(args.config) if args.config else None,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
    }

    gen_kwargs = None
    if args.gen_kwargs:
        try:
            gen_kwargs = json.loads(args.gen_kwargs)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Failed to parse --gen-kwargs JSON: {exc}") from exc

    results = evaluator.simple_evaluate(
        model="local",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.eval_batch_size or args.batch_size,
        device=args.device,
        use_cache=str(args.use_cache) if args.use_cache else None,
        limit=args.limit,
        verbosity=args.verbosity,
        gen_kwargs=gen_kwargs,
    )

    print(json.dumps(results, indent=2))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
