"""Subprocess worker for parallel GPU rollouts.

Launched by ``run_batch_simulation_multi_gpu`` — each invocation runs a
chunk of environments on a single GPU and writes ``recorded_positions.npz``
into *--output-dir*.

Usage::

    python -m humanoid_dart.dart.rollout_worker \
        --task Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation \
        --motion-file /abs/path/to/batched.npz \
        --output-dir /abs/path/to/sub_dir \
        --num-envs 750 \
        --device cuda:1 \
        --checkpoint /abs/path/to/model.pt \
        --viewer headless
"""

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-GPU rollout worker")
    parser.add_argument("--task", required=True)
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--viewer", default="headless")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--targets-file", default=None)
    args = parser.parse_args()

    from humanoid_dart.scripts.play import PlayConfig, run_play

    batch_targets = None
    if args.targets_file:
        with open(args.targets_file) as f:
            batch_targets = json.load(f)

    cfg_kwargs: dict = {
        "motion_file": args.motion_file,
        "num_envs": args.num_envs,
        "device": args.device,
        "viewer": args.viewer,
        "output_dir": args.output_dir,
        "skip_onnx_export": True,
    }
    if args.checkpoint:
        cfg_kwargs["checkpoint_file"] = args.checkpoint
    if args.max_steps is not None:
        cfg_kwargs["max_steps"] = args.max_steps

    cfg = PlayConfig(**cfg_kwargs)
    run_play(args.task, cfg, batch_target_values=batch_targets)


if __name__ == "__main__":
    main()
